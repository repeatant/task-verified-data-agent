"""Text-JSON model adapters for the task-verified LangGraph runner.

The graph owns generation and repair. These adapters perform at most one model
request per call; they do not contain another agent or retry loop. Importing the
module does not read credentials, initialize DB-GPT, or open network connections.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from .qwen_config import QwenClientError, QwenConfig

GraphMessages = list[dict[str, str]]
GraphAction = dict[str, str]


class GraphModelError(QwenClientError):
    """Safe error code and whether a fresh, budgeted repair call is permitted."""

    def __init__(self, code: str, *, retryable: bool = False):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,119}", code, flags=re.ASCII):
            code = "GRAPH_MODEL_ERROR_REDACTED"
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class GraphModel(Protocol):
    """Small model boundary; checkpoint state contains no live client object."""

    client_kind: str

    async def generate(self, messages: GraphMessages) -> GraphAction: ...

    async def aclose(self) -> None: ...


class GraphRequestBudget(Protocol):
    """Existing cumulative budgets and durable budget proxies share this API."""

    def reserve(self, prompt_chars: int) -> int: ...

    def record_tokens(self, usage: dict[str, Any]) -> None: ...

    def stop(self, reason: str) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_graph_action(raw: str | dict[str, Any]) -> GraphAction:
    """Validate one SQL action, without treating model text as executable Python."""
    if isinstance(raw, str):
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (ValueError, TypeError, RecursionError):
            raise GraphModelError("GRAPH_INVALID_ACTION_JSON", retryable=True) from None
    else:
        value = raw
    if not isinstance(value, dict) or set(value) - {"sql", "task_id", "thought"}:
        raise GraphModelError("GRAPH_INVALID_ACTION_FIELDS", retryable=True)
    if (
        type(value.get("sql")) is not str
        or not value["sql"].strip()
        or len(value["sql"]) > 20_000
        or type(value.get("task_id")) is not str
        or not value["task_id"].strip()
        or len(value["task_id"]) > 80
        or type(value.get("thought", "")) is not str
        or len(value.get("thought", "")) > 2_000
    ):
        raise GraphModelError("GRAPH_INVALID_ACTION_FIELDS", retryable=True)
    return {"sql": value["sql"], "task_id": value["task_id"], "thought": value.get("thought", "")}


def _validated_messages(messages: GraphMessages) -> GraphMessages:
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or type(item.get("role")) is not str
            or item.get("role") not in {"system", "user", "assistant"}
            or type(item.get("content")) is not str
            for item in messages
        )
    ):
        raise GraphModelError("GRAPH_TEXT_MESSAGES_REQUIRED")
    if sum(len(item["content"]) for item in messages) > 40_000:
        raise GraphModelError("GRAPH_SINGLE_REQUEST_INPUT_TOO_LARGE")
    return deepcopy(messages)


class ScriptedGraphModel:
    """Deterministic offline actions; an exhausted script is an explicit failure."""

    client_kind = "offline_scripted_graph_model"

    def __init__(self, responses: list[GraphAction | str]):
        if not isinstance(responses, list) or not 1 <= len(responses) <= 32:
            raise ValueError("responses must contain 1..32 offline actions")
        self._responses = deepcopy(responses)
        self.requests: list[GraphMessages] = []
        self.http_events: list[dict[str, Any]] = []
        self.usage: list[dict[str, Any]] = []

    async def generate(self, messages: GraphMessages) -> GraphAction:
        checked = _validated_messages(messages)
        index = len(self.requests)
        self.requests.append(checked)
        if index >= len(self._responses):
            raise GraphModelError("GRAPH_OFFLINE_SCRIPT_EXHAUSTED")
        return parse_graph_action(deepcopy(self._responses[index]))

    async def aclose(self) -> None:
        return None


class QwenGraphModel:
    """One metered Qwen HTTP POST, followed by strict JSON action validation.

    A malformed action can be repaired by the graph after its usage is recorded.
    Transport, response-envelope, truncation and accounting errors stop the
    cumulative budget. Unknown usage is never silently retried or reset.
    """

    client_kind = "real_qwen_graph_http"

    def __init__(
        self,
        config: QwenConfig,
        budget: GraphRequestBudget,
        *,
        http_client: httpx.AsyncClient | None = None,
        max_tokens: int = 900,
        timeout_seconds: float = 40,
    ):
        # Revalidate even when a caller passes a structural settings object.
        self.config = QwenConfig(config.api_key, config.base_url, config.model)
        self.model_name = self.config.model
        if self.config.api_key in self.config.model or self.config.api_key in self.config.base_url:
            raise GraphModelError("GRAPH_CREDENTIAL_IN_SETTINGS")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 900:
            raise ValueError("max_tokens must be in 1..900")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be in 0..60")
        output_cap = budget.snapshot().get("max_output_tokens")
        if output_cap is not None and (type(output_cap) is not int or max_tokens > output_cap):
            raise ValueError("max_tokens cannot exceed the budget output reservation")
        self.budget = budget
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._owns_client = http_client is None
        self.requests: list[GraphMessages] = []
        self.http_events: list[dict[str, Any]] = []
        self.usage: list[dict[str, Any]] = []

    async def generate(self, messages: GraphMessages) -> GraphAction:
        checked = _validated_messages(messages)
        if any(self.config.api_key in item["content"] for item in checked):
            raise GraphModelError("GRAPH_CREDENTIAL_IN_MESSAGES")
        self.requests.append(checked)
        try:
            attempt = self.budget.reserve(sum(len(item["content"]) for item in checked))
        except QwenClientError as error:
            safe_code = str(error)
            if self.config.api_key in safe_code:
                safe_code = "GRAPH_BUDGET_RESERVATION_REJECTED"
            raise GraphModelError(safe_code) from None
        event = self._new_event(attempt)
        self.http_events.append(event)
        text = await self._post_action(checked, event)
        try:
            return parse_graph_action(text)
        except GraphModelError as error:
            # The HTTP response and accounting succeeded. This is a candidate
            # formatting failure, so another budgeted graph turn may repair it.
            event["action_error_code"] = error.code
            raise

    def _new_event(self, attempt: int) -> dict[str, Any]:
        return {
            "attempt": attempt,
            "model": self.config.model,
            "status": "reserved",
            "post_started_at_utc": None,
            "post_finished_at_utc": None,
            "usage": {
                name: None for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "usage_complete": False,
        }

    async def _post_action(self, messages: GraphMessages, event: dict[str, Any]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "enable_thinking": False,
            "preserve_thinking": False,
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(retries=0),
                    follow_redirects=False,
                    timeout=self.timeout_seconds,
                )
            event.update(status="started", post_started_at_utc=_utc_now())
            try:
                response = await self._http_client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.config.api_key},
                    json=payload,
                    timeout=self.timeout_seconds,
                    follow_redirects=False,
                )
            finally:
                event["post_finished_at_utc"] = _utc_now()
            event["http_status"] = response.status_code
            if response.status_code != 200:
                raise GraphModelError(f"GRAPH_QWEN_HTTP_{response.status_code}")
            text = self._read_response(response, event)
            event["status"] = "completed"
            return text
        except GraphModelError as error:
            self.budget.stop("PROVIDER_ERROR")
            event.update(status="failed", error_code=error.code)
            raise
        except QwenClientError:
            self.budget.stop("ACCOUNTING_ERROR")
            event.update(status="failed", error_code="GRAPH_BUDGET_ACCOUNTING_ERROR")
            raise GraphModelError("GRAPH_BUDGET_ACCOUNTING_ERROR") from None
        except httpx.HTTPError as error:
            self.budget.stop("TRANSPORT_ERROR")
            code = "GRAPH_QWEN_TRANSPORT_" + type(error).__name__.upper()
            event.update(status="failed", error_code=code)
            raise GraphModelError(code) from None
        except asyncio.CancelledError:
            self.budget.stop("REQUEST_CANCELLED")
            event.update(status="cancelled", error_code="GRAPH_QWEN_CANCELLED_USAGE_UNKNOWN")
            raise
        except Exception:
            # A supplied transport or durable budget proxy can also fail. Its
            # exception may contain a key, URL or local path; expose no raw text.
            self.budget.stop("CLIENT_ERROR")
            event.update(status="failed", error_code="GRAPH_UNEXPECTED_CLIENT_ERROR")
            raise GraphModelError("GRAPH_UNEXPECTED_CLIENT_ERROR") from None
        finally:
            event["seconds"] = round(time.perf_counter() - started, 4)

    def _read_response(self, response: httpx.Response, event: dict[str, Any]) -> str:
        try:
            body = response.json()
            raw_usage = body.get("usage") if isinstance(body, dict) else None
            usage = {
                name: raw_usage.get(name)
                if isinstance(raw_usage, dict)
                and type(raw_usage.get(name)) is int
                and raw_usage[name] >= 0
                else None
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            event["usage"] = usage
            self.usage.append(deepcopy(usage))
            self.budget.record_tokens(usage)
            complete = all(value is not None for value in usage.values())
            consistent = (
                complete
                and usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
            )
            event["usage_complete"] = bool(consistent)
            if not consistent:
                raise GraphModelError("GRAPH_QWEN_USAGE_UNAVAILABLE_OR_INCONSISTENT")
            if self.budget.snapshot().get("stopped_reason"):
                raise GraphModelError("GRAPH_BUDGET_STOPPED_AFTER_USAGE")
            choice = body["choices"][0]
            message = choice["message"]
            text = message["content"]
            finish_reason = choice["finish_reason"]
            if not isinstance(text, str) or not text.strip() or message.get("tool_calls"):
                raise ValueError
            if self.config.api_key in text:
                raise GraphModelError("GRAPH_QWEN_RESPONSE_CONTAINS_CREDENTIAL")
            if finish_reason == "length":
                raise GraphModelError("GRAPH_QWEN_RESPONSE_TRUNCATED")
            if finish_reason != "stop":
                raise GraphModelError("GRAPH_QWEN_RESPONSE_NOT_COMPLETED")
            event["finish_reason"] = finish_reason
            return text
        except (ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError):
            raise GraphModelError("GRAPH_QWEN_INVALID_RESPONSE_ENVELOPE") from None

    async def aclose(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()


class InjectedFirstGraphModel:
    """Controlled first fault; every subsequent response comes from the delegate."""

    client_kind = "injected_first_fault_then_real_qwen_graph_http"

    def __init__(
        self, delegate: QwenGraphModel, candidate: GraphAction, *, already_injected: bool = False
    ):
        self.delegate = delegate
        self._candidate = parse_graph_action(candidate)
        self._injected = already_injected
        self.requests: list[GraphMessages] = []
        self.turn_origins: list[str] = []

    @property
    def budget(self) -> GraphRequestBudget:
        return self.delegate.budget

    @property
    def model_name(self) -> str:
        return self.delegate.model_name

    @property
    def http_events(self) -> list[dict[str, Any]]:
        return self.delegate.http_events

    @property
    def usage(self) -> list[dict[str, Any]]:
        return self.delegate.usage

    async def generate(self, messages: GraphMessages) -> GraphAction:
        self.requests.append(_validated_messages(messages))
        if not self._injected:
            self._injected = True
            self.turn_origins.append("injected_test_fault_no_api_call")
            return deepcopy(self._candidate)
        self.turn_origins.append("real_qwen_graph_http")
        return await self.delegate.generate(messages)

    async def aclose(self) -> None:
        await self.delegate.aclose()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
