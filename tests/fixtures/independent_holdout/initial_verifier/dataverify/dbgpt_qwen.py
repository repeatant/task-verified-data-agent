"""Bounded ordinary Qwen HTTP client for DB-GPT's text-JSON Action protocol.

No credential lookup or network I/O on import. Native generate_stream consumes
one complete non-streaming HTTP response; it is not token streaming. Upstream
may retry a failed client, so reserve the *actual POST* in a shared budget and
stop further POSTs after an operational error. Never log raw HTTP exceptions.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import re
from threading import Lock
import time
from urllib.parse import urlsplit

import httpx
from dbgpt.core import LLMClient, ModelMessage, ModelMetadata, ModelOutput


class QwenClientError(RuntimeError):
    """Safe diagnostic text only: upstream includes this message in its logs."""


@dataclass(frozen=True)
class QwenConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str = "qwen3.8-flash"

    def __post_init__(self):
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise QwenClientError("QWEN_API_KEY_NOT_CONFIGURED")
        if self.api_key.startswith("sk-sp-"):
            raise QwenClientError("ORDINARY_API_REQUIRED_SUBSCRIPTION_KEY_REJECTED")
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except (ValueError, TypeError, AttributeError):
            raise QwenClientError("ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED") from None
        host = parsed.hostname or ""
        ordinary_host = host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or (
            host.endswith(".maas.aliyuncs.com") and "token-plan" not in host
        )
        if (
            parsed.scheme != "https" or not ordinary_host
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or port not in (None, 443)
            or parsed.path.rstrip("/") != "/compatible-mode/v1"
        ):
            raise QwenClientError("ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED")
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 200:
            raise QwenClientError("INVALID_QWEN_MODEL")

    @classmethod
    def from_local_environment(cls):
        """Read only three named variables; use Windows user scope if absent.

        GUI processes may predate setx. Do not print/copy a key into project files
        just to refresh that process environment. No .env file is auto-loaded.
        """
        values = {name: os.environ.get(name, "") for name in (
            "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL",
        )}
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
                    for name in values:
                        if not values[name]:
                            try:
                                values[name] = winreg.QueryValueEx(handle, name)[0]
                            except FileNotFoundError:
                                pass
            except OSError:
                pass
        return cls(
            values["DASHSCOPE_API_KEY"],
            values["DASHSCOPE_BASE_URL"] or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            values["DASHSCOPE_MODEL"] or "qwen3.8-flash",
        )


@dataclass
class RequestBudget:
    """Shared across conversations; reserve before every actual HTTP attempt."""

    limit: int = 6
    max_input_chars: int = 150_000
    max_reported_tokens: int = 30_000
    used: int = field(default=0, init=False)
    input_chars: int = field(default=0, init=False)
    reported_tokens: int = field(default=0, init=False)
    stopped_reason: str | None = field(default=None, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        for value in (self.limit, self.max_input_chars, self.max_reported_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Budget limits must be positive integers")

    def reserve(self, prompt_chars: int) -> int:
        with self._lock:
            if self.stopped_reason:
                raise QwenClientError("LIVE_BUDGET_STOPPED_" + self.stopped_reason)
            if self.used >= self.limit:
                raise QwenClientError("LIVE_HTTP_REQUEST_LIMIT")
            if self.input_chars + prompt_chars > self.max_input_chars:
                raise QwenClientError("LIVE_INPUT_CHARACTER_LIMIT")
            if self.reported_tokens >= self.max_reported_tokens:
                raise QwenClientError("LIVE_REPORTED_TOKEN_LIMIT")
            self.used += 1
            self.input_chars += prompt_chars
            return self.used

    def stop(self, reason: str):
        with self._lock:
            self.stopped_reason = self.stopped_reason or reason

    def record_tokens(self, usage: dict):
        with self._lock:
            tokens = usage.get("total_tokens")
            if type(tokens) is int:
                self.reported_tokens += tokens

    def snapshot(self):
        with self._lock:
            return {
                "http_attempts": self.used, "http_attempt_limit": self.limit,
                "input_chars": self.input_chars, "input_char_limit": self.max_input_chars,
                "reported_total_tokens": self.reported_tokens,
                "reported_token_stop_threshold": self.max_reported_tokens,
                "stopped_reason": self.stopped_reason,
            }


class QwenLLMClient(LLMClient):
    """One ordinary /chat/completions POST for each native model request."""

    client_kind = "real_qwen_http"

    def __init__(
        self, config: QwenConfig, budget: RequestBudget, *,
        http_client: httpx.AsyncClient | None = None, max_tokens: int = 900,
        timeout_seconds: float = 40,
    ):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2000:
            raise ValueError("max_tokens must be in 1..2000")
        if not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be in 0..60")
        self.config, self.budget = config, budget
        self.model_name = config.model
        self.max_tokens, self.timeout_seconds = max_tokens, timeout_seconds
        self._http_client = http_client
        self._owns_client = http_client is None
        self._cache: dict = {}
        self.requests: list[list[dict]] = []
        self.http_events: list[dict] = []
        self.usage: list[dict] = []

    @property
    def cache(self):
        return self._cache

    async def models(self):
        # A conservative client input allowance, not a claim of model specs.
        return [ModelMetadata(model=self.model_name, context_length=32768)]

    async def count_token(self, model, prompt):
        # Local planning proxy only. Billing/report totals use API usage, never this.
        return max(1, len(prompt) // 2)

    @staticmethod
    def normalize_usage(raw):
        raw = raw if isinstance(raw, dict) else {}
        return {name: raw.get(name) if type(raw.get(name)) is int and raw[name] >= 0 else None
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")}

    def _safe_trace_value(self, value, pattern, max_length):
        # Provider metadata is untrusted: never persist arbitrary response fields.
        if (type(value) is str and 0 < len(value) <= max_length
                and self.config.api_key not in value
                and re.fullmatch(pattern, value, flags=re.ASCII)):
            return value
        return None

    def _endpoint_metadata(self):
        metadata = {"endpoint_host": None, "endpoint_region": None, "endpoint_workspace": None}
        # Check before hostname normalization as well, which may lowercase a key.
        if self.config.api_key in self.config.base_url:
            return metadata
        label = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        host = self._safe_trace_value(urlsplit(self.config.base_url).hostname, rf"{label}(?:\.{label})*", 253)
        metadata["endpoint_host"] = host
        if host is not None:
            parts = host.split(".")
            if len(parts) == 5 and parts[2:] == ["maas", "aliyuncs", "com"]:
                metadata["endpoint_workspace"] = self._safe_trace_value(parts[0], label, 63)
                metadata["endpoint_region"] = self._safe_trace_value(parts[1], r"[a-z]{2}-[a-z]+(?:-[1-9][0-9]?)?", 63)
        # Shared/intl hosts do not encode a workspace or an unambiguous region.
        return metadata

    def _record_response_metadata(self, event, response, body=None):
        identifier_pattern = r"[A-Za-z0-9][A-Za-z0-9._:-]*"
        if event["request_id"] is None:
            for name in ("x-request-id", "x-dashscope-request-id"):
                value = self._safe_trace_value(response.headers.get(name), identifier_pattern, 128)
                if value is not None:
                    event["request_id"] = value
                    break
        if isinstance(body, dict):
            event["response_id"] = self._safe_trace_value(body.get("id"), identifier_pattern, 128)
            event["response_model"] = self._safe_trace_value(body.get("model"), r"[A-Za-z0-9][A-Za-z0-9._:/-]*", 200)
            if event["request_id"] is None:
                event["request_id"] = self._safe_trace_value(body.get("request_id"), identifier_pattern, 128)

    async def _generate(self, request):
        if request.model != self.model_name:
            raise QwenClientError("UNCONFIGURED_MODEL_REQUEST")
        if getattr(request, "tools", None):
            raise QwenClientError("NATIVE_JSON_ACTION_CLIENT_DOES_NOT_ACCEPT_PROVIDER_TOOLS")
        try:
            messages = ModelMessage.to_common_messages(request.get_messages(), support_media_content=False)
        except Exception:
            raise QwenClientError("INVALID_NATIVE_TEXT_MESSAGES") from None
        if not messages or any(
            m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str) or m.get("tool_calls")
            for m in messages
        ):
            raise QwenClientError("ONLY_TEXT_JSON_ACTION_MESSAGES_SUPPORTED")
        prompt_chars = sum(len(m["content"]) for m in messages)
        if prompt_chars > 40_000:
            raise QwenClientError("SINGLE_REQUEST_INPUT_TOO_LARGE")
        self.requests.append(deepcopy(messages))
        request_max = request.max_new_tokens if type(request.max_new_tokens) is int and request.max_new_tokens > 0 else self.max_tokens
        payload = {
            "model": self.model_name, "messages": messages, "stream": False,
            "temperature": 0, "max_tokens": min(request_max, self.max_tokens),
            "enable_thinking": False, "preserve_thinking": False,
            "response_format": {"type": "json_object"},
        }
        attempt = self.budget.reserve(prompt_chars)
        requested_model = self._safe_trace_value(self.model_name, r"[A-Za-z0-9][A-Za-z0-9._:/-]*", 200)
        event = {"attempt": attempt, "model": requested_model, "requested_model": requested_model,
                 "response_model": None, "response_id": None, "request_id": None,
                 "post_started_at_utc": None, "post_finished_at_utc": None,
                 **self._endpoint_metadata(), "status": "started",
                 "usage": self.normalize_usage(None), "usage_complete": False}
        self.http_events.append(event)
        started = time.perf_counter()
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(retries=0), follow_redirects=False,
                    timeout=self.timeout_seconds,
                )
            event["post_started_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            try:
                response = await self._http_client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.config.api_key}, json=payload,
                    timeout=self.timeout_seconds, follow_redirects=False,
                )
            finally:
                event["post_finished_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            event["http_status"] = response.status_code
            self._record_response_metadata(event, response)
            if response.status_code != 200:
                # Error bodies may carry an audit ID, but must never be logged.
                try:
                    self._record_response_metadata(event, response, response.json())
                except (ValueError, TypeError):
                    pass
                raise QwenClientError(f"QWEN_HTTP_{response.status_code}")
            try:
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError
                self._record_response_metadata(event, response, body)
                usage = self.normalize_usage(body.get("usage"))
                event["usage"] = usage
                event["usage_complete"] = all(value is not None for value in usage.values())
                self.usage.append(usage)
                self.budget.record_tokens(usage)
                choice = body["choices"][0]
                message = choice["message"]
                text = message["content"]
                finish_reason = choice.get("finish_reason")
                if not isinstance(text, str) or not text.strip() or message.get("tool_calls"):
                    raise ValueError
                if not isinstance(finish_reason, str) or finish_reason not in {"stop", "length", "content_filter"}:
                    raise ValueError
            except (ValueError, KeyError, IndexError, TypeError):
                raise QwenClientError("QWEN_INVALID_RESPONSE_ENVELOPE") from None
            if self.config.api_key in text:
                raise QwenClientError("QWEN_RESPONSE_CONTAINS_CREDENTIAL")
            if finish_reason == "length":
                raise QwenClientError("QWEN_RESPONSE_TRUNCATED")
            event.update(status="completed", finish_reason=finish_reason)
            return ModelOutput(error_code=0, text=text, usage=usage, finish_reason=finish_reason)
        except QwenClientError as error:
            self.budget.stop("PROVIDER_ERROR")
            event.update(status="failed", error_code=str(error))
            raise
        except httpx.HTTPError as error:
            self.budget.stop("TRANSPORT_ERROR")
            event.update(status="failed", error_code="QWEN_TRANSPORT_" + type(error).__name__)
            raise QwenClientError(event["error_code"]) from None
        except asyncio.CancelledError:
            self.budget.stop("REQUEST_CANCELLED")
            event.update(status="cancelled", error_code="QWEN_REQUEST_CANCELLED_USAGE_UNKNOWN")
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - started, 4)

    async def generate(self, request, message_converter=None):
        return await self._generate(request)

    async def generate_stream(self, request, message_converter=None):
        yield await self._generate(request)

    async def aclose(self):
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()


class InjectedFirstCandidateClient(LLMClient):
    """Controlled fault test: first Action is injected, all repairs are real Qwen.

    This must never be described as a naturally occurring model error or counted
    as a paid model response. Request origins are explicitly retained in traces.
    """

    client_kind = "injected_first_fault_then_real_qwen_http"

    def __init__(self, delegate: QwenLLMClient, candidate: dict):
        self.delegate = delegate
        self.candidate = deepcopy(candidate)
        self.requests: list[list[dict]] = []
        self.turn_origins: list[str] = []

    @property
    def cache(self):
        return self.delegate.cache

    @property
    def http_events(self):
        return self.delegate.http_events

    @property
    def usage(self):
        return self.delegate.usage

    @property
    def model_name(self):
        return self.delegate.model_name

    async def models(self):
        return await self.delegate.models()

    async def count_token(self, model, prompt):
        return await self.delegate.count_token(model, prompt)

    async def generate(self, request, message_converter=None):
        self.requests.append(ModelMessage.to_common_messages(request.get_messages(), support_media_content=False))
        if len(self.requests) == 1:
            self.turn_origins.append("injected_test_fault_no_api_call")
            return ModelOutput(error_code=0, text=json.dumps(self.candidate), usage=None)
        self.turn_origins.append("real_qwen_http")
        return await self.delegate.generate(request, message_converter)

    async def generate_stream(self, request, message_converter=None):
        yield await self.generate(request, message_converter)

    async def aclose(self):
        await self.delegate.aclose()
