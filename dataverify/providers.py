"""Interchangeable planners; offline mode never claims to be a language model.

The scripted planner tests graph transitions and deterministic verification.
Qwen uses the separately configured ordinary Model Studio API. Offline tests
never make a remote call; live validation must be run explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .contracts import METRIC_CATALOG

JOIN = "FROM orders AS o INNER JOIN order_lines AS l ON o.order_id = l.order_id"
CASES = {
    "join_fanout": (f"SELECT SUM(o.total) AS revenue {JOIN}", "order_revenue"),
    "distinct_fix": (f"SELECT SUM(DISTINCT o.total) AS revenue {JOIN}", "order_revenue"),
    "clean_order_total": ("SELECT SUM(total) AS revenue FROM orders", "order_revenue"),
    "clean_line_total": (f"SELECT SUM(l.amount) AS revenue {JOIN}", "line_revenue"),
    "unsupported": ("SELECT SUM(total) AS revenue FROM orders WHERE total > 90", "order_revenue"),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Read the registered tables, metrics, grain keys and relation. Use before querying.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metric",
            "description": "Execute one supported SUM query and verify metric grain. Unverified values must not be reported as facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "maxLength": 4096},
                    "metric_id": {"type": "string", "enum": ["order_revenue", "line_revenue"]},
                },
                "required": ["sql", "metric_id"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You answer a metric question using a read-only experimental database.
Use inspect_schema first. Metrics and supported SQL are explicit, not inferred permissions.
Call only one tool at a time. Read structured verification feedback before proposing a repair.
A successful SQL execution is not a verified statistic. Do not use SUM(DISTINCT measure)
as a general repair for duplicated grain. Unsupported shapes require a narrower query
or an explicit inability to verify. Do not write or alter data. Stop after verified evidence.
This prototype supports only an unfiltered single SUM over a registered column, with
zero or one registered INNER JOIN. Do not use subqueries, CTEs, arithmetic or other functions.
"""


class ProviderError(RuntimeError):
    """An operational error whose message deliberately excludes credentials."""


@dataclass
class Action:
    kind: str
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    text: str = ""

    def assistant_message(self) -> dict[str, Any]:
        if self.kind == "final":
            return {"role": "assistant", "content": self.text}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(self.arguments, ensure_ascii=False),
                    },
                }
            ],
        }


class Planner(Protocol):
    name: str
    real_model: bool

    def choose(self, messages: list[dict[str, Any]]) -> Action: ...


class OfflineScriptedPlanner:
    """Known scripts exercise recovery; they do not measure LLM intelligence."""

    name = "offline-scripted"
    real_model = False

    def __init__(self, case: str = "join_fanout") -> None:
        if case not in CASES:
            raise ValueError(f"Unknown offline case: {case}")
        self.case = case

    def choose(self, messages: list[dict[str, Any]]) -> Action:
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            return Action("tool", "inspect_schema", {}, "offline-schema")
        queries = [m for m in tool_messages if m.get("name") == "query_metric"]
        if not queries:
            sql, metric = CASES[self.case]
            return Action(
                "tool", "query_metric", {"sql": sql, "metric_id": metric}, "offline-query-1"
            )
        feedback = json.loads(queries[-1]["content"])
        if feedback.get("verification_status") == "verified":
            return Action("final", text="The tool returned verified metric evidence.")
        if self.case == "unsupported":
            return Action(
                "final", text="This query shape is outside the declared verification scope."
            )
        # This known repair is deliberately scripted. Real Qwen selection is separate.
        # Preserve the metric selected by the case: repairing a line-level
        # query into an order-level total would answer a different question,
        # even when both aggregates happen to have the same numeric value.
        metric_id = CASES[self.case][1]
        metric = METRIC_CATALOG[metric_id]
        return Action(
            "tool",
            "query_metric",
            {
                "sql": f"SELECT SUM({metric.value_column}) AS revenue FROM {metric.source_table}",
                "metric_id": metric_id,
            },
            f"offline-query-{len(queries) + 1}",
        )


class QwenPlanner:
    """OpenAI-compatible HTTP adapter. No external call occurs at import time."""

    name = "qwen"
    real_model = True

    def __init__(self, *, api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置；请先使用 provider=offline，不要将密钥发到聊天中。"
            )
        self.base_url = os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-flash")
        self.client = client
        self.usage: list[dict[str, Any]] = []

    def choose(self, messages: list[dict[str, Any]]) -> Action:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            # Direct HTTP puts these fields at the top level, not extra_body.
            # Keep this minimal adapter non-thinking: it does not round-trip
            # reasoning_content across successive tool calls.
            "enable_thinking": False,
            "preserve_thinking": False,
            "temperature": 0,
            "max_tokens": 700,
        }
        owned = self.client is None
        client = self.client or httpx.Client(timeout=40)
        try:
            response = client.post(
                self.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            # A successful HTTP status does not guarantee a usable protocol
            # envelope. Validate containers before accessing .get() so a bad
            # remote response becomes a safe ProviderError, not an uncaught
            # AttributeError that bypasses the runner's explicit stop path.
            if not isinstance(body, dict):
                raise ValueError("model response must be an object")
            raw_usage = body.get("usage")
            self.usage.append(raw_usage if isinstance(raw_usage, dict) else {})
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("model choices must contain a message object")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("model message must be an object")
            calls = message.get("tool_calls")
            if calls is None:
                calls = []
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list")
            if not calls:
                content = message.get("content")
                if content is not None and not isinstance(content, str):
                    raise ValueError("model content must be text or null")
                return Action("final", text=content or "")
            if len(calls) != 1:
                raise ProviderError(
                    "The minimal adapter supports one tool call per model response."
                )
            call = calls[0]
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise ValueError("tool call must be a function object")
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("tool function must be an object")
            call_id = call.get("id")
            tool_name = function.get("name")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("tool call id must be non-empty text")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool name must be non-empty text")
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            return Action("tool", tool_name, arguments, call_id)
        except ProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            # Status is useful for diagnosis; the server body/headers may not
            # be safe to print and are deliberately excluded.
            raise ProviderError(
                f"Model HTTP {exc.response.status_code}; check local model, region, API key and account access."
            ) from None
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # No raw response/request/header is included in logs or errors.
            raise ProviderError(
                f"Model request/response failed ({type(exc).__name__}); inspect local API configuration."
            ) from None
        finally:
            if owned:
                client.close()
