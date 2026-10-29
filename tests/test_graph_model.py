"""Graph model protocol tests use fake responses only: no keys or paid calls."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from dataverify.graph_model import (
    GraphModelError,
    InjectedFirstGraphModel,
    QwenGraphModel,
    ScriptedGraphModel,
    parse_graph_action,
)
from dataverify.qwen_config import QwenClientError, QwenConfig

SECRET = "sk-test-never-a-real-credential"
ENDPOINT = "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
ACTION = {"sql": "SELECT SUM(amount) FROM order_lines", "task_id": "graph-test"}
MESSAGES = [{"role": "system", "content": "Return a JSON SQL action."}]


@dataclass
class FakeBudget:
    """Small accounting spy, independent of optional DB-GPT dependencies."""

    limit: int = 2
    used: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str | None = None
    max_output_tokens: int = 900

    def reserve(self, prompt_chars):
        assert type(prompt_chars) is int and prompt_chars >= 0
        if self.stopped_reason:
            raise QwenClientError("LIVE_BUDGET_STOPPED_" + self.stopped_reason)
        if self.used >= self.limit:
            raise QwenClientError("LIVE_HTTP_REQUEST_LIMIT")
        self.used += 1
        return self.used

    def record_tokens(self, usage):
        self.usage.append(dict(usage))

    def stop(self, reason):
        self.stopped_reason = self.stopped_reason or reason

    def snapshot(self):
        return {
            "http_attempts": self.used,
            "stopped_reason": self.stopped_reason,
            "max_output_tokens": self.max_output_tokens,
        }


def config():
    return QwenConfig(SECRET, ENDPOINT, "qwen-test")


def response_body(text=None, *, finish_reason="stop", usage="default"):
    result = {
        "choices": [
            {"message": {"content": text or json.dumps(ACTION)}, "finish_reason": finish_reason}
        ],
    }
    result["usage"] = (
        {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        if usage == "default"
        else usage
    )
    return result


def test_action_returns_own_normalized_values():
    raw = dict(ACTION)
    result = parse_graph_action(raw)
    assert result == {**ACTION, "thought": ""}
    result["sql"] = "changed"
    assert raw == ACTION


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"sql":"one","sql":"two","task_id":"x"}',
        "[]",
        {"sql": "", "task_id": "x"},
        {"sql": "SELECT 1", "task_id": " "},
        {"sql": "SELECT 1", "task_id": "x" * 81},
        {"sql": "x" * 20_001, "task_id": "x"},
        {**ACTION, "thought": None},
        {**ACTION, "tools": []},
        {**ACTION, "thought": "x" * 2_001},
        {"sql": True, "task_id": "x"},
    ],
)
def test_action_rejects_ambiguous_or_wrongly_typed_content(raw):
    with pytest.raises(GraphModelError) as caught:
        parse_graph_action(raw)
    assert caught.value.retryable is True
    assert caught.value.code.startswith("GRAPH_INVALID_ACTION_")


def test_script_is_deterministic_and_does_not_reuse_a_mutable_action():
    async def scenario():
        model = ScriptedGraphModel([dict(ACTION), {**ACTION, "thought": "repair"}])
        first = await model.generate(MESSAGES)
        first["sql"] = "changed"
        assert (await model.generate(MESSAGES))["thought"] == "repair"
        assert model.http_events == model.usage == []
        with pytest.raises(GraphModelError, match="SCRIPT_EXHAUSTED") as caught:
            await model.generate(MESSAGES)
        assert caught.value.retryable is False
        await model.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": [], "content": "x"}],
        [{"role": "user", "content": 1}],
        [{"role": "assistant", "content": "x", "tool_calls": []}],
        [{"role": "user", "content": "x" * 40_001}],
    ],
)
def test_invalid_messages_never_use_the_budget_or_transport(messages):
    async def scenario():
        budget = FakeBudget()
        model = QwenGraphModel(config(), budget)
        with pytest.raises(GraphModelError):
            await model.generate(messages)
        assert budget.used == 0
        assert model.http_events == []
        await model.aclose()

    asyncio.run(scenario())


def test_single_http_request_uses_json_protocol_and_reserves_before_post():
    async def scenario():
        budget = FakeBudget()
        seen = []

        def handler(request):
            assert budget.used == 1
            assert str(request.url) == ENDPOINT + "/chat/completions"
            assert request.headers["Authorization"] == "Bearer " + SECRET
            payload = json.loads(request.content)
            seen.append(payload)
            assert payload["messages"] == MESSAGES
            assert payload["max_tokens"] == 900
            assert payload["temperature"] == 0
            assert payload["stream"] is False
            assert payload["enable_thinking"] is False
            assert payload["response_format"] == {"type": "json_object"}
            assert "tools" not in payload
            return httpx.Response(200, json=response_body())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = QwenGraphModel(config(), budget, http_client=http)
            assert await model.generate(MESSAGES) == {**ACTION, "thought": ""}
            await model.aclose()
            assert not http.is_closed  # Injected clients remain caller-owned.
            assert len(seen) == 1
            event = model.http_events[0]
            assert event["status"] == "completed"
            assert event["usage_complete"] is True
            assert event["post_started_at_utc"].endswith("Z")
            assert event["post_finished_at_utc"].endswith("Z")
            assert budget.usage == [
                {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
            ]
            assert SECRET not in json.dumps(model.http_events)

    asyncio.run(scenario())


def test_malformed_action_keeps_accounting_and_permits_a_fresh_graph_repair():
    async def scenario():
        budget = FakeBudget()
        replies = iter([response_body("not json"), response_body()])
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=next(replies)))
        ) as http:
            model = QwenGraphModel(config(), budget, http_client=http)
            with pytest.raises(GraphModelError) as caught:
                await model.generate(MESSAGES)
            assert caught.value.retryable is True
            assert budget.stopped_reason is None
            assert len(budget.usage) == 1
            assert model.http_events[0]["status"] == "completed"
            assert model.http_events[0]["action_error_code"] == "GRAPH_INVALID_ACTION_JSON"
            assert (await model.generate(MESSAGES))["sql"] == ACTION["sql"]
            assert budget.used == 2
            assert len(model.http_events) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body,code",
    [
        (response_body(finish_reason="length"), "RESPONSE_TRUNCATED"),
        (response_body(finish_reason="content_filter"), "RESPONSE_NOT_COMPLETED"),
        (response_body(usage=None), "USAGE_UNAVAILABLE_OR_INCONSISTENT"),
        (
            response_body(usage={"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}),
            "USAGE_UNAVAILABLE_OR_INCONSISTENT",
        ),
        (
            response_body(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 9}),
            "USAGE_UNAVAILABLE_OR_INCONSISTENT",
        ),
        (
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            "INVALID_RESPONSE_ENVELOPE",
        ),
        (response_body(SECRET), "RESPONSE_CONTAINS_CREDENTIAL"),
    ],
)
def test_operational_response_error_stops_budget_without_hidden_retries(body, code):
    async def scenario():
        budget = FakeBudget()
        seen = []

        def handler(_):
            seen.append(True)
            return httpx.Response(200, json=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = QwenGraphModel(config(), budget, http_client=http)
            with pytest.raises(GraphModelError, match=code) as caught:
                await model.generate(MESSAGES)
            assert caught.value.retryable is False
            assert budget.stopped_reason
            with pytest.raises(GraphModelError, match="LIVE_BUDGET_STOPPED"):
                await model.generate(MESSAGES)
            assert len(seen) == 1
            assert len(model.http_events) == 1
            assert SECRET not in json.dumps(model.http_events)

    asyncio.run(scenario())


def test_provider_tool_calls_are_rejected_even_if_content_is_valid():
    async def scenario():
        body = response_body()
        body["choices"][0]["message"]["tool_calls"] = [{"id": "not-this-protocol"}]
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ) as http:
            model = QwenGraphModel(config(), FakeBudget(), http_client=http)
            with pytest.raises(GraphModelError, match="INVALID_RESPONSE_ENVELOPE"):
                await model.generate(MESSAGES)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_errors_do_not_follow_redirects_log_bodies_or_retry(status):
    async def scenario():
        seen = []

        def handler(_):
            seen.append(True)
            return httpx.Response(status, text=SECRET, headers={"Location": "https://example.test"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = QwenGraphModel(config(), FakeBudget(), http_client=http)
            with pytest.raises(GraphModelError, match=f"GRAPH_QWEN_HTTP_{status}"):
                await model.generate(MESSAGES)
            assert len(seen) == 1
            assert SECRET not in json.dumps(model.http_events)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["transport", "unexpected", "cancelled"])
def test_exceptions_have_only_safe_codes_and_stop_the_shared_ledger(failure):
    async def scenario():
        def handler(_):
            if failure == "transport":
                raise httpx.ReadTimeout(SECRET)
            if failure == "cancelled":
                raise asyncio.CancelledError(SECRET)
            raise RuntimeError(SECRET)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            budget = FakeBudget()
            model = QwenGraphModel(config(), budget, http_client=http)
            if failure == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await model.generate(MESSAGES)
            else:
                with pytest.raises(GraphModelError) as caught:
                    await model.generate(MESSAGES)
                assert SECRET not in str(caught.value)
            assert budget.used == 1
            assert budget.stopped_reason
            assert SECRET not in json.dumps(model.http_events)

    asyncio.run(scenario())


def test_credential_in_prompt_is_rejected_without_recording_it():
    async def scenario():
        budget = FakeBudget()
        model = QwenGraphModel(config(), budget)
        with pytest.raises(GraphModelError, match="CREDENTIAL_IN_MESSAGES"):
            await model.generate([{"role": "user", "content": SECRET}])
        assert budget.used == 0
        assert model.requests == []

    asyncio.run(scenario())


def test_client_output_cap_must_match_its_preflight_reservation():
    with pytest.raises(ValueError, match="output reservation"):
        QwenGraphModel(config(), FakeBudget(max_output_tokens=400), max_tokens=900)


def test_injected_first_fault_is_explicit_and_not_billed():
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response_body()))
        ) as http:
            budget = FakeBudget()
            live = QwenGraphModel(config(), budget, http_client=http)
            model = InjectedFirstGraphModel(live, {**ACTION, "sql": "SELECT bad"})
            assert (await model.generate(MESSAGES))["sql"] == "SELECT bad"
            assert budget.used == 0
            assert (await model.generate(MESSAGES))["sql"] == ACTION["sql"]
            assert budget.used == 1
            assert len(model.http_events) == 1
            assert model.budget is budget
            assert model.turn_origins == ["injected_test_fault_no_api_call", "real_qwen_graph_http"]
            await model.aclose()

    asyncio.run(scenario())
