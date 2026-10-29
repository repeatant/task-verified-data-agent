"""Transport-contract checks use an in-process mock, not a remote model."""

import json

import httpx
import pytest

from dataverify.providers import ProviderError, QwenPlanner


def test_qwen_transport_encodes_tools_without_leaking_key_to_body():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "test-key-only" not in request.content.decode()
        assert request.headers["Authorization"] == "Bearer test-key-only"
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": "inspect_schema", "arguments": "{}"}}]}}], "usage": {"total_tokens": 10}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        action = planner.choose([{"role": "user", "content": "test"}])
    assert action.tool_name == "inspect_schema" and action.arguments == {}
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["inspect_schema", "query_metric"]
    assert requests[0]["enable_thinking"] is False
    assert requests[0]["preserve_thinking"] is False
    assert requests[0]["parallel_tool_calls"] is False
    assert "extra_body" not in requests[0]


def test_qwen_error_does_not_include_credentials_or_response_body():
    def handler(request):
        return httpx.Response(401, text="sensitive-server-text test-key-only", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        with pytest.raises(ProviderError) as failure:
            planner.choose([{"role": "user", "content": "test"}])
    assert "test-key-only" not in str(failure.value)
    assert "sensitive-server-text" not in str(failure.value)
    assert "HTTP 401" in str(failure.value)


def test_qwen_uses_workspace_endpoint_and_records_usage(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen3.8-flash")

    def handler(request):
        assert str(request.url) == "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
        assert json.loads(request.content)["model"] == "qwen3.8-flash"
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        assert planner.choose([{"role": "user", "content": "test"}]).kind == "final"
    assert planner.usage == [{"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}]


def test_qwen_refuses_multiple_tool_calls_in_one_response():
    def handler(request):
        call = {"id": "call-1", "function": {"name": "inspect_schema", "arguments": "{}"}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call, {**call, "id": "call-2"}]}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        with pytest.raises(ProviderError, match="one tool call"):
            planner.choose([{"role": "user", "content": "test"}])


@pytest.mark.parametrize("message", [
    None,
    "sensitive-response-sentinel",
    [],
    {"tool_calls": {"sensitive-response-sentinel": "not-a-list"}},
    {"tool_calls": [None]},
    {"tool_calls": [{"id": "call-1", "function": None}]},
    {"tool_calls": [{"id": "", "function": {"name": "inspect_schema", "arguments": "{}"}}]},
    {"tool_calls": [{"id": None, "function": {"name": "inspect_schema", "arguments": "{}"}}]},
    {"tool_calls": [{"id": "call-1", "function": {"name": None, "arguments": "{}"}}]},
    {"tool_calls": [{"id": "call-1", "function": {"name": "inspect_schema", "arguments": "[]"}}]},
    {"content": {"sensitive-response-sentinel": "not-text"}},
])
def test_malformed_qwen_messages_fail_as_sanitized_provider_errors(message):
    """Remote protocol drift must not escape the runner as AttributeError."""
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": message}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        with pytest.raises(ProviderError, match="request/response failed") as failure:
            planner.choose([{"role": "user", "content": "test"}])
    assert "test-key-only" not in str(failure.value)
    assert "sensitive-response-sentinel" not in str(failure.value)


@pytest.mark.parametrize("body", [None, [], {"choices": None}, {"choices": [None]}])
def test_malformed_qwen_envelopes_fail_closed(body):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        with pytest.raises(ProviderError, match="request/response failed"):
            planner.choose([{"role": "user", "content": "test"}])
