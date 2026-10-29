"""Offline protocol/ledger/native-loop tests. No environment keys or live calls."""

import asyncio
import importlib.util
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import httpx
import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)

from dbgpt.core import ModelMessage, ModelRequest

from dataverify.dbgpt_bridge import BAD_ORDER_SUM, GOOD_ORDER_SUM, run_native_case, sql_candidate
from dataverify.dbgpt_qwen import (
    InjectedFirstCandidateClient,
    QwenClientError,
    QwenConfig,
    QwenLLMClient,
    RequestBudget,
)
from dataverify.guarded_task import ConfirmedMetricTask

SECRET = "sk-test-never-a-real-credential"
ENDPOINT = "https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def config():
    return QwenConfig(SECRET, ENDPOINT, "qwen-test")


def request(max_tokens=4096):
    return ModelRequest(
        model="qwen-test",
        max_new_tokens=max_tokens,
        messages=[
            ModelMessage(role="system", content="Return one JSON action."),
            ModelMessage(role="human", content="Order revenue"),
            ModelMessage(role="ai", content="Previous JSON action"),
            ModelMessage(role="human", content="Feedback"),
        ],
    )


def body(text=None, usage="default"):
    response = {
        "choices": [
            {
                "message": {"content": text or json.dumps(sql_candidate(GOOD_ORDER_SUM))},
                "finish_reason": "stop",
            }
        ]
    }
    if usage == "default":
        response["usage"] = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    elif usage is not None:
        response["usage"] = usage
    return response


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "https://coding.dashscope.aliyuncs.com/v1",
        "https://fake.example/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1?key=hidden",
        "https://user:pass@dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com:sk-test-never-a-real-credential/compatible-mode/v1",
    ],
)
def test_reject_nonordinary_or_credential_bearing_endpoint(endpoint):
    with pytest.raises(QwenClientError, match="ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED"):
        QwenConfig(SECRET, endpoint)


def test_config_hides_key_and_rejects_subscription_key():
    assert SECRET not in repr(config())
    with pytest.raises(QwenClientError, match="SUBSCRIPTION_KEY_REJECTED"):
        QwenConfig("sk-sp-fake", ENDPOINT)
    with pytest.raises(QwenClientError, match="NOT_CONFIGURED"):
        QwenConfig("", ENDPOINT)


@pytest.mark.parametrize("stream", [False, True])
def test_single_post_roles_token_cap_and_actual_usage(stream):
    seen = []

    async def scenario():
        def handler(incoming):
            payload = json.loads(incoming.content)
            seen.append(payload)
            assert str(incoming.url) == ENDPOINT + "/chat/completions"
            assert incoming.headers["Authorization"] == "Bearer " + SECRET
            assert payload["max_tokens"] == 900
            assert payload["stream"] is False
            assert payload["enable_thinking"] is False
            assert [m["role"] for m in payload["messages"]] == [
                "system",
                "user",
                "assistant",
                "user",
            ]
            assert "tools" not in payload
            assert SECRET not in json.dumps(payload)
            return httpx.Response(200, json=body())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            output = (
                [o async for o in client.generate_stream(request())][0]
                if stream
                else await client.generate(request())
            )
            assert output.usage == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
            assert client.budget.snapshot()["http_attempts"] == 1
            assert SECRET not in json.dumps(client.http_events)
            assert client.http_events[0]["usage_complete"] is True
        assert len(seen) == 1

    asyncio.run(scenario())


def test_http_event_records_actual_post_times_route_and_server_metadata():
    async def scenario():
        response = body()
        response.update(
            id="chatcmpl-0123_abcd", request_id="body-request-ignored", model="qwen-test-2026-10-01"
        )

        def handler(_):
            event = client.http_events[0]
            assert event["post_started_at_utc"].endswith("Z")
            assert event["post_finished_at_utc"] is None
            return httpx.Response(
                200,
                json=response,
                headers={
                    "x-request-id": "request-0123-abcd",
                    "x-dashscope-request-id": "dashscope-ignored",
                    "x-secret": SECRET,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            before = datetime.now(timezone.utc)
            await client.generate(request())
            after = datetime.now(timezone.utc)
            event = client.http_events[0]
            started = datetime.fromisoformat(event["post_started_at_utc"].replace("Z", "+00:00"))
            finished = datetime.fromisoformat(event["post_finished_at_utc"].replace("Z", "+00:00"))
            assert before <= started <= finished <= after
            assert event["model"] == event["requested_model"] == "qwen-test"
            assert event["response_model"] == "qwen-test-2026-10-01"
            assert event["response_id"] == "chatcmpl-0123_abcd"
            assert event["request_id"] == "request-0123-abcd"
            assert event["endpoint_host"] == "example.cn-beijing.maas.aliyuncs.com"
            assert event["endpoint_region"] == "cn-beijing"
            assert event["endpoint_workspace"] == "example"
            assert not {"headers", "body", "raw_response", "api_key"}.intersection(event)
            assert SECRET not in json.dumps(event)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "headers,body_id,expected",
    [
        ({"x-dashscope-request-id": "dashscope-123"}, "body-123", "dashscope-123"),
        ({}, "body-123", "body-123"),
        (
            {"x-request-id": "not an id", "x-dashscope-request-id": "dashscope-123"},
            "body-123",
            "dashscope-123",
        ),
        ({"x-request-id": SECRET}, "body-123", "body-123"),
        ({"request-id": "not-whitelisted", "x-arbitrary-id": "not-whitelisted"}, None, None),
        ({}, None, None),
    ],
)
def test_request_id_uses_only_valid_whitelisted_metadata(headers, body_id, expected):
    async def scenario():
        response = body()
        response["request_id"] = body_id
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=response, headers=headers),
            )
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            await client.generate(request())
            event = client.http_events[0]
            assert event["request_id"] == expected
            assert event["response_id"] is None
            assert event["response_model"] is None
            assert SECRET not in json.dumps(event)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        123,
        {},
        [],
        "",
        "x" * 201,
        "bad id",
        "bad\nid",
        "bad\rid",
        "中文",
        "<script>evil()</script>",
        "https://example.test/?key=hidden",
        SECRET,
        "prefix-" + SECRET + "-suffix",
    ],
)
def test_untrusted_response_identifiers_and_model_cannot_enter_trace(value):
    async def scenario():
        response = body()
        response.update(id=value, request_id=value, model=value)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            await client.generate(request())
            event = client.http_events[0]
            assert event["response_id"] is event["request_id"] is event["response_model"] is None
            assert SECRET not in json.dumps(event)

    asyncio.run(scenario())


def test_metadata_length_limits_accept_boundary_and_reject_overlong_identifiers():
    async def scenario():
        for identifier_length in (128, 129):
            response = body()
            response.update(
                id="r" * identifier_length, request_id="q" * identifier_length, model="m" * 200
            )
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
            ) as http:
                client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
                await client.generate(request())
                event = client.http_events[0]
                assert event["response_id"] == ("r" * 128 if identifier_length == 128 else None)
                assert event["request_id"] == ("q" * 128 if identifier_length == 128 else None)
                assert event["response_model"] == "m" * 200

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "headers",
    [
        {"x-request-id": "x" * 129},
        {"x-request-id": "bad\nid"},
        {"x-request-id": "bad id"},
        {"x-request-id": "prefix-" + SECRET},
        {"x-dashscope-request-id": SECRET},
    ],
)
def test_untrusted_request_id_headers_cannot_enter_trace(headers):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=body(), headers=headers),
            )
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            await client.generate(request())
            assert client.http_events[0]["request_id"] is None
            assert SECRET not in json.dumps(client.http_events)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "endpoint,host,region,workspace",
    [
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "dashscope.aliyuncs.com", None, None),
        (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "dashscope-intl.aliyuncs.com",
            None,
            None,
        ),
        (
            "https://ws-example.us-east-1.maas.aliyuncs.com/compatible-mode/v1",
            "ws-example.us-east-1.maas.aliyuncs.com",
            "us-east-1",
            "ws-example",
        ),
        ("https://bad!host.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", None, None, None),
        (
            "https://" + SECRET + ".cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            None,
            None,
            None,
        ),
    ],
)
def test_endpoint_metadata_is_validated_and_does_not_guess_shared_route(
    endpoint, host, region, workspace
):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body()))
        ) as http:
            client = QwenLLMClient(
                QwenConfig(SECRET, endpoint, "qwen-test"), RequestBudget(1), http_client=http
            )
            await client.generate(request())
            event = client.http_events[0]
            assert event["endpoint_host"] == host
            assert event["endpoint_region"] == region
            assert event["endpoint_workspace"] == workspace
            assert SECRET not in json.dumps(event)

    asyncio.run(scenario())


@pytest.mark.parametrize("model", ["qwen-" + SECRET, "bad model\nmetadata"])
def test_configured_model_is_not_an_unfiltered_trace_channel(model):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body()))
        ) as http:
            client = QwenLLMClient(
                QwenConfig(SECRET, ENDPOINT, model), RequestBudget(1), http_client=http
            )
            native_request = ModelRequest(
                model=model, messages=[ModelMessage(role="human", content="One action")]
            )
            await client.generate(native_request)
            event = client.http_events[0]
            assert event["requested_model"] is event["model"] is None
            assert model not in json.dumps(event)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "raw", [None, {}, {"prompt_tokens": True, "completion_tokens": -1, "total_tokens": "17"}]
)
def test_missing_or_bad_usage_is_unknown_not_zero(raw):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body(usage=raw)))
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            output = await client.generate(request(100))
            assert output.usage == {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            assert client.http_events[0]["usage_complete"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_failed_request_spends_budget_and_stops_following_posts_without_secret(status):
    async def scenario():
        posts = []

        def handler(incoming):
            posts.append(incoming)
            return httpx.Response(
                status,
                json={
                    "error": SECRET,
                    "id": "error-response-123",
                    "request_id": "error-request-123",
                },
                headers={"secret": SECRET},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            budget = RequestBudget(6)
            client = QwenLLMClient(config(), budget, http_client=http)
            with pytest.raises(QwenClientError) as error:
                await client.generate(request())
            assert str(error.value) == f"QWEN_HTTP_{status}"
            with pytest.raises(QwenClientError, match="LIVE_BUDGET_STOPPED"):
                await client.generate(request())
            second_client = QwenLLMClient(config(), budget, http_client=http)
            with pytest.raises(QwenClientError, match="LIVE_BUDGET_STOPPED"):
                await second_client.generate(request())
            assert len(posts) == 1
            assert budget.snapshot()["http_attempts"] == 1
            event = client.http_events[0]
            assert event["response_id"] == "error-response-123"
            assert event["request_id"] == "error-request-123"
            assert event["post_started_at_utc"].endswith("Z")
            assert event["post_finished_at_utc"].endswith("Z")
            assert SECRET not in json.dumps(client.http_events)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": {"content": ["bad"]}}]},
    ],
)
def test_bad_response_envelope_is_safe_and_accounted(payload):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            with pytest.raises(QwenClientError, match="INVALID_RESPONSE_ENVELOPE"):
                await client.generate(request())
            assert client.http_events[0]["status"] == "failed"
            assert client.budget.used == 1

    asyncio.run(scenario())


def test_credential_echo_and_truncated_output_are_not_forwarded():
    async def scenario():
        for response, error in ((body(SECRET), "CONTAINS_CREDENTIAL"), (body(), "TRUNCATED")):
            if error == "TRUNCATED":
                response["choices"][0]["finish_reason"] = "length"
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
            ) as http:
                client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
                with pytest.raises(QwenClientError, match=error):
                    await client.generate(request())
                assert SECRET not in json.dumps(client.http_events)

    asyncio.run(scenario())


@pytest.mark.parametrize("finish_reason", [SECRET, {"secret": SECRET}, [SECRET], None])
def test_untrusted_finish_reason_cannot_enter_model_output_or_trace(finish_reason):
    async def scenario():
        response = body()
        response["choices"][0]["finish_reason"] = finish_reason
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
            with pytest.raises(QwenClientError, match="INVALID_RESPONSE_ENVELOPE"):
                await client.generate(request())
            assert SECRET not in json.dumps(client.http_events)

    asyncio.run(scenario())


def test_timeout_and_cancellation_count_actual_attempts():
    async def scenario():
        for cancellation in (False, True):

            async def handler(incoming):
                if cancellation:
                    raise asyncio.CancelledError
                raise httpx.ReadTimeout(SECRET, request=incoming)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                client = QwenLLMClient(config(), RequestBudget(1), http_client=http)
                with pytest.raises(asyncio.CancelledError if cancellation else QwenClientError):
                    await client.generate(request())
                assert client.budget.used == 1
                event = client.http_events[0]
                assert event["usage_complete"] is False
                assert event["post_started_at_utc"].endswith("Z")
                assert event["post_finished_at_utc"].endswith("Z")
                assert (
                    event["response_id"] is event["request_id"] is event["response_model"] is None
                )
                assert SECRET not in json.dumps(client.http_events)

    asyncio.run(scenario())


def test_shared_budget_atomic_across_concurrent_clients():
    async def scenario():
        posts = []

        async def handler(incoming):
            posts.append(incoming)
            await asyncio.sleep(0)
            return httpx.Response(200, json=body())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            budget = RequestBudget(1)
            clients = [QwenLLMClient(config(), budget, http_client=http) for _ in range(2)]
            outputs = await asyncio.gather(
                *(client.generate(request()) for client in clients), return_exceptions=True
            )
            assert len(posts) == 1
            assert sum(isinstance(output, QwenClientError) for output in outputs) == 1
            assert budget.used == 1

    asyncio.run(scenario())


def test_token_and_character_stop_do_not_send_extra_post():
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body()))
        ) as http:
            client = QwenLLMClient(
                config(), RequestBudget(6, max_reported_tokens=1), http_client=http
            )
            await client.generate(request())
            with pytest.raises(QwenClientError, match="REPORTED_TOKEN_LIMIT"):
                await client.generate(request())
            assert client.budget.used == 1
            client = QwenLLMClient(config(), RequestBudget(6, max_input_chars=1), http_client=http)
            with pytest.raises(QwenClientError, match="INPUT_CHARACTER_LIMIT"):
                await client.generate(request())
            assert client.budget.used == 0

    asyncio.run(scenario())


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "orders.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total REAL);"
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER,amount REAL);"
            "INSERT INTO orders VALUES(1,100),(2,100),(3,80);"
            "INSERT INTO order_lines VALUES(1,1,40),(2,1,60),(3,2,100),(4,3,20),(5,3,30),(6,3,30);"
        )
    return path


@pytest.mark.parametrize("injected", [False, True])
def test_native_loop_receives_http_client_and_feedback(database, tmp_path, injected):
    async def scenario():
        posts = []

        def handler(incoming):
            posts.append(json.loads(incoming.content))
            sql = GOOD_ORDER_SUM if injected or len(posts) == 2 else BAD_ORDER_SUM
            return httpx.Response(200, json=body(json.dumps(sql_candidate(sql))))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = QwenLLMClient(config(), RequestBudget(2), http_client=http)
            wrapped = (
                InjectedFirstCandidateClient(client, sql_candidate(BAD_ORDER_SUM))
                if injected
                else client
            )
            record = await run_native_case(
                database,
                ConfirmedMetricTask("Total order revenue", "order_revenue"),
                llm_client=wrapped,
                output_dir=tmp_path / "runtime",
            )
            assert record["publication"]["accepted_value"] == 280
            assert record["llm_calls"] == 2
            assert len(posts) == (1 if injected else 2)
            assert "max_multiplicity" in json.dumps(posts[-1]["messages"])
            assert record["actions"][0]["error_code"] == "GRAIN_MULTIPLICATION"
            assert record["model_usage"][0]["total_tokens"] == 17
            assert SECRET not in json.dumps(record)
            if injected:
                assert record["turn_origins"] == [
                    "injected_test_fault_no_api_call",
                    "real_qwen_http",
                ]

    asyncio.run(scenario())


def test_native_http_failure_never_publishes_and_does_not_leak_key(
    database, tmp_path, monkeypatch, caplog
):
    import dbgpt.agent.core.base_agent as native

    sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await sleep(0)

    monkeypatch.setattr(native.asyncio, "sleep", fast_sleep)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(401, text=SECRET))
        ) as http:
            client = QwenLLMClient(config(), RequestBudget(6), http_client=http)
            record = await run_native_case(
                database,
                ConfirmedMetricTask("Revenue", "order_revenue"),
                llm_client=client,
                output_dir=tmp_path / "runtime",
            )
            assert record["publication"]["accepted_value"] is None
            assert record["upstream_success"] is False
            assert client.budget.used == 1
            assert SECRET not in json.dumps(record)
            assert SECRET not in caplog.text

    asyncio.run(scenario())
