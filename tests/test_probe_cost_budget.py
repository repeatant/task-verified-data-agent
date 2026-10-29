"""Budget/config checks only; fake credentials, no live endpoint calls."""

import asyncio
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import httpx
import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)

from dbgpt.core import ModelMessage, ModelRequest

from dataverify.dbgpt_qwen import QwenClientError, QwenConfig, QwenLLMClient
from dataverify.probe_cost_budget import MonetaryRequestBudget, validate_pricing

SECRET = "sk-test-never-real-budget-credential"
BEIJING = "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def usage(prompt=1200, completion=120):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


@pytest.mark.parametrize("endpoint", [BEIJING, "https://dashscope.aliyuncs.com/compatible-mode/v1"])
def test_only_approved_route_and_model_have_price_metadata(endpoint):
    result = validate_pricing(QwenConfig(SECRET, endpoint))
    assert result["model"] == "qwen3.8-flash"
    assert result["region"] == "cn-beijing"
    assert SECRET not in json.dumps(result)
    assert endpoint not in json.dumps(result)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://ws-test.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1",
        "https://ws-test.cn-beijing.maas.aliyuncs.com.fake.maas.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_unpriced_regions_and_suffix_confusion_rejected(endpoint):
    with pytest.raises(QwenClientError, match="PRICE_REGION_NOT_APPROVED"):
        validate_pricing(QwenConfig(SECRET, endpoint))


def test_other_model_and_subscription_key_rejected():
    with pytest.raises(QwenClientError, match="PRICE_MODEL_NOT_APPROVED"):
        validate_pricing(QwenConfig(SECRET, BEIJING, "qwen-test"))
    with pytest.raises(QwenClientError, match="SUBSCRIPTION_KEY_REJECTED"):
        validate_pricing(QwenConfig("sk-sp-fake", BEIJING))


@pytest.mark.parametrize(
    "amount", [0, -1, 5.01, float("nan"), float("inf"), True, "5", Decimal("0.0000000001")]
)
def test_invalid_or_above_authorized_money_allowance_rejected(amount):
    with pytest.raises(ValueError):
        MonetaryRequestBudget(money_limit_cny=amount)


@pytest.mark.parametrize("output", [0, 901, True, 3.2])
def test_output_cap_must_fit_fixed_experiment_allowance(output):
    with pytest.raises(ValueError):
        MonetaryRequestBudget(max_output_tokens=output)


@pytest.mark.parametrize("chars", [-1, True, 1.2, "3"])
def test_bad_character_count_does_not_reserve(chars):
    budget = MonetaryRequestBudget()
    with pytest.raises(ValueError):
        budget.reserve(chars)
    assert budget.snapshot()["http_attempts"] == 0


def test_reserves_conservative_cost_before_request_and_does_not_refund():
    budget = MonetaryRequestBudget()
    assert budget.reserve(1000) == 1
    before = budget.snapshot()
    assert before["reserved_upper_cny"] == pytest.approx((8096 * 0.8 + 900 * 2.7) / 1_000_000)
    assert not before["reported_cost_complete"]
    budget.record_tokens(usage())
    after = budget.snapshot()
    assert after["reported_total_tokens"] == 1320
    assert after["estimated_reported_cny"] == pytest.approx((1200 * 0.8 + 120 * 2.7) / 1_000_000)
    assert after["reserved_upper_cny"] == before["reserved_upper_cny"]
    assert after["reported_cost_complete"]
    assert after["complete_count"] == 1


def test_tenth_of_yuan_example_has_exact_decimal_accounting():
    budget = MonetaryRequestBudget(max_output_tokens=900)
    for _ in range(20):
        budget.reserve(1500)
        budget.record_tokens(usage(4500, 500))
    result = budget.snapshot()
    assert result["reported_total_tokens"] == 100000
    assert result["estimated_reported_cny"] == 0.099


def test_money_limit_prevents_post_without_incrementing_ledger():
    budget = MonetaryRequestBudget(money_limit_cny=0.01)
    budget.reserve(0)
    budget.record_tokens(usage(1, 1))
    previous = budget.snapshot()
    with pytest.raises(QwenClientError, match="MONEY_RESERVATION_LIMIT"):
        budget.reserve(1000)
    current = budget.snapshot()
    assert current["http_attempts"] == previous["http_attempts"]
    assert current["reserved_upper_cny"] == previous["reserved_upper_cny"] <= 0.01


def test_parent_request_limit_remains_active():
    budget = MonetaryRequestBudget(limit=1)
    budget.reserve(10)
    budget.record_tokens(usage(1, 1))
    with pytest.raises(QwenClientError, match="LIVE_HTTP_REQUEST_LIMIT"):
        budget.reserve(10)


def test_parent_input_character_limit_remains_active():
    budget = MonetaryRequestBudget(max_input_chars=10)
    with pytest.raises(QwenClientError, match="LIVE_INPUT_CHARACTER_LIMIT"):
        budget.reserve(11)
    assert budget.used == 0


def test_parent_reported_token_stop_remains_active():
    budget = MonetaryRequestBudget(max_reported_tokens=3)
    budget.reserve(10)
    budget.record_tokens(usage(2, 1))
    with pytest.raises(QwenClientError, match="LIVE_REPORTED_TOKEN_LIMIT"):
        budget.reserve(10)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        None,
        {"prompt_tokens": 2, "completion_tokens": None, "total_tokens": 2},
        {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
        {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
        {"prompt_tokens": 1.0, "completion_tokens": 1, "total_tokens": 2},
    ],
)
def test_missing_invalid_usage_is_unknown_and_stops_next_call(bad):
    budget = MonetaryRequestBudget()
    budget.reserve(10)
    reserved = budget.snapshot()["reserved_upper_cny"]
    budget.record_tokens(bad)
    result = budget.snapshot()
    assert result["unknown_count"] == 1 and result["complete_count"] == 0
    assert not result["reported_cost_complete"]
    assert result["estimated_reported_cny"] == 0
    assert result["reserved_upper_cny"] == reserved
    with pytest.raises(QwenClientError, match="LIVE_BUDGET_STOPPED_USAGE_UNAVAILABLE"):
        budget.reserve(10)


def test_inconsistent_usage_is_not_accepted_as_a_free_call():
    budget = MonetaryRequestBudget()
    budget.reserve(10)
    budget.record_tokens({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 1})
    assert budget.snapshot()["stopped_reason"] == "USAGE_INCONSISTENT"
    assert budget.snapshot()["unknown_count"] == 1


@pytest.mark.parametrize("prompt,completion", [(4200, 1), (1, 901)])
def test_actual_usage_above_reservation_is_recorded_but_stops(prompt, completion):
    budget = MonetaryRequestBudget()
    budget.reserve(1)  # 4,100 reserved input tokens
    budget.record_tokens(usage(prompt, completion))
    result = budget.snapshot()
    assert result["stopped_reason"] == "USAGE_EXCEEDED_RESERVATION"
    assert result["complete_count"] == 1
    assert result["estimated_reported_cny"] > 0


def test_provider_failure_retains_reservation_as_unknown():
    budget = MonetaryRequestBudget()
    budget.reserve(123)
    budget.stop("TRANSPORT_ERROR")
    result = budget.snapshot()
    assert result["unknown_count"] == 1
    assert result["reservation_ledger"][0]["status"] == "usage_unknown"
    assert result["reserved_upper_cny"] > 0
    assert not result["reported_cost_complete"]


def test_stop_reason_and_snapshot_do_not_export_credential_shaped_text():
    budget = MonetaryRequestBudget()
    budget.reserve(10)
    budget.stop(SECRET)
    serialized = json.dumps(budget.snapshot())
    assert SECRET not in serialized
    assert "STOP_REASON_REDACTED" in serialized


def test_unmatched_usage_stops_without_double_counting():
    budget = MonetaryRequestBudget()
    budget.reserve(10)
    budget.record_tokens(usage(1, 1))
    with pytest.raises(QwenClientError, match="UNRESERVED_USAGE"):
        budget.record_tokens(usage(1, 1))
    result = budget.snapshot()
    assert result["complete_count"] == 1
    assert result["reported_total_tokens"] == 2
    assert result["unmatched_usage_events"] == 1
    assert not result["reported_cost_complete"]


def test_concurrent_reserves_do_not_misattribute_out_of_order_usage():
    budget = MonetaryRequestBudget()

    def reserve_one(_):
        try:
            return budget.reserve(10)
        except QwenClientError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve_one, range(8)))
    assert results.count(1) == 1
    assert results.count("LIVE_IN_FLIGHT_RESERVATION") == 7
    budget.record_tokens(usage(1, 1))
    assert budget.reserve(10) == 2


def test_snapshot_cannot_mutate_private_reservation_ledger():
    budget = MonetaryRequestBudget()
    budget.reserve(10)
    result = budget.snapshot()
    result["reservation_ledger"][0]["input_token_upper"] = 0
    assert budget.snapshot()["reservation_ledger"][0]["input_token_upper"] == 4136


def test_existing_http_client_reserves_before_post_and_records_usage():
    async def scenario():
        budget = MonetaryRequestBudget(limit=1)
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            assert budget.snapshot()["http_attempts"] == 1
            assert budget.snapshot()["reserved_upper_cny"] > 0
            assert payload["max_tokens"] == 900
            seen.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"sql":"SELECT 1"}'}, "finish_reason": "stop"}
                    ],
                    "usage": usage(500, 20),
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            config = QwenConfig(SECRET, BEIJING)
            validate_pricing(config)
            client = QwenLLMClient(config, budget, http_client=http, max_tokens=900)
            request = ModelRequest(
                model=config.model, messages=[ModelMessage(role="human", content="Return SQL")]
            )
            output = await client.generate(request)
            assert output.usage == usage(500, 20)
            assert budget.snapshot()["complete_count"] == 1
            assert budget.snapshot()["estimated_reported_cny"] == 0.000454
            assert SECRET not in json.dumps(client.http_events)
        assert seen == [1]

    asyncio.run(scenario())


def test_insufficient_money_causes_no_actual_post():
    async def scenario():
        seen = []
        budget = MonetaryRequestBudget(money_limit_cny=0.0001)

        def handler(_):
            seen.append(1)
            raise AssertionError("No HTTP post may occur after failed reservation")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            config = QwenConfig(SECRET, BEIJING)
            client = QwenLLMClient(config, budget, http_client=http, max_tokens=900)
            request = ModelRequest(
                model=config.model, messages=[ModelMessage(role="human", content="Return SQL")]
            )
            with pytest.raises(QwenClientError, match="MONEY_RESERVATION_LIMIT"):
                await client.generate(request)
        assert not seen
        assert budget.snapshot()["http_attempts"] == 0

    asyncio.run(scenario())


def test_missing_usage_blocks_the_next_real_http_attempt():
    async def scenario():
        seen = []
        budget = MonetaryRequestBudget()

        def handler(_):
            seen.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"sql":"SELECT 1"}'}, "finish_reason": "stop"}
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            config = QwenConfig(SECRET, BEIJING)
            client = QwenLLMClient(config, budget, http_client=http)
            request = ModelRequest(
                model=config.model, messages=[ModelMessage(role="human", content="Return SQL")]
            )
            await client.generate(request)
            assert client.http_events[0]["usage_complete"] is False
            with pytest.raises(QwenClientError, match="STOPPED_USAGE_UNAVAILABLE"):
                await client.generate(request)
        assert seen == [1]
        assert not budget.snapshot()["reported_cost_complete"]

    asyncio.run(scenario())
