"""Mock-only strict-shape protocol checks; no provider capability claim or paid API."""

import asyncio
import json

import httpx
import pytest
from test_graph_model import ACTION, MESSAGES, SECRET, FakeBudget, config, response_body

from dataverify.agent.discovery_model import QwenReviewModel
from dataverify.agent.graph_model import GraphModelError, InjectedFirstGraphModel, QwenGraphModel
from dataverify.agent.structured_schemas import reviewer_schema, writer_schema
from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import DiscoveryTask, validate_plan


class RecordingBudget(FakeBudget):
    def __init__(self):
        super().__init__(max_output_tokens=1800)
        self.reservations = []

    def reserve(self, prompt_chars):
        self.reservations.append(prompt_chars)
        return super().reserve(prompt_chars)


def accepts(schema, value, root=None):
    """Test-only interpretation of the small keyword subset used by our schema.

    Production uses its independent existing action/DSL/fixture validators.
    This helper is not a replacement general-purpose JSON Schema library.
    """
    root = root or schema
    if "$ref" in schema:
        return accepts(root["$defs"][schema["$ref"].split("/")[-1]], value, root)
    if "anyOf" in schema and not any(accepts(branch, value, root) for branch in schema["anyOf"]):
        return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    kinds = {
        "object": lambda x: type(x) is dict,
        "array": lambda x: type(x) is list,
        "string": lambda x: type(x) is str,
        "number": lambda x: type(x) in (int, float),
        "integer": lambda x: type(x) is int,
        "null": lambda x: x is None,
    }
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(kinds[kind](value) for kind in types):
            return False
    if type(value) is dict:
        props = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value):
            return False
        if not schema.get("minProperties", 0) <= len(value) <= schema.get("maxProperties", 10**9):
            return False
        for key, item in value.items():
            child = props.get(key, schema.get("additionalProperties", True))
            if child is False or (isinstance(child, dict) and not accepts(child, item, root)):
                return False
    if type(value) is list:
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10**9):
            return False
        if "items" in schema and not all(accepts(schema["items"], item, root) for item in value):
            return False
    if type(value) is str and not schema.get("minLength", 0) <= len(value) <= schema.get(
        "maxLength", 10**9
    ):
        return False
    if type(value) in (int, float) and not schema.get(
        "minimum", -float("inf")
    ) <= value <= schema.get("maximum", float("inf")):
        return False
    return True


PLAN = {
    "action": "plan",
    "basis": [{"source": "question", "quote": "Sum rows."}],
    "assumptions": [],
    "reference": "[[sum(r['amount'] for r in tables['t'])]]",
    "fixtures": [],
}
PATCH = {
    "action": "probe",
    "reason": "Check sign handling.",
    "fixtures": [
        {
            "name": "negative",
            "data": {
                "base": "bound_source",
                "changes": [{"table": "t", "row": 0, "values": {"amount": -2}}],
            },
        }
    ],
}


@pytest.mark.parametrize(
    "action",
    [
        PLAN,
        PATCH,
        {
            "action": "probe",
            "reason": "Check a new dataset.",
            "fixtures": [{"name": "plain", "data": {"t": [{"amount": 2}]}}],
        },
        {"action": "finish", "reason": "All observed calculations agree."},
        {"action": "repair", "reason": "An observed difference needs attribution."},
        {"action": "clarify", "reason": "The denominator is unspecified."},
        {
            "action": "attribute",
            "target": "candidate",
            "reason": "Observed contradiction.",
            "basis": [{"source": "question", "quote": "Sum rows."}],
            "checks": ["data_example"],
        },
    ],
)
def test_all_review_action_shapes_are_supported(action):
    assert accepts(reviewer_schema(), action)


@pytest.mark.parametrize(
    "action",
    [
        {"action": "finish", "reason": "Done", "reference": "[[1]]"},
        {"action": "review", "reason": "Done"},
        {"action": "probe", "reason": "x", "fixtures": []},
        {
            "action": "probe",
            "reason": "x",
            "fixtures": [{"name": "x", "data": {"base": "file", "changes": []}}],
        },
        {
            "action": "probe",
            "reason": "x",
            "fixtures": [
                {
                    "name": "x",
                    "data": {
                        "base": "bound_source",
                        "changes": [{"table": "t", "row": -1, "values": {"x": 1}}],
                    },
                }
            ],
        },
        {
            "action": "probe",
            "reason": "x",
            "fixtures": [
                {
                    "name": "x",
                    "data": {
                        "base": "bound_source",
                        "changes": [
                            {"table": "t", "row": 0, "values": {"x": 1}, "expected": [[1]]}
                        ],
                    },
                }
            ],
        },
        {"action": "attribute", "target": "sql", "reason": "x", "basis": [], "checks": []},
        {**PLAN, "assumptions": "none"},
        {**PLAN, "unexpected": "field"},
    ],
)
def test_schema_rejects_known_structural_errors(action):
    assert not accepts(reviewer_schema(), action)


def test_writer_optional_thought_and_no_extra_fields():
    assert accepts(writer_schema(), ACTION)
    assert accepts(writer_schema(), {**ACTION, "thought": ""})
    assert not accepts(writer_schema(), {**ACTION, "reason": "unsupported"})
    assert not accepts(writer_schema(), {**ACTION, "task_id": 42})


@pytest.mark.parametrize(
    "model_class, action, schema",
    [
        (QwenGraphModel, ACTION, writer_schema()),
        (QwenReviewModel, PLAN, reviewer_schema()),
    ],
)
def test_transport_reserves_schema_chars_and_records_exact_schema(model_class, action, schema):
    async def scenario():
        budget = RecordingBudget()
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            fmt = payload["response_format"]
            expected_chars = sum(len(item["content"]) for item in MESSAGES) + len(
                json.dumps(fmt, ensure_ascii=False, separators=(",", ":"))
            )
            assert budget.reservations == [expected_chars]
            assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
            assert fmt["json_schema"]["schema"] == schema
            assert accepts(schema, action)
            assert payload["max_tokens"] == 1800 and "tools" not in payload
            seen.append(payload)
            return httpx.Response(200, json=response_body(json.dumps(action)))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = model_class(
                config(), budget, http_client=http, max_tokens=1800, structured_output=True
            )
            await model.generate(MESSAGES)
            event = model.http_events[0]
            assert event["schema_digest"] == digest_json(schema)
            assert event["reserved_input_chars"] == budget.reservations[0]
            assert event["schema_chars"] + event["message_chars"] == budget.reservations[0]
            assert event["response_format"] == "json_schema"
            assert SECRET not in json.dumps(event)
            assert len(seen) == 1
            await model.aclose()

    asyncio.run(scenario())


def test_default_mode_does_not_add_schema_reservation_or_audit_fields():
    async def scenario():
        budget = RecordingBudget()

        def handler(request):
            assert json.loads(request.content)["response_format"] == {"type": "json_object"}
            return httpx.Response(200, json=response_body())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = QwenGraphModel(config(), budget, http_client=http)
            await model.generate(MESSAGES)
            assert budget.reservations == [sum(len(item["content"]) for item in MESSAGES)]
            assert "schema_digest" not in model.http_events[0]
            assert model.client_kind == "real_qwen_graph_http"
            await model.aclose()

    asyncio.run(scenario())


def test_strict_mode_cannot_bypass_local_validation_or_usage_accounting():
    async def scenario():
        budget = RecordingBudget()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=response_body('{"sql":1,"task_id":"x"}'))
            )
        ) as http:
            model = QwenGraphModel(config(), budget, http_client=http, structured_output=True)
            with pytest.raises(GraphModelError, match="INVALID_ACTION_FIELDS"):
                await model.generate(MESSAGES)
            assert len(budget.usage) == 1
            assert model.http_events[0]["status"] == "completed"
            assert model.http_events[0]["action_error_code"] == "GRAPH_INVALID_ACTION_FIELDS"
            await model.aclose()

    asyncio.run(scenario())


def test_schema_does_not_prove_dsl_or_business_semantics():
    bad = {**PLAN, "reference": "__import__('os')"}
    assert accepts(reviewer_schema(), bad)
    task = DiscoveryTask(
        "x", "Sum rows.", "All rows.", schema={"t": ["amount"]}, allow_source_only_plan=True
    )
    with pytest.raises(ValueError):
        validate_plan(bad, task)


def test_mode_changes_checkpoint_identity_including_injection_wrappers():
    plain = QwenGraphModel(config(), RecordingBudget())
    strict = QwenGraphModel(config(), RecordingBudget(), structured_output=True)
    reviewer = QwenReviewModel(config(), RecordingBudget(), structured_output=True)
    assert plain.client_kind != strict.client_kind != reviewer.client_kind
    assert (
        InjectedFirstGraphModel(plain, ACTION).client_kind
        != InjectedFirstGraphModel(strict, ACTION).client_kind
    )
    assert InjectedFirstReviewModel(reviewer, PLAN).client_kind.endswith("_json_schema_v1")


def test_schema_input_is_bounded_before_transport():
    async def scenario():
        budget = RecordingBudget()
        model = QwenReviewModel(config(), budget, structured_output=True)
        with pytest.raises(GraphModelError, match="INPUT_TOO_LARGE"):
            await model.generate([{"role": "user", "content": "x" * 39999}])
        assert budget.used == 0
        await model.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status,finish_reason,code",
    [
        (400, "stop", "GRAPH_QWEN_HTTP_400"),
        (200, "length", "GRAPH_QWEN_RESPONSE_TRUNCATED"),
    ],
)
def test_strict_provider_rejection_or_truncation_never_falls_back(status, finish_reason, code):
    async def scenario():
        budget = RecordingBudget()
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(status, json=response_body(finish_reason=finish_reason))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            model = QwenGraphModel(
                config(), budget, http_client=http, structured_output=True, max_tokens=1800
            )
            with pytest.raises(GraphModelError, match=code):
                await model.generate(MESSAGES)
            assert len(calls) == 1 and budget.used == 1
            assert budget.stopped_reason == "PROVIDER_ERROR"
            assert model.http_events[0]["schema_chars"] > 0
            await model.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [1, "true", None])
def test_strict_mode_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        QwenGraphModel(config(), RecordingBudget(), structured_output=value)
