"""Prompt copies retain every execution observation while deduplicating schema."""

from copy import deepcopy

from dataverify.agent.discovery_context import model_context
from dataverify.agent.discovery_model import plan_validation_feedback
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.relational_data import RelationalData


def example():
    schema = {"items": ["id", "amount"]}
    data = RelationalData(
        schema, {"items": [[1, -3]]}, {"items": ["INTEGER", "INTEGER"]}, {"items": ["id"]}
    )
    task = DiscoveryTask(
        "task",
        "Sum amounts.",
        "Include negatives.",
        schema=schema,
        column_types=data.column_types,
        primary_keys=data.primary_keys,
    )
    return task, data


def test_context_preserves_rows_failures_ids_and_durable_evidence():
    task, data = example()
    original = {
        "task": task.public_input(),
        "samples": [data.to_dict()],
        "checks": [
            {
                "check_id": "c1",
                "data": data.to_dict(),
                "candidate": {"status": "error", "code": "bad"},
                "reference": {"rows": [[-3]]},
            }
        ],
    }
    before = deepcopy(original)
    compact = model_context(original)
    assert original == before
    assert compact["task"] == original["task"]
    assert compact["samples"] == [{"items": [[1, -3]]}]
    assert compact["checks"][0]["data"] == {"items": [[1, -3]]}
    assert compact["checks"][0]["candidate"] == original["checks"][0]["candidate"]
    assert compact["checks"][0]["reference"] == original["checks"][0]["reference"]
    assert compact["checks"][0]["check_id"] == "c1"


def test_legacy_data_and_arbitrary_mappings_are_unchanged():
    payload = {
        "orders": [[1, 3, "web"]],
        "lines": [],
        "nested": {"schema": "column value", "tables": "column value"},
    }
    assert model_context(payload) == payload


def test_constraint_feedback_contains_metadata_not_computed_answers():
    task, _ = example()
    feedback = plan_validation_feedback({}, "RELATIONAL_PRIMARY_KEY", task)
    assert feedback["primary_keys"] == {"items": ["id"]}
    assert set(feedback["schema"]) == {"items"}
    assert "expected" not in feedback and "reference" not in feedback
    assert "unique/non-null" in feedback["instruction"]
