"""Replay the frozen live cohort offline, including refusals and reference errors.

The archived fixture contains real model programs from one trial per case.
These tests reproduce that record, not new model requests or production accuracy.
"""

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from dataverify.evaluation.frozen_cases import grade_frozen_run
from dataverify.evaluation.frozen_experiment import aggregate, baseline_gates, experiment_cases
from dataverify.runtime.reporting import summarize_usage

EVIDENCE = json.loads(
    (Path(__file__).parent / "fixtures/discovery/v1_frozen_evaluation.json").read_text(
        encoding="utf-8"
    )
)
ROWS = EVIDENCE["rows"]
CASES = {case.case_id: case for case in experiment_cases()}


def reconstruct_record(row):
    """Restore only the fields consumed by the post-run scorer."""
    programs = row["programs"]
    return {
        "status": row["status"],
        "candidate": {"sql": programs["candidate"]},
        "publication": {
            "status": row["status"],
            "sql": programs["published_sql"],
            "rows": programs["published_rows"],
        },
        "plan": {"reference": programs["reference"]},
        "audit_plan": {"reference": programs["audit"]},
    }


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["case_id"])
def test_public_programs_reproduce_the_recorded_score_and_baselines(row):
    case = CASES[row["case_id"]]
    assert grade_frozen_run(case, reconstruct_record(row)) == row["score"]
    assert baseline_gates(case) == row["baseline"]
    assert row["family"] == case.family
    assert row["condition"] == case.condition
    assert row["score"]["scoring_dataset_count"] == 31
    assert summarize_usage(row["http_events"]) == row["usage"]


def test_cohort_keeps_all_twenty_attempts_with_four_distinct_intents():
    assert EVIDENCE["planned"] == EVIDENCE["attempted"] == len(ROWS) == 20
    assert EVIDENCE["complete"] is True
    assert [row["case_id"] for row in ROWS] == list(CASES)
    assert len({row["case_id"] for row in ROWS}) == 20
    assert len({row["family"] for row in ROWS}) == EVIDENCE["distinct_intents"] == 4
    assert EVIDENCE["independent_trials_per_case"] == 1
    assert Counter(row["condition"] for row in ROWS) == {
        "sql_wrong": 4,
        "reference_wrong": 4,
        "correct": 4,
        "correlated_wrong": 4,
        "natural": 4,
    }
    assert all(row["runner_error"] is None for row in ROWS)
    computed = aggregate(ROWS)
    for key, value in computed.items():
        assert EVIDENCE[key] == value


def test_all_seventyeight_http_attempts_and_tokens_are_retained():
    events = [event for row in ROWS for event in row["http_events"]]
    assert [event["attempt"] for event in events] == list(range(1, 79))
    assert len(events) == 78
    assert all(event["usage_complete"] is True for event in events)
    assert all(event["http_status"] == 200 for event in events)
    assert {event["model"] for event in events} == {EVIDENCE["model"]}
    assert {event["role"] for event in events} == {"writer", "reviewer", "auditor"}
    for event in events:
        usage = event["usage"]
        assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
    usage = summarize_usage(events)
    assert usage == EVIDENCE["usage"]
    assert usage["event_count"] == 78
    assert usage["unknown_usage_events"] == 0
    assert usage["usage_complete"] is True
    assert usage["prompt_tokens_known"] == 117632
    assert usage["completion_tokens_known"] == 25450
    assert usage["total_tokens_known"] == 143082


def test_cost_is_usage_estimate_not_a_cloud_bill():
    usage, cost = EVIDENCE["usage"], EVIDENCE["incremental_cost"]
    pricing = cost["pricing"]
    estimated = (
        Decimal(usage["prompt_tokens_known"])
        * Decimal(str(pricing["input_cny_per_million_tokens"]))
        + Decimal(usage["completion_tokens_known"])
        * Decimal(str(pricing["output_cny_per_million_tokens"]))
    ) / Decimal(1_000_000)
    assert estimated == Decimal(str(cost["estimated_cny"])) == Decimal("0.1628206")
    assert cost["reserved_cny"] >= cost["estimated_cny"]
    assert cost["is_cloud_bill"] is False


def test_eight_single_faults_are_fixed_on_the_correct_side():
    sql_wrong = [row for row in ROWS if row["condition"] == "sql_wrong"]
    reference_wrong = [row for row in ROWS if row["condition"] == "reference_wrong"]
    assert len(sql_wrong) == len(reference_wrong) == 4
    for row in sql_wrong:
        assert row["score"]["joint_validated_delivery"]
        assert row["sql_text_changed"]
        assert row["reference_revisions"] == 0
    for row in reference_wrong:
        assert row["score"]["joint_validated_delivery"]
        assert not row["sql_text_changed"]
        assert row["reference_revisions"] == 1


def test_correlated_faults_are_blocked_but_not_counted_as_repaired():
    rows = [row for row in ROWS if row["condition"] == "correlated_wrong"]
    assert len(rows) == 4
    for row in rows:
        assert row["baseline"]["sql_executable_wrong_release"]
        assert row["baseline"]["single_reference_wrong_release"]
        assert row["status"] == "needs_clarification"
        assert row["reason"] == "ATTRIBUTION_CITES_NO_RELEVANT_DISAGREEMENT"
        assert not row["score"]["correct_delivery"]
        assert not row["score"]["candidate"]["passes_all"]
        assert not row["score"]["reference"]["passes_all"]
        assert row["score"]["audit"]["passes_all"]
        assert not row["score"]["wrong_release"]
        # The publication envelope retains the candidate for diagnosis; its
        # non-reviewed status and absent rows mean it is NOT an accepted result.
        assert row["programs"]["published_sql"] == row["programs"]["candidate"]
        assert row["programs"]["published_rows"] is None


def test_correct_sql_is_not_damaged_but_one_correct_pair_is_not_delivered():
    protected = [row for row in ROWS if row["condition"] in {"correct", "reference_wrong"}]
    assert len(protected) == 8
    assert all(row["score"]["initial_candidate"]["passes_all"] for row in protected)
    assert all(row["score"]["candidate"]["passes_all"] for row in protected)
    assert not any(row["score"]["correct_sql_damaged"] for row in protected)
    assert sum(row["score"]["correct_delivery"] for row in protected) == 7
    correct_pairs = [row for row in protected if row["condition"] == "correct"]
    assert sum(row["score"]["correct_delivery"] for row in correct_pairs) == 3
    blocked = [row for row in correct_pairs if not row["score"]["correct_delivery"]]
    assert [row["case_id"] for row in blocked] == ["money_units__correct"]
    assert blocked[0]["status"] == "inconclusive"
    assert blocked[0]["reason"] == "REVIEW_BUDGET_EXHAUSTED"
    assert blocked[0]["score"]["initial_correct_sql_not_delivered"]


def test_natural_sql_delivery_and_joint_validation_have_different_numerators():
    natural = [row for row in ROWS if row["condition"] == "natural"]
    assert len(natural) == 4
    assert all(row["score"]["initial_candidate"] is None for row in natural)
    assert all(row["baseline"] == {} for row in natural)
    assert sum(row["score"]["correct_delivery"] for row in natural) == 3
    assert sum(row["score"]["joint_validated_delivery"] for row in natural) == 2
    reference_failure = next(row for row in natural if row["case_id"] == "order_average__natural")
    assert reference_failure["score"]["correct_delivery"]
    assert not reference_failure["score"]["reference"]["passes_all"]
    assert (
        "noninteger_average_and_units" in reference_failure["score"]["reference"]["failed_datasets"]
    )
    refused = next(row for row in natural if row["case_id"] == "money_units__natural")
    assert refused["status"] == "inconclusive"
    assert refused["score"]["candidate"]["passes_all"]
    assert not refused["score"]["correct_delivery"]


def test_public_projection_contains_no_model_messages_or_endpoint_fields():
    forbidden_keys = {
        "api_key",
        "base_url",
        "model_contexts",
        "writer_messages",
        "reviewer_messages",
        "audit_messages",
        "prior_journal",
        "request_id",
        "authorization",
    }

    def check_keys(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(EVIDENCE)
    assert not any(row["score"]["wrong_release"] for row in ROWS)
    assert len([row for row in ROWS if row["status"] != "reviewed"]) == 6
