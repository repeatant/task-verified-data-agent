"""Re-execute public v1 model evidence without network access or gold injection.

The stored results are development observations, including a failed first run.
Scoring answers live only in this test and the evaluation package; replaying a
record does not establish the accuracy of new model calls or unseen tasks.
"""

import json
from pathlib import Path

import pytest

from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.evaluation.probe_cases import get_case
from dataverify.evaluation.probe_grading import grade_probe_record
from dataverify.evaluation.reference_audit_cases import AuditScenario, audit_scenarios
from dataverify.evaluation.reference_audit_scoring import grade_audit_run
from dataverify.verification.discovery_runtime import check_candidate, fixture_data

EVIDENCE = json.loads(
    (Path(__file__).parent / "fixtures/discovery/v1_live_validation.json").read_text(
        encoding="utf-8"
    )
)
RUNS = EVIDENCE["runs_including_failures"]
REVIEWED = [record for record in RUNS if record["status"] == "reviewed"]


def scoring_scenario(record):
    """Bind the recorded task to an author's evaluation-only scoring assumption."""
    if record["track"] == "natural":
        case = next(c for c in get_discovery_cases() if c.name == record["scenario"])
        scenario = AuditScenario(
            case.name,
            case.task,
            case.source,
            case.good_sql,
            case.good_sql,
            case.offline_plan(),
            case.offline_plan(),
            None,
            case.expected,
        )
    else:
        scenario = next(s for s in audit_scenarios() if s.name == record["scenario"])
    assert scenario.task.public_input() == record["task"]
    assert scenario.source == fixture_data(record["source"])
    return scenario


@pytest.mark.parametrize("record", REVIEWED, ids=lambda record: record["scenario"])
def test_published_sql_and_both_references_match_seven_scoring_datasets(record):
    grade = grade_audit_run(scoring_scenario(record), record)
    assert grade == record["three_way_development_grade"]
    assert grade["candidate_passes_all"]
    assert grade["reference_passes_all"]
    assert grade["independent_passes_all"]
    assert len(grade["checks"]) == 7


@pytest.mark.parametrize("record", REVIEWED, ids=lambda record: record["scenario"])
def test_model_generated_fixtures_replay_with_the_published_candidate(record):
    result = check_candidate(
        record["publication"]["sql"],
        record["plan"],
        record["fixtures"],
        source=fixture_data(record["source"]),
    )
    assert result["all_match"]


def test_failed_first_audit_stays_unpublished_and_visible():
    failed = [record for record in RUNS if record["status"] != "reviewed"]
    assert len(failed) == 1
    record = failed[0]
    assert record["run"] == "01-wrong-reference"
    assert record["status"] == record["publication"]["status"] == "inconclusive"
    assert record["publication"]["rows"] is None
    assert record["audit_plan"] is None
    assert record["reference_revisions"] == 0
    assert any(step.get("error") for step in record["trace"])
    grade = grade_audit_run(scoring_scenario(record), record)
    assert grade == record["three_way_development_grade"]
    assert grade["candidate_passes_all"]
    assert not grade["reference_passes_all"]
    assert not grade["independent_passes_all"]


def test_wrong_reference_is_replaced_without_rewriting_correct_sql_or_old_version():
    record = next(r for r in REVIEWED if r["scenario"] == "wrong_reference_rounding")
    versions = record["plan_versions"]
    assert record["publication"]["sql"] == record["initial_sql"]
    assert record["reference_revisions"] == 1
    assert [(v["version"], v["status"]) for v in versions] == [
        (1, "superseded"),
        (2, "active"),
    ]
    assert versions[-1]["plan"] == record["plan"]
    assert versions[0]["plan"]["reference"] != record["plan"]["reference"]
    assert record["plan"]["reference"] == record["audit_plan"]["reference"]
    assert {f["name"] for f in versions[0]["plan"]["fixtures"]} <= {
        f["name"] for f in record["fixtures"]
    }
    assert not check_candidate(
        record["initial_sql"],
        versions[0]["plan"],
        record["fixtures"],
        source=fixture_data(record["source"]),
    )["all_match"]
    replacement = next(t for t in record["trace"] if t["node"] == "replace_reference")
    assert replacement["candidate_unchanged"]


def test_wrong_sql_is_repaired_without_replacing_the_reference():
    record = next(r for r in REVIEWED if r["scenario"] == "wrong_sql_grain")
    assert record["publication"]["sql"] != record["initial_sql"]
    assert record["reference_revisions"] == 0
    assert len(record["plan_versions"]) == 1
    assert not check_candidate(
        record["initial_sql"],
        record["plan"],
        record["fixtures"],
        source=fixture_data(record["source"]),
    )["all_match"]
    attribution = next(t["decision"] for t in record["trace"] if t["node"] == "attribute")
    assert attribution["target"] == "candidate"
    assert attribution["evidence_level"] == "model_supported_not_proven"


def test_requested_rounding_and_natural_generation_are_not_mislabeled():
    rounded = next(r for r in REVIEWED if r["scenario"] == "requested_rounding")
    assert rounded["publication"]["sql"] == rounded["initial_sql"]
    assert rounded["reference_revisions"] == 0
    natural = next(r for r in REVIEWED if r["track"] == "natural")
    assert natural["scenario"] == "distinct_orders"
    assert natural["publication"]["rows"] == [[3]]
    assert natural["controlled_injections"]["initial_sql"] is False
    assert natural["controlled_injections"]["initial_reference"] is False
    assert {event["role"] for event in natural["http_events"]} == {
        "writer",
        "reviewer",
        "auditor",
    }


def test_fixed_graph_resume_publishes_rechecked_sql_without_new_model_calls():
    paused, resumed = EVIDENCE["fixed_checkpoint_processes"]
    assert paused["status"] == "paused"
    assert paused["publication"]["rows"] is None
    assert resumed["status"] == "completed"
    assert paused["model_attempts"] == resumed["model_attempts"] == 1
    assert len(paused["http_events"]) == 1
    assert resumed["http_events"] == []
    assert resumed["source_integrity_ok"] and resumed["audit_ledger_complete"]
    grade = grade_probe_record(get_case(resumed["case_id"]), resumed)
    assert grade["correct_task"]
    assert grade["source_correct"] and grade["boundary_correct"]
    assert grade["publication_matches_sql"] and not grade["wrong_release"]


def test_usage_counts_failed_calls_once_and_omits_private_runtime_fields():
    records = RUNS + EVIDENCE["fixed_checkpoint_processes"]
    events = [event for record in records for event in record["http_events"]]
    usage = EVIDENCE["incremental_usage"]
    assert len(events) == usage["http_requests"] == 26
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert sum(event["usage"][field] for event in events) == usage[field]
    assert usage["total_tokens"] == 53194
    assert all(set(event) == {"role", "status", "usage"} for event in events)
    assert all(event["status"] == "completed" for event in events)
    assert all(
        event["usage"]["prompt_tokens"] + event["usage"]["completion_tokens"]
        == event["usage"]["total_tokens"]
        for event in events
    )

    forbidden = {
        "api_key",
        "base_url",
        "endpoint",
        "request_id",
        "writer_messages",
        "reviewer_messages",
        "audit_messages",
        "model_contexts",
        "budget",
        "budget_start",
        "budget_end",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(EVIDENCE)
