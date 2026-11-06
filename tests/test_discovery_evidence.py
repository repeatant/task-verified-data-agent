"""Replay recorded generated checks without API calls or hidden answer injection."""

import json
from pathlib import Path

import pytest

from dataverify.verification.discovery_runtime import check_candidate, fixture_data

EVIDENCE = json.loads(
    (Path(__file__).parent / "fixtures/discovery/live_evidence.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "trace", EVIDENCE["successful_traces"], ids=lambda x: x["run"] + "-" + x["task"]["task_id"]
)
def test_recorded_generated_tests_replay(trace):
    result = check_candidate(
        trace["sql_after"], trace["plan"], trace["fixtures"], source=fixture_data(trace["source"])
    )
    assert result["all_match"]
    if trace["run"] in ("final_controlled", "final_money"):
        assert not check_candidate(
            trace["sql_before"],
            trace["plan"],
            trace["fixtures"],
            source=fixture_data(trace["source"]),
        )["all_match"]


def test_wrong_oracle_challenge_preserves_actual_disagreement_and_clarification():
    challenge = EVIDENCE["oracle_fault"]
    state = challenge["state"]
    result = check_candidate(state["candidate"]["sql"], state["plan"], state["fixtures"])
    assert not result["all_match"]
    assert challenge["decision"]["action"] == "clarify"


AUDIT_EVIDENCE = json.loads(
    (Path(__file__).parent / "fixtures/discovery/reference_audit_evidence.json").read_text(
        encoding="utf-8"
    )
)
SELECTED = {(item["run"], item["scenario"]) for item in AUDIT_EVIDENCE["selected_successes"]}


@pytest.mark.parametrize(
    "record",
    [r for r in AUDIT_EVIDENCE["runs_including_failures"] if (r["run"], r["scenario"]) in SELECTED],
    ids=lambda r: r["scenario"],
)
def test_real_attribution_records_replay_both_sql_and_reference(record):
    from dataverify.evaluation.reference_audit_cases import audit_scenarios
    from dataverify.evaluation.reference_audit_scoring import grade_audit_run

    scenario = next(s for s in audit_scenarios() if s.name == record["scenario"])
    grade = grade_audit_run(scenario, record)
    assert record["status"] == "reviewed"
    assert grade["candidate_passes_all"]
    assert grade["reference_passes_all"]
    assert grade["independent_passes_all"]
    if record["scenario"] == "wrong_reference_rounding":
        assert record["sql"] == record["initial_sql"]
        assert record["reference_revisions"] == 1
        assert len(record["plan_versions"]) == 2


def test_historical_live_rounding_failure_stays_visible_after_fix():
    from dataverify.evaluation.reference_audit_cases import audit_scenarios
    from dataverify.evaluation.reference_audit_scoring import grade_audit_run

    record = next(
        r
        for r in AUDIT_EVIDENCE["runs_including_failures"]
        if r["run"] == "reference-audit-live-20261107" and r["scenario"] == "requested_rounding"
    )
    scenario = next(s for s in audit_scenarios() if s.name == "requested_rounding")
    grade = grade_audit_run(scenario, record)
    assert grade["candidate_passes_all"]
    assert not grade["reference_passes_all"]
    assert not grade["independent_passes_all"]
