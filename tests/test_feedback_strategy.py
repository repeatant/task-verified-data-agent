"""Feedback ablation and adaptive escalation never change the acceptance gate.

Tests exercise observed execution histories: evidence selection, repeated SQL,
changing failure categories, an absent witness, and stale publication state.
No model client, credential or evaluation target enters the runtime selector.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hypothesis")

from dataverify.probe_capabilities import policy_features
from dataverify.probe_cases import get_case
from dataverify.probe_guard import ProbeGuard, source_digest


@pytest.mark.parametrize(
    "policy,capability,guidance",
    [
        ("legacy_v1", False, False),
        ("capability_v2", True, True),
        ("capability_only", True, False),
        ("guidance_only", False, True),
    ],
)
def test_ablation_flags_are_independent_and_isolated(policy, capability, guidance):
    flags = policy_features(policy)
    assert flags == {"capability_description": capability, "classified_guidance": guidance}
    flags["classified_guidance"] = not guidance
    assert policy_features(policy)["classified_guidance"] is guidance


@pytest.mark.parametrize("policy", [None, True, "unknown"])
def test_unknown_policy_cannot_silently_enable_components(policy):
    with pytest.raises(ValueError):
        policy_features(policy)


def test_guidance_switch_preserves_every_other_feedback_and_verdict():
    case = get_case("d10")
    results = {}
    for policy in ("legacy_v1", "capability_v2", "capability_only", "guidance_only"):
        guard = ProbeGuard(case.task, case.source, feedback_policy=policy)
        before = source_digest(case.task, case.source)
        results[policy] = guard.execute(case.candidate_sql, case.task.task_id)
        assert results[policy]["status"] == "unverified"
        assert guard.publish()["rows"] is None
        assert before == source_digest(case.task, case.source)
    assert results["capability_only"] == results["legacy_v1"]
    assert results["guidance_only"] == results["capability_v2"]
    guided = dict(results["guidance_only"])
    assert guided.pop("repair_guidance")["classification"] == "task_requirement"
    assert guided == results["legacy_v1"]


@pytest.mark.parametrize("mode", ["full", "compact", "code"])
def test_fixed_modes_never_escalate_or_gain_strategy_metadata(mode):
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source, mode=mode)
    for _ in range(2):
        result = guard.execute(case.candidate_sql, case.task.task_id)
        assert "feedback_strategy" not in result
        if mode == "code":
            assert "diagnostic_evidence" not in result
        else:
            evidence = guard.last_result["probe_report"]["counterexample"]
            assert result["diagnostic_evidence"] == evidence[mode + "_feedback"]
        assert guard.publish()["rows"] is None


def test_adaptive_starts_compact_then_normalized_repeat_uses_full():
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    first = guard.execute(case.candidate_sql, case.task.task_id)
    evidence = guard.last_result["probe_report"]["counterexample"]
    assert first["diagnostic_evidence"] == evidence["compact_feedback"]
    assert first["feedback_strategy"] == {
        "selected_mode": "compact",
        "escalation_reason": None,
        "evidence_available": True,
    }
    equivalent = (
        case.candidate_sql.replace("SELECT", "select").replace(" FROM ", "\nFROM ")
        + " /*same candidate*/"
    )
    second = guard.execute(equivalent, case.task.task_id)
    evidence = guard.last_result["probe_report"]["counterexample"]
    assert second["diagnostic_evidence"] == evidence["full_feedback"]
    assert second["feedback_strategy"]["escalation_reason"] == "repeated_failed_candidate"
    assert second["feedback_strategy"]["selected_mode"] == "full"
    assert guard.audit[0]["feedback"]["feedback_strategy"]["selected_mode"] == "compact"
    assert guard.publish()["rows"] is None


def test_different_candidates_with_same_error_escalate_without_gold():
    case, other = get_case("d06"), get_case("d07")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    first = guard.execute(case.candidate_sql, case.task.task_id)
    second = guard.execute(other.candidate_sql, case.task.task_id)
    assert first["error_code"] == second["error_code"]
    assert second["feedback_strategy"]["escalation_reason"] == "persistent_error_code"
    assert second["feedback_strategy"]["selected_mode"] == "full"


def test_two_distinct_failure_categories_escalate_on_no_progress():
    case, other = get_case("d06"), get_case("d05")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    first = guard.execute(case.candidate_sql, case.task.task_id)
    second = guard.execute(other.candidate_sql, case.task.task_id)
    assert first["error_code"] != second["error_code"]
    assert second["feedback_strategy"]["selected_mode"] == "full"
    assert second["feedback_strategy"]["escalation_reason"] == "consecutive_failures"


def test_success_resets_failure_streak_without_discarding_regressions():
    case, other = get_case("d06"), get_case("d05")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    guard.execute(case.candidate_sql, case.task.task_id)
    repaired = guard.execute(case.reference_sql, case.task.task_id)
    assert repaired["status"] == "verified"
    assert guard.publish()["rows"] == [[35000]]
    assert guard.last_result["regression_checks"]
    assert all(check["status"] == "passed" for check in guard.last_result["regression_checks"])
    later = guard.execute(other.candidate_sql, case.task.task_id)
    assert later["feedback_strategy"]["selected_mode"] == "compact"
    assert later["feedback_strategy"]["escalation_reason"] is None
    assert guard.publish()["rows"] is None
    assert guard.publish()["verified_sql"] is None


def test_escalation_stays_full_and_a_fresh_guard_has_no_history():
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    guard.execute(case.candidate_sql, case.task.task_id)
    guard.execute(case.candidate_sql, case.task.task_id)
    accepted = guard.execute(case.reference_sql, case.task.task_id)
    assert accepted["status"] == "verified"
    assert accepted["feedback_strategy"]["selected_mode"] == "full"
    later = guard.execute(get_case("d05").candidate_sql, case.task.task_id)
    assert later["feedback_strategy"]["selected_mode"] == "full"
    fresh = ProbeGuard(case.task, case.source, mode="adaptive")
    first = fresh.execute(case.candidate_sql, case.task.task_id)
    assert first["feedback_strategy"]["selected_mode"] == "compact"
    assert first["feedback_strategy"]["escalation_reason"] is None


def test_missing_counterexample_is_reported_without_inventing_evidence():
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    # DISTINCT is deliberately outside the supported runtime grammar.
    sql = case.candidate_sql.replace("SUM(l.amount)", "SUM(DISTINCT l.amount)")
    for _ in range(2):
        result = guard.execute(sql, case.task.task_id)
        assert result["status"] == "unverified"
        assert "diagnostic_evidence" not in result
        assert result["feedback_strategy"]["evidence_available"] is False
    assert result["feedback_strategy"]["selected_mode"] == "full"
    assert guard.publish()["rows"] is None


def test_candidate_identity_preserves_literals_and_extra_statements():
    fingerprint = ProbeGuard._candidate_fingerprint
    assert fingerprint("SELECT 'web'") != fingerprint("SELECT 'store'")
    assert fingerprint("SELECT 1") != fingerprint("SELECT 2")
    assert fingerprint("SELECT 1; DELETE FROM orders") != fingerprint("SELECT 1")
    assert fingerprint("SELECT 1 /*comment*/") == fingerprint("select 1")
    assert fingerprint("SELECT (") != fingerprint("SELECT [")


def test_invalid_action_and_task_drift_never_reuse_accepted_value():
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source, mode="adaptive")
    guard.execute(case.reference_sql, case.task.task_id)
    malformed = guard.invalidate()
    assert malformed["error_code"] == "INVALID_ACTION_FORMAT"
    assert malformed["feedback_strategy"]["evidence_available"] is False
    assert guard.publish()["rows"] is None
    changed = guard.execute(case.reference_sql, "invented-task")
    assert changed["error_code"] == "TASK_DRIFT"
    assert changed["feedback_strategy"]["selected_mode"] == "full"
    assert guard.publish()["rows"] is None
    serialized = json.dumps(changed)
    assert "reference_sql" not in serialized and "expected_rows" not in serialized
