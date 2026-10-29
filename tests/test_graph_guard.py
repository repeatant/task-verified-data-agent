"""Continuation keeps exact diagnostics but never restores accepted results."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

pytest.importorskip("hypothesis", reason="Optional probe experiment")

from dataverify.feedback_strategy import FeedbackStrategy
from dataverify.graph_guard import dump_guard, restore_guard
from dataverify.probe_cases import get_case
from dataverify.probe_guard import ProbeGuard
from dataverify.probe_runtime import ProbeData


def failed_guard(mode="adaptive"):
    case = get_case("d06")
    guard = ProbeGuard(case.task, case.source, mode)
    guard.execute(case.candidate_sql, case.task.task_id)
    return case, guard


def resume(state, case, mode="adaptive", **options):
    return restore_guard(state, case.task, case.source, mode, **options)


def test_roundtrip_preserves_exact_witness_and_adaptive_failure_progress():
    case, guard = failed_guard()
    snapshot = json.loads(json.dumps(dump_guard(guard)))
    restored = resume(snapshot, case)
    assert dump_guard(restored) == snapshot
    assert restored.audit == []
    assert restored.last_result == {}
    assert restored.publish()["rows"] is None
    feedback = restored.execute(case.candidate_sql, case.task.task_id)
    assert feedback["feedback_strategy"]["selected_mode"] == "full"
    assert feedback["feedback_strategy"]["escalation_reason"] == "repeated_failed_candidate"
    assert restored.last_result["regression_checks"]


def test_resume_rechecks_historical_witnesses_before_accepting_repair():
    case, guard = failed_guard()
    restored = resume(dump_guard(guard), case)
    restored.execute(case.reference_sql, case.task.task_id)
    assert restored.publish()["status"] == "verified"
    assert all(item["status"] == "passed" for item in restored.last_result["regression_checks"])


@pytest.mark.parametrize("mode", ["full", "compact", "code", "adaptive"])
def test_previously_accepted_values_are_absent_after_restore(mode):
    case = get_case("d01")
    guard = ProbeGuard(case.task, case.source, mode)
    guard.execute(case.candidate_sql, case.task.task_id)
    assert guard.publish()["status"] == "verified"
    state = dump_guard(guard)
    assert "verified_sql" not in json.dumps(state)
    assert "accepted" not in state and "rows" not in state
    restored = resume(state, case, mode)
    assert restored.publish()["status"] == "unverified"
    assert restored.publish()["verified_sql"] is None
    restored.execute(case.candidate_sql, case.task.task_id)
    assert restored.publish()["rows"] == guard.publish()["rows"]


def test_snapshot_and_restoration_do_not_share_mutable_witnesses():
    case, guard = failed_guard()
    state = dump_guard(guard)
    original = deepcopy(state)
    restored = resume(state, case)
    state["regressions"][0]["base"]["orders"][0][1] += 10
    state["feedback_strategy"]["failed_candidates"].clear()
    assert dump_guard(guard) == original
    assert dump_guard(restored) == original


def test_structurally_legal_witness_with_wrong_boundary_is_rejected():
    case, guard = failed_guard()
    state = dump_guard(guard)
    # PK/FK/type checks still pass, but this no longer probes exclude_end.
    state["regressions"][0]["added_lines"][0][4] = case.task.start
    with pytest.raises(ValueError, match="property preconditions"):
        resume(state, case)


@pytest.mark.parametrize("change", ["task", "data", "mode", "seed", "policy"])
def test_resume_rejects_task_data_and_configuration_drift(change):
    case, guard = failed_guard()
    state = dump_guard(guard)
    task, data, mode, seed, policy = case.task, case.source, "adaptive", 17, "capability_v2"
    if change == "task":
        task = replace(task, question=task.question + " changed")
    elif change == "data":
        data = ProbeData(data.orders + ((999, 0, None),), data.lines)
    elif change == "mode":
        mode = "compact"
    elif change == "seed":
        seed = 18
    else:
        policy = "legacy_v1"
    with pytest.raises(ValueError, match="changed"):
        restore_guard(state, task, data, mode, seed, feedback_policy=policy)


@pytest.mark.parametrize(
    "path,value",
    [
        (("version",), True),
        (("version",), 2),
        (("terminal_code",), "arbitrary failure"),
        (("binding", "seed"), True),
        (("feedback_strategy", "mode"), []),
        (("feedback_strategy", "selected_mode"), []),
        (("feedback_strategy", "escalation_reason"), []),
        (("feedback_strategy", "failed_candidates"), ["ast:wrong"]),
        (("feedback_strategy", "consecutive_failures"), True),
        (("feedback_strategy", "previous_error_code"), "not a safe code"),
        (("regressions", 0, "property_id"), "preserve_zero_orders"),
        (("regressions", 0, "record_count"), False),
        (("regressions", 0, "added_lines"), "not rows"),
        (("regressions", 0, "base", "orders"), [[0, 0, None]]),
    ],
)
def test_invalid_json_state_is_rejected_before_restoration(path, value):
    case, guard = failed_guard()
    state = dump_guard(guard)
    node = state
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        resume(state, case)


@pytest.mark.parametrize("kind", ["duplicate", "too_many", "extra_field", "non_json", "oversize"])
def test_bounded_checkpoint_schema(kind):
    case, guard = failed_guard()
    state = dump_guard(guard)
    if kind == "duplicate":
        state["regressions"] *= 2
    elif kind == "too_many":
        state["regressions"] *= 33
    elif kind == "extra_field":
        state["accepted"] = {"rows": [[123]]}
    elif kind == "non_json":
        state["binding"]["seed"] = object()
    else:
        state["binding"]["source_sha256"] = "x" * 1_000_001
    with pytest.raises(ValueError):
        resume(state, case)


def test_restore_requires_a_fresh_guard_and_keeps_terminal_rejection():
    case, guard = failed_guard()
    state = dump_guard(guard)
    with pytest.raises(ValueError, match="fresh"):
        guard.restore_checkpoint_state(state)
    state["terminal_code"] = "SOURCE_OR_TASK_CHANGED"
    restored = resume(state, case)
    feedback = restored.execute(case.reference_sql, case.task.task_id)
    assert feedback["error_code"] == "SOURCE_OR_TASK_CHANGED"
    assert restored.publish()["rows"] is None


def test_changed_live_source_cannot_be_checkpointed():
    case = get_case("d01")
    task = replace(case.task)
    guard = ProbeGuard(task, case.source)
    object.__setattr__(task, "question", "Changed after binding")
    with pytest.raises(ValueError, match="changed task source"):
        dump_guard(guard)


@pytest.mark.parametrize(
    "reason", ["repeated_failed_candidate", "persistent_error_code", "consecutive_failures"]
)
def test_escalated_strategy_retains_selection_after_success(reason):
    strategy = FeedbackStrategy("adaptive")
    strategy.select("FIRST", "SELECT 1")
    if reason == "repeated_failed_candidate":
        strategy.select("SECOND", "SELECT 1")
    elif reason == "persistent_error_code":
        strategy.select("FIRST", "SELECT 2")
    else:
        strategy.select("SECOND", "SELECT 2")
    assert strategy.escalation_reason == reason
    strategy.select(None)
    restored = FeedbackStrategy.from_checkpoint(strategy.checkpoint_state())
    assert restored.checkpoint_state() == strategy.checkpoint_state()


@pytest.mark.parametrize("mutation", ["duplicate", "inconsistent", "fixed_progress", "too_many"])
def test_strategy_rejects_inconsistent_history(mutation):
    strategy = FeedbackStrategy("adaptive")
    strategy.select("FIRST", "SELECT 1")
    state = strategy.checkpoint_state()
    if mutation == "duplicate":
        state["failed_candidates"] *= 2
    elif mutation == "inconsistent":
        state["consecutive_failures"] = 2
    elif mutation == "fixed_progress":
        state["mode"] = "compact"
    else:
        state["failed_candidates"] = ["ast:" + f"{index:064x}" for index in range(33)]
    with pytest.raises(ValueError):
        FeedbackStrategy.from_checkpoint(state)
