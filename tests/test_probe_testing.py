from dataclasses import replace
import json

import pytest

pytest.importorskip("hypothesis", reason="optional probe experiment requires requirements-probes.txt")
from hypothesis import find, settings

from dataverify.probe_cases import get_case
from dataverify.probe_runtime import ProbeData
from dataverify.probe_testing import (
    ExecutionBudget, ProbeBudgetExceeded, ProbePair, _equal, evaluate_pair,
    pair_strategy, probe_candidate, rank_properties,
)


@pytest.mark.parametrize("case_id", ["d01", "d02", "d03", "d04", "d16"])
def test_correct_candidates_do_not_violate_registered_properties(case_id):
    case = get_case(case_id)
    report = probe_candidate(case.candidate_sql, case.task, max_queries=240, max_examples=4)
    assert report["static"]["accepted"]
    assert report["status"] == "no_counterexample_found"
    assert report["acceptance"]
    assert report["counterexample"] is None


@pytest.mark.parametrize("case_id", ["d05", "d06", "d07", "d10", "d12", "d13", "d14", "d15"])
def test_already_static_rejected_errors_are_diagnostics_not_new_discoveries(case_id):
    case = get_case(case_id)
    report = probe_candidate(case.candidate_sql, case.task, max_queries=260, max_examples=4)
    assert not report["static"]["accepted"]
    assert report["status"] == "violation"
    assert not report["new_detection_beyond_static"]
    assert not report["acceptance"]
    evidence = report["counterexample"]
    assert evidence["full_verdict"]["failure_type"] == evidence["compact_verdict"]["failure_type"]
    assert evidence["compact"]["record_count"] <= evidence["full"]["record_count"]
    assert "reference_sql" not in evidence["compact_feedback"]
    assert evidence["compact"]["added_orders"] == evidence["full"]["added_orders"]
    assert evidence["compact"]["added_lines"] == evidence["full"]["added_lines"]


def test_query_budget_is_global_across_generation_and_shrinking():
    case = get_case("d01")
    report = probe_candidate(case.candidate_sql, case.task, max_queries=2)
    assert report["status"] == "budget_exhausted"
    assert report["queries"] == 2
    assert not report["acceptance"]


@pytest.mark.parametrize("queries,seconds", [(1, 3), (True, 3), (2001, 3), (3, 0), (3, float("nan"))])
def test_invalid_budget_rejected(queries, seconds):
    with pytest.raises(ValueError):
        ExecutionBudget(queries, seconds)


def test_budget_reserves_failed_queries_too():
    budget = ExecutionBudget(2)
    budget.reserve()
    budget.reserve()
    with pytest.raises(ProbeBudgetExceeded):
        budget.reserve()
    assert budget.queries == 2


def test_no_float_tolerance_can_hide_missing_minor_unit():
    assert not _equal(100_000_000_000, 100_000_000_001)


def test_task_condition_absent_from_ast_still_selects_status_probe():
    case = get_case("d05")
    assert rank_properties(case.candidate_sql, case.task, "directed", 17)[0] == "exclude_status"


def test_pair_generation_has_valid_parent_child_keys_and_stable_seed_inputs():
    case = get_case("d06")
    pair = find(pair_strategy(case.task, "exclude_end"), lambda value: True,
                settings=settings(database=None, deadline=None))
    ids = {row[0] for row in pair.followup.orders}
    assert all(row[1] in ids for row in pair.followup.lines)
    assert pair.added_lines[0][4] == case.task.end
    assert pair.base != pair.followup


def test_status_probe_can_generate_cancelled_not_just_pending():
    case = get_case("d14")
    pair = find(pair_strategy(case.task, "exclude_status"),
                lambda value: value.added_lines[0][3] == "cancelled",
                settings=settings(database=None, deadline=None))
    verdict = evaluate_pair(case.candidate_sql, case.task, pair, ExecutionBudget())
    assert verdict["failure_type"] == "PROPERTY_EXCLUDE_STATUS"


def test_invalid_pair_cannot_become_counterexample():
    base = ProbeData(((1, 1, "web"),), ((1, 1, 1, "settled", "2026-09-02T00:00:00"),))
    with pytest.raises(ValueError):
        ProbePair("exclude_end", base, ((1, 1, "web"),), ())


def test_empty_base_is_not_original_failure():
    case = get_case("d06")
    pair = ProbePair("exclude_end", ProbeData((), ()),
                     ((2, 1, "web"),), ((2, 2, 1, "settled", case.task.end),))
    verdict = evaluate_pair(case.candidate_sql, case.task, pair, ExecutionBudget())
    assert verdict["status"] == "not_applicable"


def test_invalid_transformation_cannot_falsely_reject_correct_query():
    case = get_case("d01")
    # Label says excluded upper boundary, but this row is actually eligible.
    pair = ProbePair("exclude_end", case.source,
                     ((3, 1, "web"),), ((30, 3, 1, "settled", "2026-09-20T00:00:00"),))
    budget = ExecutionBudget()
    assert evaluate_pair(case.candidate_sql, case.task, pair, budget)["status"] == "not_applicable"
    assert budget.queries == 0


def test_empty_property_set_is_not_an_acceptance_proof():
    case = get_case("d01")
    task = replace(case.task, properties=())
    report = probe_candidate(case.candidate_sql, task)
    assert report["status"] == "not_applicable"
    assert report["queries"] == 0
    assert not report["acceptance"]


def test_scope_rejected_query_never_executes():
    case = get_case("d01")
    report = probe_candidate("DELETE FROM orders", case.task)
    assert report["status"] == "unsupported"
    assert report["queries"] == 0


def test_same_seed_replays_same_witness_not_elapsed_time():
    case = get_case("d06")
    left = probe_candidate(case.candidate_sql, case.task)
    right = probe_candidate(case.candidate_sql, case.task)
    assert left["counterexample"] == right["counterexample"]
    assert left["queries"] == right["queries"]


def test_feedback_is_json_replayable_and_not_target_sql():
    case = get_case("d06")
    report = probe_candidate(case.candidate_sql, case.task)
    feedback = json.loads(report["counterexample"]["compact_feedback"])
    assert "instruction" in feedback
    assert case.reference_sql not in report["counterexample"]["compact_feedback"]
