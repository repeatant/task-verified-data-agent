"""Scoring measures actual delivery and avoids pseudo-replicated statistics."""

import sqlite3

import pytest

from dataverify.evaluation.expanded_cases import expanded_cases
from dataverify.evaluation.expanded_scoring import aggregate_runs, score_controlled, score_public
from dataverify.verification.discovery_runtime import execute_readonly


def _record(case, *, sql=None, reference=None, auditor=None, status="reviewed"):
    sql = sql or case.good_sql
    return {
        "status": status,
        "candidate": {"sql": sql},
        "publication": {"sql": sql, "rows": execute_readonly(sql, case.source)}
        if status == "reviewed"
        else {},
        "plan": {"reference": reference or case.good_reference},
        "audit_plan": {"reference": auditor or case.good_reference},
    }


def test_all_three_programs_scored_independently():
    case = expanded_cases()[0]
    score = score_controlled(case, _record(case, reference=case.bad_reference))
    assert score["correct_delivery"]
    assert not score["reference"]["correct"]
    assert score["auditor"]["correct"]
    assert not score["joint_correct_delivery"]
    assert score["candidate"]["total"] == 25


def test_correct_but_withheld_is_not_success():
    case = next(case for case in expanded_cases() if case.condition == "correct")
    score = score_controlled(case, _record(case, status="needs_clarification"))
    assert score["final_sql_correct"]
    assert score["correct_task_rejected"]
    assert not score["delivered"]
    assert not score["correct_delivery"]
    assert not score["wrong_release"]


def test_initially_correct_sql_damaged_is_explicit():
    case = next(case for case in expanded_cases() if case.condition == "correct")
    score = score_controlled(case, _record(case, sql=case.bad_sql))
    assert score["initial_sql_correct"]
    assert score["correct_sql_damaged"]
    assert score["wrong_release"]


def test_candidate_correct_does_not_mask_wrong_published_sql():
    case = expanded_cases()[0]
    record = _record(case)
    record["publication"]["sql"] = case.bad_sql
    score = score_controlled(case, record)
    assert score["final_sql_correct"]
    assert not score["correct_delivery"]
    assert score["wrong_release"]


def test_published_rows_must_equal_executed_source():
    case = expanded_cases()[0]
    record = _record(case)
    record["publication"]["rows"] = [[-10000000]]
    score = score_controlled(case, record)
    assert score["publication"]["correct"]
    assert not score["publication_rows_consistent"]
    assert score["wrong_release"]


def test_missing_reference_does_not_receive_a_false_joint_grade():
    case = expanded_cases()[0]
    record = _record(case)
    record.pop("plan")
    record.pop("audit_plan")
    score = score_controlled(case, record)
    assert score["correct_delivery"]
    assert score["reference"]["correct"] is None
    assert not score["joint_evaluable"]
    assert score["joint_correct_delivery"] is None


def test_sql_syntax_error_is_a_score_not_an_evaluator_crash():
    case = expanded_cases()[0]
    record = _record(case)
    record["candidate"]["sql"] = "not sql at all"
    score = score_controlled(case, record)
    assert not score["final_sql_correct"]
    assert score["candidate"]["passed"] == 0


@pytest.fixture
def public_case(tmp_path):
    database = tmp_path / "original.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE t(x); INSERT INTO t VALUES(1),(1),(2);")
    return {"db_path": database, "gold_rows": [[1], [1], [2]], "initial_sql": "SELECT x FROM t"}


def test_public_scoring_uses_original_data_and_official_set_metric(public_case):
    record = {
        "status": "reviewed",
        "candidate": {"sql": "SELECT DISTINCT x FROM t"},
        "publication": {"sql": "SELECT DISTINCT x FROM t", "rows": [[999]]},
        "plan": {"reference": "[[1]]"},
    }
    score = score_public(public_case, record)
    assert score["correct_delivery"]
    assert score["publication"]["bird_execution_match"]
    assert not score["publication"]["bag_match"]
    assert score["reference"]["correct"] is None
    assert score["publication_rows_consistent"] is None


def test_public_wrong_release_and_correct_damage(public_case):
    score = score_public(
        public_case,
        {
            "status": "reviewed",
            "candidate": {"sql": "SELECT 1 FROM t"},
            "publication": {"sql": "SELECT 1 FROM t"},
        },
    )
    assert score["wrong_release"] and score["correct_sql_damaged"]


def test_public_rejection_cannot_improve_accuracy(public_case):
    score = score_public(
        public_case, {"status": "inconclusive", "candidate": {"sql": "SELECT x FROM t"}}
    )
    assert score["correct_task_rejected"]
    assert not score["correct_delivery"]


def _row(case_id, cluster, method, correct, *, kind="controlled", initial=True):
    return {
        "case_id": case_id,
        "cluster_id": cluster,
        "family": cluster,
        "kind": kind,
        "method": method,
        "elapsed_seconds": 2,
        "score": {
            "correct_delivery": correct,
            "wrong_release": not correct,
            "initial_sql_correct": initial,
            "correct_sql_damaged": not correct,
            "correct_task_rejected": False,
            "joint_evaluable": False,
            "joint_correct_delivery": None,
        },
    }


def test_denominators_are_not_all_trajectories_for_protection_metric():
    rows = [_row("a", "a", "full", True), _row("b", "b", "full", False, initial=False)]
    result = aggregate_runs(rows, bootstrap_repetitions=20)["populations"]["controlled"]["methods"][
        "full"
    ]
    assert result["metrics"]["correct_sql_damaged"]["denominator"] == 1
    assert result["metrics"]["correct_delivery"]["denominator"] == 2
    assert result["metrics"]["joint_correct_delivery"]["rate"] is None
    assert result["clusters"] == 2


def test_duplicating_correlated_variants_does_not_narrow_cluster_ci():
    one = [_row("a", "a", "full", True), _row("b", "b", "full", False)]
    many = [
        _row(f"{cluster}{i}", cluster, "full", correct)
        for cluster, correct in [("a", True), ("b", False)]
        for i in range(5)
    ]

    def ci(rows):
        return aggregate_runs(rows, bootstrap_repetitions=200)["populations"]["controlled"][
            "methods"
        ]["full"]["metrics"]["correct_delivery"]["cluster_bootstrap_ci95"]

    assert ci(one) == ci(many)


def test_paired_comparison_uses_intersection_not_different_populations():
    rows = [_row("a", "a", "full", True), _row("a", "a", "b0", False), _row("b", "b", "full", True)]
    paired = aggregate_runs(rows, bootstrap_repetitions=20)["populations"]["controlled"]["paired"][
        "b0"
    ]
    assert paired["paired_tasks"] == 1
    assert paired["reference_only_tasks"] == 1
    assert paired["metrics"]["correct_delivery"]["reference_minus_method"] == 1


def test_public_and_controlled_are_never_pooled():
    rows = [_row("a", "a", "full", True), _row("b", "b", "full", False, kind="public")]
    result = aggregate_runs(rows, bootstrap_repetitions=20)
    assert set(result["populations"]) == {"controlled", "public"}
    assert "overall_accuracy" not in result


def test_duplicate_case_method_requires_explicit_repeat_handling():
    row = _row("a", "a", "full", True)
    with pytest.raises(ValueError, match="DUPLICATE_CASE_METHOD"):
        aggregate_runs([row, row], bootstrap_repetitions=0)


def test_pair_cluster_mismatch_is_rejected():
    with pytest.raises(ValueError, match="PAIRED_CLUSTER_MISMATCH"):
        aggregate_runs(
            [_row("a", "a", "full", True), _row("a", "b", "b0", True)], bootstrap_repetitions=0
        )


def test_bootstrap_seed_is_reproducible():
    rows = [
        _row(str(i), str(i // 5), method, bool(i % 2))
        for i in range(10)
        for method in ("b0", "full")
    ]
    assert aggregate_runs(rows, bootstrap_repetitions=20) == aggregate_runs(
        rows, bootstrap_repetitions=20
    )


def test_shared_initial_calls_are_deduplicated_only_for_actual_usage():
    from dataverify.runtime.reporting import summarize_usage

    event = {
        "event_id": "lane0:1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    rows = [_row("a", "a", method, True) for method in ("direct", "full")]
    for row in rows:
        row["http_events"] = []
        row["usage"] = summarize_usage([])
        row["shared_initial_http_events"] = [event]
        row["shared_initial_usage"] = summarize_usage([event])
    result = aggregate_runs(
        rows, bootstrap_repetitions=0, initial_records=[{"http_events": [event]}]
    )
    assert result["actual_logged_usage"]["event_count"] == 1
    assert result["actual_logged_usage"]["total_tokens_known"] == 12
    methods = result["populations"]["controlled"]["methods"]
    assert methods["direct"]["cold_start_effective_usage"]["total_tokens_known"] == 12
    assert methods["full"]["cold_start_effective_usage"]["total_tokens_known"] == 12


def test_event_id_conflicting_usage_cannot_be_silently_deduplicated():
    rows = [_row("a", "a", method, True) for method in ("direct", "full")]
    for index, row in enumerate(rows):
        row["http_events"] = [{"event_id": "lane0:1", "usage": {"total_tokens": index}}]
    with pytest.raises(ValueError, match="EVENT_ID_USAGE_CONFLICT"):
        aggregate_runs(rows, bootstrap_repetitions=0)


def test_public_parser_error_is_scored_as_failure(public_case):
    score = score_public(
        public_case,
        {
            "status": "reviewed",
            "candidate": {"sql": "SELECT ("},
            "publication": {"sql": "SELECT ("},
        },
    )
    assert score["wrong_release"]
