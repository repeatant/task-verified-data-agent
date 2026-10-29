"""Offline final-publication scoring; no native runtime or API is required."""

import ast
import inspect
import json

import pytest

from dataverify import benchmark_grader
from dataverify.benchmark_grader import grade_record
from dataverify.benchmark_tasks import Fixture, Task, get_task

ORDER_SQL = "SELECT SUM(total) AS revenue FROM orders"
LINE_JOIN = "SELECT SUM(l.amount) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id"


def original(sql=ORDER_SQL, value=300, *, data=None, success=True):
    content = {"sql": sql, "data": [{"revenue": value}] if data is None else data}
    if not success:
        content["status"] = "refused"
    return {
        "mode": "original",
        "upstream_success": success,
        "native_reply": {"action_report": {"content": json.dumps(content)}},
    }


def guarded(
    sql=ORDER_SQL, value=300, *, mode="evidence", status="verified", metric="order_revenue"
):
    return {
        "mode": mode,
        "upstream_success": True,
        "publication": {
            "status": status,
            "verified_sql": sql,
            "accepted_value": value,
            "metric_id": metric,
            "error_code": "INVALID_SOURCE_GRAIN" if status == "unverified" else None,
        },
    }


def task_for(fixture, metric="order_revenue"):
    return Task("test", "natural", fixture, "Total revenue", metric, None)


def test_grader_has_no_checker_or_sql_executor_import():
    tree = ast.parse(inspect.getsource(benchmark_grader))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        "checker" in name or "sqlite" in name or "guarded_task" in name for name in imports
    )


@pytest.mark.parametrize("mode", ["original", "code", "evidence"])
def test_correct_final_answer_for_all_native_arms(mode):
    record = original() if mode == "original" else guarded(mode=mode)
    result = grade_record(get_task("n01"), record)
    assert result["accepted"] and result["correct_answer"]
    assert not result["wrong_release"] and not result["unscorable"]
    assert result["source_grains"] == result["aggregate_rows"] == 3


@pytest.mark.parametrize(
    "task_id,value,correct",
    [
        ("c01", 300, True),
        ("c02", 420, False),
        ("c03", 100, False),
        ("c04", 300, False),
        ("c05", 330, True),
        ("c06", 330, False),
        ("c07", 300, False),
        ("c08", 200, False),
        ("c09", 380, False),
        ("c10", 180, False),
    ],
)
def test_frozen_candidate_grade_is_not_based_on_native_success(task_id, value, correct):
    task = get_task(task_id)
    result = grade_record(task, original(task.initial_candidate["sql"], value))
    assert result["accepted"]
    assert result["correct_answer"] is correct
    assert result["wrong_release"] is (not correct)
    assert not result["unscorable"]


def test_sum_distinct_rejected_even_when_its_scalar_is_exactly_correct():
    result = grade_record(get_task("c04"), original("SELECT SUM(DISTINCT total) FROM orders", 300))
    assert result["expected_value"] == result["observed_value"] == 300
    assert result["wrong_release"]
    assert result["reason"] == "value_distinct_is_not_grain_sum"


def test_matching_order_and_line_totals_do_not_hide_wrong_source():
    result = grade_record(get_task("c07"), original("SELECT SUM(amount) FROM order_lines", 300))
    assert result["expected_value"] == result["observed_value"] == 300
    assert result["metric_drift"] and result["wrong_release"]
    assert result["aggregate_source"] == {"table": "order_lines", "column": "amount"}
    assert result["reason"] == "aggregate_source_mismatch"


def test_fanout_and_omission_with_equal_values_cannot_cancel_in_grain_multiset():
    fixture = Fixture(((1, 100), (2, 100), (3, 50)), ((11, 1, 40), (12, 1, 60), (13, 3, 50)))
    sql = "SELECT SUM(o.total) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id"
    result = grade_record(task_for(fixture), original(sql, 250))
    assert result["expected_value"] == result["observed_value"] == 250
    assert result["aggregate_rows"] == result["source_rows"] == 3
    assert result["missing_grains"] == [2]
    assert result["extra_grains"] == [1]
    assert result["wrong_release"] and result["reason"] == "incomplete_or_repeated_grain"


def test_detail_inner_join_is_valid_despite_order_fanout():
    result = grade_record(get_task("c05"), original(LINE_JOIN, 330))
    assert result["correct_answer"]
    assert result["aggregate_rows"] == 4
    assert result["inferred_metric_id"] == "line_revenue"


def test_orphan_line_omission_is_caught_even_if_released_value_is_forged_correct():
    result = grade_record(get_task("c06"), original(LINE_JOIN, 355))
    assert result["observed_value"] == result["expected_value"] == 355
    assert result["missing_grains"] == [15]
    assert result["wrong_release"]


@pytest.mark.parametrize(
    "fixture",
    [
        Fixture((), ()),
        Fixture(((None, 10),), ()),
        Fixture(((1, 10), (1, 20)), ()),
        Fixture(((1, None),), ()),
        Fixture(((1, "10"),), ()),
        Fixture(((1, True),), ()),
        Fixture(((1, float("nan")),), ()),
        Fixture(((1, float("inf")),), ()),
    ],
)
@pytest.mark.parametrize("mode", ["original", "code", "evidence"])
def test_damaged_source_requires_refusal(fixture, mode):
    task = task_for(fixture)
    record = (
        original(success=False)
        if mode == "original"
        else guarded(mode=mode, status="unverified", value=None)
    )
    refused = grade_record(task, record)
    assert refused["correct_refusal"] and not refused["target_defined"]
    assert not refused["accepted"] and not refused["wrong_release"]
    record = original() if mode == "original" else guarded(mode=mode)
    released = grade_record(task, record)
    assert released["wrong_release"] and not released["correct_answer"]
    assert released["reason"].startswith("undefined_target:")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(total) FROM orders WHERE order_id > 0",
        "SELECT SUM(total) FROM orders GROUP BY order_id",
        "SELECT SUM(total) OVER () FROM orders",
        "WITH source AS (SELECT * FROM orders) SELECT SUM(total) FROM source",
        "SELECT SUM(total) FROM (SELECT * FROM orders)",
        "SELECT SUM(total), COUNT(*) FROM orders",
        "SELECT SUM(1) FROM orders",
        "SELECT SUM(total + 0) FROM orders",
        "SELECT SUM(total) FROM secrets",
        "SELECT SUM(total) FROM main.orders",
        "SELECT SUM(o.total) FROM orders o LEFT JOIN order_lines l ON o.order_id=l.order_id",
        "SELECT SUM(o.total) FROM orders o INNER JOIN order_lines l USING(order_id)",
        "SELECT SUM(o.total) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id INNER JOIN orders p ON l.order_id=p.order_id",
        "SELECT SUM(o.total) FROM orders o CROSS JOIN order_lines l",
        "SELECT SUM(total) FROM orders; SELECT 1",
        "SELECT SUM(total) FROM orders UNION SELECT SUM(amount) FROM order_lines",
        "SELECT SUM(nonsense) FROM orders",
        "SELECT SUM(order_id) FROM orders o JOIN order_lines l ON o.order_id=l.order_id",
    ],
)
def test_out_of_scoring_grammar_remains_separate_from_wrong_release(sql):
    result = grade_record(get_task("n01"), original(sql))
    assert result["accepted"] and result["unscorable"]
    assert not result["correct_answer"] and not result["wrong_release"]
    assert result["final_sql"] == sql


@pytest.mark.parametrize(
    "data", [[], [{"v": 100}, {"v": 200}], [{"v": 300, "other": 1}], [{}], [300], [[300, 1]]]
)
def test_native_success_is_not_evidence_of_a_single_cell_result(data):
    result = grade_record(get_task("n01"), original(data=data))
    assert result["accepted"] and result["wrong_release"]
    assert not result["correct_answer"]


@pytest.mark.parametrize(
    "value", [None, True, False, "300", float("nan"), float("inf"), -float("inf")]
)
@pytest.mark.parametrize("mode", ["original", "code", "evidence"])
def test_nonfinite_nonnumeric_and_boolean_releases_are_never_correct(value, mode):
    record = original(value=value) if mode == "original" else guarded(value=value, mode=mode)
    result = grade_record(get_task("n01"), record)
    assert result["wrong_release"] and not result["correct_answer"]
    assert result["observed_value"] is None
    json.dumps(result, allow_nan=False)


def test_native_action_content_is_the_final_output_not_top_level_or_audit_claims():
    record = original("SELECT SUM(amount) FROM order_lines", 300)
    record["native_reply"]["content"] = json.dumps({"sql": ORDER_SQL, "data": [{"v": 300}]})
    record["actions"] = [
        {"sql": ORDER_SQL, "verification_status": "verified", "accepted_value": 300}
    ]
    result = grade_record(get_task("n01"), record)
    assert result["wrong_release"] and result["metric_drift"]


def test_guarded_uses_only_publication_and_does_not_score_private_rejected_scalar():
    record = guarded(status="unverified", value=None, sql=None)
    record["actions"] = [{"sql": ORDER_SQL, "accepted_value": 300}]
    record["native_reply"] = original()["native_reply"]
    result = grade_record(get_task("n01"), record)
    assert not result["accepted"] and not result["correct_answer"]
    assert not result["wrong_release"]
    assert result["final_sql"] is None and result["observed_value"] is None


def test_guarded_declared_metric_must_match_even_with_correct_sql_and_number():
    result = grade_record(get_task("n01"), guarded(metric="line_revenue"))
    assert result["metric_drift"] and result["wrong_release"]
    assert result["reason"] == "published_metric_mismatch"


def test_refusal_of_defined_target_is_not_correct_or_wrong_release():
    result = grade_record(get_task("n01"), original(success=False))
    assert not any(
        result[key]
        for key in ("accepted", "correct_answer", "correct_refusal", "wrong_release", "unscorable")
    )
    assert result["reason"] == "refused_defined_target"


def test_correct_sql_still_needs_the_right_final_scalar():
    result = grade_record(get_task("n01"), original(value=301))
    assert result["wrong_release"] and result["reason"] == "released_value_mismatch"


def test_registered_nonmeasure_column_does_not_become_a_metric():
    result = grade_record(get_task("n01"), original("SELECT SUM(order_id) FROM orders", 300))
    assert result["metric_drift"] and result["wrong_release"]
    assert result["inferred_metric_id"] is None


def test_aliases_case_and_harmless_parentheses_are_supported():
    sql = 'SELECT SUM(("O"."TOTAL")) AS "R" FROM "ORDERS" AS "O"'
    assert grade_record(get_task("n01"), original(sql))["correct_answer"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.order_id=l.line_id",
        "SELECT SUM(o.total) FROM orders o JOIN orders p ON o.order_id=p.order_id",
    ],
)
def test_unregistered_join_relationship_is_unscorable_even_if_scalar_would_match(sql):
    fixture = Fixture(((1, 120), (2, 80), (3, 100)), ((1, 1, 120), (2, 2, 80), (3, 3, 100)))
    result = grade_record(task_for(fixture), original(sql, 300))
    assert result["unscorable"] and result["accepted"]
    assert not result["wrong_release"] and not result["correct_answer"]
    assert result["reason"] == "requires_registered_order_relation"


@pytest.mark.parametrize(
    "sql", ["SELECT SUM(total) FROM orders /*", "SELECT SUM(total) FROM orders AS 'unfinished"]
)
def test_sqlglot_lexical_errors_are_unscorable_not_operational_failures(sql):
    result = grade_record(get_task("n01"), original(sql))
    assert result["accepted"] and result["unscorable"]
    assert not result["operational_failure"] and not result["wrong_release"]
    assert result["reason"] == "unparseable_final_sql"


@pytest.mark.parametrize("content", ["not JSON", "[]", "null", None])
def test_malformed_native_action_content_cannot_inherit_upstream_correctness(content):
    record = original()
    record["native_reply"]["action_report"]["content"] = content
    result = grade_record(get_task("n01"), record)
    assert result["wrong_release"] and result["reason"] == "invalid_native_action_content"


def test_mode_must_be_explicit_or_come_from_native_record():
    record = original()
    del record["mode"]
    with pytest.raises(ValueError):
        grade_record(get_task("n01"), record)
    assert grade_record(get_task("n01"), record, mode="original")["correct_answer"]


@pytest.mark.parametrize(
    "failure",
    [
        {"execution_status": "failed"},
        {"execution_status": "timeout"},
        {"error_code": "LIVE_HTTP_REQUEST_LIMIT"},
        {"error_code": "LIVE_BUDGET_STOPPED_PROVIDER_ERROR"},
        {"error": "RuntimeError"},
        {"http_events": [{"status": "failed", "error_code": "QWEN_HTTP_429"}]},
        {"actions": [{"execution_status": "error", "error_code": "QUERY_EXECUTION_ERROR"}]},
        {"native_reply": {"execution_status": "failed"}},
        {"operational_failure": True},
    ],
)
def test_operational_failure_never_earns_credit_for_refusal(failure):
    record = guarded(status="unverified", value=None, sql=None)
    record.update(failure)
    result = grade_record(get_task("c09"), record)
    assert result["operational_failure"]
    assert not result["correct_refusal"] and not result["refusal_explicit"]


@pytest.mark.parametrize("mode", ["original", "code", "evidence"])
@pytest.mark.parametrize("execution_status", ["timeout", "failed"])
def test_native_context_failure_overrides_an_earlier_explicit_source_refusal(
    mode, execution_status
):
    record = (
        original(success=False)
        if mode == "original"
        else guarded(mode=mode, status="unverified", value=None, sql=None)
    )
    record["actions"] = [
        {
            "execution_status": "executed",
            "verification_status": "rejected",
            "error_code": "INVALID_SOURCE_GRAIN",
        }
    ]
    record.setdefault("native_reply", {})["context"] = {"execution_status": execution_status}
    result = grade_record(get_task("c09"), record)
    assert not result["target_defined"] and not result["accepted"]
    assert result["operational_failure"] and result["reason"] == "operational_failure"
    assert not result["correct_refusal"] and not result["refusal_explicit"]


@pytest.mark.parametrize(
    "context", [None, "not a context mapping", {"execution_status": "completed"}]
)
def test_absent_or_successful_native_context_preserves_semantic_refusal(context):
    record = guarded(status="unverified", value=None, sql=None)
    record["native_reply"] = {"context": context}
    result = grade_record(get_task("c09"), record)
    assert result["correct_refusal"] and result["refusal_explicit"]
    assert not result["operational_failure"]


@pytest.mark.parametrize(
    "record",
    [
        {"mode": "evidence"},
        {
            "mode": "evidence",
            "publication": {"status": "unverified", "error_code": "NO_VERIFIED_EVIDENCE"},
        },
        {"mode": "original", "upstream_success": False},
        {
            "mode": "original",
            "upstream_success": False,
            "native_reply": {
                "action_report": {"is_exe_success": False, "content": "Error: SQL execution failed"}
            },
        },
    ],
)
def test_missing_or_failed_output_is_not_an_explicit_correct_refusal(record):
    result = grade_record(get_task("c09"), record)
    assert not result["accepted"] and not result["correct_refusal"]
