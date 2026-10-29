"""Meaningful boundaries for the separate method-transfer SQL runtime."""

from dataclasses import FrozenInstanceError

import pytest

import dataverify.probe_runtime as runtime
from dataverify.probe_runtime import ProbeData, ProbeTask, execute_sql, static_check

START = "2026-09-01T00:00:00"
END = "2026-10-01T00:00:00"
BOUNDS = f"status='settled' AND settled_at >= '{START}' AND settled_at < '{END}'"
GOOD = f"SELECT COALESCE(SUM(amount),0) AS revenue FROM order_lines WHERE {BOUNDS}"


@pytest.fixture
def task():
    return ProbeTask(
        "time", "九月已结算明细净收入", ("exclude_end", "exclude_status", "include_start")
    )


@pytest.fixture
def data():
    return ProbeData(
        ((1, 100, "web"), (2, 30, None), (3, 70, "其他"), (4, 50, "empty")),
        (
            (11, 1, 100, "settled", START),
            (12, 1, -10, "settled", "2026-09-09T12:00:00"),
            (13, 2, 30, "settled", "2026-09-10T00:00:00"),
            (14, 3, 70, "settled", "2026-09-25T00:00:00"),
            (15, 1, 900, "settled", END),
            (16, 2, 700, "pending", "2026-09-20T00:00:00"),
        ),
    )


def test_real_sql_execution_uses_confirmed_bounds_and_status(task, data):
    assert static_check(GOOD, task)["accepted"]
    result = execute_sql(GOOD, data, task)
    assert result == {"status": "ok", "code": None, "columns": ["revenue"], "rows": [[190]]}


def test_reversed_literal_comparisons_are_supported(task, data):
    sql = f"SELECT SUM(amount) FROM order_lines WHERE 'settled'=status AND '{START}' <= settled_at AND '{END}' > settled_at"
    assert execute_sql(sql, data, task)["rows"] == [[190]]


def test_nested_and_parentheses_remain_within_closed_grammar(task, data):
    sql = f"SELECT SUM(amount) FROM order_lines WHERE (status='settled' AND (settled_at >= '{START}' AND settled_at < '{END}'))"
    assert execute_sql(sql, data, task)["rows"] == [[190]]


def test_legal_left_join_retains_zero_orders(data):
    task = ProbeTask(
        "zeros", "按订单显示九月收入，保留零收入订单", ("preserve_zero_orders",), "orders"
    )
    sql = f"SELECT o.order_id, COALESCE(SUM(l.amount),0) AS revenue FROM orders o LEFT JOIN order_lines l ON o.order_id=l.order_id AND l.status='settled' AND l.settled_at >= '{START}' AND l.settled_at < '{END}' GROUP BY o.order_id"
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert sorted(result["rows"]) == [[1, 90], [2, 30], [3, 70], [4, 0]]


def test_channel_normalization_really_combines_null_and_literal(data):
    task = ProbeTask("channels", "NULL渠道与其他合并", ("merge_other_channels",), "channels")
    sql = f"SELECT COALESCE(o.channel,'其他') AS channel, SUM(l.amount) AS revenue FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id WHERE l.status='settled' AND l.settled_at >= '{START}' AND l.settled_at < '{END}' GROUP BY COALESCE(o.channel,'其他')"
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert dict(result["rows"]) == {"web": 90, "其他": 100}


@pytest.mark.parametrize(
    "sql,code",
    [
        ("SELECT SUM(total) FROM orders", "METRIC_SOURCE_MISMATCH"),
        ("SELECT SUM(DISTINCT amount) FROM order_lines", "DISTINCT_GRAIN_MISMATCH"),
        ("SELECT amount FROM order_lines", "UNSUPPORTED_AGGREGATE"),
        ("SELECT SUM(amount) FROM sqlite_master", "UNKNOWN_TABLE"),
        ("DELETE FROM order_lines", "SQL_NOT_READ_ONLY"),
        ("SELECT SUM(amount) FROM order_lines; DELETE FROM orders", "MULTIPLE_STATEMENTS"),
        ("SELECT SUM(amount) FROM order_lines LIMIT 1", "UNSUPPORTED_SQL"),
        ("SELECT SUM(amount) FROM order_lines ORDER BY amount", "UNSUPPORTED_SQL"),
        ("SELECT SUM(amount) FROM order_lines WHERE amount>0 OR amount<0", "UNSUPPORTED_PREDICATE"),
        ("SELECT SUM(amount) FROM order_lines WHERE amount=amount", "UNREGISTERED_RELATION"),
        ("SELECT SUM(amount) FROM order_lines WHERE amount=1.5", "INVALID_LITERAL"),
        ("SELECT SUM(amount) FROM order_lines WHERE amount='1'", "PREDICATE_TYPE_MISMATCH"),
        (
            "SELECT SUM(amount) FROM order_lines WHERE settled_at>='2026-09-01'",
            "INVALID_TIME_LITERAL",
        ),
        ("SELECT SUM(amount) FROM order_lines WHERE status='unknown'", "INVALID_STATUS_LITERAL"),
        ("SELECT SUM(amount) FROM order_lines WHERE status=NULL", "INVALID_PREDICATE"),
        ("SELECT SUM(abs(amount)) FROM order_lines", "UNSUPPORTED_EXPRESSION"),
        ("WITH x AS (SELECT * FROM order_lines) SELECT SUM(amount) FROM x", "UNSUPPORTED_SQL"),
        (
            "SELECT SUM(amount) FROM order_lines UNION SELECT SUM(amount) FROM order_lines",
            "SQL_NOT_READ_ONLY",
        ),
        (
            "SELECT SUM(amount) FROM order_lines l JOIN orders o ON l.amount=o.total",
            "UNREGISTERED_RELATION",
        ),
        ("SELECT SUM(amount) FROM order_lines l CROSS JOIN orders o", "UNSUPPORTED_JOIN"),
    ],
)
def test_closed_structure_blocks_unsupported_or_unsafe_sql(sql, code, task, data):
    checked = static_check(sql, task, static_semantics=False)
    assert not checked["accepted"]
    assert checked["code"] == code
    result = execute_sql(sql, data, task, static_semantics=False)
    assert result["status"] == "blocked" and result["rows"] == []


@pytest.mark.parametrize(
    "sql,code",
    [
        (
            f"SELECT SUM(amount) FROM order_lines WHERE settled_at >= '{START}' AND settled_at < '{END}'",
            "MISSING_STATUS_CONDITION",
        ),
        (
            f"SELECT SUM(amount) FROM order_lines WHERE status='pending' AND settled_at >= '{START}' AND settled_at < '{END}'",
            "STATUS_CONDITION_MISMATCH",
        ),
        ("SELECT SUM(amount) FROM order_lines WHERE status='settled'", "MISSING_TIME_CONDITION"),
        (GOOD.replace(f"settled_at < '{END}'", f"settled_at <= '{END}'"), "TIME_BOUNDARY_MISMATCH"),
        (
            GOOD.replace(f"settled_at >= '{START}'", f"settled_at > '{START}'"),
            "TIME_BOUNDARY_MISMATCH",
        ),
    ],
)
def test_obvious_semantics_are_not_left_for_dynamic_testing(sql, code, task):
    assert static_check(sql, task, static_semantics=False)["accepted"]
    strong = static_check(sql, task)
    assert not strong["accepted"] and strong["code"] == code


def test_explicit_weak_baseline_does_not_silently_replace_strong(task, data):
    bad = GOOD.replace(f"settled_at < '{END}'", f"settled_at <= '{END}'")
    assert execute_sql(bad, data, task)["status"] == "blocked"
    assert execute_sql(bad, data, task, static_semantics=False)["rows"] == [[1090]]


def test_negative_amount_filter_is_rejected_by_declared_task(data):
    task = ProbeTask("refund", "已结算净收入含退款", ("include_negative",))
    sql = GOOD + " AND amount > 0"
    assert static_check(sql, task)["code"] == "AMOUNT_FILTER_MISMATCH"
    assert execute_sql(sql, data, task, static_semantics=False)["rows"] == [[200]]


def test_zero_order_where_filter_rejected_before_execution(data):
    task = ProbeTask("zeros", "保留全部订单", ("preserve_zero_orders",), "orders")
    sql = f"SELECT o.order_id, COALESCE(SUM(l.amount),0) FROM orders o LEFT JOIN order_lines l ON o.order_id=l.order_id WHERE l.status='settled' AND l.settled_at >= '{START}' AND l.settled_at < '{END}' GROUP BY o.order_id"
    assert static_check(sql, task)["code"] == "ZERO_ORDER_WHERE_FILTER"
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok" and len(weak["rows"]) == 3


def test_normalization_mismatch_is_structurally_detected_not_a_new_dynamic_gain(data):
    task = ProbeTask("channels", "NULL与其他合并", ("merge_other_channels",), "channels")
    sql = f"SELECT COALESCE(o.channel,'其他'), SUM(l.amount) FROM orders o JOIN order_lines l ON o.order_id=l.order_id WHERE l.status='settled' AND l.settled_at >= '{START}' AND l.settled_at < '{END}' GROUP BY o.channel"
    assert static_check(sql, task)["code"] == "GROUP_PROJECTION_MISMATCH"
    assert not static_check(sql, task, static_semantics=False)["accepted"]


def test_empty_fixture_keeps_sql_null_policy_visible(task):
    empty = ProbeData((), ())
    assert execute_sql(GOOD, empty, task)["rows"] == [[0]]
    assert execute_sql(GOOD.replace("COALESCE(SUM(amount),0)", "SUM(amount)"), empty, task)[
        "rows"
    ] == [[None]]


def test_data_owns_nested_rows_and_rejects_mutation():
    orders, lines = [[1, 10, "web"]], [[11, 1, 10, "settled", START]]
    fixture = ProbeData(orders, lines)
    orders[0][1] = 999
    lines[0][2] = 999
    assert fixture.orders == ((1, 10, "web"),)
    assert fixture.lines[0][2] == 10
    with pytest.raises(FrozenInstanceError):
        fixture.orders = ()


@pytest.mark.parametrize(
    "orders,lines",
    [
        (((1, 10, "web"), (1, 20, None)), ()),
        (((True, 10, "web"),), ()),
        (((1, 1.5, "web"),), ()),
        (((1, 10, ""),), ()),
        (((1, 10, "web"),), ((1, 2, 10, "settled", START),)),
        (((1, 10, "web"),), ((1, 1, True, "settled", START),)),
        (((1, 10, "web"),), ((1, 1, 10, "unknown", START),)),
        (((1, 10, "web"),), ((1, 1, 10, "settled", "2026-09-01"),)),
        (((1, 10, "web"),), ((1, 1, 10, "settled", "2026-09-01T00:00:00+00:00"),)),
        (((1, 10, "web"),), ((1, 1, 10, "settled", START), (1, 1, 20, "settled", START))),
    ],
)
def test_invalid_data_never_becomes_a_counterexample(orders, lines):
    with pytest.raises(ValueError):
        ProbeData(orders, lines)


def test_task_fields_are_frozen_validated_and_contain_no_gold():
    names = ["exclude_end"]
    task = ProbeTask("safe", "question", names)
    names.append("exclude_status")
    assert task.properties == ("exclude_end",)
    assert set(task.to_dict()) == {
        "task_id",
        "question",
        "properties",
        "output_kind",
        "start",
        "end",
        "status",
        "other_label",
    }
    assert set(task.public_input()) == {"task", "schema"}
    for props in (("not_real",), ("exclude_end", "exclude_end")):
        with pytest.raises(ValueError):
            ProbeTask("bad", "question", props)
    with pytest.raises(ValueError):
        ProbeTask("bad", "question", ("preserve_zero_orders",))
    with pytest.raises(ValueError):
        ProbeTask("bad", "question", ("include_start",), output_kind="orders")


def test_unknown_task_and_invalid_sql_fail_closed(data, task):
    assert static_check(GOOD, None)["code"] == "INVALID_TASK"
    for sql in (None, "", " " * 16001):
        assert static_check(sql, task)["code"] == "INVALID_SQL"
    assert execute_sql(GOOD, {}, task)["code"] == "INVALID_DATA"


def test_repeated_runs_have_no_shared_database_state(task, data):
    assert execute_sql(GOOD, data, task)["rows"] == [[190]]
    assert execute_sql(GOOD, ProbeData((), ()), task)["rows"] == [[0]]
    assert execute_sql(GOOD, data, task)["rows"] == [[190]]


def test_actual_output_budget_is_enforced_without_truncating_to_success(monkeypatch, data):
    task = ProbeTask("zeros", "保留全部订单", ("preserve_zero_orders",), "orders")
    sql = f"SELECT o.order_id, COALESCE(SUM(l.amount),0) FROM orders o LEFT JOIN order_lines l ON o.order_id=l.order_id AND l.status='settled' AND l.settled_at >= '{START}' AND l.settled_at < '{END}' GROUP BY o.order_id"
    monkeypatch.setattr(runtime, "_MAX_ROWS", 1)
    result = execute_sql(sql, data, task)
    assert result == {
        "status": "error",
        "code": "OUTPUT_BUDGET_EXCEEDED",
        "columns": [],
        "rows": [],
    }


def test_sqlite_progress_handler_enforces_the_real_execution_deadline(monkeypatch, task):
    data = ProbeData(
        ((1, 1000, "web"),), tuple((n, 1, 1, "settled", START) for n in range(1, 1001))
    )
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks, 2.0))
    result = execute_sql(GOOD, data, task)
    assert result == {"status": "error", "code": "QUERY_BUDGET_EXCEEDED", "columns": [], "rows": []}
