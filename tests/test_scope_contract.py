"""General scope and outer-join regressions after the frozen evaluation.

These are regression tests, not an additional holdout score. The source below
uses a different cross-midnight interval, statuses, keys and channels from the
frozen challenges. Expected rows are specified independently by hand. All SQL
runs locally against immutable data; no model client or API is constructed.
"""

from dataclasses import replace

import pytest

from dataverify.probe_runtime import ProbeData, ProbeTask, execute_sql, static_check

START = "2026-12-03T22:40:00"
END = "2026-12-04T03:10:00"
LABEL = "Unassigned"
EXPECTED_CHANNELS = [["Unassigned", 13], ["courier", 0], ["kiosk", 0], ["partner", 17]]


def task_for(kind="channels", status="pending"):
    properties = {
        "channels": ("merge_other_channels",),
        "orders": ("preserve_zero_orders",),
        "scalar": ("exclude_end", "exclude_status", "include_start", "include_negative"),
    }[kind]
    return ProbeTask(
        "scope-overnight-" + kind + "-" + status,
        "Sum eligible line amounts in the confirmed interval and retain genuine zero-valued groups.",
        properties,
        output_kind=kind,
        start=START,
        end=END,
        status=status,
        other_label=LABEL,
    )


def source_for(status="pending"):
    return ProbeData(
        (
            (401, -99, "kiosk"),
            (402, 7, None),
            (403, 333, LABEL),
            (404, 0, "courier"),
            (405, 19, "partner"),
            (406, 12, "inactive"),
            (407, 0, "quiet"),
        ),
        (
            (8101, 401, 31, status, START),
            (8102, 401, -31, status, "2026-12-03T23:01:00"),
            (8103, 402, 18, status, "2026-12-04T00:02:00"),
            (8104, 403, -5, status, "2026-12-04T03:09:59"),
            (8105, 404, 0, status, "2026-12-04T01:20:00"),
            (8106, 405, 24, status, "2026-12-04T01:40:00"),
            (8107, 405, -7, status, "2026-12-04T02:30:00"),
            (8108, 406, 700, "settled", "2026-12-04T00:30:00"),
            (8109, 407, 800, status, END),
            (8110, 407, 900, status, "2026-12-03T22:39:59"),
            (8111, 401, 1000, "settled", "2026-12-04T00:35:00"),
            (8112, 402, 900, status, END),
        ),
    )


def qualifications(task):
    return (
        f"l.status='{task.status}'",
        f"l.settled_at >= '{task.start}' AND l.settled_at < '{task.end}'",
    )


def select_and_group(task, wrapped=True):
    aggregate = "COALESCE(SUM(l.amount),0)" if wrapped else "SUM(l.amount)"
    if task.output_kind == "scalar":
        return "SELECT " + aggregate, ""
    key = (
        "o.order_id"
        if task.output_kind == "orders"
        else f"COALESCE(o.channel,'{task.other_label}')"
    )
    return "SELECT " + key + ", " + aggregate, " GROUP BY " + key


def query_for(task, shape="inner-where", wrapped=True):
    select, group = select_and_group(task, wrapped)
    status, times = qualifications(task)
    key = "o.order_id=l.order_id"
    shapes = {
        "inner-where": f"orders o INNER JOIN order_lines l ON {key} WHERE {status} AND {times}",
        "inner-on": f"order_lines l INNER JOIN orders o ON {key} AND {status} AND {times}",
        "left-where": f"orders o LEFT JOIN order_lines l ON {key} WHERE {status} AND {times}",
        "left-on-status": f"orders o LEFT JOIN order_lines l ON {key} AND {status} WHERE {times}",
        "left-on-time": f"orders o LEFT JOIN order_lines l ON {key} AND {times} WHERE {status}",
        "left-on-all-with-null-rejection": (
            f"orders o LEFT JOIN order_lines l ON {key} AND {status} AND {times} WHERE {status}"
        ),
        "reverse-left-where": f"order_lines l LEFT JOIN orders o ON {key} WHERE {status} AND {times}",
        "left-on-all": f"orders o LEFT JOIN order_lines l ON {key} AND {status} AND {times}",
    }
    return select + " FROM " + shapes[shape] + group


def assert_blocked(sql, task, data, code):
    # The grammar-only baseline remains explicit. Business requirements must
    # reject this candidate before it can become a published numeric result.
    assert static_check(sql, task, static_semantics=False)["accepted"]
    checked = static_check(sql, task)
    assert checked["accepted"] is False
    assert checked["code"] == code
    result = execute_sql(sql, data, task)
    assert result["status"] == "blocked"
    assert result["code"] == code
    assert result["rows"] == []


@pytest.mark.parametrize(
    "extra",
    [
        "o.channel='kiosk'",
        "o.channel IS NULL",
        "o.order_id=401",
        "o.total>0",
        "l.order_id=405",
        "l.line_id=8101",
        "l.amount >= 0",
        "0 < l.amount",
    ],
)
@pytest.mark.parametrize("location", ["on", "where"])
def test_undeclared_parent_and_child_scope_filters_are_blocked(extra, location):
    task, data = task_for(), source_for()
    select, group = select_and_group(task)
    status, times = qualifications(task)
    on = "o.order_id=l.order_id" + (" AND " + extra if location == "on" else "")
    where = status + " AND " + times + (" AND " + extra if location == "where" else "")
    sql = select + " FROM orders o INNER JOIN order_lines l ON " + on + " WHERE " + where + group
    assert_blocked(sql, task, data, "UNDECLARED_SCOPE_FILTER")


def test_existing_refund_filter_error_keeps_priority_over_generic_scope_error():
    task, data = task_for("scalar", "cancelled"), source_for("cancelled")
    sql = query_for(task) + " AND o.channel='partner' AND l.amount > 0"
    assert_blocked(sql, task, data, "AMOUNT_FILTER_MISMATCH")


def test_amount_filter_without_refund_property_still_changes_undeclared_scope():
    task = replace(task_for("scalar"), properties=("exclude_end",))
    sql = query_for(task) + " AND l.amount > 0"
    assert_blocked(sql, task, source_for(), "UNDECLARED_SCOPE_FILTER")


@pytest.mark.parametrize("wrapped", [False, True], ids=["sum", "coalesced-sum"])
@pytest.mark.parametrize("status", ["pending", "cancelled"])
def test_parent_left_join_cannot_create_channels_without_eligible_lines(wrapped, status):
    task, data = task_for(status=status), source_for(status)
    sql = query_for(task, "left-on-all", wrapped)
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok"
    assert {row[0] for row in weak["rows"]} == {
        "Unassigned",
        "courier",
        "inactive",
        "kiosk",
        "partner",
        "quiet",
    }
    assert_blocked(sql, task, data, "UNMATCHED_GROUP_COVERAGE")


LEGAL_GROUP_SHAPES = [
    "inner-where",
    "inner-on",
    "left-where",
    "left-on-status",
    "left-on-time",
    "left-on-all-with-null-rejection",
    "reverse-left-where",
]


@pytest.mark.parametrize("shape", LEGAL_GROUP_SHAPES)
@pytest.mark.parametrize("status", ["pending", "cancelled"])
def test_real_zero_groups_survive_supported_join_and_filter_locations(shape, status):
    task, data = task_for(status=status), source_for(status)
    sql = query_for(task, shape)
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok", result
    # kiosk is a cancelling positive/refund pair; courier has an eligible zero
    # amount. Both belong in the result, unlike inactive/quiet.
    assert sorted(result["rows"]) == EXPECTED_CHANNELS


@pytest.mark.parametrize("shape", ["inner-where", "left-where", "reverse-left-where"])
@pytest.mark.parametrize("empty", [False, True], ids=["no-eligible-lines", "no-orders"])
def test_channels_without_eligible_lines_are_an_empty_relation(shape, empty):
    task, full = task_for(status="cancelled"), source_for("cancelled")
    data = (
        ProbeData((), ())
        if empty
        else ProbeData(
            full.orders,
            tuple(
                row for row in full.lines if row[3] != task.status or not (START <= row[4] < END)
            ),
        )
    )
    sql = query_for(task, shape)
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert result["rows"] == []


@pytest.mark.parametrize("kind", ["scalar", "channels"])
@pytest.mark.parametrize("on_filter", ["status", "time", "all"])
def test_reverse_left_join_on_cannot_filter_the_preserved_line_rows(kind, on_filter):
    task, data = task_for(kind, "cancelled"), source_for("cancelled")
    select, group = select_and_group(task)
    status, times = qualifications(task)
    on, where = {
        "status": (status, times),
        "time": (times, status),
        "all": (status + " AND " + times, None),
    }[on_filter]
    sql = select + " FROM order_lines l LEFT JOIN orders o ON l.order_id=o.order_id AND " + on
    if where:
        sql += " WHERE " + where
    sql += group
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok"
    assert weak["rows"] != ([[30]] if kind == "scalar" else EXPECTED_CHANNELS)
    assert_blocked(sql, task, data, "FILTER_PLACEMENT_MISMATCH")


def test_legal_reverse_left_scalar_filters_the_driving_lines_in_where():
    task, data = task_for("scalar", "cancelled"), source_for("cancelled")
    result = execute_sql(query_for(task, "reverse-left-where"), data, task)
    assert result["status"] == "ok"
    assert result["rows"] == [[30]]


def test_scope_rejection_clears_previous_publication_and_correct_query_can_recover():
    pytest.importorskip("hypothesis", reason="Optional probe experiment")
    from dataverify.probe_guard import ProbeGuard

    task, data = task_for(status="cancelled"), source_for("cancelled")
    guard = ProbeGuard(task, data)
    good = query_for(task, "reverse-left-where")
    assert guard.execute(good, task.task_id)["status"] == "verified"
    assert sorted(guard.publish()["rows"]) == EXPECTED_CHANNELS
    rejected = guard.execute(query_for(task, "left-on-all"), task.task_id)
    assert rejected["error_code"] == "UNMATCHED_GROUP_COVERAGE"
    assert rejected["rows"] is None
    assert guard.publish()["rows"] is None
    assert guard.publish()["verified_sql"] is None
    recovered = guard.execute(good, task.task_id)
    assert recovered["status"] == "verified"
    assert sorted(recovered["rows"]) == EXPECTED_CHANNELS


def witness_query(task, driving, witness):
    select, group = select_and_group(task)
    status, times = qualifications(task)
    tables = (
        "orders o LEFT JOIN order_lines l"
        if driving == "orders"
        else "order_lines l LEFT JOIN orders o"
    )
    return (
        select
        + " FROM "
        + tables
        + " ON o.order_id=l.order_id AND "
        + status
        + " AND "
        + times
        + " WHERE "
        + witness
        + group
    )


@pytest.mark.parametrize(
    "extra",
    [
        "o.order_id > 0",
        "o.order_id >= 1",
        "0 < l.line_id",
        "1 <= l.order_id",
    ],
)
def test_schema_implied_positive_key_filters_do_not_change_inner_join_scope(extra):
    task, data = task_for(), source_for()
    select, group = select_and_group(task)
    status, times = qualifications(task)
    sql = (
        select
        + " FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id WHERE "
        + status
        + " AND "
        + times
        + " AND "
        + extra
        + group
    )
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert sorted(result["rows"]) == EXPECTED_CHANNELS


@pytest.mark.parametrize(
    "driving,witness",
    [
        ("orders", "l.line_id > 0"),
        ("orders", "l.order_id >= 1"),
        ("orders", "0 < l.line_id"),
        ("orders", "1 <= l.order_id"),
        ("lines", "o.order_id > 0"),
        ("lines", "o.order_id >= 1"),
    ],
)
@pytest.mark.parametrize("status", ["pending", "cancelled"])
def test_nullable_side_positive_key_witness_applies_on_filters_without_losing_real_zeros(
    driving, witness, status
):
    task, data = task_for(status=status), source_for(status)
    sql = witness_query(task, driving, witness)
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok", result
    # The predicate is true on every matched stored row, but UNKNOWN on the
    # outer join's NULL extension. Keeping that distinction removes fake groups
    # while preserving both kinds of genuine zero-valued channel above.
    assert sorted(result["rows"]) == EXPECTED_CHANNELS


@pytest.mark.parametrize("witness", ["o.order_id > 0", "o.order_id >= 1"])
def test_parent_match_witness_makes_reverse_left_on_filters_effective_for_scalar(witness):
    task, data = task_for("scalar", "cancelled"), source_for("cancelled")
    sql = witness_query(task, "lines", witness)
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert result["rows"] == [[30]]


@pytest.mark.parametrize(
    "driving,witness,code",
    [
        ("orders", "o.order_id > 0", "UNMATCHED_GROUP_COVERAGE"),
        ("orders", "o.order_id >= 1", "UNMATCHED_GROUP_COVERAGE"),
        ("lines", "l.line_id > 0", "FILTER_PLACEMENT_MISMATCH"),
        ("lines", "l.order_id >= 1", "FILTER_PLACEMENT_MISMATCH"),
    ],
)
def test_positive_driving_key_is_not_a_nullable_side_match_witness(driving, witness, code):
    task, data = task_for(status="cancelled"), source_for("cancelled")
    sql = witness_query(task, driving, witness)
    # A positive key on the preserved side is also true for unmatched outer
    # rows. It cannot make the other side's status/date ON predicates effective.
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok"
    assert sorted(weak["rows"]) != EXPECTED_CHANNELS
    assert_blocked(sql, task, data, code)


@pytest.mark.parametrize(
    "driving,witness",
    [
        ("orders", "l.line_id > 1"),
        ("orders", "l.order_id > 1"),
        ("orders", "1 < l.line_id"),
        ("lines", "o.order_id > 1"),
        ("lines", "o.order_id >= 2"),
        ("lines", "2 <= o.order_id"),
    ],
)
def test_stricter_id_threshold_is_not_a_schema_implied_match_witness(driving, witness):
    task = task_for(status="cancelled")
    data = ProbeData(
        ((1, 10, "first"), (9, 4, "later"), (10, 0, "quiet")),
        (
            (1, 1, 0, task.status, START),
            (2, 9, 17, task.status, "2026-12-04T03:09:59"),
            (3, 10, 99, "settled", "2026-12-04T01:00:00"),
            (4, 10, 75, task.status, END),
        ),
    )
    baseline = execute_sql(query_for(task), data, task)
    assert sorted(baseline["rows"]) == [["first", 0], ["later", 17]]
    sql = witness_query(task, driving, witness)
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok"
    assert weak["rows"] == [["later", 17]]
    assert_blocked(sql, task, data, "UNDECLARED_SCOPE_FILTER")


@pytest.mark.parametrize("witness", ["l.line_id > 0", "l.order_id >= 1"])
def test_nullable_child_match_witness_still_violates_preserve_all_orders(witness):
    task, data = task_for("orders", "cancelled"), source_for("cancelled")
    sql = witness_query(task, "orders", witness)
    weak = execute_sql(sql, data, task, static_semantics=False)
    assert weak["status"] == "ok"
    assert {row[0] for row in weak["rows"]} == {401, 402, 403, 404, 405}
    assert_blocked(sql, task, data, "ZERO_ORDER_WHERE_FILTER")


@pytest.mark.parametrize("witness", ["o.order_id > 0", "o.order_id >= 1"])
def test_schema_implied_driving_key_where_can_preserve_all_orders(witness):
    task, data = task_for("orders", "cancelled"), source_for("cancelled")
    sql = witness_query(task, "orders", witness)
    assert static_check(sql, task)["accepted"]
    result = execute_sql(sql, data, task)
    assert result["status"] == "ok"
    assert sorted(result["rows"]) == [
        [401, 0],
        [402, 18],
        [403, -5],
        [404, 0],
        [405, 17],
        [406, 0],
        [407, 0],
    ]
