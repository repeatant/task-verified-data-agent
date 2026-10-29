"""Capability feedback must help repair without changing the acceptance gate.

The fixture is deliberately independent of the disclosed d03/d10 development
data: it uses July, a non-default confirmed status, refunds, empty parents and
both kinds of excluded child records. Expected rows are specified by hand.
No credentials are read and no real model client is constructed.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import pytest

pytest.importorskip("hypothesis", reason="Optional probe experiment")

from dataverify.probe_capabilities import POLICIES, repair_guidance, sql_capabilities
from dataverify.probe_guard import ProbeGuard, source_digest
from dataverify.probe_runtime import ProbeData, ProbeTask, execute_sql, static_check


TASK = ProbeTask(
    "capability-july-cancelled",
    "按订单累计七月已取消明细的净额，保留没有符合条件明细的订单与合法负金额。",
    ("preserve_zero_orders",), output_kind="orders",
    start="2026-07-01T00:00:00", end="2026-08-01T00:00:00", status="cancelled",
)
DATA = ProbeData(
    ((101, 999, "web"), (102, 0, None), (103, 88, "app"),
     (104, 0, "web"), (105, 20, "app")),
    (
        (11, 101, 100, "cancelled", "2026-07-01T00:00:00"),
        (12, 101, -25, "cancelled", "2026-07-15T12:00:00"),
        (13, 101, 500, "cancelled", "2026-08-01T00:00:00"),
        (14, 101, 70, "settled", "2026-07-15T12:00:00"),
        (31, 103, 88, "pending", "2026-07-15T12:00:00"),
        (41, 104, 999, "cancelled", "2026-08-01T00:00:00"),
        (51, 105, -9, "cancelled", "2026-07-20T12:00:00"),
    ),
)
EXPECTED = [[101, 75], [102, 0], [103, 0], [104, 0], [105, -9]]
PREDICATES = (
    "l.status = 'cancelled' AND l.settled_at >= '2026-07-01T00:00:00' "
    "AND l.settled_at < '2026-08-01T00:00:00'"
)
GOOD = (
    "SELECT o.order_id, COALESCE(SUM(l.amount),0) FROM orders o "
    "LEFT JOIN order_lines l ON o.order_id=l.order_id AND " + PREDICATES
    + " GROUP BY o.order_id"
)
BAD_WHERE = (
    "SELECT o.order_id, COALESCE(SUM(l.amount),0) FROM orders o "
    "LEFT JOIN order_lines l ON o.order_id=l.order_id WHERE " + PREDICATES
    + " GROUP BY o.order_id"
)
CASE_AGGREGATE = (
    "SELECT o.order_id, COALESCE(SUM(CASE WHEN " + PREDICATES
    + " THEN l.amount ELSE 0 END),0) FROM orders o "
    "LEFT JOIN order_lines l ON o.order_id=l.order_id GROUP BY o.order_id"
)


def ordered(rows):
    return sorted(rows, key=lambda row: row[0])


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("mode", ["code", "full", "compact"])
def test_supported_on_filters_preserve_zeros_refunds_and_bounds(policy, mode):
    before = source_digest(TASK, DATA)
    guard = ProbeGuard(TASK, DATA, mode, feedback_policy=policy)
    public = guard.execute(GOOD, TASK.task_id)
    assert public["status"] == "verified"
    assert ordered(public["rows"]) == EXPECTED
    assert ordered(guard.publish()["rows"]) == EXPECTED
    assert public["error_code"] is None
    assert source_digest(TASK, DATA) == before


@pytest.mark.parametrize("policy", POLICIES)
def test_syntax_feedback_does_not_expand_the_actual_sql_whitelist(policy):
    # This CASE formulation is legitimate SQL, but still outside this runner's
    # declared subset. Guidance must not silently make it executable.
    assert not static_check(CASE_AGGREGATE, TASK)["accepted"]
    assert execute_sql(CASE_AGGREGATE, DATA, TASK)["status"] == "blocked"
    guard = ProbeGuard(TASK, DATA, feedback_policy=policy)
    public = guard.execute(CASE_AGGREGATE, TASK.task_id)
    assert public["status"] == "unverified"
    assert public["error_code"] == "UNSUPPORTED_EXPRESSION"
    assert public["rows"] is None
    assert guard.publish()["verified_sql"] is None
    assert guard.publish()["rows"] is None
    serialized = json.dumps(public, ensure_ascii=False)
    for hidden in ("reference_sql", "fault_label", "expected_rows", GOOD):
        assert hidden not in serialized


@pytest.mark.parametrize("policy", POLICIES)
def test_where_filter_that_discards_zero_orders_stays_rejected(policy):
    guard = ProbeGuard(TASK, DATA, feedback_policy=policy)
    result = guard.execute(BAD_WHERE, TASK.task_id)
    assert result["error_code"] == "ZERO_ORDER_WHERE_FILTER"
    assert result["status"] == "unverified"
    assert guard.publish()["rows"] is None


def test_v2_classifies_semantic_and_supported_syntax_repairs_separately():
    semantic = ProbeGuard(TASK, DATA, feedback_policy="capability_v2").execute(BAD_WHERE, TASK.task_id)
    syntax = ProbeGuard(TASK, DATA, feedback_policy="capability_v2").execute(CASE_AGGREGATE, TASK.task_id)
    assert semantic["repair_guidance"]["classification"] == "task_requirement"
    assert syntax["repair_guidance"]["classification"] == "unsupported_syntax"
    for result in (semantic, syntax):
        guidance = result["repair_guidance"]
        assert guidance["retryable"] is True
        assert guidance["allowed_alternatives"]
        assert all(type(item) is str and item.strip() for item in guidance["allowed_alternatives"])
        assert guidance["preserve_requirements"]
        assert all(type(item) is str and item.strip() for item in guidance["preserve_requirements"])


def test_legacy_keeps_previous_feedback_without_new_guidance():
    legacy = ProbeGuard(TASK, DATA, feedback_policy="legacy_v1").execute(BAD_WHERE, TASK.task_id)
    enhanced = ProbeGuard(TASK, DATA, feedback_policy="capability_v2").execute(BAD_WHERE, TASK.task_id)
    assert "repair_guidance" not in legacy
    # New instructions may help a future model turn, not change this verdict or
    # expose a different counterexample in the comparison arm.
    enhanced.pop("repair_guidance")
    assert enhanced == legacy


@pytest.mark.parametrize("mode", ["code", "full", "compact"])
def test_repair_keeps_old_witness_and_publishes_only_fresh_accepted_result(mode):
    guard = ProbeGuard(TASK, DATA, mode, feedback_policy="capability_v2")
    first = guard.execute(BAD_WHERE, TASK.task_id)
    assert first["status"] == "unverified"
    repaired = guard.execute(GOOD, TASK.task_id)
    assert repaired["status"] == "verified"
    assert ordered(repaired["rows"]) == EXPECTED
    assert guard.last_result["regression_checks"]
    assert all(check["status"] == "passed" for check in guard.last_result["regression_checks"])
    # A later unsupported action cannot reuse the previously verified answer.
    guard.execute(CASE_AGGREGATE, TASK.task_id)
    assert guard.publish()["status"] == "unverified"
    assert guard.publish()["rows"] is None


def test_guidance_does_not_authorize_different_task_even_with_good_sql():
    guard = ProbeGuard(TASK, DATA, feedback_policy="capability_v2")
    guard.execute(BAD_WHERE, TASK.task_id)
    changed = guard.execute(GOOD, "replacement-task")
    assert changed["error_code"] == "TASK_DRIFT"
    assert guard.publish()["rows"] is None
    assert guard.task == TASK


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("source", [
    ProbeData((), ()),
    ProbeData(((201, 100, "web"),),
              ((201, 201, 100, "pending", "2026-07-15T12:00:00"),)),
], ids=["empty-source", "no-eligible-lines"])
def test_scalar_empty_result_never_publishes_null_as_a_verified_amount(policy, source):
    scalar_task = replace(TASK, task_id="scalar-empty-case", question="七月已取消明细净额，无匹配时为零。",
                          output_kind="scalar",
                          properties=("exclude_end", "exclude_status", "include_start", "include_negative"))
    naked_sum = "SELECT SUM(l.amount) FROM order_lines l WHERE " + PREDICATES
    guard = ProbeGuard(scalar_task, source, feedback_policy=policy)
    result = guard.execute(naked_sum, scalar_task.task_id)
    assert result["status"] == "unverified"
    assert guard.publish()["rows"] is None
    assert guard.publish()["verified_sql"] is None


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("source", [
    ProbeData((), ()),
    ProbeData(((201, 100, "web"),),
              ((201, 201, 100, "pending", "2026-07-15T12:00:00"),)),
], ids=["empty-source", "no-eligible-lines"])
def test_supported_scalar_zero_wrapper_accepts_empty_or_nonmatching_source(policy, source):
    scalar_task = replace(TASK, task_id="scalar-zero-case", question="七月已取消明细净额，无匹配时为零。",
                          output_kind="scalar",
                          properties=("exclude_end", "exclude_status", "include_start", "include_negative"))
    wrapped_sum = "SELECT COALESCE(SUM(l.amount),0) FROM order_lines l WHERE " + PREDICATES
    guard = ProbeGuard(scalar_task, source, feedback_policy=policy)
    result = guard.execute(wrapped_sum, scalar_task.task_id)
    assert result["status"] == "verified"
    assert result["rows"] == [[0]]
    assert guard.publish()["rows"] == [[0]]


@pytest.mark.parametrize("policy", POLICIES)
def test_legitimate_empty_grouped_result_is_distinct_from_null_scalar(policy):
    # An empty set of parent IDs is a valid empty relation. The scalar NULL fix
    # must not become a blanket "nonempty result" requirement.
    guard = ProbeGuard(TASK, ProbeData((), ()), feedback_policy=policy)
    result = guard.execute(GOOD, TASK.task_id)
    assert result["status"] == "verified"
    assert result["rows"] == []
    assert guard.publish()["rows"] == []


def test_capability_and_guidance_objects_cannot_be_mutated_across_tasks():
    first = sql_capabilities()
    original = deepcopy(first)
    first["aggregate"]["argument"] = "orders.total"
    first["unsupported"].clear()
    assert sql_capabilities() == original
    guidance = repair_guidance("ZERO_ORDER_WHERE_FILTER", TASK)
    clean = deepcopy(guidance)
    guidance["allowed_alternatives"].clear()
    assert repair_guidance("ZERO_ORDER_WHERE_FILTER", TASK) == clean


@pytest.mark.parametrize("policy", ["", "unknown", None, True])
def test_unknown_feedback_policy_fails_before_any_action(policy):
    with pytest.raises(ValueError):
        ProbeGuard(TASK, DATA, feedback_policy=policy)


@pytest.fixture
def native_types():
    pytest.importorskip("dbgpt", reason="Optional pinned DB-GPT runtime")
    from dataverify.probe_agent_bridge import ProbeResource, probe_sql_candidate, run_probe_native_case
    return ProbeResource, probe_sql_candidate, run_probe_native_case


def test_resource_contract_matches_what_guard_accepts(native_types):
    resource_type, _, _ = native_types
    resource = resource_type(TASK, feedback_policy="capability_v2")
    try:
        payload = json.loads(resource.get_schema_link(resource._db_name))
        contract = payload["execution_contract"]
        assert contract == sql_capabilities()
        assert contract["version"] == "probe-sql-v1"
        assert contract["aggregate"]["argument"] == "order_lines.amount"
        assert {"WHERE", "JOIN_ON"}.issubset(contract["filter_locations"])
        assert "CASE" in contract["unsupported"]
        assert static_check(GOOD, TASK)["accepted"]
        assert not static_check(CASE_AGGREGATE, TASK)["accepted"]
        assert payload["confirmed_task"] == TASK.to_dict()
        assert "reference_sql" not in json.dumps(payload)
    finally:
        resource.close()


def test_legacy_resource_contains_no_v2_contract(native_types):
    resource_type, _, _ = native_types
    resource = resource_type(TASK, feedback_policy="legacy_v1")
    try:
        payload = json.loads(resource.get_schema_link(resource._db_name))
        assert "execution_contract" not in payload
        assert payload["confirmed_task"] == TASK.to_dict()
    finally:
        resource.close()


def test_native_retry_receives_guidance_without_reference_answer(native_types, tmp_path):
    _, candidate, run_native = native_types
    record = asyncio.run(run_native(
        TASK, DATA, feedback_policy="capability_v2", output_dir=tmp_path,
        responses=[candidate(BAD_WHERE, TASK.task_id), candidate(GOOD, TASK.task_id)],
    ))
    assert record["feedback_policy"] == "capability_v2"
    assert record["publication"]["status"] == "verified"
    assert ordered(record["publication"]["rows"]) == EXPECTED
    assert record["llm_calls"] == 2
    assert record["http_events"] == []
    assert record["raw_resource_queries"] == 0
    assert record["source_unchanged"]
    retry_context = json.dumps(record["llm_requests"][1], ensure_ascii=False)
    assert "repair_guidance" in retry_context
    assert "execution_contract" in retry_context
    assert GOOD not in retry_context
    assert "reference_sql" not in retry_context
    assert "fault_label" not in retry_context
    assert record["actions"][1]["regression_checks"]
