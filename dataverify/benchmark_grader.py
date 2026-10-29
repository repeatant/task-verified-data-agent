"""Independent final-output grading for the disclosed DB-GPT development suite.

No checker, executor, SQL execution, or model verdict is used. A small SQLGlot
parser establishes column lineage; Python fixture records establish the exact
multiset of grains entering SUM. Equal scalars alone never prove correctness.

The scoring grammar is one SUM(column), optionally aliased, from registered
tables with at most one INNER JOIN on the registered order_id relation. SUM DISTINCT
is understood and explicitly incorrect for these row-grain metrics. Other SQL
is unscorable, retained in the denominator, and never called a correct answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from .benchmark_tasks import Task, oracle


# Independent, explicit evaluation definitions; do not import checker policy.
_COLUMNS = {
    "orders": ("order_id", "total"),
    "order_lines": ("line_id", "order_id", "amount"),
}
_METRICS = {
    "order_revenue": ("orders", "total", "order_id"),
    "line_revenue": ("order_lines", "amount", "line_id"),
}


class _Unscorable(ValueError):
    pass


@dataclass(frozen=True)
class _Plan:
    sources: tuple[tuple[str, str], ...]  # (SQL alias, registered table)
    value: tuple[str, str]  # (SQL alias, column)
    join: tuple[tuple[str, str], tuple[str, str]] | None
    distinct: bool


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _only_args(node: exp.Expression, allowed: set[str]) -> None:
    if any(key not in allowed and value not in (None, False, []) for key, value in node.args.items()):
        raise _Unscorable("unsupported_sql_clause")


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _source(node: exp.Expression) -> tuple[str, str]:
    if not isinstance(node, exp.Table) or not isinstance(node.this, exp.Identifier):
        raise _Unscorable("unsupported_table_expression")
    _only_args(node, {"this", "alias"})
    table = node.name.lower()
    if table not in _COLUMNS:
        raise _Unscorable("unregistered_table")
    alias = node.args.get("alias")
    if alias is not None:
        _only_args(alias, {"this"})
        if not isinstance(alias.this, exp.Identifier):
            raise _Unscorable("unsupported_table_alias")
    return node.alias_or_name.lower(), table


def _column(node: exp.Expression, sources: dict[str, str]) -> tuple[str, str]:
    node = _unparen(node)
    if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier):
        raise _Unscorable("sum_or_join_requires_column")
    _only_args(node, {"this", "table"})
    name, qualifier = node.name.lower(), node.table.lower()
    matches = [alias for alias, table in sources.items()
               if name in _COLUMNS[table] and (not qualifier or qualifier == alias)]
    if len(matches) != 1:
        raise _Unscorable("unknown_or_ambiguous_column")
    return matches[0], name


def _parse(sql: Any) -> _Plan:
    if not isinstance(sql, str) or not sql.strip():
        raise _Unscorable("missing_final_sql")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, ValueError, TypeError, RecursionError):
        raise _Unscorable("unparseable_final_sql") from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise _Unscorable("requires_one_select")
    query = statements[0]
    _only_args(query, {"expressions", "from_", "joins"})
    if len(query.expressions) != 1:
        raise _Unscorable("requires_one_sum")
    aggregate = query.expressions[0]
    if isinstance(aggregate, exp.Alias):
        _only_args(aggregate, {"this", "alias"})
        aggregate = aggregate.this
    aggregate = _unparen(aggregate)
    if not isinstance(aggregate, exp.Sum):
        raise _Unscorable("requires_one_sum")
    _only_args(aggregate, {"this"})
    argument = _unparen(aggregate.this)
    distinct = isinstance(argument, exp.Distinct)
    if distinct:
        _only_args(argument, {"expressions"})
        if len(argument.expressions) != 1:
            raise _Unscorable("requires_one_sum_column")
        argument = argument.expressions[0]
    source = query.args.get("from_")
    if not isinstance(source, exp.From):
        raise _Unscorable("missing_source_table")
    _only_args(source, {"this"})
    sources = [_source(source.this)]
    joins = query.args.get("joins") or []
    if len(joins) > 1:
        raise _Unscorable("more_than_one_join")
    if joins:
        join = joins[0]
        _only_args(join, {"this", "kind", "on"})
        if (join.args.get("kind") or "").upper() not in ("", "INNER"):
            raise _Unscorable("requires_inner_join")
        sources.append(_source(join.this))
    if len({alias for alias, _ in sources}) != len(sources):
        raise _Unscorable("duplicate_table_alias")
    aliases = dict(sources)
    value = _column(argument, aliases)
    equality = None
    if joins:
        condition = _unparen(joins[0].args.get("on"))
        if not isinstance(condition, exp.EQ):
            raise _Unscorable("requires_key_equality_join")
        left, right = _column(condition.this, aliases), _column(condition.expression, aliases)
        relationship = {(aliases[alias], column) for alias, column in (left, right)}
        if left[0] == right[0] or relationship != {("orders", "order_id"), ("order_lines", "order_id")}:
            raise _Unscorable("requires_registered_order_relation")
        equality = left, right
    return _Plan(tuple(sources), value, equality, distinct)


def _participating_grains(task: Task, plan: _Plan) -> list[Any]:
    tables = {"orders": task.fixture.orders, "order_lines": task.fixture.lines}
    sources = dict(plan.sources)

    def cell(rows: dict, reference: tuple[str, str]) -> Any:
        alias, column = reference
        return rows[alias][_COLUMNS[sources[alias]].index(column)]

    first_alias, first_table = plan.sources[0]
    combinations = [{first_alias: row} for row in tables[first_table]]
    if plan.join is not None:
        second_alias, second_table = plan.sources[1]
        joined = []
        for left in combinations:
            for right in tables[second_table]:
                rows = {**left, second_alias: right}
                a, b = (cell(rows, reference) for reference in plan.join)
                # These fixtures use integer key cells. Unsupported SQLite
                # affinity/collation cases must not silently become truth.
                if any(value is not None and type(value) is not int for value in (a, b)):
                    raise _Unscorable("unsupported_join_key_value")
                if a is not None and b is not None and a == b:
                    joined.append(rows)
        combinations = joined
    value_alias = plan.value[0]
    grain_column = "order_id" if sources[value_alias] == "orders" else "line_id"
    return [cell(rows, (value_alias, grain_column)) for rows in combinations]


def _released_output(record: dict, mode: str) -> tuple[bool, Any, Any, Any, str | None]:
    if mode != "original":
        publication = record.get("publication")
        if not isinstance(publication, dict):
            return False, None, None, None, None
        return (publication.get("status") == "verified", publication.get("verified_sql"),
                publication.get("accepted_value"), publication.get("metric_id"), None)
    accepted = record.get("upstream_success") is True
    reply = record.get("native_reply") or {}
    report = reply.get("action_report") if isinstance(reply, dict) else None
    content = report.get("content") if isinstance(report, dict) else None
    try:
        content = json.loads(content) if isinstance(content, str) else None
    except (ValueError, TypeError, RecursionError):
        content = None
    if not isinstance(content, dict):
        return accepted, None, None, None, "invalid_native_action_content"
    sql, data = content.get("sql"), content.get("data")
    if not isinstance(data, list) or len(data) != 1:
        return accepted, sql, None, None, "requires_one_result_row"
    row = data[0]
    if not isinstance(row, (dict, list)) or len(row) != 1:
        return accepted, sql, None, None, "requires_one_result_column"
    value = next(iter(row.values())) if isinstance(row, dict) else row[0]
    return accepted, sql, value, None, None


def _refusal_state(record: dict, mode: str) -> tuple[bool, bool]:
    """Separate an observed refusal from absence of output after a run error.

    Verification metadata proves only that the arm declined publication; the
    independent row oracle must still establish whether that refusal is right.
    """
    failure_statuses = {"error", "failed", "timeout", "timed_out", "cancelled", "budget_exhausted", "operational_failure"}
    operational_codes = {
        "QUERY_EXECUTION_ERROR", "SOURCE_UNAVAILABLE", "SOURCE_CHANGED_DURING_TASK",
        "UPSTREAM_DID_NOT_FINISH", "TIMEOUT", "BUDGET_EXHAUSTED",
    }
    publication = record.get("publication")
    publication = publication if isinstance(publication, dict) else {}
    actions = record.get("actions") or []
    last_action = actions[-1] if isinstance(actions, list) and actions and isinstance(actions[-1], dict) else {}
    reply = record.get("native_reply")
    reply = reply if isinstance(reply, dict) else {}
    context = reply.get("context")
    context = context if isinstance(context, dict) else {}
    report = reply.get("action_report")
    report = report if isinstance(report, dict) else {}
    terminal = (record, publication, last_action, reply, context, report)
    operational = any(
        str(item.get("execution_status", "")).lower() in failure_statuses
        or str(item.get("status", "")).lower() in failure_statuses
        or str(item.get("error_code", "")) in operational_codes
        or any(token in str(item.get("error_code", "")).upper()
               for token in ("HTTP_", "LIVE_BUDGET", "LIVE_HTTP", "REQUEST_LIMIT", "TIMEOUT", "TRANSPORT_"))
        for item in terminal
    )
    operational = operational or any(
        isinstance(event, dict) and (event.get("status") == "failed" or bool(event.get("error_code")))
        for event in (record.get("http_events") or [])
    )
    operational = operational or bool(record.get("error") or record.get("exception") or record.get("operational_failure"))
    if mode != "original":
        code = str(publication.get("error_code") or "")
        explicit = publication.get("status") == "unverified" and (
            (bool(code) and code not in {"NO_VERIFIED_EVIDENCE", "INVALID_ACTION_FORMAT"})
            or last_action.get("verification_status") == "rejected"
        )
    else:
        try:
            content = json.loads(report.get("content", ""))
        except (ValueError, TypeError, RecursionError):
            content = {}
        explicit = isinstance(content, dict) and str(content.get("status", "")) in {"refused", "unverified"}
        # Native ChartAction's failed SQL/format path is not a semantic refusal.
        if report.get("is_exe_success") is False and not explicit:
            operational = True
    return bool(operational), bool(explicit and not operational)


def grade_record(task: Task, record: dict, *, mode: str | None = None) -> dict[str, Any]:
    """Grade the actual final publication, never candidates/private check logs.

    ``accepted`` records release behavior, not correctness. In the original arm
    native success is merely the upstream acceptance flag; its final Action
    content must separately pass shape, scalar, SQL lineage and row checks.
    Unscorable releases are a separate category, with all correctness flags
    false, and must remain in any overall task denominator.
    """
    mode = mode or record.get("mode")
    if mode not in {"original", "code", "evidence", "guarded"}:
        raise ValueError("mode must be original, code, evidence or guarded")
    expected = oracle(task)
    accepted, sql, value, metric_id, output_error = _released_output(record, mode)
    operational_failure, refusal_explicit = _refusal_state(record, mode)
    result = {
        "accepted": accepted, "target_defined": expected["target_defined"],
        "correct_answer": False, "correct_refusal": False, "wrong_release": False,
        "unscorable": False, "metric_drift": metric_id is not None and metric_id != task.confirmed_metric_id,
        "final_sql": sql, "observed_value": value if _finite(value) else None,
        "reason": None, "expected_value": expected["metric_value"],
        "confirmed_metric_id": task.confirmed_metric_id, "published_metric_id": metric_id,
        "source_rows": expected["source_rows"], "source_grains": expected["source_grains"],
        "operational_failure": operational_failure, "refusal_explicit": refusal_explicit,
    }
    if not accepted:
        correct_refusal = not expected["target_defined"] and refusal_explicit
        reason = "operational_failure" if operational_failure else (
            "correct_refusal_of_undefined_target" if correct_refusal else (
                "refused_defined_target" if expected["target_defined"] else "no_explicit_refusal"))
        result.update(correct_refusal=correct_refusal, reason=reason)
        return result
    if not expected["target_defined"]:
        result.update(wrong_release=True, reason="undefined_target:" + expected["reason"])
        return result
    if output_error or not _finite(value):
        result.update(wrong_release=True, reason=output_error or "invalid_released_scalar")
        return result
    try:
        plan = _parse(sql)
        participating = Counter(_participating_grains(task, plan))
    except _Unscorable as error:
        result.update(unscorable=True, reason=str(error))
        return result
    source_table = dict(plan.sources)[plan.value[0]]
    observed_source = source_table, plan.value[1]
    metric_source = _METRICS[task.confirmed_metric_id][:2]
    result["aggregate_source"] = {"table": source_table, "column": plan.value[1]}
    result["inferred_metric_id"] = next((name for name, definition in _METRICS.items()
                                          if definition[:2] == observed_source), None)
    result["metric_drift"] = result["metric_drift"] or observed_source != metric_source
    source_rows = task.fixture.orders if metric_source[0] == "orders" else task.fixture.lines
    required = Counter(row[0] for row in source_rows)
    result["aggregate_rows"] = sum(participating.values())
    result["grain_multiplicities"] = [{"grain": grain, "count": count} for grain, count in participating.items()]
    result["missing_grains"] = list((required - participating).elements())
    result["extra_grains"] = list((participating - required).elements())
    reason = None
    if mode != "original" and metric_id != task.confirmed_metric_id:
        reason = "published_metric_mismatch"
    elif observed_source != metric_source:
        reason = "aggregate_source_mismatch"
    elif plan.distinct:
        reason = "value_distinct_is_not_grain_sum"
    elif participating != required:
        reason = "incomplete_or_repeated_grain"
    elif not math.isclose(value, expected["metric_value"], rel_tol=1e-12, abs_tol=1e-9):
        reason = "released_value_mismatch"
    result.update(correct_answer=reason is None, wrong_release=reason is not None,
                  reason=reason or "correct_answer")
    return result


# Short alias for callers that already carry the task and native record.
grade = grade_record
