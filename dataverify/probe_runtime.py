"""Bounded runtime for the method-transfer development experiment.

This is a separate SQL subset, not an implicit expansion of ``checker.py``.
Properties are confirmed task requirements, never a reference SQL or gold
answer. The default static baseline checks the obvious requirements too;
dynamic tests must not claim those already-rejected cases as new discoveries.

All execution uses a new in-memory SQLite database. No supplied path or network
resource is accepted, and candidate SQL runs only after a closed AST check.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError, TokenError

TABLE_SCHEMA = {
    "orders": ("order_id", "total", "channel"),
    "order_lines": ("line_id", "order_id", "amount", "status", "settled_at"),
}
PROPERTY_IDS = frozenset(
    {
        "exclude_end",
        "exclude_status",
        "include_start",
        "preserve_zero_orders",
        "merge_other_channels",
        "include_negative",
    }
)
STATUSES = frozenset({"settled", "pending", "cancelled"})
_MAX_ROWS = 1_000
_MAX_INT = 1_000_000_000
_MAX_SQL = 16_000


def _iso_second(value: Any) -> bool:
    if type(value) is not str or len(value) != 19:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is None and parsed.strftime("%Y-%m-%dT%H:%M:%S") == value


def _integer(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and (-_MAX_INT <= value <= _MAX_INT) and (not positive or value > 0)


@dataclass(frozen=True)
class ProbeData:
    """Immutable legal fixture; integer amounts avoid floating-point oracles.

    Amounts are integer minor units. Negative line amounts are valid (refunds).
    ``total`` is independent source data and need not equal summed line amounts.
    """

    orders: tuple[tuple[int, int, str | None], ...]
    lines: tuple[tuple[int, int, int, str, str], ...]

    def __post_init__(self) -> None:
        for name, width in (("orders", 3), ("lines", 5)):
            try:
                rows = tuple(tuple(row) for row in getattr(self, name))
            except TypeError as exc:
                raise ValueError(f"{name} must contain immutable scalar rows") from exc
            if len(rows) > _MAX_ROWS or any(len(row) != width for row in rows):
                raise ValueError(f"{name}: at most {_MAX_ROWS} rows of width {width}")
            object.__setattr__(self, name, rows)
        keys: set[int] = set()
        for order_id, total, channel in self.orders:
            if not _integer(order_id, positive=True) or order_id in keys:
                raise ValueError("orders.order_id must be a unique positive integer")
            if not _integer(total):
                raise ValueError("orders.total must be a bounded integer amount")
            if channel is not None and (
                type(channel) is not str or not channel or len(channel) > 80
            ):
                raise ValueError("orders.channel must be NULL or a nonempty short string")
            keys.add(order_id)
        line_keys: set[int] = set()
        for line_id, order_id, amount, status, settled_at in self.lines:
            if not _integer(line_id, positive=True) or line_id in line_keys:
                raise ValueError("order_lines.line_id must be a unique positive integer")
            if not _integer(order_id, positive=True) or order_id not in keys:
                raise ValueError("order_lines.order_id must reference an existing order")
            if not _integer(amount):
                raise ValueError("order_lines.amount must be a bounded integer amount")
            if type(status) is not str or status not in STATUSES:
                raise ValueError("order_lines.status is not registered")
            if not _iso_second(settled_at):
                raise ValueError("settled_at must use canonical timezone-free ISO seconds")
            line_keys.add(line_id)

    def to_dict(self) -> dict[str, list[list[Any]]]:
        return {
            "orders": [list(row) for row in self.orders],
            "lines": [list(row) for row in self.lines],
        }


@dataclass(frozen=True)
class ProbeTask:
    """Confirmed *partial* task requirements, with no SQL or answer field."""

    task_id: str
    question: str
    properties: tuple[str, ...]
    output_kind: str = "scalar"
    start: str = "2026-09-01T00:00:00"
    end: str = "2026-10-01T00:00:00"
    status: str = "settled"
    other_label: str = "其他"

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id or len(self.task_id) > 80:
            raise ValueError("task_id must be a nonempty short string")
        if type(self.question) is not str or not self.question or len(self.question) > 6_000:
            raise ValueError("question must be a nonempty bounded string")
        if not isinstance(self.properties, (tuple, list)):
            raise ValueError("properties must be a list or tuple of registered names")
        properties = tuple(self.properties)
        if any(type(item) is not str or item not in PROPERTY_IDS for item in properties):
            raise ValueError("Unknown task property")
        if len(set(properties)) != len(properties):
            raise ValueError("Task properties must not repeat")
        object.__setattr__(self, "properties", properties)
        if self.output_kind not in ("scalar", "orders", "channels"):
            raise ValueError("output_kind must be scalar, orders, or channels")
        if not _iso_second(self.start) or not _iso_second(self.end) or self.start >= self.end:
            raise ValueError("Task requires an ordered canonical ISO time range")
        if type(self.status) is not str or self.status not in STATUSES:
            raise ValueError("Task status is not registered")
        if type(self.other_label) is not str or not self.other_label or len(self.other_label) > 80:
            raise ValueError("other_label must be a nonempty short string")
        if "preserve_zero_orders" in properties and self.output_kind != "orders":
            raise ValueError("preserve_zero_orders requires orders output")
        if "merge_other_channels" in properties and self.output_kind != "channels":
            raise ValueError("merge_other_channels requires channels output")
        scalar_properties = {"exclude_end", "exclude_status", "include_start", "include_negative"}
        if self.output_kind != "scalar" and scalar_properties.intersection(properties):
            raise ValueError("Scalar delta properties require scalar output in this version")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["properties"] = list(self.properties)
        return result

    def public_input(self) -> dict[str, Any]:
        return {
            "task": self.to_dict(),
            "schema": {name: list(cols) for name, cols in TABLE_SCHEMA.items()},
        }


class _Rejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _reject(code: str, message: str) -> None:
    raise _Rejected(code, message)


def _only(node: exp.Expression, allowed: set[str]) -> None:
    extras = [
        key
        for key, value in node.args.items()
        if value not in (None, False, [], "") and key not in allowed
    ]
    if extras:
        _reject("UNSUPPORTED_SQL", f"Unsupported {node.key} options: {', '.join(extras)}")


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        _only(node, {"this"})
        node = node.this
    return node


def _binding(node: exp.Expression) -> tuple[str, str]:
    if not isinstance(node, exp.Table) or not isinstance(node.this, exp.Identifier):
        _reject("UNSUPPORTED_SOURCE", "Only registered direct tables are supported")
    _only(node, {"this", "alias"})
    name = node.name.casefold()
    if name not in TABLE_SCHEMA:
        _reject("UNKNOWN_TABLE", "Table is not registered")
    alias_node = node.args.get("alias")
    if alias_node is not None:
        if not isinstance(alias_node, exp.TableAlias) or not isinstance(
            alias_node.this, exp.Identifier
        ):
            _reject("UNSUPPORTED_ALIAS", "Only ordinary aliases are supported")
        _only(alias_node, {"this"})
    return (alias_node.name if alias_node else node.name).casefold(), name


def _column(node: exp.Expression, bindings: dict[str, str]) -> tuple[str, str]:
    if not isinstance(node, exp.Column) or not isinstance(node.this, exp.Identifier):
        _reject("UNSUPPORTED_EXPRESSION", "Expected a direct registered column")
    _only(node, {"this", "table"})
    name = node.name.casefold()
    if node.table:
        alias = node.table.casefold()
        if alias not in bindings or name not in TABLE_SCHEMA[bindings[alias]]:
            _reject("UNKNOWN_COLUMN", "Unknown qualified column")
        return bindings[alias], name
    matches = [table for table in bindings.values() if name in TABLE_SCHEMA[table]]
    if len(matches) != 1:
        _reject("AMBIGUOUS_COLUMN", "Unknown or ambiguous column")
    return matches[0], name


def _literal(node: exp.Expression) -> int | str | None:
    if isinstance(node, exp.Null):
        return None
    negative = isinstance(node, exp.Neg)
    if negative:
        _only(node, {"this"})
        node = node.this
    if not isinstance(node, exp.Literal):
        _reject("UNSUPPORTED_EXPRESSION", "Only literal predicate operands are supported")
    _only(node, {"this", "is_string"})
    if node.is_string:
        if negative or len(node.this) > 100:
            _reject("INVALID_LITERAL", "Invalid string literal")
        return node.this
    try:
        number = int(node.this)
    except (ValueError, TypeError):
        _reject("INVALID_LITERAL", "Numeric literals must be bounded integers")
    if str(number) != node.this or not _integer(-number if negative else number):
        _reject("INVALID_LITERAL", "Numeric literals must be bounded integers")
    return -number if negative else number


def _flatten(node: exp.Expression) -> list[exp.Expression]:
    node = _unparen(node)
    if isinstance(node, exp.And):
        _only(node, {"this", "expression"})
        return _flatten(node.this) + _flatten(node.expression)
    return [node]


_OP_NAMES = {
    exp.EQ: "eq",
    exp.NEQ: "neq",
    exp.GT: "gt",
    exp.GTE: "gte",
    exp.LT: "lt",
    exp.LTE: "lte",
    exp.Is: "is",
}
_REVERSED = {"eq": "eq", "neq": "neq", "gt": "lt", "gte": "lte", "lt": "gt", "lte": "gte"}


def _predicates(
    node: exp.Expression, bindings: dict[str, str], *, join: bool = False
) -> tuple[list[dict[str, Any]], int]:
    predicates: list[dict[str, Any]] = []
    key_equalities = 0
    for atom in _flatten(node):
        op = _OP_NAMES.get(type(atom))
        if op is None:
            _reject(
                "UNSUPPORTED_PREDICATE", "Only AND of direct comparisons and IS NULL is supported"
            )
        _only(atom, {"this", "expression"})
        left, right = atom.this, atom.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            pair = {_column(left, bindings), _column(right, bindings)}
            if (
                not join
                or op != "eq"
                or pair != {("orders", "order_id"), ("order_lines", "order_id")}
            ):
                _reject(
                    "UNREGISTERED_RELATION", "Only the registered order key equality is supported"
                )
            key_equalities += 1
            continue
        if not isinstance(left, exp.Column):
            if not isinstance(right, exp.Column) or op not in _REVERSED:
                _reject("UNSUPPORTED_PREDICATE", "Comparison must contain a direct column")
            left, right, op = right, left, _REVERSED[op]
        table, column = _column(left, bindings)
        value = _literal(right)
        if op == "is" and value is not None:
            _reject("UNSUPPORTED_PREDICATE", "Only IS NULL is supported")
        if op != "is" and value is None:
            _reject("INVALID_PREDICATE", "Use IS NULL rather than comparing NULL")
        if value is not None:
            text_column = column in {"channel", "status", "settled_at"}
            if text_column != (type(value) is str):
                _reject(
                    "PREDICATE_TYPE_MISMATCH", "Predicate literal type does not match its field"
                )
            if column == "settled_at" and not _iso_second(value):
                _reject("INVALID_TIME_LITERAL", "Date comparisons require canonical ISO seconds")
            if column == "status" and value not in STATUSES:
                _reject("INVALID_STATUS_LITERAL", "Unknown status literal")
        predicates.append({"table": table, "column": column, "op": op, "value": value})
    return predicates, key_equalities


def _expression(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Alias):
        _only(node, {"this", "alias"})
        if not isinstance(node.args.get("alias"), exp.Identifier):
            _reject("UNSUPPORTED_ALIAS", "Only ordinary output aliases are supported")
        return _unparen(node.this)
    return _unparen(node)


def _group_key(node: exp.Expression, bindings: dict[str, str]) -> tuple[str, str, str | None]:
    node = _unparen(node)
    if isinstance(node, exp.Coalesce):
        _only(node, {"this", "expressions"})
        if len(node.expressions) != 1 or _column(node.this, bindings) != ("orders", "channel"):
            _reject("UNSUPPORTED_GROUP_KEY", "Only COALESCE(orders.channel, label) is supported")
        label = _literal(node.expressions[0])
        if type(label) is not str or not label:
            _reject("UNSUPPORTED_GROUP_KEY", "Channel default must be a nonempty string")
        return "orders", "channel", label
    table, column = _column(node, bindings)
    if table != "orders" or column not in {"order_id", "channel"}:
        _reject("UNSUPPORTED_GROUP_KEY", "Group key must be order ID or channel")
    return table, column, None


def _aggregate(node: exp.Expression, bindings: dict[str, str]) -> bool:
    node = _expression(node)
    zero_default = False
    if isinstance(node, exp.Coalesce):
        _only(node, {"this", "expressions"})
        if len(node.expressions) != 1 or _literal(node.expressions[0]) != 0:
            _reject("UNSUPPORTED_AGGREGATE", "Only COALESCE(SUM(amount), 0) is supported")
        zero_default, node = True, node.this
    if not isinstance(node, exp.Sum):
        _reject("UNSUPPORTED_AGGREGATE", "Exactly one SUM of the line amount is required")
    _only(node, {"this"})
    if isinstance(node.this, exp.Distinct):
        _reject("DISTINCT_GRAIN_MISMATCH", "Amount DISTINCT does not preserve line grain")
    if _column(node.this, bindings) != ("order_lines", "amount"):
        _reject("METRIC_SOURCE_MISMATCH", "This experiment sums order_lines.amount")
    return zero_default


def _schema_implies_predicate(predicate: dict[str, Any]) -> bool:
    """Prove a numeric predicate true on every stored row, not padded NULLs.

    Bounds come from ProbeData validation. Such a comparison may be harmless
    on a driving table, or a legitimate match witness on the nullable side of
    LEFT JOIN. Its placement still matters; it is not globally true after JOIN.
    """
    field = (predicate["table"], predicate["column"])
    bounds = {
        ("orders", "order_id"): (1, _MAX_INT),
        ("order_lines", "order_id"): (1, _MAX_INT),
        ("order_lines", "line_id"): (1, _MAX_INT),
        ("orders", "total"): (-_MAX_INT, _MAX_INT),
        ("order_lines", "amount"): (-_MAX_INT, _MAX_INT),
    }
    value, operator = predicate["value"], predicate["op"]
    if field not in bounds or type(value) is not int:
        return False
    low, high = bounds[field]
    return {
        "eq": low == high == value,
        "neq": value < low or value > high,
        "gt": low > value,
        "gte": low >= value,
        "lt": high < value,
        "lte": high <= value,
    }.get(operator, False)


def _null_rejecting_on(predicate: dict[str, Any], table: str) -> bool:
    return (
        predicate["table"] == table
        and predicate["value"] is not None
        and predicate["op"] in {"eq", "neq", "gt", "gte", "lt", "lte"}
    )


def _semantics(task: ProbeTask, features: dict[str, Any]) -> None:
    if not task.properties:
        return
    predicates = features["predicates"]
    # The status and half-open range are confirmed facts on these development
    # tasks. Check obvious cases directly rather than claiming them as a new
    # capability of dynamic testing.
    status = [p for p in predicates if (p["table"], p["column"]) == ("order_lines", "status")]
    if not status:
        _reject("MISSING_STATUS_CONDITION", "Confirmed status condition is missing")
    if any(p["op"] != "eq" or p["value"] != task.status for p in status):
        _reject("STATUS_CONDITION_MISMATCH", "Status predicate contradicts the confirmed task")
    times = [p for p in predicates if (p["table"], p["column"]) == ("order_lines", "settled_at")]
    wanted = {("gte", task.start), ("lt", task.end)}
    actual = {(p["op"], p["value"]) for p in times}
    if not times:
        _reject("MISSING_TIME_CONDITION", "Confirmed time range is missing")
    if actual != wanted:
        _reject("TIME_BOUNDARY_MISMATCH", "Task requires start-inclusive, end-exclusive bounds")
    if "preserve_zero_orders" in task.properties:
        if features["join_type"] != "left" or features["first_table"] != "orders":
            _reject("ZERO_ORDER_COVERAGE", "All orders require orders LEFT JOIN order_lines")
        if any(
            p["table"] != "orders" or not _schema_implies_predicate(p)
            for p in features["where_predicates"]
        ):
            _reject("ZERO_ORDER_WHERE_FILTER", "WHERE filtering may remove required zero orders")
        if not features["zero_default"]:
            _reject("MISSING_ZERO_DEFAULT", "Zero-order output requires COALESCE(SUM(amount),0)")
    if "merge_other_channels" in task.properties:
        if features["group_signature"] != ("orders", "channel", task.other_label):
            _reject(
                "CHANNEL_NORMALIZATION",
                "NULL and the declared other channel must be grouped together",
            )
    if "include_negative" in task.properties and any(
        (p["table"], p["column"]) == ("order_lines", "amount") and not _schema_implies_predicate(p)
        for p in predicates
    ):
        _reject("AMOUNT_FILTER_MISMATCH", "A source amount filter may remove confirmed refunds")
    # These tasks confirm a status/time eligibility scope, not arbitrary
    # channel/ID/amount restrictions. Correct required predicates do not make
    # extra predicates legitimate: they can silently narrow the task universe.
    allowed_scope_fields = {("order_lines", "status"), ("order_lines", "settled_at")}
    if any(
        (p["table"], p["column"]) not in allowed_scope_fields and not _schema_implies_predicate(p)
        for p in predicates
    ):
        _reject(
            "UNDECLARED_SCOPE_FILTER",
            "An eligibility filter was not confirmed by the caller; preserve the full task population",
        )
    if features["join_type"] == "left" and features["first_table"] == "order_lines":
        # ON only determines the optional parent match. It does not discard
        # ineligible driving child rows, even when every FK is valid.
        where_facts = {
            (p["table"], p["column"], p["op"], p["value"]) for p in features["where_predicates"]
        }
        required = {
            ("order_lines", "status", "eq", task.status),
            ("order_lines", "settled_at", "gte", task.start),
            ("order_lines", "settled_at", "lt", task.end),
        }
        matched_parent = any(_null_rejecting_on(p, "orders") for p in features["where_predicates"])
        if not required.issubset(where_facts) and not matched_parent:
            _reject(
                "FILTER_PLACEMENT_MISMATCH",
                "LEFT JOIN preserves driving line rows; their confirmed eligibility filters must apply in WHERE",
            )
    if (
        task.output_kind == "channels"
        and features["join_type"] == "left"
        and features["first_table"] == "orders"
        and not any(_null_rejecting_on(p, "order_lines") for p in features["where_predicates"])
    ):
        # A legitimate zero-sum group has eligible line rows. Parent-only
        # channels are not groups in this task, regardless of COALESCE(0).
        # Keep equivalent LEFT+WHERE forms: confirmed child comparisons are
        # null-rejecting, so they remove unmatched parents without dropping
        # genuine zero-amount lines.
        _reject(
            "UNMATCHED_GROUP_COVERAGE",
            "Channel output requires an eligible line; unmatched parents must not create zero-valued groups",
        )


def _parse_select(sql: str) -> exp.Select:
    """Parse exactly one SELECT within the closed execution grammar."""
    if type(sql) is not str or not sql.strip() or len(sql) > _MAX_SQL:
        _reject("INVALID_SQL", "Expected one nonempty bounded SQL string")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError) as exc:
        raise _Rejected("SQL_PARSE_ERROR", "SQL could not be parsed") from exc
    if len(statements) != 1 or statements[0] is None:
        _reject("MULTIPLE_STATEMENTS", "Exactly one SELECT is supported")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        _reject("SQL_NOT_READ_ONLY", "Only a supported SELECT is allowed")
    _only(tree, {"expressions", "from_", "joins", "where", "group"})
    return tree


def _analyse_relations(tree: exp.Select) -> tuple[dict[str, str], str, str, list[dict[str, Any]]]:
    """Resolve aliases and validate the single registered parent/child join."""
    source = tree.args.get("from_")
    if not isinstance(source, exp.From):
        _reject("MISSING_SOURCE", "FROM is required")
    _only(source, {"this"})
    alias, first = _binding(source.this)
    bindings = {alias: first}
    joins = tree.args.get("joins") or []
    if len(joins) > 1:
        _reject("UNSUPPORTED_JOIN", "At most one registered join is supported")
    join_type = "none"
    on_predicates: list[dict[str, Any]] = []
    if joins:
        join = joins[0]
        _only(join, {"this", "on", "kind", "side"})
        side, kind = (join.side or "").upper(), (join.kind or "").upper()
        if (side, kind) not in {("", ""), ("", "INNER"), ("LEFT", ""), ("LEFT", "OUTER")}:
            _reject("UNSUPPORTED_JOIN", "Only INNER or LEFT joins are supported")
        second_alias, second = _binding(join.this)
        if second_alias in bindings or second == first:
            _reject("UNSUPPORTED_JOIN", "Repeated aliases and self joins are unsupported")
        bindings[second_alias] = second
        if join.args.get("on") is None:
            _reject("UNREGISTERED_RELATION", "A registered ON key is required")
        on_predicates, key_count = _predicates(join.args["on"], bindings, join=True)
        if key_count != 1:
            _reject("UNREGISTERED_RELATION", "ON must contain exactly one registered key equality")
        join_type = "left" if side == "LEFT" else "inner"
    if "order_lines" not in bindings.values():
        _reject("METRIC_SOURCE_MISMATCH", "The line amount source is required")
    return bindings, first, join_type, on_predicates


def _analyse_where(tree: exp.Select, bindings: dict[str, str]) -> list[dict[str, Any]]:
    """Collect supported WHERE predicates after table names are resolved."""
    where = tree.args.get("where")
    where_predicates: list[dict[str, Any]] = []
    if where is not None:
        if not isinstance(where, exp.Where):
            _reject("UNSUPPORTED_PREDICATE", "Invalid WHERE")
        _only(where, {"this"})
        where_predicates, _ = _predicates(where.this, bindings)
    return where_predicates


def _analyse_projection(
    tree: exp.Select,
    bindings: dict[str, str],
    task: ProbeTask,
) -> tuple[bool, tuple | None]:
    """Match the projected measure and grouping key to the confirmed task."""
    group = tree.args.get("group")
    group_signature = None
    if task.output_kind == "scalar":
        if group is not None or len(tree.expressions) != 1:
            _reject("OUTPUT_SHAPE_MISMATCH", "Scalar tasks require one aggregate without grouping")
        zero_default = _aggregate(tree.expressions[0], bindings)
    else:
        if not isinstance(group, exp.Group) or len(tree.expressions) != 2:
            _reject("OUTPUT_SHAPE_MISMATCH", "Grouped tasks require key, SUM, and one GROUP BY key")
        _only(group, {"expressions"})
        if len(group.expressions) != 1:
            _reject("UNSUPPORTED_GROUP", "Exactly one group key is supported")
        projected = _group_key(_expression(tree.expressions[0]), bindings)
        grouped = _group_key(group.expressions[0], bindings)
        if projected != grouped:
            _reject("GROUP_PROJECTION_MISMATCH", "Projected and actual grouping expressions differ")
        wanted_column = "order_id" if task.output_kind == "orders" else "channel"
        if projected[1] != wanted_column:
            _reject("OUTPUT_SHAPE_MISMATCH", "Output key does not match the confirmed task")
        group_signature = grouped
        zero_default = _aggregate(tree.expressions[1], bindings)
    return zero_default, group_signature


def _analyse(sql: str, task: ProbeTask, *, static_semantics: bool) -> tuple[str, dict[str, Any]]:
    """Read SQL in stages: syntax, relations, filters, output, task semantics."""
    if not isinstance(task, ProbeTask):
        _reject("INVALID_TASK", "Expected a confirmed ProbeTask")
    tree = _parse_select(sql)
    bindings, first, join_type, on_predicates = _analyse_relations(tree)
    where_predicates = _analyse_where(tree, bindings)
    zero_default, group_signature = _analyse_projection(tree, bindings, task)
    where = tree.args.get("where")
    features = {
        "tables": list(bindings.values()),
        "first_table": first,
        "join_type": join_type,
        "has_where": where is not None,
        "predicates": on_predicates + where_predicates,
        "where_predicates": where_predicates,
        "group_signature": group_signature,
        "zero_default": zero_default,
        "output_kind": task.output_kind,
        "static_semantics": static_semantics,
    }
    if static_semantics:
        _semantics(task, features)
    return tree.sql(dialect="sqlite"), features


def static_check(sql: str, task: ProbeTask, *, static_semantics: bool = True) -> dict[str, Any]:
    """Closed grammar plus, by default, obvious confirmed-task checks.

    ``static_semantics=False`` is an explicitly weaker *structural* comparator,
    not a production recommendation and never evidence against the strong arm.
    """
    try:
        _, features = _analyse(sql, task, static_semantics=static_semantics)
        return {"accepted": True, "code": None, "features": features}
    except _Rejected as exc:
        return {"accepted": False, "code": exc.code, "message": str(exc), "features": {}}
    except (ValueError, TypeError, AttributeError, RecursionError):
        return {
            "accepted": False,
            "code": "INVALID_SQL_STRUCTURE",
            "message": "Unsupported SQL structure",
            "features": {},
        }


def execute_sql(
    sql: str, data: ProbeData, task: ProbeTask, *, static_semantics: bool = True
) -> dict[str, Any]:
    """Execute one accepted candidate on an isolated, legal in-memory fixture."""
    result: dict[str, Any] = {"status": "blocked", "code": None, "columns": [], "rows": []}
    if not isinstance(data, ProbeData):
        result["code"] = "INVALID_DATA"
        return result
    checked = static_check(sql, task, static_semantics=static_semantics)
    if not checked["accepted"]:
        result.update(code=checked["code"], message=checked.get("message", ""))
        return result
    canonical, _ = _analyse(sql, task, static_semantics=static_semantics)
    connection = sqlite3.connect(":memory:")
    query_budget_exceeded = False
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, total INTEGER NOT NULL, channel TEXT)"
        )
        connection.execute(
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(order_id), amount INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('settled','pending','cancelled')), settled_at TEXT NOT NULL)"
        )
        connection.executemany("INSERT INTO orders VALUES (?,?,?)", data.orders)
        connection.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?)", data.lines)
        connection.commit()
        connection.execute("PRAGMA query_only=ON")
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, _MAX_SQL)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        deadline = time.monotonic() + 1.0
        callbacks = 0

        def progress() -> int:
            nonlocal callbacks, query_budget_exceeded
            callbacks += 1
            query_budget_exceeded = callbacks > 100 or time.monotonic() > deadline
            return int(query_budget_exceeded)

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and table in TABLE_SCHEMA:
                # Some SQLite versions report a registered-table row scan
                # (not a system-table read) with an empty column and no DB.
                if (database == "main" and column in TABLE_SCHEMA[table]) or (
                    database in {"main", None} and column == ""
                ):
                    return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (column or "").casefold() in {
                "sum",
                "coalesce",
            }:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        connection.set_progress_handler(progress, 1_000)
        cursor = connection.execute(canonical)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchmany(_MAX_ROWS + 1)
        if len(rows) > _MAX_ROWS:
            result.update(status="error", code="OUTPUT_BUDGET_EXCEEDED")
            return result
        for row in rows:
            for value in row:
                if type(value) not in (int, float, str, type(None)):
                    raise ValueError("Unsupported result scalar")
                if type(value) is float and not math.isfinite(value):
                    raise ValueError("Nonfinite query result")
        result.update(status="ok", code=None, columns=columns, rows=[list(row) for row in rows])
    except (sqlite3.Error, ValueError, OverflowError):
        code = "QUERY_BUDGET_EXCEEDED" if query_budget_exceeded else "QUERY_EXECUTION_ERROR"
        result.update(status="error", code=code, columns=[], rows=[])
    finally:
        connection.close()
    return result
