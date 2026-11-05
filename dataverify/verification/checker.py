"""Fail-closed verification for a small, explicitly supported SQL subset.

AST validation is a semantic boundary, not a SQL security sandbox. Execution also
uses a read-only SQLite URI, query_only, an authorizer, and a progress budget.
The experiment supports whole-table additive metrics only; it does not claim to
understand arbitrary SQL or infer unknown business semantics.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp
from sqlglot.errors import ParseError, TokenError

from .contracts import METRIC_CATALOG, RELATIONS, TABLE_COLUMNS, MetricContract


class _UnsupportedSQL(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _MetricMismatch(_UnsupportedSQL):
    pass


@dataclass(frozen=True)
class _QueryShape:
    metric: MetricContract
    source_alias: str
    from_sql: str
    distinct: bool


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _only_args(node: exp.Expression, allowed: set[str]) -> None:
    """Reject even syntactically valid features outside the declared grammar."""
    extra = [
        key
        for key, value in node.args.items()
        if value not in (None, False, [], "") and key not in allowed
    ]
    if extra:
        raise _UnsupportedSQL(
            "UNSUPPORTED_SQL", f"不支持 {node.key} 的语法选项：{', '.join(extra)}。"
        )


def _table_binding(table: exp.Expression) -> tuple[str, str]:
    if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier):
        raise _UnsupportedSQL(
            "UNSUPPORTED_SOURCE", "只允许直接读取已登记表，不支持表函数或子查询。"
        )
    _only_args(table, {"this", "alias"})
    name = table.name.casefold()
    if name not in TABLE_COLUMNS:
        raise _UnsupportedSQL("UNKNOWN_TABLE", f"表 {name!r} 不在允许目录内。")
    alias = table.args.get("alias")
    if alias is not None:
        if not isinstance(alias, exp.TableAlias) or not isinstance(alias.this, exp.Identifier):
            raise _UnsupportedSQL("UNSUPPORTED_ALIAS", "只支持普通表别名。")
        _only_args(alias, {"this"})
    return (alias.name if alias is not None else table.name).casefold(), name


def _resolve_column(column: exp.Expression, bindings: dict[str, str]) -> tuple[str, str, str]:
    if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier):
        raise _UnsupportedSQL(
            "UNSUPPORTED_EXPRESSION", "SUM 和 JOIN 只能引用直接字段，不能使用表达式或函数。"
        )
    _only_args(column, {"this", "table"})
    name = column.name.casefold()
    if column.table:
        alias = column.table.casefold()
        if alias not in bindings or name not in TABLE_COLUMNS[bindings[alias]]:
            raise _UnsupportedSQL(
                "UNKNOWN_COLUMN", f"字段 {column.sql()!r} 未登记或引用了未知别名。"
            )
    else:
        matches = [alias for alias, table in bindings.items() if name in TABLE_COLUMNS[table]]
        if len(matches) != 1:
            raise _UnsupportedSQL(
                "AMBIGUOUS_COLUMN", f"字段 {name!r} 未知或不明确，请使用表别名限定。"
            )
        alias = matches[0]
    return alias, bindings[alias], name


def _inspect_sql(sql: str, metric: MetricContract) -> _QueryShape:
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 16_000:
        raise _UnsupportedSQL("INVALID_SQL_INPUT", "SQL 必须是非空字符串，且不超过 16000 字符。")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError) as exc:
        raise _UnsupportedSQL("SQL_PARSE_ERROR", f"SQL 解析失败：{exc}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise _UnsupportedSQL("MULTIPLE_STATEMENTS", "只允许一条 SELECT。")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise _UnsupportedSQL("SQL_NOT_READ_ONLY", "只允许受支持的 SELECT，禁止写入和管理语句。")
    _only_args(tree, {"expressions", "from_", "joins"})
    if len(tree.expressions) != 1:
        raise _UnsupportedSQL("UNSUPPORTED_PROJECTION", "只允许输出一个 SUM(column)。")
    aggregate = tree.expressions[0]
    if isinstance(aggregate, exp.Alias):
        _only_args(aggregate, {"this", "alias"})
        if not isinstance(aggregate.args.get("alias"), exp.Identifier):
            raise _UnsupportedSQL("UNSUPPORTED_ALIAS", "只支持普通输出别名。")
        aggregate = aggregate.this
    if not isinstance(aggregate, exp.Sum):
        raise _UnsupportedSQL("UNSUPPORTED_AGGREGATE", "首版只支持 SUM(column)，不支持任意函数。")
    _only_args(aggregate, {"this"})
    from_node = tree.args.get("from_")
    if not isinstance(from_node, exp.From):
        raise _UnsupportedSQL("MISSING_SOURCE", "缺少 FROM 数据源。")
    _only_args(from_node, {"this"})
    first_alias, first_table = _table_binding(from_node.this)
    bindings = {first_alias: first_table}
    joins = tree.args.get("joins") or []
    if len(joins) > 1:
        raise _UnsupportedSQL("UNSUPPORTED_JOIN", "首版最多支持一个已登记 INNER JOIN。")
    if joins:
        join = joins[0]
        _only_args(join, {"this", "on", "kind"})
        if (join.args.get("kind") or "").upper() not in ("", "INNER"):
            raise _UnsupportedSQL("UNSUPPORTED_JOIN", "只支持 INNER JOIN。")
        second_alias, second_table = _table_binding(join.this)
        if second_alias in bindings or second_table == first_table:
            raise _UnsupportedSQL("UNSUPPORTED_JOIN", "首版不支持重复别名或自连接。")
        bindings[second_alias] = second_table
        condition = join.args.get("on")
        if not isinstance(condition, exp.EQ):
            raise _UnsupportedSQL(
                "UNREGISTERED_RELATION", "JOIN 必须使用已登记字段的单个等值条件。"
            )
        _only_args(condition, {"this", "expression"})
        _, left_table, left_column = _resolve_column(condition.this, bindings)
        _, right_table, right_column = _resolve_column(condition.expression, bindings)
        actual = {(left_table, left_column), (right_table, right_column)}
        registered = [
            {
                (relation.left_table, relation.left_column),
                (relation.right_table, relation.right_column),
            }
            for relation in RELATIONS
        ]
        if len(actual) != 2 or actual not in registered:
            raise _UnsupportedSQL("UNREGISTERED_RELATION", "JOIN 条件与已登记表关系不匹配。")
    operand = aggregate.this
    distinct = isinstance(operand, exp.Distinct)
    if distinct:
        _only_args(operand, {"expressions"})
        if len(operand.expressions) != 1:
            raise _UnsupportedSQL("UNSUPPORTED_EXPRESSION", "只支持单字段 DISTINCT 的诊断。")
        operand = operand.expressions[0]
    source_alias, source_table, value_column = _resolve_column(operand, bindings)
    if source_table != metric.source_table or value_column != metric.value_column:
        raise _MetricMismatch(
            "METRIC_SOURCE_MISMATCH",
            f"指标 {metric.metric_id} 要求 SUM({metric.source_table}.{metric.value_column})，"
            f"当前读取 {source_table}.{value_column}。",
        )
    # Re-render only validated AST fragments for the independent evidence probes.
    from_sql = "FROM " + from_node.this.sql(dialect="sqlite")
    if joins:
        from_sql += " " + joins[0].sql(dialect="sqlite")
    return _QueryShape(metric, source_alias, from_sql, distinct)


class CheckedQueryExecutor:
    """Execute and verify whole-source additive metrics against declared grain."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @staticmethod
    def _authorize(
        action: int, arg1: str | None, arg2: str | None, database: str | None, trigger: str | None
    ) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            # SQLite reports COUNT(*) as a table read with an empty column;
            # some versions leave the database field unset for that event.
            if arg1 in TABLE_COLUMNS and (
                (database == "main" and arg2 in TABLE_COLUMNS[arg1])
                or (database in {"main", None} and not arg2)
            ):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").casefold() in {
            "sum",
            "count",
            "typeof",
        }:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_TRANSACTION and (arg1 or "").upper() in {"BEGIN", "ROLLBACK"}:
            return sqlite3.SQLITE_OK
        # PRAGMA, ATTACH, writes, schema/system reads and unknown operations fail closed.
        return sqlite3.SQLITE_DENY

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(self._authorize)
        deadline = time.monotonic() + 2.0
        callbacks = 0

        def progress() -> int:
            nonlocal callbacks
            callbacks += 1
            return int(callbacks > 250 or time.monotonic() > deadline)

        connection.set_progress_handler(progress, 1_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        return connection

    def execute(self, sql: str, metric_id: str = "order_revenue") -> dict:
        result = {
            "sql": sql,
            "metric_id": metric_id,
            "execution_status": "blocked",
            "verification_status": "unsupported",
            "value": None,
            "accepted_value": None,
            "checks": [],
            "error_code": None,
            "feedback": "",
            "evidence": [],
        }
        if not isinstance(metric_id, str):
            result.update(
                error_code="INVALID_METRIC_INPUT", feedback="指标ID必须是已登记的字符串。"
            )
            return result
        metric = METRIC_CATALOG.get(metric_id)
        if metric is None:
            result.update(
                error_code="UNKNOWN_METRIC",
                feedback="指标未登记；不能根据模型建议自动推断指标口径。",
            )
            return result
        try:
            shape = _inspect_sql(sql, metric)
        except _UnsupportedSQL as exc:
            result.update(error_code=exc.code, feedback=str(exc))
            if isinstance(exc, _MetricMismatch):
                result["verification_status"] = "rejected"
            return result
        result["checks"].append({"name": "supported_ast_and_contract", "passed": True})
        connection = None
        try:
            connection = self._connect()
            # Keep aggregate and evidence probes on the same SQLite snapshot.
            # A concurrent commit is also detected by the runner's source fingerprint.
            connection.execute("BEGIN")
            row = connection.execute(sql).fetchone()
            value = None if row is None or row[0] is None else float(row[0])
            if value is not None and not math.isfinite(value):
                raise ValueError("聚合结果不是有限数值。")
            result.update(execution_status="executed", value=value)
            self._verify(connection, shape, result)
        except (sqlite3.Error, ValueError, OverflowError) as exc:
            result.update(
                execution_status="error",
                verification_status="unsupported",
                accepted_value=None,
                error_code="QUERY_EXECUTION_ERROR",
                feedback=f"执行或证据查询失败，未接受结果：{exc}",
            )
        finally:
            if connection is not None:
                connection.close()
        return result

    @staticmethod
    def _verify(connection: sqlite3.Connection, shape: _QueryShape, result: dict) -> None:
        metric = shape.metric
        table = _quote(metric.source_table)
        grain_name = _quote(metric.grain_column)
        source_grain = _quote(shape.source_alias) + "." + grain_name
        source_value = _quote(shape.source_alias) + "." + _quote(metric.value_column)
        measure = _quote(metric.value_column)
        invalid_measure = (
            f"{measure} IS NULL OR typeof({measure}) NOT IN ('integer','real') "
            f"OR {measure} > ? OR {measure} < ?"
        )
        invalid_values = [
            dict(row)
            for row in connection.execute(
                f"SELECT {grain_name} AS grain_value, {measure} AS measure_value "
                f"FROM {table} WHERE {invalid_measure} LIMIT 3",
                (
                    float.fromhex("0x1.fffffffffffffp+1023"),
                    -float.fromhex("0x1.fffffffffffffp+1023"),
                ),
            )
        ]
        for example in invalid_values:
            value = example["measure_value"]
            if isinstance(value, float) and not math.isfinite(value):
                example["measure_value"] = repr(value)
        result["checks"].append(
            {"name": "source_measure_numeric_finite_non_null", "passed": not invalid_values}
        )
        if invalid_values:
            result["evidence"].append({"kind": "invalid_measure_value", "examples": invalid_values})
            result.update(
                verification_status="rejected",
                error_code="INVALID_MEASURE_VALUE",
                feedback="指标金额包含空值、非数值或非有限值；SQLite SUM 可能忽略损坏文本，不能接受当前结果。",
            )
            return
        base = connection.execute(
            f"SELECT COUNT(*) AS n, COUNT({grain_name}) AS non_null, COUNT(DISTINCT {grain_name}) AS unique_n FROM {table}"
        ).fetchone()
        grain_valid = base["n"] == base["non_null"] == base["unique_n"]
        result["checks"].append(
            {
                "name": "source_grain_unique_non_null",
                "passed": grain_valid,
                "source_rows": base["n"],
            }
        )
        if not grain_valid:
            result.update(
                verification_status="rejected",
                error_code="INVALID_SOURCE_GRAIN",
                feedback="源表粒度字段不唯一或包含 NULL，当前契约不能成立。",
            )
            return
        duplicates = [
            dict(row)
            for row in connection.execute(
                f"SELECT {source_grain} AS grain_value, COUNT(*) AS multiplicity {shape.from_sql} "
                f"GROUP BY {source_grain} HAVING COUNT(*) > 1 ORDER BY multiplicity DESC, grain_value LIMIT 3"
            )
        ]
        result["checks"].append({"name": "one_row_per_metric_grain", "passed": not duplicates})
        if duplicates:
            result["evidence"].append(
                {
                    "kind": "duplicate_grain",
                    "source_table": metric.source_table,
                    "grain_column": metric.grain_column,
                    "examples": duplicates,
                    "max_multiplicity": max(row["multiplicity"] for row in duplicates),
                }
            )
        observed = connection.execute(
            f"SELECT COUNT(DISTINCT {source_grain}) {shape.from_sql}"
        ).fetchone()[0]
        coverage_ok = observed == base["n"]
        result["checks"].append(
            {
                "name": "all_source_grains_covered",
                "passed": coverage_ok,
                "source_grains": base["n"],
                "query_grains": observed,
            }
        )
        if not coverage_ok:
            # Counts establish incomplete coverage; concrete source keys make
            # the feedback useful to the planner and to a human reviewing it.
            # source_grain is non-null because the source contract was checked
            # above, so NOT IN cannot acquire SQL's NULL/UNKNOWN ambiguity.
            missing = [
                dict(row)
                for row in connection.execute(
                    f"SELECT {grain_name} AS grain_value FROM {table} "
                    f"WHERE {grain_name} NOT IN (SELECT {source_grain} {shape.from_sql}) "
                    f"ORDER BY {grain_name} LIMIT 3"
                )
            ]
            result["evidence"].append(
                {
                    "kind": "missing_grain_coverage",
                    "source_table": metric.source_table,
                    "grain_column": metric.grain_column,
                    "source_grains": base["n"],
                    "query_grains": observed,
                    "examples": missing,
                }
            )
        result["checks"].append(
            {"name": "additive_metric_without_value_distinct", "passed": not shape.distinct}
        )
        if shape.distinct:
            collisions = [
                dict(row)
                for row in connection.execute(
                    f"SELECT {source_value} AS measure_value, COUNT(DISTINCT {source_grain}) AS distinct_grain_count "
                    f"{shape.from_sql} GROUP BY {source_value} HAVING COUNT(DISTINCT {source_grain}) > 1 LIMIT 3"
                )
            ]
            result["evidence"].append(
                {"kind": "value_collision_across_grains", "examples": collisions}
            )
            result.update(
                verification_status="rejected",
                error_code="VALUE_DISTINCT_NOT_GRAIN_DEDUP",
                feedback="SUM(DISTINCT 金额)按金额去重，不能代替按业务粒度去重；不同订单可以金额相同。"
                f"契约要求每个 {metric.grain_column} 累计一次，请从指标源表聚合。",
            )
        elif duplicates:
            example = duplicates[0]
            result.update(
                verification_status="rejected",
                error_code="GRAIN_MULTIPLICATION",
                feedback=f"关联后重复累计：{metric.grain_column}={example['grain_value']} 出现 {example['multiplicity']} 次。"
                f"请按指标粒度每个 {metric.grain_column} 累计一次；没有明细筛选需求时直接聚合 {metric.source_table}。",
            )
        elif not coverage_ok:
            result.update(
                verification_status="rejected",
                error_code="MISSING_GRAIN",
                feedback="JOIN 丢失了部分源表粒度，违反包含所有源记录的指标口径。",
            )
        elif result["value"] is None:
            result.update(
                verification_status="rejected",
                error_code="EMPTY_RESULT",
                feedback="聚合结果为空，未接受数值。",
            )
        else:
            result.update(
                verification_status="verified",
                accepted_value=result["value"],
                feedback=f"已验证：字段符合 {metric.metric_id} 契约，源粒度唯一、无重复且完整覆盖。",
            )
