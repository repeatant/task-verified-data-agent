"""Execute generated checks; distinguish observed disagreement from truth.

The legacy verifier stays unchanged. This opt-in runtime extends its SQL scope
to read-only SELECTs (including grouped subqueries). It does not reuse the six
property IDs as an allowlist for business expectations. SQLite and an explicit
reference-expression interpreter execute independent calculation paths.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import expressions as exp

from dataverify.runtime.reporting import digest_json
from dataverify.verification.probe_runtime import TABLE_SCHEMA, ProbeData, ProbeTask
from dataverify.verification.reference_expression import calculate, parse_expression
from dataverify.verification.relational_data import (
    RelationalData,
    SQLiteSource,
    normalize_keys,
    normalize_schema,
    normalize_types,
)


@dataclass(frozen=True)
class DiscoveryTask:
    task_id: str
    question: str
    definitions: str = "Amounts are integer minor units. orders.total need not equal line amounts."
    legacy_task: ProbeTask | None = None
    schema: dict | None = None
    column_types: dict | None = None
    data_constraints: str | None = None
    primary_keys: dict | None = None
    foreign_keys: tuple | list = ()
    allow_source_only_plan: bool = False

    def __post_init__(self):
        if type(self.allow_source_only_plan) is not bool or (
            self.allow_source_only_plan and self.schema is None
        ):
            raise ValueError("INVALID_SOURCE_ONLY_PLAN_POLICY")
        for value, maximum in ((self.task_id, 80), (self.question, 6000), (self.definitions, 4000)):
            if type(value) is not str or not 1 <= len(value) <= maximum:
                raise ValueError("INVALID_DISCOVERY_TASK")
        if self.legacy_task is not None and (
            self.legacy_task.task_id != self.task_id or self.legacy_task.question != self.question
        ):
            raise ValueError("LEGACY_TASK_BINDING_MISMATCH")
        if self.schema is not None:
            schema = normalize_schema(self.schema)
            object.__setattr__(self, "schema", schema)
            object.__setattr__(self, "column_types", normalize_types(schema, self.column_types))
            keys, links = normalize_keys(schema, self.primary_keys, self.foreign_keys)
            object.__setattr__(self, "primary_keys", keys)
            object.__setattr__(self, "foreign_keys", links)
        if self.data_constraints is not None and (
            type(self.data_constraints) is not str or len(self.data_constraints) > 8000
        ):
            raise ValueError("INVALID_DATA_CONSTRAINTS")

    def public_input(self) -> dict:
        if self.schema is not None:
            return {
                "task_id": self.task_id,
                "question": self.question,
                "definitions": self.definitions,
                "schema": {table: list(columns) for table, columns in self.schema.items()},
                "column_types": {table: list(types) for table, types in self.column_types.items()},
                "primary_keys": {
                    table: list(columns) for table, columns in self.primary_keys.items()
                },
                "foreign_keys": list(self.foreign_keys),
                "allow_source_only_plan": self.allow_source_only_plan,
                "data_constraints": self.data_constraints
                or "Fixtures must respect column types, declared primary/foreign keys and nulls. "
                "Provided source rows may be a bounded preview, not the full database.",
            }
        return {
            "task_id": self.task_id,
            "question": self.question,
            "definitions": self.definitions,
            "schema": {name: list(columns) for name, columns in TABLE_SCHEMA.items()},
            "data_constraints": "Unique positive primary keys; valid order_id foreign keys; "
            "status in settled/pending/cancelled; timestamps YYYY-MM-DDTHH:MM:SS; "
            "channel string or null; negative amounts allowed.",
        }


def fixture_data(
    value: dict, schema=None, column_types=None, primary_keys=None, foreign_keys=()
) -> ProbeData | RelationalData:
    if isinstance(value, dict) and {"schema", "tables"} <= set(value):
        allowed = {"schema", "tables", "column_types", "primary_keys", "foreign_keys"}
        if not set(value) <= allowed:
            raise ValueError("RELATIONAL_FIXTURE_FIELDS")
        if schema is not None and normalize_schema(value["schema"]) != normalize_schema(schema):
            raise ValueError("RELATIONAL_SCHEMA_MISMATCH")
        # A model cannot weaken trusted task metadata through its envelope.
        data = RelationalData(
            schema if schema is not None else value["schema"],
            value["tables"],
            column_types if schema is not None else value.get("column_types"),
            primary_keys if schema is not None else value.get("primary_keys"),
            foreign_keys if schema is not None else value.get("foreign_keys", ()),
        )
    elif schema is not None:
        data = RelationalData(schema, value, column_types, primary_keys, foreign_keys)
    else:
        data = None
    if data is not None:
        if (
            any(len(rows) > 50 for rows in data.tables.values())
            or sum(map(len, data.tables.values())) > 500
        ):
            raise ValueError("FIXTURE_ROW_LIMIT")
        return data
    if not isinstance(value, dict) or set(value) not in (
        {"orders", "lines"},
        {"orders", "order_lines"},
    ):
        raise ValueError("FIXTURE_FIELDS")
    if any(not isinstance(value[k], list) or len(value[k]) > 50 for k in value):
        raise ValueError("FIXTURE_ROW_LIMIT")
    converted = {}
    for table, key in (
        ("orders", "orders"),
        ("order_lines", "lines" if "lines" in value else "order_lines"),
    ):
        rows = []
        for row in value[key]:
            if isinstance(row, dict):
                if set(row) != set(TABLE_SCHEMA[table]):
                    raise ValueError("FIXTURE_ROW_FIELDS")
                row = [row[column] for column in TABLE_SCHEMA[table]]
            if not isinstance(row, (list, tuple)):
                raise ValueError("FIXTURE_ROW_TYPE")
            rows.append(row)
        converted[table] = rows
    return ProbeData(converted["orders"], converted["order_lines"])


def named_tables(data: ProbeData | RelationalData) -> dict:
    if isinstance(data, RelationalData):
        return data.named_tables()
    if isinstance(data, SQLiteSource):
        raise ValueError("FULL_DATABASE_REFERENCE_NOT_SUPPORTED")
    return {
        table: [dict(zip(TABLE_SCHEMA[table], row)) for row in rows]
        for table, rows in (("orders", data.orders), ("order_lines", data.lines))
    }


def validate_fixtures(
    fixtures: list, schema=None, column_types=None, primary_keys=None, foreign_keys=()
) -> list[dict]:
    if not isinstance(fixtures, list) or not 1 <= len(fixtures) <= 3:
        raise ValueError("EXPECTED_ONE_TO_THREE_FIXTURES")
    normalized = []
    for fixture in fixtures:
        if not isinstance(fixture, dict) or set(fixture) != {"name", "data"}:
            raise ValueError("FIXTURE_FIELDS")
        if type(fixture["name"]) is not str or not 1 <= len(fixture["name"]) <= 120:
            raise ValueError("FIXTURE_NAME")
        data = fixture_data(fixture["data"], schema, column_types, primary_keys, foreign_keys)
        normalized.append(
            {**fixture, "data": data.to_dict()} if isinstance(data, RelationalData) else fixture
        )
    return normalized


def _source_patch(value: dict, task: DiscoveryTask, source) -> dict:
    """Expand bounded cell edits against trusted rows; no insertion or schema edits."""
    if task.schema is None or not isinstance(source, RelationalData):
        raise ValueError("SOURCE_PATCH_REQUIRES_GENERIC_BOUND_SOURCE")
    if task.schema != source.schema:
        raise ValueError("SOURCE_PATCH_SCHEMA_MISMATCH")
    if set(value) != {"base", "changes"} or value["base"] != "bound_source":
        raise ValueError("SOURCE_PATCH_FIELDS")
    changes = value["changes"]
    if not isinstance(changes, list) or not 1 <= len(changes) <= 12:
        raise ValueError("SOURCE_PATCH_CHANGE_LIMIT")
    expanded = source.to_dict()  # Fresh lists; the immutable bound source stays unchanged.
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"table", "row", "values"}:
            raise ValueError("SOURCE_PATCH_CHANGE_FIELDS")
        table, row, values = change["table"], change["row"], change["values"]
        if type(table) is not str or table not in task.schema:
            raise ValueError("SOURCE_PATCH_TABLE")
        if type(row) is not int or not 0 <= row < len(expanded["tables"][table]):
            raise ValueError("SOURCE_PATCH_ROW")
        if not isinstance(values, dict) or not values or not set(values) <= set(task.schema[table]):
            raise ValueError("SOURCE_PATCH_COLUMNS")
        for column, scalar in values.items():
            expanded["tables"][table][row][task.schema[table].index(column)] = scalar
    # The normal fixture validator applies scalar/type limits and the task's
    # original PK/FK constraints after all edits, including coordinated key edits.
    return expanded


def validate_model_fixtures(
    fixtures: list, task: DiscoveryTask, source: RelationalData | None = None
) -> list[dict]:
    """Bind untrusted model rows to the task, never to model-authored metadata.

    Internal replay accepts normalized generic envelopes. A legacy task still
    has the fixed ProbeData schema, however, and must not gain a new schema
    simply because model JSON contains a generic envelope.
    """
    if not isinstance(fixtures, list) or not 1 <= len(fixtures) <= 3:
        raise ValueError("EXPECTED_ONE_TO_THREE_FIXTURES")
    if task.schema is None:
        for fixture in fixtures:
            if isinstance(fixture, dict) and isinstance(fixture.get("data"), dict):
                if set(fixture["data"]) not in ({"orders", "lines"}, {"orders", "order_lines"}):
                    raise ValueError("MODEL_FIXTURE_TASK_SCHEMA_MISMATCH")
    expanded = []
    for fixture in fixtures:
        if (
            isinstance(fixture, dict)
            and isinstance(fixture.get("data"), dict)
            and isinstance(fixture["data"].get("base"), str)
        ):
            expanded.append({**fixture, "data": _source_patch(fixture["data"], task, source)})
        else:
            expanded.append(fixture)
    return validate_fixtures(
        expanded, task.schema, task.column_types, task.primary_keys, task.foreign_keys
    )


def validate_plan(plan: dict, task: DiscoveryTask) -> dict:
    if set(plan) != {"action", "basis", "assumptions", "reference", "fixtures"}:
        raise ValueError("PLAN_FIELDS")
    if plan["action"] != "plan" or not isinstance(plan["basis"], list) or not plan["basis"]:
        raise ValueError("PLAN_BASIS_REQUIRED")
    if len(plan["basis"]) > 8:
        raise ValueError("PLAN_BASIS_LIMIT")
    for basis in plan["basis"]:
        if not isinstance(basis, dict) or set(basis) != {"source", "quote"}:
            raise ValueError("BASIS_FIELDS")
        if type(basis["source"]) is not str:
            raise ValueError("BASIS_FIELDS")
        text = {"question": task.question, "definitions": task.definitions}.get(basis["source"])
        if (
            not text
            or type(basis["quote"]) is not str
            or not basis["quote"].strip()
            or basis["quote"] not in text
        ):
            raise ValueError("BASIS_NOT_IN_SOURCE")
    if (
        not isinstance(plan["assumptions"], list)
        or any(type(s) is not str or not 1 <= len(s) <= 500 for s in plan["assumptions"])
        or len(plan["assumptions"]) > 8
    ):
        raise ValueError("PLAN_ASSUMPTIONS")
    parse_expression(plan["reference"])
    # This option relaxes only initial fixture construction. The program must
    # still parse, and the graph executes it on bound source rows before any
    # publication. Subsequent probe actions retain their non-empty contract.
    if task.allow_source_only_plan and plan["fixtures"] == []:
        fixtures = []
    else:
        fixtures = validate_model_fixtures(plan["fixtures"], task)
    plan = {**plan, "fixtures": fixtures}
    for fixture in plan["fixtures"]:
        validate_rows(calculate(plan["reference"], named_tables(fixture_data(fixture["data"]))))
    # Matching source quotes establishes provenance only, NOT semantic correctness.
    return json.loads(json.dumps(plan, allow_nan=False))


def validate_rows(rows: Any) -> list:
    if not isinstance(rows, list) or len(rows) > 200:
        raise ValueError("RESULT_ROWS")
    for row in rows:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 8:
            raise ValueError("RESULT_COLUMNS")
        for value in row:
            if type(value) not in (int, float, str, type(None)):
                raise ValueError("RESULT_SCALAR_REQUIRED")
            if type(value) in (int, float) and (not math.isfinite(value) or abs(value) > 10**15):
                raise ValueError("RESULT_NUMERIC_LIMIT")
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError("RESULT_STRING_LIMIT")
    return [list(row) for row in rows]


def rows_equal(left: list, right: list) -> bool:
    """Compare multisets, retaining duplicates; ordering is outside this version."""

    def cell(a, b):
        if type(a) in (int, float) and type(b) in (int, float):
            if type(a) is int and type(b) is int:
                return a == b
            # Relative tolerance grows with the amount and can hide whole-unit
            # errors in large totals, including caller-confirmed examples.
            return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        return type(a) is type(b) and a == b

    remaining = list(right)
    for row in left:
        for index, other in enumerate(remaining):
            if len(row) == len(other) and all(cell(a, b) for a, b in zip(row, other)):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def execute_readonly(sql: str, data: ProbeData | RelationalData | SQLiteSource) -> list:
    """A fresh, fixture-only database; no paths or arbitrary Python are accepted."""
    schema = data.schema if isinstance(data, (RelationalData, SQLiteSource)) else TABLE_SCHEMA
    validate_sql(sql, schema)
    functions = {"sum", "avg", "count", "min", "max", "coalesce", "nullif", "abs", "round"}
    if isinstance(data, (RelationalData, SQLiteSource)):
        functions |= {
            "substr",
            "substring",
            "instr",
            "strftime",
            "replace",
            "trim",
            "ltrim",
            "rtrim",
            "lower",
            "upper",
            "length",
            "date",
            "datetime",
            "julianday",
            "ifnull",
            "like",
            "glob",
            "iif",
        }
        connection = data.connect()
    else:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total INTEGER,channel TEXT)"
        )
        connection.execute(
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER,amount INTEGER,status TEXT,settled_at TEXT)"
        )
        connection.executemany("INSERT INTO orders VALUES(?,?,?)", data.orders)
        connection.executemany("INSERT INTO order_lines VALUES(?,?,?,?,?)", data.lines)
        connection.commit()
    with closing(connection):
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.enable_load_extension(False)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 100_000)

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and arg1 in schema:
                if (database == "main" and arg2 in schema[arg1]) or (
                    database in ("main", None) and arg2 == ""
                ):
                    return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in functions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        started, ticks = time.monotonic(), 0

        def stop():
            nonlocal ticks
            ticks += 1
            return int(ticks > 1000 or time.monotonic() - started > 1.0)

        connection.set_authorizer(authorize)
        connection.set_progress_handler(stop, 1000)
        try:
            rows = connection.execute(sql).fetchmany(201)
            return validate_rows(rows)
        except sqlite3.Error:
            raise ValueError("SQL_EXECUTION_REJECTED") from None


def validate_sql(sql: str, schema: dict) -> None:
    """Shared structural scope check; permission to execute still needs SQLite's authorizer."""
    if type(sql) is not str or not 1 <= len(sql) <= 12_000:
        raise ValueError("SQL_LENGTH")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        raise ValueError("SQL_PARSE") from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("READONLY_SELECT_REQUIRED")
    tree = statements[0]
    if len(list(tree.walk())) > 500 or tree.find(exp.With, exp.Window, exp.Into, exp.Union):
        raise ValueError("SQL_SCOPE")
    if any(
        table.name.casefold() not in {name.casefold() for name in schema}
        or table.db
        or table.catalog
        for table in tree.find_all(exp.Table)
    ):
        raise ValueError("UNKNOWN_TABLE")
    if not list(tree.find_all(exp.Table)):
        raise ValueError("SOURCE_TABLE_REQUIRED")


def check_candidate(
    sql: str, plan: dict, fixtures: list[dict], *, source: ProbeData | None = None
) -> dict:
    """Replay immutable expectations. A disagreement is not a proven SQL bug."""
    checks = []
    options = (
        (source.schema, source.column_types, source.primary_keys, source.foreign_keys)
        if isinstance(source, (RelationalData, SQLiteSource))
        else ()
    )
    items = [(fixture["name"], fixture_data(fixture["data"], *options)) for fixture in fixtures]
    if source is not None:
        items.append(("bound_source", source))
    for name, data in items:
        try:
            expected = validate_rows(calculate(plan["reference"], named_tables(data)))
        except ValueError as error:
            checks.append({"fixture": name, "status": "invalid_reference", "code": str(error)})
            continue
        try:
            actual = execute_readonly(sql, data)
            checks.append(
                {
                    "fixture": name,
                    "status": "match" if rows_equal(actual, expected) else "disagreement",
                    "actual": actual,
                    "expected": expected,
                }
            )
        except ValueError as error:
            checks.append({"fixture": name, "status": "sql_error", "code": str(error)})
    return {
        "sql_digest": digest_json(sql),
        "plan_digest": digest_json(plan),
        "checks": checks,
        "all_match": bool(checks) and all(c["status"] == "match" for c in checks),
        "interpretation_status": "model_proposed_not_proven",
    }
