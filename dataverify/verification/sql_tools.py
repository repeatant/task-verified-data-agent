"""Bounded SQL observations over a host-bound source, never a correctness oracle.

The same executor serves initial candidates, diagnostic queries and revisions.
It intentionally retains discovery_runtime's SELECT subset and function policy;
an experiment must not attribute a more permissive executor to agent reasoning.
Only the trusted host supplies a database handle. Model output supplies SQL and
bound scalar values, not paths, schema metadata, SQLite settings or Python code.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass

from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import validate_sql
from dataverify.verification.probe_runtime import TABLE_SCHEMA, ProbeData
from dataverify.verification.relational_data import RelationalData, SQLiteSource

SQLSource = ProbeData | RelationalData | SQLiteSource
_BASE_FUNCTIONS = frozenset(
    {"sum", "avg", "count", "min", "max", "coalesce", "nullif", "abs", "round"}
)
_RELATIONAL_FUNCTIONS = frozenset(
    {
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
)


@dataclass(frozen=True)
class SQLLimits:
    """Trusted per-query ceilings. Model requests cannot raise these limits."""

    timeout_seconds: float = 1.0
    max_vm_steps: int = 1_000_000
    max_rows: int = 200
    max_columns: int = 8
    max_result_chars: int = 16_000
    max_cell_chars: int = 1_000

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not 0 < self.timeout_seconds <= 1.0
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("SQL_TIMEOUT_LIMIT")
        ceilings = {
            "max_vm_steps": 1_000_000,
            "max_rows": 200,
            "max_columns": 8,
            "max_result_chars": 100_000,
            "max_cell_chars": 1_000,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("SQL_RESOURCE_LIMIT")


def _fixture_source(source: ProbeData) -> RelationalData:
    """Materialize trusted controlled data without adding business fault rules."""
    return RelationalData(
        schema=TABLE_SCHEMA,
        tables={"orders": source.orders, "order_lines": source.lines},
        column_types={
            "orders": ("INTEGER", "INTEGER", "TEXT"),
            "order_lines": ("INTEGER", "INTEGER", "INTEGER", "TEXT", "TEXT"),
        },
        primary_keys={"orders": ("order_id",), "order_lines": ("line_id",)},
        foreign_keys=(
            {
                "table": "order_lines",
                "columns": ("order_id",),
                "references": "orders",
                "reference_columns": ("order_id",),
            },
        ),
    )


def _parameters(values) -> tuple:
    if not isinstance(values, (list, tuple)) or len(values) > 100:
        raise ValueError("SQL_PARAMETERS")
    for value in values:
        if type(value) not in (int, float, str, type(None)):
            raise ValueError("SQL_PARAMETER_TYPE")
        if type(value) in (int, float) and (abs(value) > 10**15 or not math.isfinite(value)):
            raise ValueError("SQL_PARAMETER_NUMBER")
        if type(value) is str and (len(value) > 4_000 or "\x00" in value):
            raise ValueError("SQL_PARAMETER_STRING")
    if len(json.dumps(values, ensure_ascii=False, allow_nan=False)) > 12_000:
        raise ValueError("SQL_PARAMETERS_LENGTH")
    return tuple(values)


class SQLTools:
    """Read-only observations with bounded output and source provenance.

    For SQLite files, ``source_id`` fingerprints trusted file metadata (including
    WAL), schema and location, not a cryptographic content hash or a proof of
    immutability. Stat changes before/after a query invalidate its observation.
    SQLite itself provides a consistent read snapshot during each statement. The
    host must bind a stable evaluation database, not an actively modified service.
    No source path is exposed through metadata or observations.
    """

    def __init__(self, source: SQLSource, *, limits: SQLLimits | None = None):
        if not isinstance(source, (ProbeData, RelationalData, SQLiteSource)):
            raise TypeError("SQL_SOURCE_HANDLE_REQUIRED")
        if limits is not None and not isinstance(limits, SQLLimits):
            raise TypeError("SQL_LIMITS_REQUIRED")
        self.source = _fixture_source(source) if isinstance(source, ProbeData) else source
        self.limits = limits or SQLLimits()
        self.scope = "original_database" if isinstance(source, SQLiteSource) else "provided_fixture"
        self._functions = _BASE_FUNCTIONS
        if not isinstance(source, ProbeData):
            self._functions |= _RELATIONAL_FUNCTIONS
        self._schema = {
            name.casefold(): {c.casefold() for c in cols}
            for name, cols in self.source.schema.items()
        }
        self.source_id = digest_json(self._source_version())

    def _source_version(self) -> dict:
        if isinstance(self.source, SQLiteSource):
            files = []
            for path in (
                self.source.path,
                self.source.path.with_name(self.source.path.name + "-wal"),
            ):
                if path.exists():
                    stat = path.stat()
                    files.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino))
                else:
                    files.append((str(path), None))
            return {
                "files": files,
                "schema": self.source.schema,
                "types": self.source.column_types,
                "primary_keys": self.source.primary_keys,
                "foreign_keys": self.source.foreign_keys,
            }
        return self.source.to_dict()

    def _unchanged(self) -> bool:
        try:
            return digest_json(self._source_version()) == self.source_id
        except OSError:
            return False

    def metadata(self) -> dict:
        """Return schema-only context: no rows, local paths, gold SQL or answers."""
        return {
            "schema": {name: list(cols) for name, cols in self.source.schema.items()},
            "column_types": {name: list(cols) for name, cols in self.source.column_types.items()},
            "primary_keys": {name: list(cols) for name, cols in self.source.primary_keys.items()},
            "foreign_keys": [dict(link) for link in self.source.foreign_keys],
            "source_id": self.source_id,
            "scope": self.scope,
            "limits": asdict(self.limits),
        }

    def _authorize(self, action, arg1, arg2, database, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and (arg1 or "").casefold() in self._schema:
            columns = self._schema[arg1.casefold()]
            if (database == "main" and (arg2 or "").casefold() in columns) or (
                database in ("main", None) and arg2 == ""
            ):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in self._functions:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def execute(self, sql: str, parameters=(), *, observation_id: str, purpose: str = "") -> dict:
        """Run one bounded SELECT and return evidence, never a correctness claim."""
        if type(observation_id) is not str or not 1 <= len(observation_id) <= 200:
            raise ValueError("SQL_OBSERVATION_ID")
        if type(purpose) is not str or len(purpose) > 2_000:
            raise ValueError("SQL_OBSERVATION_PURPOSE")
        started = time.monotonic()
        observation = {
            "id": observation_id,
            "sql": sql[:12_001] if isinstance(sql, str) else None,
            "parameters": [],
            "purpose": purpose,
            "source_id": self.source_id,
            "scope": self.scope,
            "status": "ok",
            "columns": [],
            "rows": [],
            "complete": False,
            "truncated": False,
            "truncation_reason": None,
            "error": None,
            "limits": asdict(self.limits),
        }

        def finish(status="ok", code=None, message=None):
            observation["status"] = status
            if code:
                observation["error"] = {"code": code, "message": message or code}
            observation["returned_rows"] = len(observation["rows"])
            observation["query_digest"] = digest_json(
                [observation["sql"], observation["parameters"]]
            )
            observation["observation_digest"] = digest_json(observation)
            observation["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            return observation

        try:
            validate_sql(sql, self.source.schema)
            parameters = _parameters(parameters)
            observation["parameters"] = list(parameters)
        except ValueError as error:
            return finish("rejected", str(error))
        if not self._unchanged():
            return finish("error", "SOURCE_CHANGED", "Bound source changed; start a new task.")

        ticks, interrupted = 0, False
        interval = min(1_000, self.limits.max_vm_steps)

        def stop():
            nonlocal ticks, interrupted
            ticks += 1
            interrupted = (
                ticks * interval >= self.limits.max_vm_steps
                or time.monotonic() - started > self.limits.timeout_seconds
            )
            return int(interrupted)

        try:
            with closing(self.source.connect()) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                connection.execute("PRAGMA busy_timeout=100")
                connection.enable_load_extension(False)
                connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 100_000)
                connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 12_000)
                connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100)
                connection.set_authorizer(self._authorize)
                connection.set_progress_handler(stop, interval)
                cursor = connection.execute(sql, parameters)
                observation["columns"] = [column[0] for column in cursor.description]
                if len(observation["columns"]) > self.limits.max_columns:
                    return finish("rejected", "RESULT_COLUMNS")
                chars = len(json.dumps(observation["columns"], ensure_ascii=False))
                for row in cursor:
                    reason = self._row_limit(row, observation["rows"], chars)
                    if reason:
                        observation["truncated"] = True
                        observation["truncation_reason"] = reason
                        break
                    observation["rows"].append(list(row))
                    chars += len(json.dumps(row, ensure_ascii=False, allow_nan=False)) + 1
                    if time.monotonic() - started > self.limits.timeout_seconds:
                        interrupted = True
                        raise sqlite3.OperationalError("interrupted")
                observation["complete"] = not observation["truncated"]
        except sqlite3.Error as error:
            # Partially streamed results cannot be used as a successful observation.
            observation["rows"], observation["complete"] = [], False
            code = getattr(error, "sqlite_errorname", "SQL_EXECUTION_ERROR")
            if interrupted or code in {
                "SQLITE_INTERRUPT",
                "SQLITE_TOOBIG",
                "SQLITE_NOMEM",
                "SQLITE_BUSY",
                "SQLITE_LOCKED",
            }:
                return finish(
                    "resource_limit", code, "SQL exceeded the configured resource limits."
                )
            if code == "SQLITE_AUTH" or "not authorized" in str(error).lower():
                return finish("rejected", "SQL_NOT_AUTHORIZED")
            return finish("error", code, str(error)[:500])
        if not self._unchanged():
            observation["rows"], observation["complete"] = [], False
            return finish("error", "SOURCE_CHANGED", "Bound source changed during execution.")
        return finish("truncated" if observation["truncated"] else "ok")

    def _row_limit(self, row: tuple, previous: list, chars: int) -> str | None:
        if len(previous) >= self.limits.max_rows:
            return "row_limit"
        for value in row:
            if type(value) not in (int, float, str, type(None)):
                return "unsupported_scalar"
            if type(value) in (int, float) and (not math.isfinite(value) or abs(value) > 10**15):
                return "numeric_limit"
            if isinstance(value, str) and len(value) > self.limits.max_cell_chars:
                return "cell_length_limit"
        if (
            chars + len(json.dumps(row, ensure_ascii=False, allow_nan=False)) + 1
            > self.limits.max_result_chars
        ):
            return "result_char_limit"
        return None


def execute_observation(
    sql: str,
    source: SQLSource,
    observation_id: str,
    purpose: str = "",
    parameters=(),
    limits: SQLLimits | None = None,
) -> dict:
    """Convenience entry point; reuse SQLTools to bind one source across a task."""
    return SQLTools(source, limits=limits).execute(
        sql, parameters, observation_id=observation_id, purpose=purpose
    )
