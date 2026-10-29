"""Independent source and boundary grading shared by Agent evaluators.

These disclosed development targets are evaluation-only. Runtime executors and
feedback generators must never import this module or receive its boundary data.
No Agent framework, model client, credentials or network access is required.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from datetime import datetime

from .probe_cases import Case, grade_rows, reference_result
from .probe_runtime import TABLE_SCHEMA, ProbeData


def boundary_fixture(case: Case) -> ProbeData:
    """Additional evaluator-only rows cover boundaries/refunds/zero/grouping.

    This is an exact-target regression fixture, not an unseen problem template.
    It is never passed to the agent, runtime properties or feedback generator.
    """
    task, source = case.task, case.source
    parent = max((row[0] for row in source.orders), default=0) + 10
    child = max((row[0] for row in source.lines), default=0) + 10
    start = datetime.fromisoformat(task.start)
    end = datetime.fromisoformat(task.end)
    interior = (start + (end - start) / 2).replace(microsecond=0).isoformat()
    extra_orders = (
        (parent, 37, None),
        (parent + 1, 11, task.other_label),
        (parent + 2, 0, "zero-only"),
    )
    pending = "pending" if task.status != "pending" else "cancelled"
    cancelled = "cancelled" if task.status != "cancelled" else "pending"
    extra_lines = (
        (child, parent, 11, task.status, task.start),
        (child + 1, parent, -7, task.status, interior),
        (child + 2, parent, 23, task.status, task.end),
        (child + 3, parent, 31, pending, interior),
        (child + 4, parent + 1, 11, task.status, interior),
        (child + 5, parent + 1, 19, cancelled, interior),
        (child + 6, parent + 2, 47, pending, interior),
    )
    return ProbeData(source.orders + extra_orders, source.lines + extra_lines)


def independent_query(sql: str, data: ProbeData) -> dict:
    """Execute published SQL for grading without consulting the runtime checker.

    SQLite authorizer, a VM/time limit and row cap enforce an independent read
    boundary. The target itself is computed from Python records, not this SQL.
    """
    if type(sql) is not str or not sql.strip() or len(sql) > 16_000:
        return {"status": "error", "code": "INVALID_PUBLISHED_SQL", "rows": []}
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total INTEGER NOT NULL,channel TEXT)"
        )
        conn.execute(
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER REFERENCES orders(order_id),amount INTEGER,status TEXT,settled_at TEXT)"
        )
        conn.executemany("INSERT INTO orders VALUES (?,?,?)", data.orders)
        conn.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?)", data.lines)
        conn.commit()
        conn.execute("PRAGMA query_only=ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        deadline = time.monotonic() + 1.0
        callbacks = 0

        def progress():
            nonlocal callbacks
            callbacks += 1
            return int(callbacks > 100 or time.monotonic() > deadline)

        def authorizer(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and table in TABLE_SCHEMA:
                if (database == "main" and column in TABLE_SCHEMA[table]) or (
                    database in {None, "main"} and column == ""
                ):
                    return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (column or "").casefold() in {
                "sum",
                "coalesce",
            }:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorizer)
        conn.set_progress_handler(progress, 1000)
        rows = conn.execute(sql).fetchmany(1001)
        if len(rows) > 1000:
            return {"status": "error", "code": "GRADING_ROW_BUDGET", "rows": []}
        return {"status": "ok", "code": None, "rows": [list(row) for row in rows]}
    except (sqlite3.Error, ValueError, OverflowError):
        return {"status": "error", "code": "GRADING_SQL_ERROR", "rows": []}
    finally:
        conn.close()


def grade_probe_record(case: Case, record: dict) -> dict:
    """Grade source answer, published-SQL consistency and boundary behaviour."""
    publication = record.get("publication") or {}
    accepted = publication.get("status") == "verified"
    task_preserved = publication.get("task_id") == case.task.task_id
    http_errors = any(event.get("status") != "completed" for event in record.get("http_events", []))
    operational = bool(
        record.get("operational_error") or http_errors or record.get("source_unchanged") is False
    )
    result = {
        "accepted": accepted,
        "task_preserved": task_preserved,
        "operational_failure": operational,
        "correct_task": False,
        "source_correct": False,
        "boundary_correct": False,
        "publication_matches_sql": False,
        "wrong_release": False,
        "unscorable": False,
        "scope": "source and additional disclosed-family boundary fixture, not arbitrary SQL proof",
    }
    if not accepted:
        return result
    sql = publication.get("verified_sql")
    expected = reference_result(case)
    source_grade = grade_rows(publication.get("rows"), expected)
    source_exec = independent_query(sql, case.source)
    hidden = boundary_fixture(case)
    hidden_exec = independent_query(sql, hidden)
    hidden_expected = reference_result(replace(case, source=hidden))
    result.update(source_grade=source_grade, source_correct=source_grade["correct"])
    if source_exec["status"] != "ok" or hidden_exec["status"] != "ok":
        result.update(
            unscorable=True,
            source_sql_status=source_exec["status"],
            boundary_sql_status=hidden_exec["status"],
        )
    else:
        published_consistency = grade_rows(publication.get("rows"), source_exec["rows"])
        boundary_grade = grade_rows(hidden_exec["rows"], hidden_expected)
        result.update(
            publication_matches_sql=published_consistency["correct"],
            boundary_grade=boundary_grade,
            boundary_correct=boundary_grade["correct"],
        )
    semantically_correct = bool(
        task_preserved
        and result["source_correct"]
        and result["publication_matches_sql"]
        and result["boundary_correct"]
    )
    result["correct_task"] = semantically_correct and not operational
    result["wrong_release"] = not semantically_correct
    return result
