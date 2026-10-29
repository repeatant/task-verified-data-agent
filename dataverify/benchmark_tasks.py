"""Small frozen *development* suite for the three DB-GPT comparison arms.

The row records and private expectations are evaluation data, never prompt data.
``Task.public_input()`` is the only task payload intended for a model message.
The target oracle uses Python records, independently of SQLite, SQLGlot, the
checker, and the older synthetic evaluation module. It does not interpret SQL.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable


class _FrozenCandidate(dict):
    """Keep the required JSON dictionary shape without mutable suite entries."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("Frozen benchmark candidates cannot be changed")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __copy__(self) -> _FrozenCandidate:
        return self

    def __deepcopy__(self, memo: dict) -> _FrozenCandidate:
        return self


@dataclass(frozen=True)
class Fixture:
    orders: tuple[tuple[Any, Any], ...]
    lines: tuple[tuple[Any, Any, Any], ...]

    def __post_init__(self) -> None:
        # Own the nested tuples even when a caller supplies lists. Cells remain
        # scalar so freezing the dataclass also freezes its entire row payload.
        for name, width in (("orders", 2), ("lines", 3)):
            rows = tuple(tuple(row) for row in getattr(self, name))
            if any(len(row) != width for row in rows):
                raise ValueError(f"{name} rows must contain {width} cells")
            if any(type(cell) not in (type(None), bool, int, float, str) for row in rows for cell in row):
                raise ValueError("Fixture cells must be immutable SQLite scalar values")
            object.__setattr__(self, name, rows)


@dataclass(frozen=True)
class Task:
    task_id: str
    track: str
    fixture: Fixture
    question: str
    confirmed_metric_id: str
    initial_candidate: dict[str, str] | None
    pair_id: str | None = None
    split: str = "development"

    def __post_init__(self) -> None:
        if self.track not in ("natural", "controlled"):
            raise ValueError("track must be natural or controlled")
        if self.split != "development":
            raise ValueError("This disclosed suite only supports the development split")
        if self.confirmed_metric_id not in ("order_revenue", "line_revenue"):
            raise ValueError("Unknown confirmed metric")
        if not isinstance(self.fixture, Fixture):
            raise TypeError("fixture must be a frozen Fixture")
        if (self.initial_candidate is None) != (self.track == "natural"):
            raise ValueError("Only controlled tasks have an initial candidate")
        if self.initial_candidate is not None:
            required = {"display_type", "sql", "metric_id", "thought"}
            if set(self.initial_candidate) != required or any(type(value) is not str for value in self.initial_candidate.values()):
                raise ValueError("Candidate must contain the four string protocol fields")
            object.__setattr__(self, "initial_candidate", _FrozenCandidate(self.initial_candidate))

    def public_input(self) -> dict[str, str]:
        """Allowlist the model input: no IDs, labels, rows, answers, or repairs."""
        return {"question": self.question, "confirmed_metric_id": self.confirmed_metric_id}


_ORDER_QUESTION = "请计算全部订单的订单收入总额。"
_LINE_QUESTION = "请计算全部订单明细的明细收入总额。"
_THOUGHT = "根据问题和已确认指标提交候选查询。"
_ORDER_JOIN = "SELECT SUM(o.total) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id"
_ORDER_DISTINCT = "SELECT SUM(DISTINCT total) FROM orders"
_LINE_JOIN = "SELECT SUM(l.amount) FROM order_lines l INNER JOIN orders o ON l.order_id=o.order_id"
_ORDER_DIRECT = "SELECT SUM(total) FROM orders"
_LINE_DIRECT = "SELECT SUM(amount) FROM order_lines"

_NORMAL = Fixture(
    ((1, 120.0), (2, 80.0), (3, 100.0)),
    ((11, 1, 120.0), (12, 2, 80.0), (13, 3, 100.0)),
)
_FANOUT = Fixture(_NORMAL.orders, _NORMAL.lines + ((14, 1, 30.0),))
_MISSING = Fixture(_NORMAL.orders, _NORMAL.lines[:-1])
_ORPHAN = Fixture(_FANOUT.orders, _FANOUT.lines + ((15, 999, 25.0),))
_EQUAL_VALUES = Fixture(((1, 100.0), (2, 100.0), (3, 100.0)), _NORMAL.lines)
_BAD_ORDER = Fixture(_NORMAL.orders + ((2, 80.0),), _NORMAL.lines)
_BAD_LINE = Fixture(_NORMAL.orders, ((11, 1, "unavailable"),) + _NORMAL.lines[1:])


def _task(
    task_id: str, fixture: Fixture, metric_id: str, sql: str | None = None,
    pair_id: str | None = None,
) -> Task:
    candidate = None if sql is None else {
        "display_type": "response_table", "sql": sql,
        "metric_id": metric_id, "thought": _THOUGHT,
    }
    return Task(
        task_id, "natural" if candidate is None else "controlled", fixture,
        _ORDER_QUESTION if metric_id == "order_revenue" else _LINE_QUESTION,
        metric_id, candidate, pair_id,
    )


# Opaque IDs keep fixture/fault names out of even accidental ID-only prompts.
# Each pair changes only database records; question, metric and candidate stay
# identical. These are disclosed development cases, not held-out evidence.
TASKS: tuple[Task, ...] = (
    _task("n01", _NORMAL, "order_revenue"),
    _task("n02", _NORMAL, "line_revenue"),
    _task("n03", _MISSING, "order_revenue"),
    _task("n04", _ORPHAN, "line_revenue"),
    _task("c01", _NORMAL, "order_revenue", _ORDER_JOIN, "p01"),
    _task("c02", _FANOUT, "order_revenue", _ORDER_JOIN, "p01"),
    _task("c03", _EQUAL_VALUES, "order_revenue", _ORDER_DISTINCT, "p02"),
    _task("c04", _NORMAL, "order_revenue", _ORDER_DISTINCT, "p02"),
    _task("c05", _FANOUT, "line_revenue", _LINE_JOIN, "p03"),
    _task("c06", _ORPHAN, "line_revenue", _LINE_JOIN, "p03"),
    # Source substitution remains invalid even though both source totals are 300.
    _task("c07", _NORMAL, "order_revenue", _LINE_DIRECT),
    _task("c08", _MISSING, "order_revenue", _ORDER_JOIN, "p01"),
    _task("c09", _BAD_ORDER, "order_revenue", _ORDER_DIRECT),
    _task("c10", _BAD_LINE, "line_revenue", _LINE_DIRECT),
)


def get_tasks(track: str | None = None) -> tuple[Task, ...]:
    if track is not None and track not in ("natural", "controlled"):
        raise ValueError("track must be natural or controlled")
    return tuple(task for task in TASKS if track is None or task.track == track)


def get_task(task_id: str) -> Task:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise KeyError(task_id)


def _numeric(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def oracle(task: Task) -> dict[str, Any]:
    """Compute the target only, from source row values and unique business grain.

    An empty, nonnumeric, NULL, nonfinite, or nonunique target source has no
    answer under this experiment's contract. Missing parent/detail relations
    alone do not remove records from an all-source metric.
    """
    records = task.fixture.orders if task.confirmed_metric_id == "order_revenue" else task.fixture.lines
    grains = [row[0] for row in records]
    amounts = [row[1] if task.confirmed_metric_id == "order_revenue" else row[2] for row in records]
    reason = None
    total = None
    if not records:
        reason = "empty_source"
    elif None in grains or len(set(grains)) != len(grains):
        reason = "invalid_source_grain"
    elif not all(_numeric(value) for value in amounts):
        reason = "invalid_measure_value"
    else:
        try:
            total = math.fsum(amounts)
        except (OverflowError, ValueError):
            reason = "nonfinite_total"
        if reason is None and not _numeric(total):
            reason = "nonfinite_total"
        if reason is not None:
            total = None
    return {
        "target_defined": reason is None, "metric_value": total,
        "source_rows": len(records),
        "source_grains": len({grain for grain in grains if grain is not None}),
        "reason": reason,
    }


@dataclass(frozen=True)
class _CandidatePlan:
    source: str
    joined: bool = False
    value_distinct: bool = False


# These are explicit semantics for exactly the frozen templates, not a SQL
# parser. Unknown or edited SQL is never silently assigned a template's truth.
_PLANS = {
    _ORDER_JOIN: _CandidatePlan("orders", joined=True),
    _ORDER_DISTINCT: _CandidatePlan("orders", value_distinct=True),
    _LINE_JOIN: _CandidatePlan("order_lines", joined=True),
    _ORDER_DIRECT: _CandidatePlan("orders"),
    _LINE_DIRECT: _CandidatePlan("order_lines"),
}


def candidate_expectation(task: Task) -> dict[str, Any]:
    """Private initial-candidate expectation, independently counted in Python.

    This cannot grade an arbitrary generated or repaired SQL query. ``oracle``
    provides the target for that separate grading step.
    """
    expected = oracle(task)
    if task.initial_candidate is None:
        return {**expected, "should_accept": None, "candidate_reason": None}
    plan = _PLANS.get(task.initial_candidate["sql"])
    if plan is None:
        raise ValueError("Only frozen candidate templates have declared semantics")
    target = "orders" if task.confirmed_metric_id == "order_revenue" else "order_lines"
    reason = expected["reason"]
    if reason is None and (plan.source != target or task.initial_candidate["metric_id"] != task.confirmed_metric_id):
        reason = "metric_source_mismatch"
    if reason is None and plan.value_distinct:
        reason = "value_distinct_not_grain"
    if reason is None and plan.joined:
        observed = []
        for order in task.fixture.orders:
            for line in task.fixture.lines:
                if order[0] is not None and line[1] is not None and order[0] == line[1]:
                    observed.append(order[0] if target == "orders" else line[0])
        records = task.fixture.orders if target == "orders" else task.fixture.lines
        if Counter(observed) != Counter(row[0] for row in records):
            reason = "incomplete_or_repeated_grain"
    return {**expected, "should_accept": reason is None, "candidate_reason": reason}


def write_fixture(path: str | Path, fixture: Fixture) -> Path:
    """Create one database exclusively; existing files are never overwritten."""
    path = Path(path)
    # Exclusive creation handles both regular files and existing symlinks before
    # SQLite connects. Do not use exists()+connect(), which has a clobber race.
    with path.open("xb"):
        pass
    with closing(sqlite3.connect(str(path))) as connection:
        # Deliberately no primary/foreign key or non-null constraints: damaged
        # records must reach the oracle/checker rather than fail fixture setup.
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER,total REAL);"
            "CREATE TABLE order_lines(line_id INTEGER,order_id INTEGER,amount REAL);"
        )
        connection.executemany("INSERT INTO orders VALUES(?,?)", fixture.orders)
        connection.executemany("INSERT INTO order_lines VALUES(?,?,?)", fixture.lines)
        connection.commit()
    return path


def manifest(tasks: Iterable[Task] = TASKS) -> dict[str, Any]:
    """Serialize private evaluation data. Never use this object as model input."""
    tasks = tuple(tasks)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Manifest task IDs must be unique")
    return {
        "manifest_version": 1, "split": "development", "task_count": len(tasks),
        "limitation": "Small disclosed development suite; not a held-out or generalization benchmark.",
        "tasks": [{
            "task_id": task.task_id, "track": task.track, "split": task.split,
            "fixture": {
                "orders": [list(row) for row in task.fixture.orders],
                "lines": [list(row) for row in task.fixture.lines],
            },
            "question": task.question, "confirmed_metric_id": task.confirmed_metric_id,
            "initial_candidate": None if task.initial_candidate is None else dict(task.initial_candidate),
            "pair_id": task.pair_id, "expected": candidate_expectation(task),
        } for task in tasks],
    }
