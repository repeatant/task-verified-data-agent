"""Contract-guided metamorphic diagnostics on isolated, generated databases.

Hypothesis supplies generation and shrinking. This module supplies task-specific
transformations, paired-data validity, failure identity and a shared SQL budget.
No model, reference SQL, evaluation label, source file or credential is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import math
import random
import time
from typing import Any

from hypothesis import HealthCheck, Phase, find, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample, Unsatisfiable
import sqlglot
from sqlglot import expressions as exp

from .probe_runtime import PROPERTY_IDS, ProbeData, ProbeTask, execute_sql, static_check


class ProbeBudgetExceeded(RuntimeError):
    """A local execution budget is exhausted; it is not a passing check."""


@dataclass
class ExecutionBudget:
    max_queries: int = 160
    max_seconds: float = 12.0
    queries: int = 0
    started: float = 0.0

    def __post_init__(self) -> None:
        if type(self.max_queries) is not int or not 2 <= self.max_queries <= 2000:
            raise ValueError("max_queries must be an integer between 2 and 2000")
        if type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        self.started = time.monotonic()

    def reserve(self) -> None:
        if self.queries >= self.max_queries or time.monotonic() - self.started > self.max_seconds:
            raise ProbeBudgetExceeded("local_query_or_wall_clock_budget_exhausted")
        # Reserve before execution. Failed executions are not free retries.
        self.queries += 1


@dataclass(frozen=True)
class ProbePair:
    property_id: str
    base: ProbeData
    added_orders: tuple[tuple, ...]
    added_lines: tuple[tuple, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, ProbeData):
            raise ValueError("base must be validated ProbeData")
        if self.property_id not in PROPERTY_IDS:
            raise ValueError("unknown property")
        object.__setattr__(self, "added_orders", tuple(tuple(row) for row in self.added_orders))
        object.__setattr__(self, "added_lines", tuple(tuple(row) for row in self.added_lines))
        # Validate both sides immediately, including fresh PKs and child FKs.
        self.followup

    @property
    def followup(self) -> ProbeData:
        return ProbeData(self.base.orders + self.added_orders, self.base.lines + self.added_lines)

    @property
    def record_count(self) -> int:
        # Count unique base+inserted records, not two copies of the base.
        return len(self.base.orders) + len(self.base.lines) + len(self.added_orders) + len(self.added_lines)

    def payload(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "base": asdict(self.base),
            "added_orders": self.added_orders,
            "added_lines": self.added_lines,
            "record_count": self.record_count,
        }


def _eligible(row: tuple, task: ProbeTask) -> bool:
    return row[3] == task.status and task.start <= row[4] < task.end


def _interior(task: ProbeTask) -> str:
    start, end = datetime.fromisoformat(task.start), datetime.fromisoformat(task.end)
    if (end - start).total_seconds() < 2:
        raise ValueError("probe interval must contain an interior second")
    return (start + (end - start) / 2).replace(microsecond=0).isoformat()


def rank_properties(sql: str, task: ProbeTask, mode: str, seed: int) -> list[str]:
    """Task properties are mandatory inputs, including conditions absent in AST.

    This is an explicit heuristic, not learned VOI or an optimal test selector.
    Both comparison modes have the same available properties and local budget.
    """
    properties = list(task.properties)
    if mode == "random":
        random.Random(seed).shuffle(properties)
        return properties
    if mode != "directed":
        raise ValueError("mode must be directed or random")
    tree = sqlglot.parse_one(sql, read="sqlite")
    columns = {column.name.casefold() for column in tree.find_all(exp.Column)}
    joins = list(tree.find_all(exp.Join))
    scores = {name: 0 for name in properties}
    if "status" not in columns:
        scores.update({p: 8 for p in properties if p == "exclude_status"})
    if any(isinstance(node, (exp.LTE, exp.Between)) for node in tree.walk()):
        scores.update({p: 7 for p in properties if p == "exclude_end"})
    if any(isinstance(node, exp.GT) for node in tree.walk()):
        scores.update({p: 6 for p in properties if p in {"include_start", "include_negative"}})
    if joins and tree.args.get("where") is not None:
        scores.update({p: 8 for p in properties if p == "preserve_zero_orders"})
    if tree.find(exp.Coalesce) is not None:
        scores.update({p: 7 for p in properties if p == "merge_other_channels"})
    # Stable tie breaking is explicit; it does not consult fault labels/gold.
    return sorted(properties, key=lambda name: (-scores[name], properties.index(name)))


@st.composite
def pair_strategy(draw: Any, task: ProbeTask, property_id: str) -> ProbePair:
    """Generate parent keys first, then valid children and one legal change."""
    if property_id not in task.properties:
        raise ValueError("property was not confirmed by this task")
    count = draw(st.integers(min_value=3, max_value=7))
    orders, lines = [], []
    interior = _interior(task)
    for order_id in range(1, count + 1):
        channel = task.other_label if order_id == 1 else draw(st.sampled_from(("web", "store", task.other_label)))
        amounts = draw(st.lists(st.integers(min_value=1, max_value=900), min_size=1, max_size=3))
        orders.append((order_id, sum(amounts), channel))
        for amount in amounts:
            lines.append((len(lines) + 1, order_id, amount, task.status, interior))
    base = ProbeData(tuple(orders), tuple(lines))
    new_order, new_line = count + 1, len(lines) + 1
    amount = draw(st.integers(min_value=1, max_value=900))
    status, date, channel = task.status, interior, "web"
    if property_id == "exclude_end":
        date = task.end
    elif property_id == "exclude_status":
        status = draw(st.sampled_from(tuple(value for value in ("pending", "cancelled", "settled") if value != task.status)))
    elif property_id == "include_start":
        date = task.start
    elif property_id == "include_negative":
        amount = -amount
    elif property_id == "preserve_zero_orders":
        status = "pending" if task.status != "pending" else "cancelled"
    elif property_id == "merge_other_channels":
        channel = None
    else:
        raise ValueError("unknown property")
    return ProbePair(
        property_id, base,
        ((new_order, abs(amount), channel),),
        ((new_line, new_order, amount, status, date),),
    )


def _precondition(pair: ProbePair, task: ProbeTask) -> bool:
    # Base witnesses prevent a reducer turning the original fault into empty
    # SUM/NULL behaviour. The inserted witnesses remain fixed throughout.
    if pair.property_id not in task.properties or len(pair.added_orders) != 1 or len(pair.added_lines) != 1:
        return False
    if not pair.base.orders or not any(_eligible(row, task) for row in pair.base.lines):
        return False
    parent, child = pair.added_orders[0], pair.added_lines[0]
    if child[1] != parent[0]:
        return False
    if pair.property_id == "exclude_end" and not (child[4] == task.end and child[3] == task.status and child[2] > 0):
        return False
    if pair.property_id == "exclude_status" and not (task.start <= child[4] < task.end and child[3] != task.status and child[2] > 0):
        return False
    if pair.property_id == "include_start" and not (child[4] == task.start and child[3] == task.status and child[2] > 0):
        return False
    if pair.property_id == "include_negative" and not (_eligible(child, task) and child[2] < 0):
        return False
    if pair.property_id == "preserve_zero_orders" and _eligible(child, task):
        return False
    if pair.property_id == "merge_other_channels":
        if parent[2] is not None or not _eligible(child, task) or child[2] <= 0:
            return False
        matching_parents = {row[0] for row in pair.base.orders if row[2] == task.other_label}
        return any(row[1] in matching_parents and _eligible(row, task) for row in pair.base.lines)
    return True


def _scalar(result: dict) -> int | float:
    rows = result["rows"]
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("expected one scalar")
    value = rows[0][0]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected finite numeric result")
    return value


def _mapping(result: dict) -> dict[Any, int | float]:
    mapped = {}
    for row in result["rows"]:
        if len(row) != 2 or row[0] in mapped:
            raise ValueError("duplicate or invalid output key")
        if type(row[1]) not in (int, float) or not math.isfinite(row[1]):
            raise ValueError("expected finite group value")
        mapped[row[0]] = row[1]
    return mapped


def _equal(a: int | float, b: int | float) -> bool:
    # Money is an integer number of minor units. Relative tolerance at large
    # totals could incorrectly forgive a missing cent, so compare exactly.
    return a == b


def evaluate_pair(sql: str, task: ProbeTask, pair: ProbePair, budget: ExecutionBudget) -> dict:
    """Necessary relation check; this function never computes a full gold SQL.

    Semantic gates are disabled ONLY for diagnostic execution of a structurally
    permitted frozen candidate. The strong gate still controls acceptance in
    probe_candidate(), so a static rejection is not relabelled a new detection.
    """
    if not _precondition(pair, task):
        return {"status": "not_applicable", "property_id": pair.property_id}
    results = []
    for data in (pair.base, pair.followup):
        budget.reserve()
        result = execute_sql(sql, data, task, static_semantics=False)
        if result["status"] != "ok":
            return {"status": "execution_error", "property_id": pair.property_id, "code": result.get("code")}
        results.append(result)
    before, after = results
    property_id = pair.property_id
    try:
        if property_id in {"exclude_end", "exclude_status", "include_start", "include_negative"}:
            delta = pair.added_lines[0][2] if property_id in {"include_start", "include_negative"} else 0
            passed = _equal(_scalar(after), _scalar(before) + delta)
        elif property_id == "preserve_zero_orders":
            previous, following = _mapping(before), _mapping(after)
            expected = dict(previous)
            expected[pair.added_orders[0][0]] = 0.0
            passed = set(expected) == set(following) and all(_equal(expected[key], following[key]) for key in expected)
        elif property_id == "merge_other_channels":
            previous, following = _mapping(before), _mapping(after)
            expected = dict(previous)
            expected[task.other_label] = expected.get(task.other_label, 0) + pair.added_lines[0][2]
            passed = set(expected) == set(following) and all(_equal(expected[key], following[key]) for key in expected)
        else:
            raise ValueError("unregistered property")
    except (ValueError, TypeError, KeyError):
        return {"status": "output_invalid", "property_id": property_id,
                "code": "UNDEFINED_OR_INVALID_OUTPUT", "before": before["rows"], "after": after["rows"]}
    return {
        "status": "passed" if passed else "violation",
        "property_id": property_id,
        "failure_type": None if passed else "PROPERTY_" + property_id.upper(),
        "before": before["rows"], "after": after["rows"],
    }


@st.composite
def _subset_strategy(draw: Any, pair: ProbePair) -> ProbePair:
    selected_parents = draw(st.sets(st.sampled_from(tuple(row[0] for row in pair.base.orders)), min_size=1))
    orders = tuple(row for row in pair.base.orders if row[0] in selected_parents)
    eligible_lines = tuple(row for row in pair.base.lines if row[1] in selected_parents)
    if not eligible_lines:
        return ProbePair(pair.property_id, ProbeData(orders, ()), pair.added_orders, pair.added_lines)
    selected_children = draw(st.sets(st.sampled_from(tuple(row[0] for row in eligible_lines)), min_size=1))
    lines = tuple(row for row in eligible_lines if row[0] in selected_children)
    return ProbePair(pair.property_id, ProbeData(orders, lines), pair.added_orders, pair.added_lines)


def reduce_pair(sql: str, task: ProbeTask, pair: ProbePair, budget: ExecutionBudget, seed: int) -> tuple[ProbePair, str]:
    """Shrink base records, keeping the inserted change and same failure type.

    The returned witness is the best validated subset observed within budget;
    there is no claim of globally minimal input or amount/value simplification.
    """
    best = pair
    target_type = "PROPERTY_" + pair.property_id.upper()

    def still_fails(candidate: ProbePair) -> bool:
        nonlocal best
        if not _precondition(candidate, task):
            return False
        verdict = evaluate_pair(sql, task, candidate, budget)
        failed = verdict.get("failure_type") == target_type
        if failed and candidate.record_count < best.record_count:
            best = candidate
        return failed

    try:
        find(
            _subset_strategy(pair), still_fails,
            settings=settings(max_examples=24, database=None, deadline=None,
                              phases=(Phase.generate, Phase.shrink),
                              suppress_health_check=(HealthCheck.too_slow,)),
            random=random.Random(seed),
        )
        return best, "completed"
    except ProbeBudgetExceeded:
        return best, "budget_exhausted_best_validated_witness"
    except (NoSuchExample, Unsatisfiable):
        return best, "no_smaller_witness_found"


def evidence_text(task: ProbeTask, pair: ProbePair, verdict: dict) -> str:
    return json.dumps({
        "task_id": task.task_id, "confirmed_property": pair.property_id,
        "property_violation": verdict.get("failure_type"),
        "data": pair.payload(), "before": verdict.get("before"), "after": verdict.get("after"),
        "instruction": "修正候选实现以满足已确认性质，并保留原任务其他要求；此处不提供正确SQL。",
    }, ensure_ascii=False, separators=(",", ":"))


def probe_candidate(
    sql: str, task: ProbeTask, *, mode: str = "directed", seed: int = 17,
    max_queries: int = 160, max_seconds: float = 12.0,
    shrink: bool = True, max_examples: int = 8,
) -> dict:
    """Probe a frozen candidate; separate diagnostics from new detections."""
    if type(max_examples) is not int or not 1 <= max_examples <= 64:
        raise ValueError("max_examples must be between 1 and 64")
    if mode not in {"directed", "random"} or type(seed) is not int:
        raise ValueError("invalid mode or seed")
    budget = ExecutionBudget(max_queries, max_seconds)
    strong = static_check(sql, task)
    structure = static_check(sql, task, static_semantics=False)
    report: dict[str, Any] = {
        "task_id": task.task_id, "candidate_sql": sql, "mode": mode, "seed": seed,
        "confirmed_task": task.to_dict(),
        "static": strong, "structural": structure, "status": "not_checked",
        "new_detection_beyond_static": False, "queries": 0,
        "scope": "registered necessary properties; no complete correctness proof",
        "property_results": [], "counterexample": None,
        "intermediate_check_errors": [],
    }
    if not structure["accepted"]:
        report.update(status="unsupported", code=structure.get("code"), acceptance=False)
        return report
    if not task.properties:
        report.update(status="not_applicable", acceptance=False)
        return report
    order = rank_properties(sql, task, mode, seed)
    report["property_order"] = order
    for index, property_id in enumerate(order):
        execution_errors = []
        witnessed = {}

        def violates(pair: ProbePair) -> bool:
            verdict = evaluate_pair(sql, task, pair, budget)
            if verdict["status"] in {"execution_error", "output_invalid"}:
                execution_errors.append(verdict)
                report["intermediate_check_errors"].append(verdict)
            if verdict["status"] == "violation":
                witnessed["pair"], witnessed["verdict"] = pair, verdict
                return True
            return False

        try:
            full = find(
                pair_strategy(task, property_id), violates,
                settings=settings(max_examples=max_examples, database=None, deadline=None,
                                  phases=(Phase.generate,),
                                  suppress_health_check=(HealthCheck.too_slow,)),
                random=random.Random(seed + index * 1009),
            )
            full_verdict = witnessed["verdict"]
            compact, reduction_status = (reduce_pair(sql, task, full, budget, seed + 7919)
                                         if shrink else (full, "disabled"))
            # Store an independently checked compact verdict. If no execution
            # budget remains, keep the already-observed full witness instead.
            try:
                compact_verdict = evaluate_pair(sql, task, compact, budget)
            except ProbeBudgetExceeded:
                compact, compact_verdict = full, full_verdict
                reduction_status = "budget_exhausted_full_witness_retained"
            if compact_verdict.get("failure_type") != full_verdict.get("failure_type"):
                compact, compact_verdict = full, full_verdict
                reduction_status = "reduction_rejected"
            full_text = evidence_text(task, full, full_verdict)
            compact_text = evidence_text(task, compact, compact_verdict)
            report.update(
                status="violation", new_detection_beyond_static=bool(strong["accepted"]),
                counterexample={
                    "full": full.payload(), "compact": compact.payload(),
                    "full_verdict": full_verdict, "compact_verdict": compact_verdict,
                    "full_feedback": full_text, "compact_feedback": compact_text,
                    "full_feedback_chars": len(full_text), "compact_feedback_chars": len(compact_text),
                    "reduction_status": reduction_status,
                },
            )
            report["property_results"].append({"property_id": property_id, "status": "violation"})
            break
        except ProbeBudgetExceeded:
            report.update(status="budget_exhausted")
            break
        except (NoSuchExample, Unsatisfiable):
            status = execution_errors[0]["status"] if execution_errors else "no_counterexample_found"
            report["property_results"].append({"property_id": property_id, "status": status,
                                                "execution_errors": execution_errors})
            if execution_errors:
                report.update(status=status)
                break
    else:
        report["status"] = "no_counterexample_found"
    report["queries"] = budget.queries
    report["elapsed_seconds"] = round(time.monotonic() - budget.started, 6)
    report["acceptance"] = bool(strong["accepted"] and report["status"] == "no_counterexample_found")
    return report
