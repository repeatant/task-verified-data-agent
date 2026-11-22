"""Frozen fault-injection cases and scoring-only arithmetic for v1 experiments.

Sixteen controlled scenarios reuse FOUR disclosed development tasks. Four
natural-generation runs can be added by the runner; neither twenty scenarios
nor the per-run scoring datasets are twenty independent business requirements.
No scoring dataset, expected output, family dispatch or gold SQL is an input to
the Agent. Freeze this module and the case manifest before making model calls.
"""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dataverify.evaluation.discovery_cases import ELIGIBLE, get_discovery_cases
from dataverify.verification.discovery_runtime import (
    DiscoveryTask,
    execute_readonly,
    named_tables,
    rows_equal,
    validate_rows,
)
from dataverify.verification.probe_runtime import ProbeData
from dataverify.verification.reference_expression import calculate

SCORING_SEED = 20261107
RANDOM_DATASETS = 24
CONDITIONS = ("sql_wrong", "reference_wrong", "correct", "correlated_wrong")
FAMILIES = ("order_average", "web_scope", "distinct_orders", "money_units")


@dataclass(frozen=True)
class FrozenCase:
    """Only task/source and explicit initial injections enter a model run."""

    case_id: str
    family: str
    condition: str
    task: DiscoveryTask
    source: ProbeData
    initial_sql: str | None
    initial_plan: dict | None


def _wrong_reference(family: str) -> str:
    """Evaluation-only faults equivalent to the corresponding authored bad SQL."""
    values = f"[r['amount'] for r in tables['order_lines'] if {ELIGIBLE}]"
    if family == "order_average":
        return f"values={values}\n[[sum(values)/len(values) if values else 0]]"
    if family == "distinct_orders":
        return f"[[len({values})]]"
    if family in {"web_scope", "money_units"}:
        return f"[[sum({values})]]"
    raise ValueError("UNKNOWN_SCORING_FAMILY")


def get_frozen_cases() -> list[FrozenCase]:
    """Cross four tasks with four controlled conditions without changing prompts."""
    scenarios = []
    for case in get_discovery_cases():
        for condition in CONDITIONS:
            plan = deepcopy(case.offline_plan())
            if condition in {"reference_wrong", "correlated_wrong"}:
                plan["reference"] = _wrong_reference(case.name)
            scenarios.append(
                FrozenCase(
                    case_id=f"{case.name}__{condition}",
                    family=case.name,
                    condition=condition,
                    task=case.task,
                    source=case.source,
                    initial_sql=(
                        case.bad_sql
                        if condition in {"sql_wrong", "correlated_wrong"}
                        else case.good_sql
                    ),
                    initial_plan=plan,
                )
            )
    return scenarios


def expected_rows(family: str, dataset: ProbeData) -> list:
    """Hand-written scoring oracle, independent of SQL and reference DSL.

    Use parsed timestamps and per-order accumulation rather than the DSL's string
    filter and ratio expression. The assumptions are the four published questions;
    this scorer is an author-maintained oracle, not a formal correctness proof.
    """
    if family not in FAMILIES:
        raise ValueError("UNKNOWN_SCORING_FAMILY")
    start, end = datetime(2026, 9, 1), datetime(2026, 10, 1)
    order_totals: dict[int, int] = defaultdict(int)
    for _, order_id, amount, status, timestamp in dataset.lines:
        if status == "settled" and start <= datetime.fromisoformat(timestamp) < end:
            order_totals[order_id] += amount
    if family == "order_average":
        value = sum(order_totals.values()) / len(order_totals) if order_totals else 0
    elif family == "distinct_orders":
        value = len(order_totals)
    elif family == "web_scope":
        value = 0
        for order_id, _, channel in dataset.orders:
            if channel == "web":
                value += order_totals[order_id] if order_id in order_totals else 0
    else:
        value = sum(order_totals.values()) / 100
    return [[value]]


def _authored_datasets() -> dict[str, ProbeData]:
    """Six compact scoring inputs with independently inspectable arithmetic."""
    september = "2026-09-15T00:00:00"
    return {
        "empty": ProbeData((), ()),
        "orders_without_lines": ProbeData(((1, 900, "web"), (2, 300, None)), ()),
        "time_and_status_boundaries": ProbeData(
            ((1, 500, "web"), (2, 800, "store"), (3, 700, None)),
            (
                (1, 1, 13, "settled", "2026-09-01T00:00:00"),
                (2, 2, 29, "settled", "2026-09-30T23:59:59"),
                (3, 1, 1001, "settled", "2026-08-31T23:59:59"),
                (4, 3, 2003, "settled", "2026-10-01T00:00:00"),
                (5, 1, 3007, "pending", september),
                (6, 2, 4001, "cancelled", september),
            ),
        ),
        "refunds_and_zero_net_order": ProbeData(
            ((1, 999, "web"), (2, 888, "store")),
            (
                (1, 1, 101, "settled", september),
                (2, 1, -101, "settled", september),
                (3, 2, -39, "settled", september),
            ),
        ),
        "join_and_grain": ProbeData(
            ((1, 777, "web"), (2, 555, "store"), (3, 333, None), (4, 222, "WEB")),
            (
                (1, 1, 103, "settled", september),
                (2, 2, 37, "settled", september),
                (3, 2, 49, "settled", september),
                (4, 3, 17, "settled", september),
                (5, 4, 11, "settled", september),
            ),
        ),
        "noninteger_average_and_units": ProbeData(
            ((1, 999, "web"), (2, 999, "web"), (3, 999, "web")),
            (
                (1, 1, 10, "settled", september),
                (2, 2, 10, "settled", september),
                (3, 3, 11, "settled", september),
            ),
        ),
    }


def scoring_datasets(case: FrozenCase) -> dict[str, ProbeData]:
    """Return source + six authored + 24 seeded inputs; never send to the Agent.

    The same additional datasets are used across all conditions/families. Variation
    in input rows increases regression coverage, not the independent sample size.
    """
    result = {"source": case.source, **_authored_datasets()}
    rng = random.Random(SCORING_SEED)
    times = (
        "2026-08-31T23:59:59",
        "2026-09-01T00:00:00",
        "2026-09-12T15:23:05",
        "2026-09-30T23:59:59",
        "2026-10-01T00:00:00",
    )
    for index in range(RANDOM_DATASETS):
        orders = tuple(
            (key, rng.randint(-500, 2500), rng.choice(("web", "store", "WEB", None)))
            for key in range(1, rng.randint(1, 6) + 1)
        )
        lines = tuple(
            (
                line_id,
                rng.choice(orders)[0],
                rng.randint(-700, 1800),
                rng.choice(("settled", "settled", "pending", "cancelled")),
                rng.choice(times),
            )
            for line_id in range(1, rng.randint(0, 24) + 1)
        )
        result[f"seeded_{index:02d}"] = ProbeData(orders, lines)
    return result


def _program_grade(family: str, datasets: dict[str, ProbeData], program: Any, *, sql: bool) -> dict:
    failures, errors = [], []
    for name, dataset in datasets.items():
        try:
            actual = (
                execute_readonly(program, dataset)
                if sql
                else validate_rows(calculate(program, named_tables(dataset)))
            )
            if not rows_equal(actual, expected_rows(family, dataset)):
                failures.append(name)
        except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError):
            failures.append(name)
            errors.append(name)
    return {
        "passed": len(datasets) - len(failures),
        "total": len(datasets),
        "passes_all": not failures,
        "failed_datasets": failures,
        "error_datasets": errors,
    }


def _reference(plan: Any) -> str | None:
    return plan.get("reference") if isinstance(plan, dict) else None


def grade_frozen_run(case: FrozenCase, record: dict) -> dict:
    """Score all final programs, preserving abstentions and wrong publications.

    Candidate grading prefers the actual published SQL. A nonpublished final
    candidate can pass diagnostic checks but never counts as correct delivery.
    SQL protection is semantic on these datasets, not byte-for-byte equality.
    """
    datasets = scoring_datasets(case)
    publication = record.get("publication")
    publication = publication if isinstance(publication, dict) else {}
    candidate = record.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    published_sql = publication.get("sql")
    sql = published_sql if isinstance(published_sql, str) else candidate.get("sql")
    candidate_grade = _program_grade(case.family, datasets, sql, sql=True)
    reference_grade = _program_grade(
        case.family, datasets, _reference(record.get("plan")), sql=False
    )
    audit_grade = _program_grade(
        case.family, datasets, _reference(record.get("audit_plan")), sql=False
    )
    initial_grade = (
        _program_grade(case.family, datasets, case.initial_sql, sql=True)
        if case.initial_sql is not None
        else None
    )
    status = record.get("status", "missing")
    published = status == "reviewed" and publication.get("status", status) == "reviewed"
    try:
        source_rows_match = rows_equal(
            validate_rows(publication.get("rows")), expected_rows(case.family, case.source)
        )
    except (ValueError, TypeError):
        source_rows_match = False
    correct_delivery = (
        published
        and isinstance(published_sql, str)
        and candidate_grade["passes_all"]
        and source_rows_match
    )
    return {
        "case_id": case.case_id,
        "family": case.family,
        "condition": case.condition,
        "scope": "Four disclosed development tasks; seeded scoring coverage is not independent task count.",
        "oracle": "Hand-written arithmetic over parsed dates and per-order totals; scoring only.",
        "run_status": status,
        "scoring_seed": SCORING_SEED,
        "scoring_dataset_count": len(datasets),
        "candidate_origin": "publication" if isinstance(published_sql, str) else "last_candidate",
        "candidate": candidate_grade,
        "reference": reference_grade,
        "audit": audit_grade,
        "initial_candidate": initial_grade,
        "source_rows_match": source_rows_match,
        "correct_delivery": correct_delivery,
        "joint_validated_delivery": correct_delivery
        and reference_grade["passes_all"]
        and audit_grade["passes_all"],
        "wrong_release": published and not correct_delivery,
        "correct_sql_damaged": bool(
            initial_grade
            and initial_grade["passes_all"]
            and isinstance(sql, str)
            and not candidate_grade["passes_all"]
        ),
        "initial_correct_sql_not_delivered": bool(
            initial_grade and initial_grade["passes_all"] and not correct_delivery
        ),
    }
