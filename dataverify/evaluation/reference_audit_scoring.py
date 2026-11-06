"""Post-run scoring of the SQL and BOTH model-proposed reference calculations.

Gold SQL and these development datasets remain evaluation-side. They must never
be passed to an agent, used as runtime fallback, or labeled independent holdout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dataverify.evaluation.discovery_cases import data
from dataverify.evaluation.reference_audit_cases import AuditScenario
from dataverify.verification.discovery_runtime import (
    execute_readonly,
    named_tables,
    rows_equal,
    validate_rows,
)
from dataverify.verification.probe_runtime import ProbeData
from dataverify.verification.reference_expression import calculate


def _scoring_data(source: ProbeData) -> dict[str, ProbeData]:
    """Exercise grain, precision, sign, empty input and filtering independently."""
    mixed = ProbeData(
        ((1, 400, "web"), (2, 500, "store"), (3, 200, None)),
        (
            (1, 1, 25, "settled", "2026-09-01T00:00:00"),
            (2, 1, -3, "settled", "2026-09-15T00:00:00"),
            (3, 2, 19, "settled", "2026-09-15T00:00:00"),
            (4, 3, 10, "settled", "2026-09-30T23:59:59"),
            (5, 1, 999, "pending", "2026-09-15T00:00:00"),
            (6, 2, 777, "settled", "2026-10-01T00:00:00"),
            (7, 3, 888, "settled", "2026-08-31T23:59:59"),
        ),
    )
    return {
        "source": source,
        "order_grain": data([100, 50, 50], order_ids=[1, 2, 2]),
        "empty": ProbeData((), ()),
        "recurring_decimal": data([10, 10, 11]),
        "half_tie": data([1, 0, 0, 0, 0, 0, 0, 0]),
        "refunds": data([100, -30, -70, -1], order_ids=[1, 1, 2, 2]),
        "mixed_boundaries": mixed,
    }


def _capture(operation: Callable[[], list]) -> dict[str, Any]:
    try:
        return {"status": "ok", "rows": operation()}
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError) as error:
        return {"status": "error", "error": str(error)}


def _reference(plan: Any, dataset: ProbeData) -> list:
    if not isinstance(plan, dict) or not isinstance(plan.get("reference"), str):
        raise ValueError("SCORING_REFERENCE_MISSING")
    return validate_rows(calculate(plan["reference"], named_tables(dataset)))


def _candidate(record: dict) -> tuple[str | None, str]:
    publication = record.get("publication")
    if isinstance(publication, dict) and isinstance(publication.get("sql"), str):
        origin = (
            "published"
            if publication.get("status", record.get("status")) == "reviewed"
            else "unpublished"
        )
        return publication["sql"], "publication_" + origin
    candidate = record.get("candidate")
    if isinstance(candidate, dict) and isinstance(candidate.get("sql"), str):
        return candidate["sql"], "last_candidate_unpublished"
    return None, "missing"


def grade_audit_run(scenario: AuditScenario, record: dict) -> dict:
    """Diagnose the last candidate and references without changing the run record.

    The scenario's final_sql is the author's scoring assumption. A source-level
    disagreement with the authored expected_rows invalidates that scoring setup;
    matching a mislabeled gold program must not produce an overall pass.
    """
    sql, origin = _candidate(record)
    checks = []
    for name, dataset in _scoring_data(scenario.source).items():
        gold = _capture(lambda: execute_readonly(scenario.final_sql, dataset))
        outputs = {
            "candidate": _capture(lambda: execute_readonly(sql, dataset)),
            "reference": _capture(lambda: _reference(record.get("plan"), dataset)),
            "independent": _capture(lambda: _reference(record.get("audit_plan"), dataset)),
        }
        for result in outputs.values():
            result["match"] = gold["status"] == result["status"] == "ok" and rows_equal(
                result["rows"], gold["rows"]
            )
        checks.append({"name": name, "gold": gold, **outputs})
    source_gold = checks[0]["gold"]
    gold_source_consistent = source_gold["status"] == "ok" and rows_equal(
        source_gold["rows"], scenario.expected_rows
    )
    valid_gold = gold_source_consistent and all(x["gold"]["status"] == "ok" for x in checks)
    return {
        "scope": "Seven authored development datasets, not independent holdout or production accuracy.",
        "gold_source": "AuditScenario.final_sql; evaluation-only author assumption, not model truth.",
        "scenario": scenario.name,
        "run_status": record.get("status"),
        "candidate_origin": origin,
        "gold_source_consistent": gold_source_consistent,
        "gold_valid_on_scoring_data": valid_gold,
        "checks": checks,
        "candidate_passes_all": valid_gold and all(x["candidate"]["match"] for x in checks),
        "reference_passes_all": valid_gold and all(x["reference"]["match"] for x in checks),
        "independent_passes_all": valid_gold and all(x["independent"]["match"] for x in checks),
    }
