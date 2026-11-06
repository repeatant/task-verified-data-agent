"""Attribute execution disagreements without making a model an answer authority.

The reference and independent calculation are both hypotheses. This module
checks their observable relationships and caller-supplied examples; it cannot
prove that a quoted requirement entails a model's interpretation. Model-only
authorizations therefore remain explicitly conditional, never semantic proofs.
"""

from __future__ import annotations

import ast
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any

from dataverify.runtime.reporting import digest_json

from .discovery_runtime import (
    DiscoveryTask,
    execute_readonly,
    fixture_data,
    named_tables,
    rows_equal,
    validate_fixtures,
    validate_rows,
)
from .probe_runtime import ProbeData
from .reference_expression import calculate, parse_expression

_ROLES = ("candidate", "reference", "independent")
_MAX_FIXTURES = 12


def _canonical(data: ProbeData) -> dict:
    """Row order is outside the current task contract; primary keys are unique."""
    return ProbeData(tuple(sorted(data.orders)), tuple(sorted(data.lines))).to_dict()


def merge_fixtures(existing: list[dict], proposed: list[dict]) -> list[dict]:
    """Keep existing datasets and append distinct legal datasets, never evicting one."""
    if not isinstance(existing, list) or not isinstance(proposed, list):
        raise ValueError("AUDIT_FIXTURE_LIST")
    if len(existing) > _MAX_FIXTURES or len(proposed) > _MAX_FIXTURES:
        raise ValueError("AUDIT_FIXTURE_LIMIT")
    merged, known = [], set()
    for fixture in existing + proposed:
        validate_fixtures([fixture])
        canonical = _canonical(fixture_data(fixture["data"]))
        digest = digest_json(canonical)
        if digest in known:
            continue
        known.add(digest)
        # Preserve the original record, not only its canonical representation.
        merged.append(json.loads(json.dumps(fixture, allow_nan=False)))
    if len(merged) > _MAX_FIXTURES:
        raise ValueError("AUDIT_FIXTURE_LIMIT")
    return merged


def validate_confirmed_examples(examples: Any) -> list[dict]:
    """Validate examples asserted by the caller, not by a model response.

    The source string is provenance, not authentication. The caller is the
    trust boundary; generated plans must never populate this parameter.
    """
    if not isinstance(examples, (list, tuple)) or len(examples) > _MAX_FIXTURES:
        raise ValueError("CONFIRMED_EXAMPLE_LIMIT")
    normalized, known = [], {}
    for item in examples:
        if not isinstance(item, dict) or set(item) != {"name", "data", "expected", "source"}:
            raise ValueError("CONFIRMED_EXAMPLE_FIELDS")
        if type(item["name"]) is not str or not 1 <= len(item["name"]) <= 120:
            raise ValueError("CONFIRMED_EXAMPLE_NAME")
        if type(item["source"]) is not str or not 1 <= len(item["source"].strip()) <= 500:
            raise ValueError("CONFIRMED_EXAMPLE_SOURCE")
        data = _canonical(fixture_data(item["data"]))
        expected = validate_rows(item["expected"])
        digest = digest_json(data)
        if digest in known and not rows_equal(known[digest], expected):
            raise ValueError("CONFLICTING_CONFIRMED_EXAMPLES")
        known[digest] = expected
        normalized.append({**item, "data": data, "expected": expected})
    return json.loads(json.dumps(normalized, allow_nan=False))


def _reference_result(reference: str, data: ProbeData) -> dict:
    try:
        rows = validate_rows(calculate(reference, named_tables(data)))
        return {"status": "ok", "rows": rows}
    except ValueError as error:
        return {"status": "error", "error": str(error)}


def _candidate_result(sql: str, data: ProbeData) -> dict:
    try:
        return {"status": "ok", "rows": execute_readonly(sql, data)}
    except ValueError as error:
        return {"status": "error", "error": str(error)}


def _same(left: dict, right: dict) -> bool:
    return left["status"] == right["status"] == "ok" and rows_equal(left["rows"], right["rows"])


def _summary(checks: list[dict]) -> dict:
    summary = {
        "all_executed": bool(checks)
        and all(check[role]["status"] == "ok" for check in checks for role in _ROLES),
        "confirmed_count": sum("confirmed_expected" in check for check in checks),
    }
    for first, second in (
        ("reference", "independent"),
        ("candidate", "reference"),
        ("candidate", "independent"),
    ):
        summary[f"{first}_matches_{second}"] = bool(checks) and all(
            _same(check[first], check[second]) for check in checks
        )
    for role in _ROLES:
        summary[f"{role}_matches_confirmed"] = all(
            _same(check[role], {"status": "ok", "rows": check["confirmed_expected"]})
            for check in checks
            if "confirmed_expected" in check
        )
    summary["candidate_difference_ids"] = [
        check["check_id"]
        for check in checks
        if _same(check["reference"], check["independent"])
        and not _same(check["candidate"], check["independent"])
    ]
    summary["reference_difference_ids"] = [
        check["check_id"]
        for check in checks
        if _same(check["candidate"], check["independent"])
        and not _same(check["reference"], check["independent"])
    ]
    return summary


def inspect_dispute(
    task: DiscoveryTask,
    source: ProbeData,
    sql: str,
    reference_plan: dict,
    independent_plan: dict,
    fixtures: list[dict],
    confirmed_examples: list[dict] | tuple = (),
) -> dict:
    """Execute all three calculations on the same retained and confirmed data.

    No outcome is supplied by the adjudicating model. The evidence records
    calculation errors as well as differences. Its summary is a convenience;
    authorization recomputes relationships from the actual result records.
    """
    examples = validate_confirmed_examples(confirmed_examples)
    datasets: dict[str, dict] = {}

    def add(data: ProbeData, origin: dict, expected: list | None = None) -> None:
        canonical = _canonical(data)
        key = digest_json(canonical)
        record = datasets.setdefault(
            key, {"check_id": "data_" + key[:16], "data": canonical, "origins": []}
        )
        record["origins"].append(origin)
        if expected is not None:
            record["confirmed_expected"] = expected

    for fixture in merge_fixtures([], fixtures):
        add(fixture_data(fixture["data"]), {"kind": "retained", "name": fixture["name"]})
    add(source, {"kind": "bound_source", "name": "bound_source"})
    for example in examples:
        add(
            fixture_data(example["data"]),
            {"kind": "caller_confirmed", "name": example["name"], "source": example["source"]},
            example["expected"],
        )
    checks = []
    for item in datasets.values():
        data = ProbeData(item["data"]["orders"], item["data"]["lines"])
        checks.append(
            {
                **item,
                "candidate": _candidate_result(sql, data),
                "reference": _reference_result(reference_plan["reference"], data),
                "independent": _reference_result(independent_plan["reference"], data),
            }
        )
    return {
        "task_digest": digest_json(task.public_input()),
        "sql_digest": digest_json(sql),
        "reference_digest": digest_json(reference_plan),
        "independent_digest": digest_json(independent_plan),
        "checks": checks,
        "summary": _summary(checks),
        "scope": "Execution agreement and disagreement, not proof of natural-language intent.",
    }


def _validate_attribution(action: dict, task: DiscoveryTask) -> None:
    fields = {"action", "target", "reason", "basis", "checks"}
    if not isinstance(action, dict) or set(action) != fields or action["action"] != "attribute":
        raise ValueError("ATTRIBUTION_FIELDS")
    if action["target"] not in ("candidate", "reference", "ambiguous"):
        raise ValueError("ATTRIBUTION_TARGET")
    if type(action["reason"]) is not str or not 1 <= len(action["reason"]) <= 1000:
        raise ValueError("ATTRIBUTION_REASON")
    if not isinstance(action["checks"], list) or len(action["checks"]) > 25:
        raise ValueError("ATTRIBUTION_CHECKS")
    if any(type(x) is not str for x in action["checks"]) or len(set(action["checks"])) != len(
        action["checks"]
    ):
        raise ValueError("ATTRIBUTION_CHECKS")
    if not isinstance(action["basis"], list) or len(action["basis"]) > 8:
        raise ValueError("ATTRIBUTION_BASIS")
    for basis in action["basis"]:
        if not isinstance(basis, dict) or set(basis) != {"source", "quote"}:
            raise ValueError("ATTRIBUTION_BASIS")
        if type(basis["source"]) is not str:
            raise ValueError("ATTRIBUTION_BASIS")
        text = {"question": task.question, "definitions": task.definitions}.get(basis["source"])
        if (
            text is None
            or type(basis["quote"]) is not str
            or not basis["quote"].strip()
            or basis["quote"] not in text
        ):
            raise ValueError("ATTRIBUTION_BASIS_NOT_IN_SOURCE")
    if action["target"] != "ambiguous" and (not action["checks"] or not action["basis"]):
        raise ValueError("ATTRIBUTION_EVIDENCE_REQUIRED")


def authorize_decision(action: dict, task: DiscoveryTask, evidence: dict) -> dict:
    """Permit a bounded change only when cited execution evidence supports it.

    A valid quote establishes provenance, not entailment. An independent model
    plus execution supports a tentative action, not a guarantee that it is right.
    Conflicting caller-confirmed examples always block model-only authorization.
    """
    _validate_attribution(action, task)
    if evidence["task_digest"] != digest_json(task.public_input()):
        raise ValueError("ATTRIBUTION_TASK_MISMATCH")
    checks = {check["check_id"]: check for check in evidence["checks"]}
    if any(name not in checks for name in action["checks"]):
        raise ValueError("ATTRIBUTION_UNKNOWN_CHECK")
    decision = {
        "target": "ambiguous",
        "reason": action["reason"],
        "evidence_level": "insufficient",
        "checks": list(action["checks"]),
    }
    target = action["target"]
    if target == "ambiguous":
        return decision
    summary = _summary(list(checks.values()))
    counterpart = "reference" if target == "candidate" else "candidate"
    difference_ids = summary[f"{target}_difference_ids"]
    # Citing an unrelated matching check must not authorize a repair elsewhere.
    if not set(action["checks"]).issubset(difference_ids):
        return {**decision, "reason": "ATTRIBUTION_CITES_NO_RELEVANT_DISAGREEMENT"}
    if not summary[f"{counterpart}_matches_independent"]:
        return {**decision, "reason": "SUPPORTING_CALCULATIONS_DISAGREE"}
    if (
        not summary[f"{counterpart}_matches_confirmed"]
        or not summary["independent_matches_confirmed"]
    ):
        return {**decision, "reason": "SUPPORTING_CALCULATIONS_VIOLATE_CONFIRMED_EXAMPLE"}
    confirmed_contradiction = any(
        "confirmed_expected" in checks[name]
        and not _same(
            checks[name][target], {"status": "ok", "rows": checks[name]["confirmed_expected"]}
        )
        for name in action["checks"]
    )
    return {
        **decision,
        "target": target,
        "evidence_level": "caller_confirmed_example"
        if confirmed_contradiction
        else "model_supported_not_proven",
    }


class _SearchLimit(Exception):
    """Keep generation and shrinking within a local wall-clock/execution budget."""


def _search_literals(sql: str, references: tuple[str, str]) -> tuple[list[str], list[str]]:
    """Use task-code constants as generation hints, never as expected answers."""
    values = set(re.findall(r"['\"]([^'\"]{1,80})['\"]", sql))
    for reference in references:
        for node in ast.walk(parse_expression(reference)):
            if isinstance(node, ast.Constant) and type(node.value) is str:
                values.add(node.value)
    dates = {"2000-01-01T00:00:00"}
    for value in values:
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", value):
            continue
        try:
            date = datetime.fromisoformat(value)
            for delta in (-1, 0, 1):
                dates.add((date + timedelta(seconds=delta)).strftime("%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            continue
    channels = ["web", "store", "other"]
    ignored = {"orders", "order_lines", "settled", "pending", "cancelled"}
    for value in sorted(values):
        if value not in ignored and re.fullmatch(r"[A-Za-z][A-Za-z_-]{0,15}", value):
            if value not in channels:
                channels.append(value)
    return sorted(dates)[:15], channels[:12]


def find_distinguishing_fixture(
    sql: str, ref1: str | dict, ref2: str | dict, max_examples: int = 40
) -> dict | None:
    """Search legal data on which the three programs differ; shrink if possible.

    The schema and resource limits are fixed, not business-error templates.
    None means no witness within this bounded search, never equivalence. A
    witness proves a disagreement only; attribution remains a separate step.
    """
    from hypothesis import HealthCheck, Phase, find, settings
    from hypothesis import strategies as st
    from hypothesis.errors import NoSuchExample, Unsatisfiable

    if type(max_examples) is not int or not 1 <= max_examples <= 200:
        raise ValueError("AUDIT_SEARCH_EXAMPLE_LIMIT")
    refs = tuple(ref["reference"] if isinstance(ref, dict) else ref for ref in (ref1, ref2))
    dates, channels = _search_literals(sql, refs)

    @st.composite
    def legal_data(draw):
        order_count = draw(st.integers(1, 8))
        amounts = st.integers(-1000, 1000)
        orders = tuple(
            (key, draw(amounts), draw(st.one_of(st.none(), st.sampled_from(channels))))
            for key in range(1, order_count + 1)
        )
        raw_lines = draw(
            st.lists(
                st.tuples(
                    st.integers(1, order_count),
                    amounts,
                    st.sampled_from(["settled", "pending", "cancelled"]),
                    st.sampled_from(dates),
                ),
                min_size=0,
                max_size=12,
            )
        )
        return ProbeData(orders, tuple((i + 1, *line) for i, line in enumerate(raw_lines)))

    started, executions, witness = time.monotonic(), 0, None

    def differs(data: ProbeData) -> bool:
        nonlocal executions, witness
        if executions >= max_examples * 12 or time.monotonic() - started > 3.0:
            raise _SearchLimit
        executions += 1
        results = [_candidate_result(sql, data), *(_reference_result(ref, data) for ref in refs)]
        valid = [result for result in results if result["status"] == "ok"]
        different = bool(valid) and (
            len(valid) != 3 or any(not _same(valid[0], result) for result in valid[1:])
        )
        if different:
            witness = data
        return different

    try:
        witness = find(
            legal_data(),
            differs,
            settings=settings(
                max_examples=max_examples,
                database=None,
                derandomize=True,
                deadline=None,
                phases=(Phase.generate, Phase.shrink),
                suppress_health_check=(HealthCheck.too_slow,),
            ),
        )
    except (NoSuchExample, Unsatisfiable, _SearchLimit):
        pass
    if witness is None:
        return None
    return {"name": "searched_disagreement", "data": witness.to_dict()}
