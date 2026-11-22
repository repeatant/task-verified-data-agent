"""Forward engineering cases for the paired termination-policy experiment.

Three authored requirements on the existing orders schema, fixed before either
arm ran. All arms receive the same one-row preview and synthetic SQL data. The
hidden fixtures and imperative scorer stay entirely on the evaluation side.
This is a scoped small challenge, not an external blind benchmark.
"""

from collections import defaultdict

from sqlglot.errors import SqlglotError

from dataverify.evaluation.dispute_recovery import DisputeCase
from dataverify.evaluation.expanded_cases import ExpandedCase
from dataverify.evaluation.expanded_scoring import _outcomes
from dataverify.evaluation.query_ablation import full_source
from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import (
    DiscoveryTask,
    execute_readonly,
    named_tables,
    rows_equal,
)
from dataverify.verification.probe_runtime import ProbeData
from dataverify.verification.reference_expression import calculate

DATE = "2026-09-15T00:00:00"
PREVIEW = ProbeData(((1, 400, "web"),), ((1, 1, 0, "pending", DATE),))
DEFINITIONS = (
    "Use the complete orders and order_lines tables. Every line belongs to one order. "
    "Amounts are signed integer minor units. The only statuses are pending, cancelled "
    "and settled. Return one integer; the count of an empty population is zero."
)


def _programs():
    question = (
        "Count the orders that have at least one pending line and no settled lines. "
        "A cancelled line neither qualifies nor disqualifies the order."
    )
    sql = (
        "SELECT COUNT(*) FROM orders o WHERE EXISTS (SELECT 1 FROM order_lines l "
        "WHERE l.order_id=o.order_id AND l.status='pending') AND NOT EXISTS "
        "(SELECT 1 FROM order_lines l WHERE l.order_id=o.order_id AND l.status='settled')"
    )
    ref = (
        "pending=set([r['order_id'] for r in tables['order_lines'] if r['status']=='pending'])\n"
        "settled=set([r['order_id'] for r in tables['order_lines'] if r['status']=='settled'])\n"
        "[[len([i for i in pending if i not in settled])]]"
    )
    yield (
        "awaiting_settlement",
        "correct",
        question,
        sql,
        sql.replace(" AND NOT EXISTS ", " AND EXISTS "),
        ref,
        ref.replace("i not in settled", "i in settled"),
    )

    question = (
        "How many individual pending lines have a strictly positive charge on orders "
        "whose channel is exactly web? A zero amount is not a charge; include each qualifying line."
    )
    sql = (
        "SELECT COUNT(*) FROM order_lines l JOIN orders o ON o.order_id=l.order_id "
        "WHERE o.channel='web' AND l.status='pending' AND l.amount>0"
    )
    ref = (
        "web=set([o['order_id'] for o in tables['orders'] if o['channel']=='web'])\n"
        "[[len([r for r in tables['order_lines'] if r['order_id'] in web "
        "and r['status']=='pending' and r['amount']>0])]]"
    )
    yield (
        "pending_web_charges",
        "sql_wrong",
        question,
        sql,
        sql.replace("amount>0", "amount>=0"),
        ref,
        ref.replace("]>0", "]>=0"),
    )

    question = (
        "Count orders that have at least two different line statuses across their lines. "
        "Two lines with the same status do not establish two different statuses."
    )
    sql = (
        "SELECT COUNT(*) FROM (SELECT order_id FROM order_lines GROUP BY order_id "
        "HAVING COUNT(DISTINCT status)>=2)"
    )
    ref = (
        "lines=tables['order_lines']\n"
        "orders=set([r['order_id'] for r in lines])\n"
        "[[len([i for i in orders if len(set([r['status'] for r in lines if r['order_id']==i]))>=2])]]"
    )
    yield (
        "mixed_line_status",
        "all_wrong",
        question,
        sql,
        sql.replace("DISTINCT status", "*"),
        ref,
        ref.replace(
            "len(set([r['status'] for r in lines if r['order_id']==i]))",
            "len([r['status'] for r in lines if r['order_id']==i])",
        ),
    )


def study_cases():
    result = []
    for index, (family, condition, question, good, bad, reference, faulty) in enumerate(
        _programs(), 1
    ):
        case_id = f"terminal-{index:02}"
        task = DiscoveryTask(
            "forward_" + digest_json([question, DEFINITIONS])[:12], question, DEFINITIONS
        )
        base = ExpandedCase(
            case_id,
            family,
            family,
            "correlated_wrong" if condition == "all_wrong" else condition,
            task,
            PREVIEW,
            good,
            bad,
            reference,
            faulty,
            1,
        )
        result.append(DisputeCase(case_id, base, condition))
    return result


def _fixture(orders, lines):
    return ProbeData(
        tuple((i + 1, 400, channel) for i, channel in enumerate(orders)),
        tuple(
            (i + 1, order, amount, status, DATE) for i, (order, amount, status) in enumerate(lines)
        ),
    )


def scoring_sources():
    """Identical fixtures across questions; their values are never sent as diagnosis hints."""
    return {
        "preview": PREVIEW,
        "full_source": full_source(PREVIEW),
        "empty": _fixture([], []),
        "bare_orders": _fixture(["web", "store", None], []),
        "duplicates_only": _fixture(["web", "store"], [(1, 10, "pending"), (1, 10, "pending")]),
        "mixed_and_inactive": _fixture(
            ["web", "web", None, "store"],
            [
                (1, 2, "pending"),
                (1, 2, "settled"),
                (2, 0, "pending"),
                (2, 5, "cancelled"),
                (3, 2, "pending"),
                (4, -7, "pending"),
                (4, 8, "pending"),
            ],
        ),
        "sign_and_channel": _fixture(
            ["web", None, "store", "web"],
            [
                (1, -4, "pending"),
                (1, 4, "pending"),
                (1, 4, "pending"),
                (2, 4, "pending"),
                (3, 4, "pending"),
                (4, 4, "cancelled"),
                (4, 1, "settled"),
            ],
        ),
    }


def expected(case, data):
    """Business definitions written as one pass over raw rows; no SQL or reference evaluation."""
    channels = {order_id: channel for order_id, _, channel in data.orders}
    seen_statuses = defaultdict(set)
    charges = 0
    for _, order_id, amount, status, _ in data.lines:
        seen_statuses[order_id].add(status)
        if status == "pending" and amount > 0 and channels[order_id] == "web":
            charges += 1
    family = case.base.family
    if family == "awaiting_settlement":
        return [
            [
                sum(
                    "pending" in states and "settled" not in states
                    for states in seen_statuses.values()
                )
            ]
        ]
    if family == "pending_web_charges":
        return [[charges]]
    if family == "mixed_line_status":
        return [[sum(len(states) >= 2 for states in seen_statuses.values())]]
    raise ValueError("Unregistered forward challenge")


def frozen_cases():
    fixtures = scoring_sources()
    return [
        {
            "id": case.case_id,
            "requirement": digest_json(case.base.task.public_input()),
            "condition": case.condition,
            "initial_sql": digest_json(case.initial_sql),
            "reference": digest_json(case.reference_plan),
            "auditor": digest_json(case.auditor_plan),
            "preview": digest_json(PREVIEW.to_dict()),
            "full_source": digest_json(fixtures["full_source"].to_dict()),
            "grading": digest_json(
                {k: [d.to_dict(), expected(case, d)] for k, d in fixtures.items()}
            ),
        }
        for case in study_cases()
    ]


def grade(case, record):
    fixtures = scoring_sources()

    def check(program, is_sql):
        present = isinstance(program, str) and bool(program.strip())
        failures = []
        if present:
            for name, data in fixtures.items():
                try:
                    actual = (
                        execute_readonly(program, data)
                        if is_sql
                        else calculate(program, named_tables(data))
                    )
                    if not rows_equal(actual, expected(case, data)):
                        failures.append(name)
                except (ValueError, TypeError, KeyError, ArithmeticError, SqlglotError):
                    failures.append(name)
        return {
            "present": present,
            "correct": not failures if present else None,
            "passed": len(fixtures) - len(failures) if present else 0,
            "total": len(fixtures),
            "failures": failures,
        }

    candidate = record.get("candidate") or {}
    publication = record.get("publication") or {}
    reference = check((record.get("plan") or {}).get("reference"), False)
    auditor = check((record.get("audit_plan") or {}).get("reference"), False)
    sql = publication.get("sql")
    consistent = False
    if sql and isinstance(publication.get("rows"), list):
        try:
            consistent = rows_equal(
                execute_readonly(sql, fixtures["full_source"]), publication["rows"]
            )
        except (ValueError, TypeError, KeyError, ArithmeticError, SqlglotError):
            pass
    result = _outcomes(
        record,
        check(case.initial_sql, True),
        check(candidate.get("sql"), True),
        check(sql, True),
        publication_rows_consistent=consistent,
    )
    both = reference["present"] and auditor["present"]
    result.update(
        reference=reference,
        auditor=auditor,
        joint_evaluable=both,
        joint_correct_delivery=(
            result["correct_delivery"] and reference["correct"] and auditor["correct"]
        )
        if both
        else None,
    )
    return result
