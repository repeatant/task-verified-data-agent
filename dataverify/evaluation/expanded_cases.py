"""Controlled semantic faults; gold computations stay in the evaluation layer.

100 trajectories are 20 parent tasks x five conditions, NOT 100 independent
requirements. Two parameter variants of each of ten mechanisms share a family.
These author-generated tasks complement, never replace, a public benchmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from dataverify.runtime.reporting import digest_json
from dataverify.verification.discovery_runtime import DiscoveryTask, execute_readonly, rows_equal
from dataverify.verification.probe_runtime import ProbeData

CONDITIONS = ("sql_wrong", "reference_wrong", "correlated_wrong", "auditor_wrong", "correct")
FAMILIES = (
    "upper_boundary",
    "lower_boundary",
    "refund_sign",
    "status_scope",
    "zero_activity",
    "null_channel",
    "duplicate_amounts",
    "aggregate_threshold",
    "population_denominator",
    "absolute_net",
)


@dataclass(frozen=True)
class ExpandedCase:
    case_id: str
    parent_id: str
    family: str
    condition: str
    task: DiscoveryTask
    source: ProbeData
    good_sql: str
    bad_sql: str
    good_reference: str
    bad_reference: str
    variant: int

    @property
    def initial_sql(self):
        return (
            self.bad_sql if self.condition in {"sql_wrong", "correlated_wrong"} else self.good_sql
        )

    def initial_plan(self, *, auditor=False):
        wrong = (
            self.condition == "auditor_wrong"
            if auditor
            else self.condition in {"reference_wrong", "correlated_wrong"}
        )
        return {
            "action": "plan",
            "basis": [{"source": "question", "quote": self.task.question}],
            "assumptions": [],
            "reference": self.bad_reference if wrong else self.good_reference,
            # Deliberately ordinary source: an injected fault may be invisible here.
            "fixtures": [{"name": "initial_example", "data": self.source.to_dict()}],
        }


def _definition(family, variant):
    month = 8 + variant
    start, end = f"2026-{month:02d}-01T00:00:00", f"2026-{month + 1:02d}-01T00:00:00"
    cutoff = 100 * variant
    eligible = f"status='settled' AND settled_at >= '{start}' AND settled_at < '{end}'"
    py = f"r['status']=='settled' and '{start}' <= r['settled_at'] < '{end}'"
    selected = f"[r for r in tables['order_lines'] if {py}]"
    pre = f"rows={selected}\n"
    total_sql = f"SELECT COALESCE(SUM(amount),0) FROM order_lines WHERE {eligible}"
    total_ref = pre + "[[sum([r['amount'] for r in rows])]]"
    descriptions = {
        "upper_boundary": (
            f"Return net settled line revenue from {start} inclusive to {end} exclusive. Include refunds.",
            total_sql,
            total_sql.replace(f"< '{end}'", f"<= '{end}'"),
            total_ref,
            total_ref.replace(f"< '{end}'", f"<= '{end}'"),
        ),
        "lower_boundary": (
            f"Return net settled line revenue from {start} inclusive to {end} exclusive. Include refunds.",
            total_sql,
            total_sql.replace(f">= '{start}'", f"> '{start}'"),
            total_ref,
            total_ref.replace(f"'{start}' <=", f"'{start}' <"),
        ),
        "refund_sign": (
            "Return the net revenue of settled lines in the specified interval, including negative refund amounts.",
            total_sql,
            total_sql + " AND amount>=0",
            total_ref,
            pre + "[[sum([r['amount'] for r in rows if r['amount']>=0])]]",
        ),
        "status_scope": (
            "Return revenue of settled lines only in the specified interval. Pending and cancelled lines contribute nothing.",
            total_sql,
            total_sql.replace("status='settled'", "status!='pending'"),
            total_ref,
            total_ref.replace("r['status']=='settled'", "r['status']!='pending'"),
        ),
        "zero_activity": (
            "Count every order, including orders having no lines. Count each order once.",
            "SELECT COUNT(*) FROM orders",
            "SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN order_lines l ON o.order_id=l.order_id",
            "[[len(tables['orders'])]]",
            "[[len(set([r['order_id'] for r in tables['order_lines']]))]]",
        ),
        "null_channel": (
            "Return settled line revenue in the specified interval for orders not explicitly marked web. A NULL channel is not web and must be included.",
            f"SELECT COALESCE(SUM(l.amount),0) FROM order_lines l JOIN orders o ON l.order_id=o.order_id WHERE {eligible} AND (o.channel!='web' OR o.channel IS NULL)",
            f"SELECT COALESCE(SUM(l.amount),0) FROM order_lines l JOIN orders o ON l.order_id=o.order_id WHERE {eligible} AND o.channel!='web'",
            pre
            + "ids=[o['order_id'] for o in tables['orders'] if o['channel']!='web']\n[[sum([r['amount'] for r in rows if r['order_id'] in ids])]]",
            pre
            + "ids=[o['order_id'] for o in tables['orders'] if o['channel'] is not None and o['channel']!='web']\n[[sum([r['amount'] for r in rows if r['order_id'] in ids])]]",
        ),
        "duplicate_amounts": (
            "Count settled line records in the specified interval. Separate records with the same amount are counted separately.",
            f"SELECT COUNT(*) FROM order_lines WHERE {eligible}",
            f"SELECT COUNT(DISTINCT amount) FROM order_lines WHERE {eligible}",
            pre + "[[len(rows)]]",
            pre + "[[len(set([r['amount'] for r in rows]))]]",
        ),
        "aggregate_threshold": (
            f"Count orders whose net settled line revenue in the specified interval exceeds {cutoff}. Apply the threshold after summing all eligible lines per order.",
            f"SELECT COUNT(*) FROM (SELECT order_id FROM order_lines WHERE {eligible} GROUP BY order_id HAVING SUM(amount)>{cutoff})",
            f"SELECT COUNT(DISTINCT order_id) FROM order_lines WHERE {eligible} AND amount>{cutoff}",
            pre
            + f"ids=set([r['order_id'] for r in rows])\n[[len([i for i in ids if sum([r['amount'] for r in rows if r['order_id']==i])>{cutoff}])]]",
            pre + f"[[len(set([r['order_id'] for r in rows if r['amount']>{cutoff}]))]]",
        ),
        "population_denominator": (
            "Compute net settled line revenue in the specified interval per every order in the orders table, including zero-activity orders in the denominator. Return zero if there are no orders. Do not round.",
            f"SELECT COALESCE((SELECT SUM(amount)*1.0 FROM order_lines WHERE {eligible})/NULLIF((SELECT COUNT(*) FROM orders),0),0)",
            f"SELECT COALESCE(SUM(amount)*1.0/NULLIF(COUNT(DISTINCT order_id),0),0) FROM order_lines WHERE {eligible}",
            pre
            + "[[sum([r['amount'] for r in rows])/len(tables['orders']) if tables['orders'] else 0]]",
            pre
            + "ids=set([r['order_id'] for r in rows])\n[[sum([r['amount'] for r in rows])/len(ids) if ids else 0]]",
        ),
        "absolute_net": (
            "Return the absolute value of total net settled line revenue in the specified interval. Refunds offset positive amounts before taking the absolute value.",
            f"SELECT ABS(COALESCE(SUM(amount),0)) FROM order_lines WHERE {eligible}",
            f"SELECT COALESCE(SUM(ABS(amount)),0) FROM order_lines WHERE {eligible}",
            pre + "[[abs(sum([r['amount'] for r in rows]))]]",
            pre + "[[sum([abs(r['amount']) for r in rows])]]",
        ),
    }
    return descriptions[family], start, end


def expanded_cases():
    cases = []
    for family in FAMILIES:
        for variant in (1, 2):
            (question, good_sql, bad_sql, good_ref, bad_ref), start, end = _definition(
                family, variant
            )
            parent = f"{family}_{variant}"
            source = ProbeData(
                ((1, 999, "store"),), ((1, 1, 11, "settled", start[:8] + "15T00:00:00"),)
            )
            definitions = f"Interval is [{start}, {end}). Amounts are integer minor units. Orders have unique positive keys and line foreign keys are valid. Use line amounts, not orders.total. Outputs are unordered; zero is the empty-sum convention."
            task = DiscoveryTask(
                "control_" + digest_json([question, definitions])[:12], question, definitions
            )
            for condition in CONDITIONS:
                cases.append(
                    ExpandedCase(
                        parent + "__" + condition,
                        parent,
                        family,
                        condition,
                        task,
                        source,
                        good_sql,
                        bad_sql,
                        good_ref,
                        bad_ref,
                        variant,
                    )
                )
    return cases


def scoring_data(case):
    """25 held-out datasets per parent; seeds/expected answers never enter prompts."""
    _, start, end = _definition(case.family, case.variant)
    middle = start[:8] + "15T00:00:00"
    base = ProbeData(
        ((1, 999, "store"), (2, 999, None), (3, 999, "web"), (4, 999, "store")),
        (
            (1, 1, 80, "settled", middle),
            (2, 1, 80, "settled", middle),
            (3, 1, -120, "settled", middle),
            (4, 2, 31, "settled", start),
            (5, 2, 47, "settled", end),
            (6, 2, 59, "cancelled", middle),
            (7, 3, 89, "settled", middle),
        ),
    )
    datasets = {
        "source": case.source,
        "boundaries_and_duplicates": base,
        "empty": ProbeData((), ()),
    }
    rng = random.Random(3107 + case.variant)
    for index in range(22):
        count = rng.randint(1, 7)
        orders = tuple(
            (i, rng.randint(-100, 1000), rng.choice((None, "web", "store", "WEB")))
            for i in range(1, count + 1)
        )
        lines = tuple(
            (
                i + 1,
                rng.randint(1, count),
                rng.choice((-211, -17, 0, 13, 13, 97, 301)),
                rng.choice(("settled", "settled", "pending", "cancelled")),
                rng.choice((start, end, middle, "2025-01-01T00:00:00")),
            )
            for i in range(rng.randint(0, 18))
        )
        datasets[f"seeded_{index:02}"] = ProbeData(orders, lines)
    return datasets


def expected(case, data):
    """Third implementation: handwritten arithmetic, independent of SQL and DSL."""
    _, start, end = _definition(case.family, case.variant)
    rows = [r for r in data.lines if r[3] == "settled" and start <= r[4] < end]
    totals = {}
    for row in rows:
        totals[row[1]] = totals.get(row[1], 0) + row[2]
    family = case.family
    if family == "zero_activity":
        value = len(data.orders)
    elif family == "null_channel":
        ids = {o[0] for o in data.orders if o[2] != "web"}
        value = sum(r[2] for r in rows if r[1] in ids)
    elif family == "duplicate_amounts":
        value = len(rows)
    elif family == "aggregate_threshold":
        value = sum(total > 100 * case.variant for total in totals.values())
    elif family == "population_denominator":
        value = sum(totals.values()) / len(data.orders) if data.orders else 0
    elif family == "absolute_net":
        value = abs(sum(totals.values()))
    else:
        value = sum(totals.values())
    return [[value]]


def grade_sql(case, sql):
    passed, total = 0, 0
    for data in scoring_data(case).values():
        total += 1
        try:
            passed += rows_equal(execute_readonly(sql, data), expected(case, data))
        except (ValueError, TypeError):
            pass
    return {"passed": passed, "total": total, "correct": passed == total}
