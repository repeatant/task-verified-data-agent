"""Disclosed development cases, never imported by the discovery runtime.

Authored reference expressions are used ONLY in the offline integration demo.
Live runs receive public_input only. Gold rows are scored after the run.
"""

from dataclasses import dataclass

from dataverify.verification.discovery_runtime import DiscoveryTask, execute_readonly, rows_equal
from dataverify.verification.probe_runtime import ProbeData

FILTER = (
    "status='settled' AND settled_at>='2026-09-01T00:00:00' AND settled_at<'2026-10-01T00:00:00'"
)
ELIGIBLE = "r['status']=='settled' and '2026-09-01T00:00:00'<=r['settled_at']<'2026-10-01T00:00:00'"
PERIOD = "统计2026年9月已结算明细，时间为9月1日零点（含）到10月1日零点（不含）。"


def data(amounts, channels=None, order_ids=None):
    order_ids = order_ids or list(range(1, len(amounts) + 1))
    keys = sorted(set(order_ids))
    channels = channels or {key: "web" for key in keys}
    return ProbeData(
        tuple((key, 999, channels[key]) for key in keys),
        tuple(
            (i + 1, key, amount, "settled", "2026-09-10T00:00:00")
            for i, (key, amount) in enumerate(zip(order_ids, amounts))
        ),
    )


@dataclass(frozen=True)
class DiscoveryCase:
    name: str
    task: DiscoveryTask
    seed: ProbeData
    distinguishing: ProbeData
    source: ProbeData
    bad_sql: str
    good_sql: str
    offline_reference: str
    expected: list

    def offline_plan(self):
        return {
            "action": "plan",
            "basis": [{"source": "question", "quote": self.task.question}],
            "assumptions": [],
            "reference": self.offline_reference,
            "fixtures": [{"name": "initial", "data": self.seed.to_dict()}],
        }

    def offline_probe(self):
        return {
            "action": "probe",
            "reason": "Disclosed authored development counterexample.",
            "fixtures": [{"name": "distinguishing", "data": self.distinguishing.to_dict()}],
        }


def get_discovery_cases() -> list[DiscoveryCase]:
    rows = "tables['order_lines']"
    sums = f"sum(r['amount'] for r in {rows} if {ELIGIBLE})"
    ids = f"set(r['order_id'] for r in {rows} if {ELIGIBLE})"
    average = DiscoveryCase(
        "order_average",
        DiscoveryTask(
            "order_average",
            PERIOD
            + "先按订单汇总明细金额，再计算有符合条件明细的订单的平均金额；无符合条件订单时返回0。",
        ),
        data([100]),
        data([100, 50, 50], order_ids=[1, 2, 2]),
        data([120, 20, 60, -20], order_ids=[1, 2, 2, 3]),
        f"SELECT COALESCE(AVG(amount),0) FROM order_lines WHERE {FILTER}",
        f"SELECT COALESCE(AVG(net),0) FROM (SELECT SUM(amount) AS net FROM order_lines WHERE {FILTER} GROUP BY order_id)",
        f"[[{sums}/len({ids}) if len({ids}) else 0]]",
        [[60]],
    )
    web = DiscoveryCase(
        "web_scope",
        DiscoveryTask(
            "web_scope",
            PERIOD + "只统计orders.channel为web的订单，返回明细金额合计；无符合条件明细时返回0。",
        ),
        data([100]),
        data([100, 70], {1: "web", 2: "store"}),
        data([90, 40, -10], {1: "web", 2: "store", 3: "web"}),
        f"SELECT COALESCE(SUM(amount),0) FROM order_lines WHERE {FILTER}",
        f"SELECT COALESCE(SUM(amount),0) FROM order_lines JOIN orders ON orders.order_id=order_lines.order_id WHERE {FILTER} AND channel='web'",
        f"[[sum(r['amount'] for r in {rows} if {ELIGIBLE} and r['order_id'] in [o['order_id'] for o in tables['orders'] if o['channel']=='web'])]]",
        [[80]],
    )
    count = DiscoveryCase(
        "distinct_orders",
        DiscoveryTask(
            "distinct_orders", PERIOD + "返回至少有一条符合条件明细的订单数量；同一订单只计算一次。"
        ),
        data([100]),
        data([100, 40, 60], order_ids=[1, 2, 2]),
        data([20, 30, 60, 10, -5], order_ids=[1, 1, 2, 3, 3]),
        f"SELECT COUNT(*) FROM order_lines WHERE {FILTER}",
        f"SELECT COUNT(DISTINCT order_id) FROM order_lines WHERE {FILTER}",
        f"[[len({ids})]]",
        [[3]],
    )
    units = DiscoveryCase(
        "money_units",
        DiscoveryTask(
            "money_units",
            PERIOD + "金额字段单位为分，返回以元为单位的净额合计，100分等于1元；无明细返回0。",
        ),
        data([0]),
        data([123]),
        data([230, -15]),
        f"SELECT COALESCE(SUM(amount),0) FROM order_lines WHERE {FILTER}",
        f"SELECT COALESCE(SUM(amount),0)/100.0 FROM order_lines WHERE {FILTER}",
        f"[[{sums}/100]]",
        [[2.15]],
    )
    return [average, web, count, units]


def grade_discovery_sql(case: DiscoveryCase, sql: str | None) -> dict:
    """Post-run scoring; neither these sources nor gold SQL enter model prompts."""
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
    sources = {
        "source": case.source,
        "distinguishing": case.distinguishing,
        "empty": ProbeData((), ()),
        "precision": data([10, 10, 11]),
        "mixed": mixed,
    }
    checks = []
    for name, source in sources.items():
        expected = execute_readonly(case.good_sql, source)
        try:
            actual = execute_readonly(sql, source)
            checks.append(
                {
                    "name": name,
                    "match": rows_equal(actual, expected),
                    "actual": actual,
                    "expected": expected,
                }
            )
        except ValueError:
            checks.append({"name": name, "match": False, "error": "NO_EXECUTABLE_CANDIDATE"})
    return {
        "scope": "Five authored scoring datasets; gold SQL is an evaluation assumption, not a formal proof.",
        "all_match": all(c["match"] for c in checks),
        "checks": checks,
    }
