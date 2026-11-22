"""Two prespecified wrong candidates. Oracles only run after the model has decided."""

from dataclasses import dataclass

from dataverify.evaluation import terminal_scenarios
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


@dataclass(frozen=True)
class Fault:
    name: str
    question: str
    preview: ProbeData
    sql: str
    reference: str
    auditor: str

    @property
    def task(self):
        # Keep the evaluator's fault category out of every model-visible task.
        return DiscoveryTask(
            "t_" + digest_json(self.question)[:12], self.question, terminal_scenarios.DEFINITIONS
        )

    def plan(self, program):
        return {
            "action": "plan",
            "basis": [{"source": "question", "quote": self.question}],
            "assumptions": [],
            "reference": program,
            "fixtures": [{"name": "initial_example", "data": self.preview.to_dict()}],
        }


def cases():
    preview = ProbeData(((1, 400, "web"),), ((1, 1, 20, "cancelled", "2026-09-15T00:00:00"),))
    count_store = (
        "channels=set([r['order_id'] for r in tables['orders'] if r['channel']=='store'])\n"
        "[[len(set([r['order_id'] for r in tables['order_lines'] "
        "if r['status']=='cancelled' and r['order_id'] in channels]))]]"
    )
    # In the shared-error control every program counts NULL as a distinct sales channel.
    wrong_ref = "[[len(set([r['channel'] for r in tables['orders']]))]]"
    return [
        Fault(
            "wrong-store",
            "Count store orders with at least one cancelled line. Count each qualifying order once.",
            preview,
            "SELECT COUNT(DISTINCT order_id) FROM order_lines WHERE status='cancelled'",
            count_store,
            count_store,
        ),
        Fault(
            "wrong-null",
            "How many distinct known sales channels occur across all orders? An unspecified channel must be excluded.",
            preview,
            "SELECT COUNT(DISTINCT COALESCE(channel,'unknown')) FROM orders",
            wrong_ref,
            wrong_ref,
        ),
    ]


def sources(case):
    return {
        **terminal_scenarios.scoring_sources(),
        "preview": case.preview,
        "full_source": full_source(case.preview),
    }


def expected(case, data):
    if case.name == "wrong-store":
        store = {i for i, _, channel in data.orders if channel == "store"}
        return [
            [len(store & {oid for _, oid, _, status, _ in data.lines if status == "cancelled"})]
        ]
    if case.name == "wrong-null":
        return [[len({channel for _, _, channel in data.orders if channel is not None})]]
    raise ValueError("Unknown control")


def grade(case, rec):
    data = sources(case)

    def check(text, sql):
        if not text:
            return False
        for d in data.values():
            got = execute_readonly(text, d) if sql else calculate(text, named_tables(d))
            if not rows_equal(got, expected(case, d)):
                return False
        return True

    sql_good = check(rec["candidate"]["sql"], True)
    reference_good = check(rec["plan"]["reference"], False)
    auditor_good = check(rec["audit_plan"]["reference"], False)
    published = rec.get("publication", {})
    correct = check(published.get("sql"), True) and rows_equal(
        published.get("rows"), expected(case, data["full_source"])
    )
    return {
        "sql_good": sql_good,
        "reference_good": reference_good,
        "auditor_good": auditor_good,
        "correct_delivery": correct,
        "wrong_release": rec["status"] == "reviewed" and not correct,
    }
