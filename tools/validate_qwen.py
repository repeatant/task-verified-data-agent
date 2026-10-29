"""Explicit, bounded live integration smoke test; never imported by pytest.

Run from the project root with ``python tools/validate_qwen.py --live``.
Only synthetic orders/schema and tool feedback are sent to the configured API.
This checks integration paths, not general SQL accuracy or benchmark gains.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.database import seed_demo_database
from dataverify.engine import AgentRunner
from dataverify.providers import JOIN, ProviderError, QwenPlanner

SCENARIOS = {
    "order_total": {
        "question": "请统计全部订单的总收入，并使用工具核验统计结果。",
        "metric_id": "order_revenue",
        "fault_value": None,
    },
    "line_total": {
        "question": "请统计全部订单明细的金额总和，使用明细粒度的指标并通过工具核验。",
        "metric_id": "line_revenue",
        "fault_value": None,
    },
    "join_fanout_review": {
        "question": f"请审查下面的候选SQL能否用于全部订单总收入。先检查契约，再原样调用query_metric执行候选SQL以获得核验反馈；若未通过，再根据反馈尝试修复并核验。指标order_revenue。候选SQL：SELECT SUM(o.total) AS revenue {JOIN}",
        "metric_id": "order_revenue",
        "fault_value": 540,
    },
    "distinct_review": {
        "question": f"请审查下面的候选SQL能否用于全部订单总收入。先检查契约，再原样调用query_metric执行候选SQL以获得核验反馈；若未通过，再根据反馈尝试修复并核验。指标order_revenue。候选SQL：SELECT SUM(DISTINCT o.total) AS revenue {JOIN}",
        "metric_id": "order_revenue",
        "fault_value": 180,
    },
}


class BudgetedPlanner(QwenPlanner):
    """Shared request-attempt cap across the selected scenarios; no retries."""

    def __init__(self, budget: dict[str, int]):
        super().__init__()
        self.budget = budget
        if self.api_key.startswith("sk-sp-") or "token-plan." in self.base_url:
            raise ValueError("This live test requires the ordinary API configuration.")

    def choose(self, messages):
        if self.budget["used"] >= self.budget["limit"]:
            raise ProviderError("Live smoke request-attempt budget exhausted.")
        self.budget["used"] += 1
        started = time.perf_counter()
        action = super().choose(messages)
        print(
            json.dumps(
                {
                    "event": "model_response",
                    "attempt": self.budget["used"],
                    "seconds": round(time.perf_counter() - started, 3),
                    "action": action.tool_name or action.kind,
                    "usage": self.usage[-1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return action


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="explicitly allow metered model API requests"
    )
    parser.add_argument("--cases", nargs="+", choices=SCENARIOS, default=["order_total"])
    parser.add_argument("--max-requests", type=int, choices=range(1, 17), default=12)
    arguments = parser.parse_args()
    if not arguments.live:
        parser.error("Pass --live explicitly to call the metered model API.")

    # Always create a new known fixture. Never send values from an existing
    # user-replaced database while labelling them synthetic.
    data = seed_demo_database(ROOT / "tmp" / "live-fixtures" / f"{uuid4().hex}.sqlite3")
    budget = {"used": 0, "limit": arguments.max_requests}
    cases = []
    for case_id in arguments.cases:
        scenario = SCENARIOS[case_id]
        planner = BudgetedPlanner(budget)
        started = time.perf_counter()
        result = AgentRunner(data, ROOT / "artifacts" / "traces", planner=planner).run(
            question=scenario["question"], provider="qwen", max_repairs=1
        )
        queries = [event for event in result["trace"] if event.get("tool") == "query_metric"]
        metric_id = queries[-1]["arguments"].get("metric_id") if queries else None
        expected_result = (
            result["status"] == "verified"
            and result["value"] == 280
            and metric_id == scenario["metric_id"]
        )
        fault_observed = (
            any(
                event["result"].get("value") == scenario["fault_value"]
                and event["result"].get("verification_status") == "rejected"
                and event["result"].get("accepted_value") is None
                for event in queries
            )
            if scenario["fault_value"] is not None
            else None
        )
        passed = (
            expected_result
            and (fault_observed if scenario["fault_value"] is not None else True)
            and bool(result["model_usage"]["total_tokens"])
        )
        cases.append(
            {
                "case": case_id,
                "fault_injected_by_test_question": scenario["fault_value"] is not None,
                "rejected_fault_observed": fault_observed,
                "passed": passed,
                "seconds": round(time.perf_counter() - started, 3),
                "result": result,
            }
        )
        if not passed or budget["used"] >= budget["limit"]:
            # A failed first request must not trigger a burst of extra charges.
            break

    usage = {
        field: sum(case["result"]["model_usage"][field] for case in cases)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    model = cases[0]["result"]["model"] if cases else None
    # Standard Beijing list price, excluding free quota/cache/discounts. This
    # is a planning estimate, not the account's actual bill.
    estimate = (
        round((usage["prompt_tokens"] * 0.8 + usage["completion_tokens"] * 2.7) / 1_000_000, 6)
        if model == "qwen3.8-flash"
        else None
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": "bounded-real-model-integration-smoke",
        "dataset": "synthetic-demo-orders-v1",
        "model": model,
        "request_attempts": budget["used"],
        "request_attempt_limit": budget["limit"],
        "usage": usage,
        "estimated_standard_price_cny": estimate,
        "cost_note": "Uncached Beijing qwen3.8-flash list-price estimate; excludes free quota, discounts and unreported/failed request usage. Consult account billing for actual cost.",
        "limitation": "Small integration smoke, not an accuracy benchmark. Repair cases explicitly inject a bad candidate SQL through the test question; no claim of naturally occurring model errors or general SQL correctness.",
        "requested_cases": arguments.cases,
        "passed": len(cases) == len(arguments.cases) and all(case["passed"] for case in cases),
        "cases": cases,
    }
    destination = ROOT / "reports" / "qwen_live_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "live_smoke_complete",
                "passed": report["passed"],
                "completed_cases": len(cases),
                "request_attempts": budget["used"],
                "usage": usage,
                "estimated_standard_price_cny": estimate,
                "report_path": str(destination),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
