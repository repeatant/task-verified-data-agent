"""Explicit, bounded live Qwen smoke through the actual DB-GPT native loop.

Only newly created synthetic orders/schema/feedback leave the computer. No
credential or raw HTTP envelope is stored. This is integration, not a benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.database import seed_demo_database
from dataverify.dbgpt_bridge import BAD_ORDER_SUM, run_native_case, sql_candidate
from dataverify.dbgpt_qwen import InjectedFirstCandidateClient, QwenConfig, QwenLLMClient, RequestBudget
from dataverify.guarded_task import ConfirmedMetricTask


SCENARIOS = {
    "natural_order": {
        "question": "请计算全部订单的收入总额，每笔订单的总金额只累计一次。",
        "metric_id": "order_revenue", "kind": "natural_language_with_confirmed_metric",
        "expected": sum([100, 100, 80]),
    },
    "natural_line": {
        "question": "请计算全部订单明细金额的总和，每条明细累计一次。",
        "metric_id": "line_revenue", "kind": "natural_language_with_confirmed_metric",
        "expected": sum([40, 60, 100, 20, 30, 30]),
    },
    "injected_fanout_repair": {
        "question": "请计算全部订单的收入总额，每笔订单的总金额只累计一次。",
        "metric_id": "order_revenue", "kind": "scripted_first_fault_real_model_repair",
        "expected": sum([100, 100, 80]),
    },
}


def write_json(path, data):
    partial = path.with_suffix(".part.json")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


async def validate(arguments):
    config = QwenConfig.from_local_environment()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # New UUID path: never send an existing user-replaced database as synthetic.
    database = seed_demo_database(output / f"synthetic-{uuid4().hex}.sqlite3")
    budget = RequestBudget(arguments.max_requests)
    summary = {
        "status": "running", "passed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": config.model,
        "type": "bounded_real_qwen_dbgpt_integration_not_accuracy_benchmark",
        "requested_cases": arguments.cases, "cases": [], "budget": budget.snapshot(),
        "limitation": "Caller confirms the metric. Repair case injects the first bad Action; only subsequent repairs are model-generated. No general accuracy or comparative improvement claim.",
    }
    write_json(output / "summary.json", summary)
    try:
        for case_id in arguments.cases:
            scenario = SCENARIOS[case_id]
            client = QwenLLMClient(config, budget)
            injected = scenario["kind"] == "scripted_first_fault_real_model_repair"
            effective_client = InjectedFirstCandidateClient(client, sql_candidate(BAD_ORDER_SUM)) if injected else client
            started = time.perf_counter()
            record = await run_native_case(
                database, ConfirmedMetricTask(scenario["question"], scenario["metric_id"]),
                llm_client=effective_client, max_attempts=2, mode="evidence", output_dir=output / "runtime",
            )
            publication = record["publication"]
            correct = publication["status"] == "verified" and publication["metric_id"] == scenario["metric_id"] and publication["accepted_value"] == scenario["expected"]
            repair_observed = None
            if injected:
                first = record["actions"][0] if record["actions"] else {}
                repair_observed = first.get("error_code") == "GRAIN_MULTIPLICATION" and len(record["actions"]) == 2
                correct = correct and repair_observed
            usage_complete = bool(record["http_events"]) and all(e["usage_complete"] for e in record["http_events"])
            case = {
                "case": case_id, "kind": scenario["kind"], "passed": bool(correct),
                "seconds": round(time.perf_counter() - started, 3),
                "published_value": publication["accepted_value"],
                "native_model_turns": record["llm_calls"], "actual_http_attempts": len(record["http_events"]),
                "repair_observed": repair_observed, "usage_complete": usage_complete,
            }
            write_json(output / (case_id + "_trace.json"), record)
            summary["cases"].append(case)
            summary["budget"] = budget.snapshot()
            write_json(output / "summary.json", summary)
            print(json.dumps(case, ensure_ascii=False), flush=True)
            if not correct or budget.stopped_reason:
                break
        # Tokens come from actual API usage; missing values remain unknown.
        events = [event for case in summary["cases"] for event in json.loads((output / (case["case"] + "_trace.json")).read_text(encoding="utf-8"))["http_events"]]
        summary["usage"] = {
            field: sum(e["usage"][field] for e in events) if events and all(e["usage"][field] is not None for e in events) else None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        summary["passed"] = len(summary["cases"]) == len(arguments.cases) and all(c["passed"] for c in summary["cases"])
        summary["status"] = "passed" if summary["passed"] else "failed"
        summary["usage_note"] = "API-reported tokens; unknown for any failed/cancelled/unreported usage. Request cap is not a monetary cap; consult account billing."
        write_json(output / "summary.json", summary)
        print(json.dumps({"status": summary["status"], "budget": budget.snapshot(), "usage": summary["usage"], "report": str(output / "summary.json")}, ensure_ascii=False))
        return 0 if summary["passed"] else 1
    except Exception as error:
        summary.update(status="failed", passed=False, error_type=type(error).__name__, budget=budget.snapshot())
        write_json(output / "summary.json", summary)
        raise


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--max-requests", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / ("dbgpt_qwen_live_" + uuid4().hex[:10]))
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit --live is required for metered API calls.")
    return asyncio.run(validate(args))


if __name__ == "__main__":
    raise SystemExit(main())
