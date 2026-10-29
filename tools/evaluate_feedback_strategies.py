"""Run crossed feedback experiments; offline unless --live is explicit."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.dbgpt_qwen import QwenConfig
from dataverify.feedback_experiment import load_prior_budget, run_feedback_experiment


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ablation", "strategy"), default="ablation")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prior-budget-json", type=Path)
    parser.add_argument(
        "--resume-after-truncation",
        action="store_true",
        help="Review one fully accounted final HTTP-200 truncation; preserve its failed trial and spending in separate reports.",
    )
    parser.add_argument(
        "--arms", nargs="+", help="Optional subset of the stage's registered comparison arms."
    )
    parser.add_argument("--controlled-case-ids", nargs="*", default=["d06", "d10", "d13"])
    parser.add_argument("--natural-case-ids", nargs="*", default=["d01", "d03", "d04"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[53])
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-extra-requests", type=int, default=24)
    parser.add_argument("--extra-reservation-cny", type=float, default=0.8)
    parser.add_argument("--money-limit-cny", type=float, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("feedback_experiment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    if args.live and not args.prior_budget_json:
        parser.error("--live requires --prior-budget-json to preserve cumulative spending")
    if args.prior_budget_json and not args.live:
        parser.error("--prior-budget-json is only valid with --live")
    if args.resume_after_truncation and not args.prior_budget_json:
        parser.error("--resume-after-truncation requires a prior live report")
    config = QwenConfig.from_local_environment() if args.live else None
    budget = (
        load_prior_budget(
            args.prior_budget_json,
            max_extra_requests=args.max_extra_requests,
            money_limit_cny=args.money_limit_cny,
            allow_truncated_resume=args.resume_after_truncation,
        )
        if args.prior_budget_json
        else None
    )
    report = asyncio.run(
        run_feedback_experiment(
            args.output,
            stage=args.stage,
            live=args.live,
            config=config,
            budget=budget,
            controlled_case_ids=tuple(args.controlled_case_ids),
            natural_case_ids=tuple(args.natural_case_ids),
            arm_ids=tuple(args.arms) if args.arms is not None else None,
            seeds=tuple(args.seeds),
            max_attempts=args.max_attempts,
            max_extra_requests=args.max_extra_requests,
            extra_reservation_cny=args.extra_reservation_cny,
            money_limit_cny=args.money_limit_cny,
        )
    )
    safe_budget_fields = (
        "http_attempts",
        "reserved_upper_cny",
        "estimated_reported_cny",
        "reported_total_tokens",
        "reported_cost_complete",
        "stopped_reason",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": report["stage"],
                "real_model_used": report["real_model_used"],
                "summary": report["summary"],
                "usage": report["usage"],
                "this_run_http_attempts": report["this_run_http_attempts"],
                "this_run_reserved_upper_cny": report["this_run_reserved_upper_cny"],
                "audit_ledger_complete": report["audit_ledger_complete"],
                "source_integrity_ok": report["source_integrity_ok"],
                "budget_end": {k: report["budget_end"][k] for k in safe_budget_fields},
                "report": str(args.output.resolve() / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return (
        0
        if report["all_cells_completed"]
        and report["source_integrity_ok"]
        and report["audit_ledger_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
