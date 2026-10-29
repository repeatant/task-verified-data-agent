"""Full/compact feedback experiment. Offline by default; --live bills Qwen."""

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
from dataverify.probe_live_benchmark import run_probe_benchmark


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--feedback-policy", choices=("legacy_v1", "capability_v2"), default="capability_v2"
    )
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 31])
    parser.add_argument(
        "--modes", nargs="+", choices=("full", "compact"), default=["full", "compact"]
    )
    tracks = parser.add_mutually_exclusive_group()
    tracks.add_argument("--natural-only", action="store_true")
    tracks.add_argument("--controlled-only", action="store_true")
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--money-limit-cny", type=float, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("probe_agent_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    config = QwenConfig.from_local_environment() if args.live else None
    report = asyncio.run(
        run_probe_benchmark(
            args.output,
            live=args.live,
            config=config,
            seeds=tuple(args.seeds),
            include_controlled=not args.natural_only,
            include_natural=not args.controlled_only,
            modes=tuple(args.modes),
            case_ids=None if args.case_ids is None else tuple(args.case_ids),
            max_requests=args.max_requests,
            max_attempts=args.max_attempts,
            money_limit_cny=args.money_limit_cny,
            feedback_policy=args.feedback_policy,
        )
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "real_model_used": report["real_model_used"],
                "summary": report["summary"],
                "usage": report["usage"],
                "budget_start": report["budget_start"],
                "budget_end": report["budget_end"],
                "this_run_http_attempts": report["this_run_http_attempts"],
                "report": str(args.output.resolve() / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_cells_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
