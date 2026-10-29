"""Run frozen three-arm development comparison. Offline by default."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent_benchmark import run_benchmark
from dataverify.benchmark_tasks import TASKS
from dataverify.dbgpt_qwen import QwenConfig


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Explicitly allow ordinary metered Qwen requests"
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Frozen task IDs; omit for all development tasks"
    )
    parser.add_argument("--max-requests", type=int, choices=range(1, 37), default=18)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / ("agent_comparison_" + uuid4().hex[:10])
    )
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    if args.list_tasks:
        print(
            json.dumps(
                [
                    {"task_id": t.task_id, "track": t.track, "pair_id": t.pair_id, "split": t.split}
                    for t in TASKS
                ],
                indent=2,
            )
        )
        return 0
    tasks = TASKS
    if args.tasks:
        by_id = {task.task_id: task for task in TASKS}
        if len(set(args.tasks)) != len(args.tasks) or any(name not in by_id for name in args.tasks):
            parser.error("Select distinct task IDs printed by --list-tasks")
        tasks = tuple(by_id[name] for name in args.tasks)
    config = QwenConfig.from_local_environment() if args.live else None
    report = asyncio.run(
        run_benchmark(
            args.output, tasks=tasks, live=args.live, config=config, max_requests=args.max_requests
        )
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "real_model_used": report["real_model_used"],
                "budget": report["budget"],
                "summary": report["summary"],
                "usage": report["usage"],
                "report": str(args.output.resolve() / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_cells_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
