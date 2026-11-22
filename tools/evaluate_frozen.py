"""Preflight (free) or execute (--live) the frozen v1 development experiment."""

import argparse
import asyncio
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation.frozen_experiment import (
    build_manifest,
    create_top_up_budget,
    run_experiment,
)
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prior-journal", type=Path)
    parser.add_argument("--authorized-top-up-cny", type=float)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; runs are never overwritten")
    if args.live and (not args.prior_journal or not args.authorized_top_up_cny):
        parser.error("Live mode requires prior journal and an explicitly approved added allowance")
    if not args.live and (args.prior_journal or args.authorized_top_up_cny):
        parser.error("Budget options require --live")
    config = QwenConfig.from_local_environment() if args.live else None
    if config:
        validate_pricing(config)
    manifest = build_manifest(ROOT)
    manifest["model"] = config.model if config else "offline_no_model_calls"
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("langgraph", "hypothesis", "sqlglot", "httpx")
        },
    }
    manifest["base_git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest["git_scope"] = (
        "Base commit plus evaluation additions identified by manifest file hashes"
    )
    manifest["authorized_added_allowance_cny"] = args.authorized_top_up_cny
    args.output.mkdir(parents=True)
    write_json(args.output / "manifest.json", manifest)
    print("Frozen manifest:", digest_json(manifest), flush=True)
    if not args.live:
        print("Offline preflight completed: 16 controlled + 4 natural cases; zero API calls")
        return
    with exclusive_run(args.prior_journal), exclusive_run(args.output / "budget.json"):
        budget, prior = create_top_up_budget(
            args.prior_journal, args.output, args.authorized_top_up_cny
        )
        report = asyncio.run(run_experiment(ROOT, args.output, config, budget, prior, manifest))
    if not report["complete"] or not report["source_integrity_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
