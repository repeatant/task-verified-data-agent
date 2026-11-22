"""Freeze a development query ablation, or execute its exact snapshot with --execute."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_evidence_recovery import accounted, choose_lane
from evaluate_structured_confirmation import immutable, load_driver

from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation import dispute_recovery as previous
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation import query_ablation as experiment
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.evaluation.structured_confirmation import cached
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json


def hash_sources(root):
    paths = [
        p
        for p in (root / "dataverify").rglob("*.py")
        if "workbench" not in p.relative_to(root).parts and p.name != "api.py"
    ]
    for name in (
        "evaluate_query_ablation.py",
        "evaluate_evidence_recovery.py",
        "evaluate_structured_confirmation.py",
    ):
        paths.append(root / "tools" / name)
    paths.extend((root / "requirements").glob("*.txt"))
    return {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)}


def freeze(main, output):
    driver = main / "tools/continue_expanded.py"
    snapshot = output / "source"
    manifest = output / "protocol.json"
    if manifest.exists():
        protocol = bounded.read_json(manifest)
        if (
            protocol["source_hashes"] != hash_sources(snapshot)
            or file_digest(driver) != protocol["budget_driver_sha256"]
        ):
            raise ValueError("Frozen ablation source or accounting driver has changed")
        return protocol
    files = hash_sources(ROOT)
    for name, checksum in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_digest(target) != checksum:
            raise ValueError("Source changed while freezing")
    cases = experiment.frozen_cases()
    calls = len(cases) * (1 + previous.MAX_ACTIONS * 2)
    protocol = {
        "format": "query-ablation-development-v1",
        "model": PRICE_MODEL,
        "cases": cases,
        "source_hashes": files,
        "budget_driver_sha256": file_digest(driver),
        "arms": list(experiment.METHODS),
        "case_order": "Original six development trajectories. Rotate arm order by case index.",
        "scope": "5 previously seen requirements, 6 controlled faults. No unseen-task or public dataset claim.",
        "source": "Seeded generic synthetic source (17 orders, 120 lines) with original preview prefix, built without fault metadata; used for all active queries and all final publications.",
        "exposure": "Dispute arms receive same preview, task and three programs; no distinguishing examples. Only tools_enabled varies. One-call self-review receives candidate execution on same full source and original task.",
        "scoring": "25 independent grading fixture executions per program (one original preview and 24 private fixtures), plus rows checked on the common full source.",
        "primary": "Correct deliveries over all trajectories. Report paired discordances, wrong delivery, correct rejection, repairs, damage, queries, time and usage together.",
        "limits": {
            "calls_ceiling": calls,
            "reserved_cny_ceiling": bounded.max_reservation(calls),
            "total_authorization_cny": 120,
        },
    }
    immutable(manifest, protocol)
    return protocol


async def campaign_run(protocol, driver, directory, receipt, output, case_limit):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    cases = previous.study_cases()
    if experiment.frozen_cases() != protocol["cases"]:
        raise ValueError("Prepared cases changed")
    (output / "records").mkdir(exist_ok=True)
    protocol_id = digest_json(protocol)
    for index, case in enumerate(cases[:case_limit]):
        rotation = index % len(experiment.METHODS)
        methods = experiment.METHODS[rotation:] + experiment.METHODS[:rotation]
        for method in methods:
            key = {
                "protocol": protocol_id,
                "case": case.case_id,
                "method": method,
                "model": config.model,
            }
            path = output / "records" / f"{case.case_id}--{method}.json"
            if cached(path, key) is not None:
                continue
            count = 1 if method == "self_review" else previous.MAX_ACTIONS
            _, lane = choose_lane(driver, directory, receipt, count)
            result = await experiment.run_cell(case, method, config, lane, output, protocol_id)
            print(case.case_id, method, result["record"]["status"], flush=True)
            experiment.summarize(output, protocol)
            if not accounted(result, lane):
                raise ValueError(
                    "Transport or accounting incident; inspect saved record before continuing"
                )
    return experiment.summarize(output, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        type=Path,
        required=True,
        help="Main repository holding the original campaign ledgers",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case-limit", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    main_repo, output = args.main.resolve(), args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    protocol = freeze(main_repo, output)
    if not args.execute:
        print(
            json.dumps(
                {"prepared": str(output), "cases": len(protocol["cases"]), **protocol["limits"]}
            )
        )
        return
    snapshot = output / "source"
    if ROOT != snapshot:
        subprocess.run(
            [
                sys.executable,
                str(snapshot / "tools/evaluate_query_ablation.py"),
                "--main",
                str(main_repo),
                "--output",
                str(output),
                "--execute",
                "--case-limit",
                str(args.case_limit),
            ],
            cwd=snapshot,
            check=True,
        )
        return
    campaign = main_repo / "reports/expanded-v1-20261107"
    if not (campaign / "budget120/authorization.json").is_file():
        raise ValueError("Existing cumulative authorization required")
    driver = load_driver(main_repo / "tools/continue_expanded.py", protocol["budget_driver_sha256"])
    with ExitStack() as locks:
        locks.enter_context(exclusive_run(campaign / "allocation"))
        for lane in range(4):
            locks.enter_context(exclusive_run(campaign / f"lane-{lane}-execution"))
        directory, receipt = driver.allocate_extension(
            campaign, campaign / "runs/main-v1", approved_limit=120
        )
        report = asyncio.run(
            campaign_run(protocol, driver, directory, receipt, output, args.case_limit)
        )
    print(
        json.dumps(
            {
                "completed_cells": report["completed_cells"],
                "paired_cases": len(report["paired_cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
