"""Freeze and run the corrected query-off arm against the original paired controls.

The archived query-enabled and self-review records are references, never new calls.
"""

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

from dataverify.agent.dispute_no_query import QueryDisabledDisputeModel, prompt_without_queries
from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation import dispute_recovery as previous
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation import query_ablation as experiment
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json


def bind(parent, output):
    old = bounded.read_json(parent / "protocol.json")
    new = output / "protocol.json"
    extras = (
        "dataverify/agent/dispute_no_query.py",
        "dataverify/evaluation/query_ablation.py",
        "tools/rerun_query_off.py",
    )
    files = {**old["source_hashes"], **{name: file_digest(ROOT / name) for name in extras}}
    controls = {}
    for case in old["cases"]:
        for method in ("query_on", "self_review", "query_off"):
            path = parent / "records" / f"{case['id']}--{method}.json"
            item = bounded.read_json(path)
            if not item.get("completed") or item["binding"]["protocol"] != digest_json(old):
                raise ValueError("Missing frozen control record")
            controls[path.name] = file_digest(path)
    snapshot = output / "source"
    if not new.exists():
        # The corrected adapter and injection point run on the original frozen graph.
        for name, digest in files.items():
            source = (ROOT if name in extras else parent / "source") / name
            if file_digest(source) != digest:
                raise ValueError("Frozen control source mismatch")
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    # On resume prefer existing frozen extras, even if a working copy has moved on.
    protocol = (
        bounded.read_json(new)
        if new.exists()
        else {
            "format": "corrected-query-off-v2",
            "parent": digest_json(old),
            "model": old["model"],
            "control_records": controls,
            "source_hashes": files,
            "cases": old["cases"],
            "change": "Off arm removes query action from its system instruction, local parser and optional provider schema. Its remaining query allowance is zero. HTTP json_object format and all other budgets and graph rules unchanged.",
            "analysis": "Paired with original query_on for six reused development trajectories; comparator is not an independent new run. Never pool old and corrected runs as additional tasks.",
            "budget_driver_sha256": old["budget_driver_sha256"],
        }
    )
    if (
        protocol["parent"] != digest_json(old)
        or controls != protocol["control_records"]
        or any(file_digest(snapshot / name) != h for name, h in protocol["source_hashes"].items())
    ):
        raise ValueError("Frozen protocol or parent changes")
    immutable(new, protocol)
    return protocol


async def execute(main_repo, output, protocol, driver, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    if experiment.frozen_cases() != protocol["cases"]:
        raise ValueError("Control bindings changed")
    output.joinpath("records").mkdir(exist_ok=True)
    prompt_without_queries()  # Fail before any payment on prompt drift.
    for case in previous.study_cases():
        exists = (output / "records" / f"{case.case_id}--query_off.json").exists()
        _, lane = choose_lane(driver, directory, receipt, 0 if exists else previous.MAX_ACTIONS)
        item = await experiment.run_cell(
            case,
            "query_off",
            config,
            lane,
            output,
            digest_json(protocol),
            model_type=QueryDisabledDisputeModel,
        )
        if not accounted(item, lane):
            raise ValueError("Inspect accounting incident before further requests")
        print(case.case_id, item["record"]["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("main", "parent", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    main_repo, parent, output = [p.resolve() for p in (args.main, args.parent, args.output)]
    if parent == output:
        raise ValueError("Use a new output directory")
    output.mkdir(exist_ok=True, parents=True)
    protocol = bind(parent, output)
    if not args.execute:
        print(
            json.dumps(
                {
                    "protocol": digest_json(protocol),
                    "max_calls": 24,
                    "max_reservation_cny": bounded.max_reservation(24),
                }
            )
        )
        return
    snapshot = output / "source"
    if ROOT != snapshot:
        subprocess.run(
            [
                sys.executable,
                str(snapshot / "tools/rerun_query_off.py"),
                "--main",
                str(main_repo),
                "--parent",
                str(parent),
                "--output",
                str(output),
                "--execute",
            ],
            cwd=snapshot,
            check=True,
        )
        return
    campaign = main_repo / "reports/expanded-v1-20261107"
    if not (campaign / "budget120/authorization.json").exists():
        raise ValueError("Existing authorization required")
    driver = load_driver(main_repo / "tools/continue_expanded.py", protocol["budget_driver_sha256"])
    with ExitStack() as locks:
        locks.enter_context(exclusive_run(campaign / "allocation"))
        for i in range(4):
            locks.enter_context(exclusive_run(campaign / f"lane-{i}-execution"))
        directory, receipt = driver.allocate_extension(
            campaign, campaign / "runs/main-v1", approved_limit=120
        )
        asyncio.run(execute(main_repo, output, protocol, driver, directory, receipt))


if __name__ == "__main__":
    main()
