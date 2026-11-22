"""Experimental SQL-only recovery runner using existing CNY120 ledgers.

This is NOT the proposed three-party disagreement resolver. Its results cannot
establish repair of reference/auditor programs or independent role consensus.

Default behavior prepares a frozen development protocol without network calls.
--execute must be explicit. Holdout execution also requires the predeclared
development gate; a disappointing completed outcome is never silently rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_structured_confirmation import immutable, load_driver

from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation import evidence_recovery as recovery
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json, write_json


def source_hashes():
    files = sorted((ROOT / "dataverify").rglob("*.py"))
    files += sorted((ROOT / "requirements").glob("*.txt"))
    files += [Path(__file__).resolve(), ROOT / "tools/evaluate_structured_confirmation.py"]
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in files}


def freeze(args):
    main = args.main.resolve()
    public_root = main / "reports/public-data"
    original_path = public_root / "bird-manifest-v2.json"
    confirmation_path = main / "reports/expanded-v1-20261107/confirmation-manifest.json"
    original, old_confirmation = map(recovery.read_json, (original_path, confirmation_path))
    selection = {
        **recovery.select_question_ids(original, old_confirmation),
        "original_manifest_sha256": file_digest(original_path),
        "previous_confirmation_sha256": file_digest(confirmation_path),
    }
    selection_path = args.selection.resolve()
    immutable(selection_path, selection)
    ids = selection["development_ids" if args.phase == "development" else "holdout_ids"]
    manifest = recovery.make_manifest(original, public_root / "mini_dev_sqlite.jsonl", ids)
    bundles = recovery.recovery_bundles(manifest)
    for name in {bundle.family for bundle in bundles}:
        database = manifest["databases"][name]
        if file_digest(Path(database["path"])) != database["sha256"]:
            raise ValueError("RECOVERY_DATABASE_CHANGED")
    driver_path = main / "tools/continue_expanded.py"
    protocol = {
        "format": "dataverify-evidence-recovery-v1",
        "status": "experimental SQL-only prototype; not three-party disagreement recovery",
        "phase": args.phase,
        "selection_digest": digest_json(selection),
        "manifest_digest": digest_json(manifest),
        "source_hashes": source_hashes(),
        "budget_driver_path": str(driver_path),
        "budget_driver_sha256": file_digest(driver_path),
        "model": PRICE_MODEL,
        "methods": list(recovery.METHODS),
        "limits": {
            "input_chars": recovery.INPUT_CHAR_CAP,
            "output_tokens": recovery.OUTPUT_TOKEN_CAP,
            "self_review_calls": 1,
            "agent_calls": recovery.ACTION_CAP,
            "agent_queries": 2,
            "agent_revisions": 1,
        },
        "primary_metric": "candidate SQL correctness on original DB, paired against the same initial SQL",
        "secondary_metrics": [
            "initially wrong SQL repaired",
            "initially correct SQL damaged",
            "correct delivery",
            "wrong release",
            "refusal",
            "unchanged SQL delivery",
            "calls/tokens/reservation",
        ],
        "fairness": "All methods receive the same initial SQL and original-DB execution under identical SQLTools limits; no_tools differs only by extra-query access.",
        "gate": "Complete 6 development questions; evidence candidate correctness exceeds both comparators, at least one repaired SQL, no initially-correct damage, wrong releases no worse than self_review.",
        "no_reroll": "Completed failures remain failures; interrupted paid stages require explicit review.",
        "limitations": [
            "Small, supported BIRD subset; new questions in existing databases, not unseen-domain generalization.",
            "Original-DB execution match is not a semantic proof; no model sees official answers.",
            "A model-reviewed delivery is not independently verified truth.",
            "self_review is a one-call competitor; no_tools is the matched maximum-call ablation.",
            "Maximum calls are equal for agent arms; actual calls may differ due to early stopping.",
            "Query observations are bounded and not evidence that unqueried cases are correct.",
        ],
        "cases": [
            {"case_id": b.case_id, "family": b.family, **recovery.task_binding(b)} for b in bundles
        ],
        "forecast": recovery.forecast(),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("initial", "records"):
        (output / name).mkdir(exist_ok=True)
    immutable(output / "manifest.json", manifest)
    immutable(output / "protocol.json", protocol)
    archive = output / "frozen-source.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
            for relative in protocol["source_hashes"]:
                saved.write(ROOT / relative, relative)
            saved.write(driver_path, "budget_driver/continue_expanded.py")
            saved.write(output / "protocol.json", "protocol.json")
    return protocol, bundles


def choose_lane(driver, directory, receipt, calls):
    """Fit one bounded stage in an existing lane; never transfer old allowances."""
    lanes = [(i, driver.load_lane(directory, i, receipt)) for i in range(4)]
    remaining = [
        (lane.snapshot()["money_limit_cny"] - lane.snapshot()["reserved_upper_cny"], i, lane)
        for i, lane in lanes
        if lane.snapshot()["stopped_reason"] is None
    ]
    if not remaining:
        raise ValueError("RECOVERY_NO_ACTIVE_EXISTING_LANE")
    available, index, lane = max(remaining, key=lambda item: item[0])
    if available < recovery.max_reservation(calls):
        raise ValueError("RECOVERY_EXISTING_LANES_CANNOT_FIT_BOUNDED_STAGE")
    return index, lane


def accounted(result, lane):
    if lane.snapshot()["stopped_reason"]:
        return False
    return all(
        e.get("status") == "completed" and e.get("usage_complete") is True
        for e in result.get("http_events", [])
    )


async def execute(args, protocol, bundles, driver, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("RECOVERY_FROZEN_SOURCE_CHANGED")
    output = args.output.resolve()
    limit = args.case_limit or len(bundles)
    stopped = None
    for bundle in bundles[:limit]:
        initial_path = output / "initial" / (bundle.case_id + ".json")
        if args.phase == "development":
            old = args.main / "reports/expanded-v1-20261107/runs/development-v2/initial"
            common = recovery.imported_initial(bundle, old / initial_path.name, initial_path)
        else:
            _, lane = choose_lane(driver, directory, receipt, 1)
            common = await recovery.shared_initial(bundle, config, lane, initial_path)
            if not accounted(common, lane):
                stopped = "initial_accounting_requires_review"
                break
        for method in recovery.METHODS:
            call_cap = 1 if method == "self_review" else recovery.ACTION_CAP
            _, lane = choose_lane(driver, directory, receipt, call_cap)
            result = await recovery.paired_cell(
                bundle, method, common, config, lane, output, digest_json(protocol)
            )
            summary = recovery.summarize(output, protocol)
            print(
                json.dumps(
                    {
                        "case": bundle.case_id,
                        "method": method,
                        "status": result["row"]["status"],
                        "completed_cells": summary["completed_cells"],
                    }
                ),
                flush=True,
            )
            if not accounted(result, lane):
                stopped = "cell_accounting_requires_review"
                break
        if stopped:
            break
    summary = recovery.summarize(output, protocol)
    gate = recovery.development_gate(summary) if args.phase == "development" else None
    write_json(output / "stop-go.json", {"stopped": stopped, "gate": gate})
    return {"stopped": stopped, "summary": summary, "gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        type=Path,
        required=True,
        help="existing published worktree with private data and ledgers",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "reports/evidence-recovery-selection.json"
    )
    parser.add_argument("--phase", choices=("development", "holdout"), default="development")
    parser.add_argument(
        "--case-limit", type=int, default=0, help="staged prefix only; 0 means full frozen phase"
    )
    parser.add_argument("--development-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.case_limit < 0:
        parser.error("case-limit must be nonnegative")
    protocol, bundles = freeze(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "prepared": str(args.output.resolve()),
                    "tasks": len(bundles),
                    "forecast": recovery.forecast(),
                }
            )
        )
        return
    if args.phase == "holdout":
        if not args.development_output:
            parser.error("holdout execution requires --development-output")
        development = recovery.read_json(args.development_output / "summary.json")
        dev_protocol = recovery.read_json(args.development_output / "protocol.json")
        if dev_protocol["source_hashes"] != protocol["source_hashes"]:
            raise ValueError("RECOVERY_DEVELOPMENT_SOURCE_CHANGED")
        if not recovery.development_gate(development)["passed"]:
            raise ValueError("RECOVERY_DEVELOPMENT_GATE_NOT_PASSED")
    campaign = args.main.resolve() / "reports/expanded-v1-20261107"
    base = campaign / "runs/main-v1"
    if not (campaign / "budget120/authorization.json").exists():
        raise ValueError("RECOVERY_REQUIRES_EXISTING_CNY120_ALLOCATION")
    driver = load_driver(Path(protocol["budget_driver_path"]), protocol["budget_driver_sha256"])
    with ExitStack() as stack:
        stack.enter_context(exclusive_run(campaign / "allocation"))
        for index in range(4):
            stack.enter_context(exclusive_run(campaign / f"lane-{index}-execution"))
        directory, receipt = driver.allocate_extension(campaign, base, approved_limit=120)
        immutable(
            args.output.resolve() / "budget-binding.json",
            {
                "existing_directory": str(directory),
                "receipt_digest": digest_json(receipt),
                "new_allowance": 0,
                "cumulative_authorized_limit_cny": 120,
            },
        )
        result = asyncio.run(execute(args, protocol, bundles, driver, directory, receipt))
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "stopped": result["stopped"],
                    "gate": result["gate"],
                    "completed_cells": result["summary"]["completed_cells"],
                }
            )
        )
        if result["stopped"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
