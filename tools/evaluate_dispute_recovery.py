"""Prepare or explicitly run a six-trajectory three-program development check.

Default and --prepare are offline. --execute alone enables model requests, under
the existing cumulative CNY120 ledger. This is a controlled disagreement-repair
experiment on seen requirements, not a test of discovering unseen fault types.
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

from evaluate_evidence_recovery import accounted, choose_lane
from evaluate_structured_confirmation import immutable, load_driver

from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation import dispute_recovery as experiment
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json, write_json


def source_hashes():
    files = sorted((ROOT / "dataverify").rglob("*.py"))
    files += sorted((ROOT / "requirements").glob("*.txt"))
    files += [
        Path(__file__).resolve(),
        ROOT / "tools/evaluate_evidence_recovery.py",
        ROOT / "tools/evaluate_structured_confirmation.py",
    ]
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in files}


def build_protocol(main):
    cases = experiment.study_cases()
    driver = main / "tools/continue_expanded.py"
    maximum_calls = len(cases) * (1 + 2 * experiment.MAX_ACTIONS)
    return {
        "format": "dataverify-dispute-development-v1",
        "status": "experimental opt-in controlled development; existing discovery graph unchanged",
        "model": PRICE_MODEL,
        "source_hashes": source_hashes(),
        "budget_driver_path": str(driver),
        "budget_driver_sha256": file_digest(driver),
        "methods": list(experiment.METHODS),
        "controlled_fault_trajectories": len(cases),
        "distinct_original_requirements": len({case.base.task.task_id for case in cases}),
        "visible_diagnostic_fixture": experiment.visible_diagnostic_fixture().to_dict(),
        "visibility": "Original source and the identical fixed diagnostic fixture are available to both methods. No fault labels, good-program annotations, expected results or scoring-only fixtures enter model messages.",
        "limits": {
            "resolver_model_calls": experiment.MAX_ACTIONS,
            "baseline_model_calls": 1,
            "legacy_three_roles_shared_calls": experiment.MAX_ACTIONS,
            "query_calls": experiment.MAX_QUERIES,
            "revision_attempts": experiment.MAX_REVISIONS,
            "request_input_chars": bounded.INPUT_CHAR_CAP,
            "request_output_tokens": bounded.OUTPUT_TOKEN_CAP,
        },
        "cost_bound": {
            "maximum_new_calls": maximum_calls,
            "maximum_reserved_cny": bounded.max_reservation(maximum_calls),
            "new_allowance": 0,
            "no_refund": "Historical reservations including unknown attempts remain intact.",
        },
        "grading": {
            "source": "Existing handwritten arithmetic expected() over scoring_data().",
            "fixtures_per_trajectory": 25,
            "fixture_visibility": "One original source plus 24 grading-only datasets. The new visible diagnostic fixture is separate from these 25.",
            "programs": ["candidate SQL", "first reference", "auditor reference"],
            "baseline_reference_and_auditor": "N/A; baseline does not produce either program.",
            "publication_row_scope": "legacy_full returns original-source rows; self_review and dispute_resolver return diagnostic-source rows. Each is checked against its actual source; all SQL and reference programs share the same 25 grading fixtures.",
            "metrics": [
                "wrong to correct",
                "correct to wrong",
                "correct SQL rejected",
                "wrong SQL delivered",
                "wrong reference delivered",
                "wrong auditor delivered",
                "joint correct delivery",
                "unchanged delivery",
                "calls/tokens/reservation",
            ],
        },
        "no_reroll_policy": "Every completed failure remains. An incomplete paid record requires explicit manual review. Do not tune against these results and overwrite them.",
        "negative_control": "All three programs share the same lower-bound mistake. Agreement is not truth; erroneous delivery must be reported unchanged. The two-revision cap cannot repair all three programs, so this control measures recognition/non-delivery/wrong release, not full-recovery capability. Candidate SQL repair is still graded separately.",
        "limitations": [
            "Six fault trajectories on five distinct previously seen requirements; no unseen-task or unseen-fault claim.",
            "The diverse visible fixture is author-supplied experiment support, not automatically discovered evidence.",
            "Candidate-only one-call self-review is a whole-method comparator, not a perfectly isolated role ablation.",
            "Matched legacy_full receives the same three initial programs and visible fixtures, sharing a four-request cap across its three roles. Its existing generated challenges remain part of that older method.",
            "Finite fixture correctness does not prove arbitrary-input correctness.",
            "The shared-error negative control may expose an unresolved common-mode failure; no hidden gold gate repairs it.",
            "The all_wrong control starts with three faulty programs but permits only two revisions. Failure to repair all three is structurally constrained and must not be interpreted as pure model-reasoning failure.",
        ],
        "cases": [
            {
                "case_id": case.case_id,
                "parent_id": case.base.parent_id,
                "task_id": case.base.task.task_id,
                "condition": case.condition,
                **experiment.case_binding(case),
            }
            for case in cases
        ],
    }


def prepare(args):
    protocol = build_protocol(args.main.resolve())
    output = args.output.resolve()
    (output / "records").mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    immutable(output / "protocol.json", protocol)
    archive = output / "frozen-source.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
            for relative in protocol["source_hashes"]:
                saved.write(ROOT / relative, relative)
            saved.write(Path(protocol["budget_driver_path"]), "budget_driver/continue_expanded.py")
            saved.write(output / "protocol.json", "protocol.json")
    return protocol


async def execute(args, protocol, driver, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    output = args.output.resolve()
    cases = experiment.study_cases()
    limit = args.case_limit or len(cases)
    stopped = None
    for case in cases[:limit]:
        for method in experiment.METHODS:
            if source_hashes() != protocol["source_hashes"]:
                raise ValueError("DISPUTE_FROZEN_SOURCE_CHANGED")
            calls = 1 if method == "self_review" else experiment.MAX_ACTIONS
            if (output / "records" / f"{case.case_id}--{method}.json").exists():
                calls = 0  # run_cell validates a cached result before any model creation.
            _, lane = choose_lane(driver, directory, receipt, calls)
            result = await experiment.run_cell(
                case, method, config, lane, output, digest_json(protocol)
            )
            summary = experiment.summarize(output, protocol)
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "method": method,
                        "status": result["row"]["status"],
                        "completed_cells": summary["completed_cells"],
                    }
                ),
                flush=True,
            )
            if not accounted(result, lane):
                stopped = "accounting_or_transport_requires_review"
                break
        if stopped:
            break
    summary = experiment.summarize(output, protocol)
    write_json(
        output / "run-status.json",
        {
            "stopped": stopped,
            "complete": summary["complete"],
            "note": "Observed outcomes, including the shared-error negative control, are retained without result-based retries.",
        },
    )
    return stopped, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--case-limit", type=int, default=0, help="prefix only, 0 means all six frozen trajectories"
    )
    args = parser.parse_args()
    if not 0 <= args.case_limit <= 6:
        parser.error("case-limit must be 0..6")
    protocol = prepare(args)
    if not args.execute:
        print(
            json.dumps(
                {
                    "prepared": str(args.output.resolve()),
                    "controlled_fault_trajectories": protocol["controlled_fault_trajectories"],
                    "distinct_original_requirements": protocol["distinct_original_requirements"],
                    "cost_bound": protocol["cost_bound"],
                    "network_calls": 0,
                }
            )
        )
        return
    campaign = args.main.resolve() / "reports/expanded-v1-20261107"
    base = campaign / "runs/main-v1"
    if not (campaign / "budget120/authorization.json").exists():
        raise ValueError("DISPUTE_REQUIRES_EXISTING_CNY120_ALLOCATION")
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
        stopped, summary = asyncio.run(execute(args, protocol, driver, directory, receipt))
        print(
            json.dumps(
                {
                    "stopped": stopped,
                    "complete": summary["complete"],
                    "completed_cells": summary["completed_cells"],
                }
            )
        )
        if stopped:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
