"""Freeze, smoke-test, then compare JSON formats using existing CNY120 lanes only."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.discovery_model import QwenReviewModel
from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.agent.structured_schemas import reviewer_schema, writer_schema
from dataverify.evaluation.expanded_experiment import SEED, _events, message, public_bundles
from dataverify.evaluation.expanded_scoring import aggregate_runs
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.evaluation.structured_confirmation import METHODS, cached, paired_cell, shared_sql
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import CellBudget
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("CONFIRMATION_FROZEN_RECORD_CHANGED")
    else:
        write_json(path, value)


def local_hashes():
    files = sorted((ROOT / "dataverify").rglob("*.py"))
    files += sorted((ROOT / "requirements").glob("*.txt")) + [Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix(): file_digest(p) for p in files}


def assert_runtime_isolated():
    """A foreign driver's import path must not replace the experimental code."""
    from dataverify.agent import discovery_model, graph_model

    for name, module in tuple(sys.modules.items()):
        if name == "dataverify" or name.startswith("dataverify."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(ROOT / "dataverify"):
                raise ValueError("CONFIRMATION_FOREIGN_RUNTIME_MODULE")
    if (
        discovery_model.QwenReviewModel is not QwenReviewModel
        or graph_model.QwenGraphModel.response_schema.__module__ != graph_model.__name__
        or QwenReviewModel.response_schema.__module__ != discovery_model.__name__
    ):
        raise ValueError("CONFIRMATION_MODEL_ADAPTER_IDENTITY_CHANGED")


def load_driver(path, digest):
    """Load the frozen accounting driver, retaining this worktree's runtime."""
    assert_runtime_isolated()
    if file_digest(path) != digest:
        raise ValueError("CONFIRMATION_BUDGET_DRIVER_CHANGED")
    spec = importlib.util.spec_from_file_location("confirmation_budget_driver", path)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    assert_runtime_isolated()
    if getattr(module, "QwenReviewModel", None) is not QwenReviewModel:
        raise ValueError("CONFIRMATION_BUDGET_DRIVER_IMPORTED_FOREIGN_MODEL")
    for name in ("allocate_extension", "load_lane", "lane_quarantines", "rollover_truncation"):
        if not hasattr(module, name):
            raise ValueError("CONFIRMATION_BUDGET_DRIVER_INTERFACE_MISSING")
    return module


def build_protocol(manifest, base_protocol, driver_path):
    bundles = public_bundles(manifest, "test")
    if len(bundles) != 40 or len({b.case_id for b in bundles}) != 40:
        raise ValueError("CONFIRMATION_REQUIRES_PRESELECTED_40_UNIQUE_TASKS")
    old = {item["case_id"] for item in base_protocol["cases"]}
    if old & {bundle.case_id for bundle in bundles}:
        raise ValueError("CONFIRMATION_OVERLAPS_MAIN_TEST_TASKS")
    return {
        "format": "dataverify-format-confirmation-v1",
        "seed": SEED,
        "model": PRICE_MODEL,
        "manifest_digest": digest_json(manifest),
        "base_protocol_digest": digest_json(base_protocol),
        "budget_driver_path": str(driver_path.resolve()),
        "budget_driver_sha256": file_digest(driver_path),
        "source_hashes": local_hashes(),
        "methods": list(METHODS),
        "databases": sorted({b.family for b in bundles}),
        "database_task_counts": {
            db: sum(b.family == db for b in bundles) for db in sorted({b.family for b in bundles})
        },
        "task_count": len(bundles),
        "expected_cells": 80,
        "shared_initial_sql": "one json_object writer call per task",
        "reference_generation": "separate by mode, same prompt, up to 3 format-repair calls",
        "verification": "same full policy, up to 12 post-initial calls, 3 attempts/3 reviews/40 search examples",
        "temperature": 0,
        "max_output_tokens": 1800,
        "schema_digests": {
            "writer": digest_json(writer_schema()),
            "reviewer": digest_json(reviewer_schema()),
        },
        "cases": [
            {
                "case_id": b.case_id,
                "task_digest": digest_json(b.task.public_input()),
                "source_digest": digest_json(b.source.to_dict()),
            }
            for b in bundles
        ],
        "no_reroll_policy": "Completed failures stay failed; incomplete paid stages require manual review. No tuning on confirmation results.",
        "limitations": [
            "JSON Schema constrains structure, not SQL/DSL meaning or business truth.",
            "Same new questions in existing domains; not an unseen-database test.",
            "References are independently sampled per format; finite-sample variability remains.",
            "Strict schema adds reserved input cost and cannot prevent token-cap truncation.",
        ],
    }


def check_frozen(protocol):
    assert_runtime_isolated()
    if local_hashes() != protocol["source_hashes"]:
        raise ValueError("CONFIRMATION_FROZEN_SOURCE_CHANGED")
    if file_digest(Path(protocol["budget_driver_path"])) != protocol["budget_driver_sha256"]:
        raise ValueError("CONFIRMATION_BUDGET_DRIVER_CHANGED")


def reservation_forecast(tasks=40):
    """Illustrative reserve planning, explicitly separated from the hard bound.

    Reservations never refund completed/unknown calls. This calculation does
    not claim a tokenizer estimate or that the full study fits the remaining
    allowance. Existing lane budgets always enforce the authorized ceiling.
    """
    fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": QwenReviewModel.schema_name,
            "strict": True,
            "schema": reviewer_schema(),
        },
    }
    schema_chars = len(json.dumps(fmt, ensure_ascii=False, separators=(",", ":")))

    def reserve(chars):
        return (4 * chars + 4096) * 0.8 / 1_000_000 + 1800 * 2.7 / 1_000_000

    calls_per_mode = 6  # Planning assumption, not the allowed maximum of 15.
    messages = 12_000
    planning = tasks * (1 + calls_per_mode) * reserve(messages) + (
        tasks * calls_per_mode + 1
    ) * reserve(messages + schema_chars)
    max_requests = tasks * (1 + 2 * (3 + 12)) + 1
    return {
        "planning_assumptions": {
            "message_chars_per_call": messages,
            "calls_per_mode_including_reference": calls_per_mode,
        },
        "planning_reserved_cny": round(planning, 6),
        "absolute_max_requests": max_requests,
        "absolute_max_reserved_cny": round(max_requests * reserve(40_000), 6),
        "caveat": "Planning scenario is not an upper bound. Full completion is not guaranteed; no budget increase, refund or selective rerun is permitted.",
    }


async def smoke(config, lane, output, protocol):
    path = output / "smoke.json"
    binding = {"protocol_digest": digest_json(protocol), "model": config.model}
    previous = cached(path, binding)
    if previous is not None:
        return previous
    write_json(path, {"binding": binding, "completed": False})
    start = lane.snapshot()
    model = QwenReviewModel(config, CellBudget(lane, 1), max_tokens=1800, structured_output=True)
    record = {"binding": binding, "passed": False}
    try:
        record["action"] = await model.generate(
            [
                message(
                    "system",
                    'Protocol compatibility smoke test. Return only JSON: {"action":"clarify","reason":"Schema compatibility check"}. Do not generate SQL or fixtures.',
                )
            ]
        )
        record["passed"] = record["action"]["action"] == "clarify"
    except Exception as error:
        record["error_type"] = type(error).__name__
    finally:
        await model.aclose()
    events = _events([("strict_schema_smoke", model)], lane)
    record["passed"] = (
        record["passed"]
        and len(events) == 1
        and events[0]["usage_complete"]
        and events[0]["status"] == "completed"
    )
    record.update(
        completed=True, http_events=events, budget_start=start, budget_end=lane.snapshot()
    )
    write_json(path, record)
    return record


def save_summary(output, protocol):
    records = [read(p) for p in (output / "records").glob("*.json")]
    rows = [r["row"] for r in records if r.get("completed") and "row" in r]
    initial = [
        read(p) for folder in ("initial", "references") for p in (output / folder).glob("*.json")
    ]
    if (output / "smoke.json").exists():
        initial.append(read(output / "smoke.json"))
    observed = {(r["case_id"], r["method"]) for r in rows}
    missing = [
        {"case_id": case["case_id"], "method": method}
        for case in protocol["cases"]
        for method in METHODS
        if (case["case_id"], method) not in observed
    ]
    value = {
        "protocol_digest": digest_json(protocol),
        "completed_cells": len(rows),
        "expected_cells": 80,
        "report_status": "complete" if not missing else "interim",
        "missing_cells": missing,
        "incomplete_records": [
            p.name for p in (output / "records").glob("*.json") if not read(p).get("completed")
        ],
        "rows": rows,
        "analysis": aggregate_runs(rows, reference_method="json_object", initial_records=initial),
        "initial_reference_acceptance": {
            method: {
                "accepted": sum(
                    r["initial_reference_accepted"] for r in rows if r["method"] == method
                ),
                "tasks": sum(r["method"] == method for r in rows),
                "validation_failures": sum(
                    r["reference_validation_failures"] for r in rows if r["method"] == method
                ),
            }
            for method in METHODS
        },
        "timing_scope": "Reference generation plus subsequent verification; shared SQL generation excluded.",
        "reference_acceptance_note": "Acceptance includes local DSL/fixture validity; it is not a pure JSON compliance rate or a correctness measure.",
    }
    write_json(output / "summary.json", value)
    return value


async def execute(manifest, protocol, output, driver, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    approval = read(output / "smoke.json")
    if approval.get("passed") is not True or approval.get("binding") != {
        "protocol_digest": digest_json(protocol),
        "model": config.model,
    }:
        raise ValueError("CONFIRMATION_REQUIRES_SUCCESSFUL_REAL_SMOKE")
    bundles = public_bundles(manifest, "test")
    random.Random(SEED).shuffle(bundles)
    stopped = asyncio.Event()

    async def worker(index):
        budget = driver.load_lane(directory, index, receipt)
        quarantined = driver.lane_quarantines(directory, index, receipt)

        def after(report):
            nonlocal budget
            if budget.snapshot()["stopped_reason"]:
                try:
                    budget = driver.rollover_truncation(
                        directory, index, budget, report, quarantined
                    )
                except ValueError:
                    stopped.set()

        try:
            for bundle in bundles[index::4]:
                if stopped.is_set():
                    break
                check_frozen(protocol)
                common = await shared_sql(
                    bundle, config, budget, output / "initial" / f"{bundle.case_id}.json"
                )
                after(common)
                if stopped.is_set():
                    break
                methods = list(METHODS)
                random.Random(SEED + int(digest_json(bundle.case_id)[:8], 16)).shuffle(methods)
                for method in methods:
                    if stopped.is_set():
                        break
                    check_frozen(protocol)
                    result = await paired_cell(bundle, method, common, config, budget, output)
                    print(
                        json.dumps(
                            {
                                "case": bundle.case_id,
                                "method": method,
                                "status": result["row"]["status"],
                                "correct": result["row"]["score"]["correct_delivery"],
                            }
                        ),
                        flush=True,
                    )
                    after(result)
        except BaseException:
            stopped.set()
            raise

    results = await asyncio.gather(*(worker(i) for i in range(4)), return_exceptions=True)
    for value in results:
        if isinstance(value, BaseException):
            print(json.dumps({"worker_error": type(value).__name__}), flush=True)
    summary = save_summary(output, protocol)
    print(
        json.dumps(
            {
                "completed_cells": summary["completed_cells"],
                "expected_cells": 80,
                "stopped": stopped.is_set(),
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/structured-confirmation-v1")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = args.base_run.resolve()
    campaign = base.parent.parent
    manifest_path = args.manifest or campaign / "confirmation-manifest.json"
    driver_path = campaign.parents[1] / "tools/continue_expanded.py"
    manifest, base_protocol = read(manifest_path), read(base / "protocol.json")
    for database in manifest["databases"].values():
        if file_digest(Path(database["path"])) != database["sha256"]:
            raise ValueError("ORIGINAL_DATABASE_CHANGED")
    protocol = build_protocol(manifest, base_protocol, driver_path)
    output = args.output.resolve()
    for folder in (
        output,
        *(output / name for name in ("initial", "references", "records", "checkpoints")),
    ):
        folder.mkdir(parents=True, exist_ok=True)
    immutable(output / "protocol.json", protocol)
    immutable(output / "manifest.json", manifest)
    archive = output / "frozen-source.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as saved:
            for relative in protocol["source_hashes"]:
                saved.write(ROOT / relative, relative)
            saved.write(driver_path, "budget_driver/continue_expanded.py")
            saved.write(output / "protocol.json", "protocol.json")
    if not args.smoke and not args.execute:
        print(
            json.dumps(
                {
                    "frozen": str(output),
                    "tasks": 40,
                    "cells": 80,
                    "databases": protocol["databases"],
                    "database_task_counts": protocol["database_task_counts"],
                    "digest": digest_json(protocol),
                    "reservation_forecast": reservation_forecast(),
                }
            )
        )
        return
    if (
        not (campaign / "budget120/authorization.json").exists()
        or not (campaign / "budget120-allocation.json").exists()
    ):
        raise ValueError("CONFIRMATION_REQUIRES_EXISTING_CNY120_ALLOCATION")
    driver = load_driver(driver_path, protocol["budget_driver_sha256"])
    with ExitStack() as stack:
        stack.enter_context(exclusive_run(campaign / "allocation"))
        for index in range(4):
            stack.enter_context(exclusive_run(campaign / f"lane-{index}-execution"))
        directory, receipt = driver.allocate_extension(campaign, base, approved_limit=120)
        immutable(
            directory / "structured-confirmation-binding.json",
            {"output": str(output), "protocol_digest": digest_json(protocol)},
        )
        immutable(
            output / "budget-authorization-binding.json",
            {
                "directory": str(directory),
                "receipt_digest": digest_json(receipt),
                "new_allowance": 0,
                "cumulative_limit_cny": 120,
            },
        )
        lane_snapshots = [driver.load_lane(directory, i, receipt).snapshot() for i in range(4)]
        print(
            json.dumps(
                {
                    "existing_remaining_reserved_cny_by_lane": [
                        round(s["money_limit_cny"] - s["reserved_upper_cny"], 6)
                        for s in lane_snapshots
                    ],
                    "reservation_forecast": reservation_forecast(),
                }
            ),
            flush=True,
        )
        if args.smoke:
            config = QwenConfig.from_local_environment()
            validate_pricing(config)
            check_frozen(protocol)
            result = asyncio.run(
                smoke(config, driver.load_lane(directory, 0, receipt), output, protocol)
            )
            print(
                json.dumps(
                    {"smoke_passed": result["passed"], "new_calls": len(result["http_events"])}
                )
            )
            if not result["passed"]:
                raise SystemExit(2)
        else:
            result = asyncio.run(execute(manifest, protocol, output, driver, directory, receipt))
            if result["completed_cells"] != 80:
                raise SystemExit(2)


if __name__ == "__main__":
    main()
