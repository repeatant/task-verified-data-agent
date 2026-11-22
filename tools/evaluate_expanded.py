"""Run an explicitly frozen expanded campaign; all raw records remain private."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation.expanded_experiment import (
    ABLATIONS,
    BASE_METHODS,
    SEED,
    build_protocol,
    controlled_bundles,
    prepare_first,
    public_bundles,
    run_cell,
    source_hashes,
)
from dataverify.evaluation.expanded_scoring import aggregate_runs
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import (
    authorize_campaign,
    continue_accounted_truncation,
    lane_budget,
)
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json, write_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "test", "stability"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--public-manifest", type=Path, default=ROOT / "reports/public-data/bird-manifest-v2.json"
    )
    parser.add_argument("--campaign", type=Path, default=ROOT / "reports/expanded-v1-20261107")
    return parser.parse_args()


def save_summary(output, protocol):
    rows = [
        json.loads(p.read_text(encoding="utf-8"))["row"]
        for p in sorted((output / "records").glob("*.json"))
    ]
    initial_records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((output / "initial").glob("*.json"))
    ]
    summary = aggregate_runs(rows, bootstrap_repetitions=2000, initial_records=initial_records)
    write_json(
        output / "summary.json",
        {"protocol_digest": digest_json(protocol), "rows": rows, "analysis": summary},
    )
    return summary


async def run(args, manifest, protocol, output, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    bundles = public_bundles(manifest, "development" if args.phase == "development" else "test")
    if args.phase in {"test", "stability"}:
        bundles += controlled_bundles()
    if args.phase == "stability":
        bundles = [
            replace(bundle, case_id=bundle.case_id + f"--rep{rep}")
            for bundle in bundles
            if bundle.case_id in protocol["stability_case_ids"]
            for rep in (2, 3)
        ]
    random.Random(SEED).shuffle(bundles)
    if args.limit is not None:
        bundles = bundles[: args.limit]
    stopped = asyncio.Event()

    async def worker_body(index):
        # A lane's lifetime lock prevents overlapping paid processes reusing it.
        lock = args.campaign / f"lane-{index}-execution"
        with exclusive_run(lock):
            budget = lane_budget(args.campaign, index, receipt)
            for bundle in bundles[index :: receipt["workers"]]:
                if stopped.is_set():
                    break
                if source_hashes(ROOT) != protocol["source_hashes"]:
                    stopped.set()
                    raise ValueError("FROZEN_SOURCE_CHANGED")
                initial = await prepare_first(
                    bundle, config, budget, output / "initial" / (bundle.case_id + ".json")
                )
                if budget.snapshot()["stopped_reason"]:
                    try:
                        budget = continue_accounted_truncation(
                            args.campaign, index, receipt, budget, initial
                        )
                    except ValueError:
                        stopped.set()
                    print(
                        json.dumps(
                            {
                                "lane": index,
                                "stopped": budget.snapshot()["stopped_reason"],
                                "after": "initial",
                                "case": bundle.case_id,
                            }
                        ),
                        flush=True,
                    )
                    if stopped.is_set():
                        break
                methods = ["full"] if args.phase == "stability" else list(BASE_METHODS)
                if args.phase == "test" and bundle.case_id in protocol["ablation_case_ids"]:
                    methods += ABLATIONS
                random.Random(SEED + int(digest_json(bundle.case_id)[:8], 16)).shuffle(methods)
                for method in methods:
                    if stopped.is_set():
                        break
                    await run_cell(bundle, method, initial, config, budget, output)
                    if budget.snapshot()["stopped_reason"]:
                        cell_report = json.loads(
                            (output / "records" / f"{bundle.case_id}--{method}.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        try:
                            budget = continue_accounted_truncation(
                                args.campaign, index, receipt, budget, cell_report
                            )
                        except ValueError:
                            stopped.set()
                        print(
                            json.dumps(
                                {
                                    "lane": index,
                                    "stopped": budget.snapshot()["stopped_reason"],
                                    "after": method,
                                    "case": bundle.case_id,
                                }
                            ),
                            flush=True,
                        )
                        if stopped.is_set():
                            break

    async def worker(index):
        try:
            return await worker_body(index)
        except BaseException:
            # Stop later cells on every lane after an unclassified persistence,
            # accounting or infrastructure failure. Do not cancel an in-flight
            # HTTP call: its outcome still needs to be recorded in its own lane.
            stopped.set()
            raise

    results = await asyncio.gather(
        *(worker(i) for i in range(receipt["workers"])), return_exceptions=True
    )
    for result in results:
        if isinstance(result, BaseException):
            print(
                json.dumps({"worker_error": type(result).__name__, "code": str(result)[:160]}),
                flush=True,
            )
    summary = save_summary(output, protocol)
    print(
        json.dumps(
            {
                "finished": not stopped.is_set()
                and not any(isinstance(r, BaseException) for r in results),
                "records": len(list((output / "records").glob("*.json"))),
                "output": str(output),
                "summary_keys": list(summary),
            }
        ),
        flush=True,
    )


def main():
    args = arguments()
    if not args.version.replace("-", "").replace("_", "").isalnum():
        raise ValueError("INVALID_VERSION_NAME")
    manifest = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    for database in manifest["databases"].values():
        if file_digest(Path(database["path"])) != database["sha256"]:
            raise ValueError("ORIGINAL_DATABASE_CHANGED")
    output = args.campaign / "runs" / args.version
    output.mkdir(parents=True, exist_ok=True)
    for name in ("records", "initial", "checkpoints"):
        (output / name).mkdir(exist_ok=True)
    path = output / "protocol.json"
    expected = build_protocol(ROOT, manifest)
    expected.update(phase=args.phase, version=args.version, limit=args.limit)
    if path.exists():
        protocol = json.loads(path.read_text(encoding="utf-8"))
        if protocol != expected:
            raise ValueError("FROZEN_PROTOCOL_CHANGED_USE_NEW_VERSION")
    else:
        protocol = expected
        write_json(path, protocol)
    archive = output / "frozen-source.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as snapshot:
            for relative in protocol["source_hashes"]:
                snapshot.write(ROOT / relative, relative)
            snapshot.write(path, "protocol.json")
    if args.freeze_only:
        print(json.dumps({"frozen": str(path), "digest": digest_json(protocol)}))
        return
    prior = [
        ROOT.parent / "github-v1-langgraph-20261107/artifacts/v1-live-20261107/budget-02.json",
        ROOT / "reports/frozen-v1-live-20261107/budget.json",
    ]
    with exclusive_run(args.campaign / "allocation"):
        receipt = authorize_campaign(args.campaign, prior)
    asyncio.run(run(args, manifest, protocol, output, receipt))


if __name__ == "__main__":
    main()
