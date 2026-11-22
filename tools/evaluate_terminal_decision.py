"""Evaluate reserving the last diagnosis action; defaults to freezing, never paying."""

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
from evaluate_query_ablation import hash_sources
from evaluate_structured_confirmation import immutable, load_driver, read

from dataverify.agent.dispute_model import is_final_decision
from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.evaluation import dispute_recovery as dev
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation import query_ablation as paired
from dataverify.evaluation import terminal_scenarios as fresh
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json

BASELINE = "query_on"
TREATMENT = "last_decision"
PARENT_DIGEST = "b434956c24baec3af1dafc1770917e67afd1203268431eb2e06432a0d0905ed1"


def schedule():
    for case in dev.study_cases():
        yield "development", case, TREATMENT
    for i, case in enumerate(fresh.study_cases()):
        for arm in (BASELINE, TREATMENT) if i % 2 == 0 else (TREATMENT, BASELINE):
            yield "new_cases", case, arm


def controls(parent):
    original = read(parent / "protocol.json")
    if digest_json(original) != PARENT_DIGEST:
        raise ValueError("Unexpected parent protocol")
    result = {}
    for case in original["cases"]:
        name = case["id"] + "--query_on.json"
        path = parent / "records" / name
        item = read(path)
        expected = {
            "protocol": PARENT_DIGEST,
            "case": case["id"],
            "model": PRICE_MODEL,
            "method": BASELINE,
        }
        if item.get("binding") != expected or item.get("completed") is not True:
            raise ValueError("Unusable control " + name)
        result[name] = file_digest(path)
    return result


def source_hashes(root):
    files = hash_sources(root)
    name = "tools/evaluate_terminal_decision.py"
    files[name] = file_digest(root / name)
    return files


def freeze(main, parent, output):
    manifest = output / "protocol.json"
    snapshot = output / "source"
    bound = controls(parent)
    driver_digest = file_digest(main / "tools/continue_expanded.py")
    if manifest.exists():
        protocol = read(manifest)
        if (
            source_hashes(snapshot) != protocol["source_hashes"]
            or protocol["controls"] != bound
            or driver_digest != protocol["driver"]
        ):
            raise ValueError("Archived code, accounting driver or controls changed")
        return protocol
    files = source_hashes(ROOT)
    for name, digest in files.items():
        dest = snapshot / name
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(ROOT / name, dest)
        if file_digest(dest) != digest:
            raise ValueError("Source changed during freeze")
    if read(parent / "protocol.json")["cases"] != paired.frozen_cases():
        raise ValueError("Historical task inputs differ")
    cells = [[group, case.case_id, arm] for group, case, arm in schedule()]
    cap = len(cells) * dev.MAX_ACTIONS
    protocol = {
        "version": 1,
        "model": PRICE_MODEL,
        "parent": PARENT_DIGEST,
        "controls": bound,
        "driver": driver_digest,
        "source_hashes": files,
        "development": paired.frozen_cases(),
        "new_cases": fresh.frozen_cases(),
        "cells": cells,
        "max_new_calls": cap,
        "max_reserved_cny": bounded.max_reservation(cap),
        "limits": {
            "actions": dev.MAX_ACTIONS,
            "queries": dev.MAX_QUERIES,
            "revisions": dev.MAX_REVISIONS,
        },
        "design": "Tools enabled in both arms. Treatment reserves final action for finish, clarify or abstain. Same initial inputs, full source, Qwen client, budget, test data. Six archived dev controls reused; alternate ordering of fresh paired runs. No contextual oracle.",
        "scope": "New requirements are authored for this evaluation on the same schema, not independent external holdout data. One run per arm, no regeneration or retuning.",
        "primary": "Paired correct deliveries; separately report wrong deliveries, abstentions, joint-correct delivery, requests and tokens. Keep the two cohorts separate.",
        "decision": "Keep investigation opt-in. Do not enable terminal restriction by default on any new wrong release or paired deterioration on new cases. Absent clear fresh-case delivery gain attributable to reaching the final turn, keep experimental. Stochastic changes before the final turn do not establish causation.",
    }
    immutable(manifest, protocol)
    return protocol


def check_bindings(protocol):
    if (
        protocol["development"] != paired.frozen_cases()
        or protocol["new_cases"] != fresh.frozen_cases()
        or protocol["cells"] != [[g, c.case_id, a] for g, c, a in schedule()]
    ):
        raise ValueError("Frozen execution bindings have changed")


async def execute(output, protocol, driver, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    seen_ids = set()
    for group, case, arm in schedule():
        arm_dir = output / arm
        (arm_dir / "records").mkdir(exist_ok=True, parents=True)
        path = arm_dir / "records" / (case.case_id + "--query_on.json")
        _, lane = choose_lane(driver, directory, receipt, 0 if path.exists() else dev.MAX_ACTIONS)
        item = await paired.run_cell(
            case,
            BASELINE,
            config,
            lane,
            arm_dir,
            digest_json(protocol),
            reserve_final_decision=arm == TREATMENT,
            grader=fresh.grade if group == "new_cases" else paired.grade,
        )
        events = item["http_events"]
        if not events or not accounted(item, lane):
            raise ValueError("Incomplete payment record; pause before any further request")
        ids = {e["event_id"] for e in events}
        if len(ids) != len(events) or ids & seen_ids:
            raise ValueError("Duplicate payment event")
        seen_ids.update(ids)
        print(group, case.case_id, arm, item["record"]["status"], len(events), flush=True)


def audit(parent, output, protocol):
    rows = []
    usage_events = []
    reused_events = []
    seen_ids = set()
    protocol_id = digest_json(protocol)
    for group, cases, grader in (
        ("development", dev.study_cases(), paired.grade),
        ("new_cases", fresh.study_cases(), fresh.grade),
    ):
        for case in cases:
            for arm in (BASELINE, TREATMENT):
                historical = group == "development" and arm == BASELINE
                folder = parent if historical else output / arm
                item = read(folder / "records" / (case.case_id + "--query_on.json"))
                binding = {
                    "case": case.case_id,
                    "protocol": PARENT_DIGEST if historical else protocol_id,
                    "method": BASELINE,
                    "model": PRICE_MODEL,
                }
                if arm == TREATMENT:
                    binding["reserve_final_decision"] = True
                events = item["http_events"]
                if (
                    not item["completed"]
                    or item["binding"] != binding
                    or not events
                    or summarize_usage(events) != item["usage"]
                    or not item["usage"]["usage_complete"]
                ):
                    raise ValueError("Record binding or usage mismatch")
                for event in events:
                    if (
                        event["event_id"] in seen_ids
                        or event["status"] != "completed"
                        or event["http_status"] != 200
                    ):
                        raise ValueError("Duplicate or failed billed HTTP event")
                    seen_ids.add(event["event_id"])
                (reused_events if historical else usage_events).extend(events)
                score = grader(case, item["record"])
                if score != item["score"]:
                    raise ValueError("Score does not reproduce")
                record = item["record"]
                if len(events) > dev.MAX_ACTIONS or bool(
                    record["policy"].get("reserve_final_decision")
                ) != (arm == TREATMENT):
                    raise ValueError("Arm or budget mismatch")
                exposed = any(is_final_decision(messages) for messages in item["requests"])
                if exposed != (arm == TREATMENT and len(events) == dev.MAX_ACTIONS):
                    raise ValueError("Reserved turn exposure differs from observed prompt")
                rows.append(
                    {
                        "group": group,
                        "case": case.case_id,
                        "arm": arm,
                        "status": record["status"],
                        "reason": record["reason"],
                        "correct": score["correct_delivery"],
                        "wrong": score["wrong_release"],
                        "joint_correct": score["joint_correct_delivery"],
                        "queries": record["query_calls"],
                        "http": len(events),
                        "tokens": item["usage"]["total_tokens_known"],
                        "historical": historical,
                        "reached_last_slot": len(events) == dev.MAX_ACTIONS,
                        "saw_terminal_prompt": exposed,
                    }
                )
    result = {
        "protocol": protocol_id,
        "new_usage": summarize_usage(usage_events),
        "reused_usage": summarize_usage(reused_events),
        "rows": rows,
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("main", "parent", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    main_repo, parent, output = (x.resolve() for x in (args.main, args.parent, args.output))
    if output == parent or output.is_relative_to(parent):
        raise ValueError("Choose a fresh output directory separate from parent controls")
    output.mkdir(parents=True, exist_ok=True)
    protocol = freeze(main_repo, parent, output)
    snapshot = output / "source"
    if not (args.execute or args.audit):
        print(
            json.dumps(
                {"protocol": digest_json(protocol), "max_new_calls": protocol["max_new_calls"]}
            )
        )
        return
    if ROOT != snapshot:
        subprocess.run(
            [
                sys.executable,
                str(snapshot / "tools/evaluate_terminal_decision.py"),
                "--main",
                str(main_repo),
                "--parent",
                str(parent),
                "--output",
                str(output),
                "--execute" if args.execute else "--audit",
            ],
            cwd=snapshot,
            check=True,
        )
        return
    check_bindings(protocol)
    if args.execute:
        campaign = main_repo / "reports/expanded-v1-20261107"
        if not (campaign / "budget120/authorization.json").is_file():
            raise ValueError("Existing cumulative authorization required")
        driver = load_driver(main_repo / "tools/continue_expanded.py", protocol["driver"])
        with ExitStack() as locks:
            locks.enter_context(exclusive_run(campaign / "allocation"))
            for lane in range(4):
                locks.enter_context(exclusive_run(campaign / f"lane-{lane}-execution"))
            directory, receipt = driver.allocate_extension(
                campaign, campaign / "runs/main-v1", approved_limit=120
            )
            asyncio.run(execute(output, protocol, driver, directory, receipt))
    print(json.dumps(audit(parent, output, protocol)["new_usage"]))


if __name__ == "__main__":
    main()
