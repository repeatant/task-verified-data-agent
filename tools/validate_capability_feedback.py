"""Freeze boundary regressions, compare legacy/v2 repairs, then run eight data conditions.

No API unless --live is explicit. Live always continues a prior complete money
ledger; it cannot silently obtain a fresh CNY 5 allowance. Runtime rules are
identical in both arms, including the separately added numeric-output gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from run_probe_workflow import restore_completed_budget, write

from dataverify.capability_cases import get_capability_cases
from dataverify.dbgpt_qwen import QwenConfig
from dataverify.probe_cases import grade_rows, reference_result
from dataverify.probe_cost_budget import MonetaryRequestBudget, validate_pricing
from dataverify.probe_guard import ProbeGuard
from dataverify.probe_live_benchmark import run_probe_benchmark


async def validate(output: Path, *, live=False, previous_workflow: Path | None = None):
    output = output.resolve()
    if not output.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Output must be inside project reports")
    if live and previous_workflow is None:
        raise ValueError("Live validation must continue a prior complete budget")
    config = QwenConfig.from_local_environment() if live else None
    pricing = validate_pricing(config) if live else None
    budget = MonetaryRequestBudget()
    if previous_workflow is not None:
        previous_workflow = previous_workflow.resolve()
        if not live or not previous_workflow.is_relative_to((ROOT / "reports").resolve()):
            raise ValueError("Invalid live continuation path")
        previous = json.loads(previous_workflow.read_text(encoding="utf-8"))
        if previous.get("real_model_used") is not True:
            raise ValueError("Prior workflow must be live")
        budget = restore_completed_budget(previous["budget"])
    # Four repairs + eight natural tasks x at most two HTTP calls. The old
    # ledger has 58 attempts, so the unchanged global 80-call cap is enough.
    if live and budget.snapshot()["http_attempt_limit"] - budget.snapshot()["http_attempts"] < 20:
        raise ValueError("Not enough remaining request allowance for the frozen plan")
    output.mkdir(parents=True, exist_ok=False)
    entries = get_capability_cases()
    sources = (
        "dataverify/probe_capabilities.py",
        "dataverify/probe_guard.py",
        "dataverify/probe_agent_bridge.py",
        "dataverify/probe_runtime.py",
        "dataverify/probe_testing.py",
        "dataverify/capability_cases.py",
        "dataverify/probe_live_benchmark.py",
        "tools/validate_capability_feedback.py",
    )
    report_dirs = [
        "legacy_seed17",
        "capability_seed17",
        "capability_seed31",
        "legacy_seed31",
        "natural_v2",
    ]
    frozen = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-change boundary regressions in existing task families; not unseen-template holdout.",
        "component_cases": [asdict(entry) for entry in entries],
        "category_counts": dict(Counter(entry.category for entry in entries)),
        "feedback_policies": ["legacy_v1", "capability_v2"],
        "runtime_rules": "Shared static checks, execution boundaries and numeric-result gate.",
        "live_plan": report_dirs,
        "max_additional_http_attempts": 20,
        "budget_start": budget.snapshot(),
        "pricing": pricing,
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
    }
    write(output / "frozen_validation.json", frozen)
    document = {
        "status": "running",
        "real_model_used": live,
        "report_dirs": report_dirs,
        "frozen_manifest": "frozen_validation.json",
        "budget": budget.snapshot(),
        "scope": frozen["scope"],
        "runs": [],
    }
    if previous_workflow is not None:
        document["continued_from"] = str(previous_workflow.relative_to(ROOT))
    write(output / "workflow.json", document)
    try:
        checks = []
        for entry in entries:
            case = entry.case
            for policy in ("legacy_v1", "capability_v2"):
                guard = ProbeGuard(case.task, case.source, feedback_policy=policy)
                guard.execute(case.candidate_sql, case.task.task_id)
                publication = guard.publish()
                accepted = publication["status"] == "verified"
                target = reference_result(case)
                grade = grade_rows(publication["rows"], target) if accepted else None
                passed = accepted == entry.expected_acceptance and (
                    not accepted or grade["correct"]
                )
                checks.append(
                    {
                        "case_id": case.case_id,
                        "category": entry.category,
                        "policy": policy,
                        "expected_acceptance": entry.expected_acceptance,
                        "accepted": accepted,
                        "passed": passed,
                        "publication": publication,
                        "independent_target": target,
                        "grade": grade,
                        "source_output_check": guard.last_result.get("source_output_check"),
                        "static": (guard.last_result.get("probe_report") or {}).get("static"),
                    }
                )
        component = {
            "checks": checks,
            "passed": sum(item["passed"] for item in checks),
            "scheduled": len(checks),
            "all_passed": all(item["passed"] for item in checks),
        }
        write(output / "component_validation.json", component)
        document["component_validation"] = {
            key: component[key] for key in ("passed", "scheduled", "all_passed")
        }
        if not component["all_passed"]:
            document["status"] = "component_failed"
            return document
        runs = []
        for directory, policy, seed in (
            ("legacy_seed17", "legacy_v1", 17),
            ("capability_seed17", "capability_v2", 17),
            ("capability_seed31", "capability_v2", 31),
            ("legacy_seed31", "legacy_v1", 31),
        ):
            result = await run_probe_benchmark(
                output / directory,
                live=live,
                config=config,
                case_ids=("d10",),
                include_natural=False,
                seeds=(seed,),
                modes=("compact",),
                budget=budget,
                feedback_policy=policy,
            )
            runs.append(result)
            document["runs"].append(
                {
                    "directory": directory,
                    "policy": policy,
                    "summary": result["summary"],
                    "usage": result["usage"],
                }
            )
            document["budget"] = budget.snapshot()
            write(output / "workflow.json", document)
        positives = tuple(entry.case for entry in entries if entry.expected_acceptance)
        natural = await run_probe_benchmark(
            output / "natural_v2",
            live=live,
            config=config,
            evaluation_cases=positives,
            natural_case_ids=tuple(case.case_id for case in positives),
            include_controlled=False,
            seeds=(17,),
            modes=("compact",),
            budget=budget,
            feedback_policy="capability_v2",
        )
        runs.append(natural)
        document["runs"].append(
            {
                "directory": "natural_v2",
                "policy": "capability_v2",
                "summary": natural["summary"],
                "usage": natural["usage"],
            }
        )
        document.update(
            status="completed"
            if all(result["all_cells_completed"] for result in runs)
            else "incomplete",
            all_ledgers_complete=all(result["audit_ledger_complete"] for result in runs),
        )
    except BaseException as exc:
        budget.stop("CAPABILITY_VALIDATION_ERROR")
        document.update(
            status="interrupted"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "error",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        document.update(
            budget=budget.snapshot(), finished_at_utc=datetime.now(timezone.utc).isoformat()
        )
        write(output / "workflow.json", document)
    return document


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--previous-workflow", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("capability_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    args = parser.parse_args()
    result = asyncio.run(
        validate(args.output, live=args.live, previous_workflow=args.previous_workflow)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "component": result.get("component_validation"),
                "budget": result["budget"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
