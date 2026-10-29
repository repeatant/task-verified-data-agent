"""Offline, disclosed development evaluation; never calls a model API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.probe_cases import get_cases, grade_rows, reference_result
from dataverify.probe_runtime import execute_sql, static_check
from dataverify.probe_testing import ExecutionBudget, probe_candidate


def run_evaluation(
    output: Path,
    *,
    seeds: tuple[int, ...] = (17,),
    max_queries: int = 160,
    max_seconds: float = 12.0,
) -> dict:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Output must remain inside this project's reports directory")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("At least one integer seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must not repeat; duplicate cells cannot share trace paths")
    ExecutionBudget(max_queries, max_seconds)
    cases = get_cases()
    modes = ("directed", "random")
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "disclosed_controlled_development",
        "real_model_used": False,
        "api_requests": 0,
        "case_count": len(cases),
        "seeds": list(seeds),
        "modes": list(modes),
        "max_queries_per_cell": max_queries,
        "max_seconds_per_cell": max_seconds,
        "task_count_is_not_case_count": "16 authored candidates cover 3 task specifications; seeds are repetitions, not new questions",
        "cases": [asdict(case) for case in cases],
        "versions": {name: importlib.metadata.version(name) for name in ("hypothesis", "sqlglot")},
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [
                ROOT / "dataverify" / name
                for name in ("probe_runtime.py", "probe_testing.py", "probe_cases.py")
            ]
            + [Path(__file__).resolve()]
        },
    }
    (output / "frozen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_grades = []
    cells = [
        {"case_id": case.case_id, "seed": seed, "mode": mode, "status": "not_run"}
        for case in cases
        for seed in seeds
        for mode in modes
    ]
    progress_path = output / "progress.json"
    progress_path.write_text(json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        strong = static_check(case.candidate_sql, case.task)
        original = execute_sql(case.candidate_sql, case.source, case.task, static_semantics=False)
        expected = reference_result(case)
        source_grade = (
            grade_rows(original["rows"], expected)
            if original["status"] == "ok"
            else {
                "correct": None,
                "code": "SOURCE_NOT_EXECUTED",
                "execution_code": original.get("code"),
            }
        )
        source_grades.append(
            {
                "case_id": case.case_id,
                "controlled_fault": case.fault_label,
                "static_accepted": strong["accepted"],
                "static_code": strong.get("code"),
                "source_execution": original,
                "source_grade": source_grade,
            }
        )
        for seed in seeds:
            for mode in modes:
                cell = next(
                    item
                    for item in cells
                    if item["case_id"] == case.case_id
                    and item["seed"] == seed
                    and item["mode"] == mode
                )
                # Persist the planned grid after each cell; interrupted work is
                # retained as incomplete rather than silently removed.
                try:
                    report = probe_candidate(
                        case.candidate_sql,
                        case.task,
                        mode=mode,
                        seed=seed,
                        max_queries=max_queries,
                        max_seconds=max_seconds,
                    )
                    cell.update(status="completed", report=report)
                except Exception as exc:
                    cell.update(status="runner_error", error_type=type(exc).__name__)
                filename = f"{case.case_id}_{mode}_{seed}.json"
                (output / "traces" / filename).write_text(
                    json.dumps(cell, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                progress_path.write_text(
                    json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    mode_summaries = {}
    for mode in modes:
        selected = [cell for cell in cells if cell["mode"] == mode]
        completed = [cell["report"] for cell in selected if cell["status"] == "completed"]
        witnesses = [report["counterexample"] for report in completed if report["counterexample"]]
        full_chars = sum(item["full_feedback_chars"] for item in witnesses)
        compact_chars = sum(item["compact_feedback_chars"] for item in witnesses)
        mode_summaries[mode] = {
            "planned_cells": len(cases) * len(seeds),
            "completed_cells": len(completed),
            "runner_errors": sum(cell["status"] == "runner_error" for cell in selected),
            "dynamic_violations": sum(report["status"] == "violation" for report in completed),
            "new_detections_beyond_strong_static": sum(
                report["new_detection_beyond_static"] for report in completed
            ),
            "budget_exhausted": sum(report["status"] == "budget_exhausted" for report in completed),
            "execution_errors": sum(report["status"] == "execution_error" for report in completed),
            "output_invalid": sum(report["status"] == "output_invalid" for report in completed),
            "probe_execution_attempts_completed_cells": sum(
                report["queries"] for report in completed
            ),
            "probe_accounting_unknown_cells": sum(
                cell["status"] != "completed" for cell in selected
            ),
            "intermediate_check_errors": sum(
                len(report["intermediate_check_errors"]) for report in completed
            ),
            "witness_count": len(witnesses),
            "full_feedback_chars": full_chars,
            "compact_feedback_chars": compact_chars,
            "feedback_char_reduction": None if not full_chars else 1 - compact_chars / full_chars,
        }
    summary = {
        "split": manifest["split"],
        "real_model_used": False,
        "api_requests": 0,
        "authored_candidates": len(cases),
        "task_specifications": len({case.task.task_id for case in cases}),
        "known_correct_candidates": sum(case.fault_label is None for case in cases),
        "known_defective_candidates": sum(case.fault_label is not None for case in cases),
        "strong_static_rejected_defects": sum(
            not row["static_accepted"] and row["controlled_fault"] is not None
            for row in source_grades
        ),
        "strong_static_false_rejections": sum(
            not row["static_accepted"] and row["controlled_fault"] is None for row in source_grades
        ),
        "source_execution_calls": len(source_grades),
        "source_grades": source_grades,
        "modes": mode_summaries,
        "interpretation": [
            "Authored development candidates and generated data are not natural LLM errors or held-out evidence.",
            "Static rejections explained by dynamic counterexamples are diagnostic evidence, not new detections.",
            "Character reduction is not token reduction, monetary savings or measured repair accuracy.",
            "No counterexample found only covers the registered properties and budget.",
            "Per-mode execution attempts cover completed probe cells only; source execution calls are separate, and failed-cell accounting is marked unknown.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--max-queries", type=int, default=160)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()
    output = args.output or ROOT / "reports" / (
        "probe_development_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    summary = run_evaluation(
        output, seeds=tuple(args.seeds), max_queries=args.max_queries, max_seconds=args.max_seconds
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "candidates": summary["authored_candidates"],
                "static_rejected_defects": summary["strong_static_rejected_defects"],
                "static_false_rejections": summary["strong_static_false_rejections"],
                "modes": summary["modes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(
        any(mode["runner_errors"] or mode["execution_errors"] for mode in summary["modes"].values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
