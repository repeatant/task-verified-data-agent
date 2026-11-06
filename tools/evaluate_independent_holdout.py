"""Score independently authored challenges against a frozen verifier offline.

The initial score is recorded without adapting the verifier to challenge cases.
This component check never calls a model or measures an agent repair loop.
LangGraph model experiments use the separate graph and discovery evaluators.
An evaluated challenge may fail without invalidating the run. Source changes
or operational errors invalidate the run and produce a nonzero CLI exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

VERIFIER_SOURCES = (
    "dataverify/verification/probe_runtime.py",
    "dataverify/verification/probe_testing.py",
    "dataverify/verification/probe_capabilities.py",
    "dataverify/verification/probe_guard.py",
    "dataverify/evaluation/probe_cases.py",
    "dataverify/agent/feedback_strategy.py",
    "tools/evaluate_independent_holdout.py",
)

if TYPE_CHECKING:
    from dataverify.verification.probe_runtime import ProbeData, ProbeTask


@dataclass(frozen=True)
class Verifier:
    """Bindings for the active verifier or the isolated historical snapshot."""

    cases: ModuleType
    runtime: ModuleType
    guard: type


def current_verifier() -> Verifier:
    from dataverify.evaluation import probe_cases
    from dataverify.verification import probe_runtime
    from dataverify.verification.probe_guard import ProbeGuard

    return Verifier(probe_cases, probe_runtime, ProbeGuard)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    partial = path.with_suffix(".part.json")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    partial.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_current_verifier() -> dict:
    """Freeze this run's component sources without rewriting historical evidence."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current component regression on disclosed cases; not a new holdout or agent benchmark.",
        "files": {name: sha(ROOT / name) for name in VERIFIER_SOURCES},
    }


def verify_source_manifest(manifest: dict) -> dict[str, bool]:
    """Validate paths and coverage before comparing source digests."""
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("Invalid verifier source manifest")
    matches = {}
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("Invalid verifier source entry")
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Invalid verifier source path or digest")
        if not source.is_file():
            raise ValueError("Verifier manifest refers to unavailable source files")
        matches[name] = sha(source) == expected
    if not set(VERIFIER_SOURCES).issubset(files):
        raise ValueError("Verifier manifest does not cover the current component sources")
    return matches


@dataclass(frozen=True)
class Challenge:
    identifier: str
    category: str
    task: ProbeTask
    data: ProbeData
    sql: str
    expected: list
    reason: str


def load_challenges(document, *, verifier: Verifier | None = None):
    verifier = verifier or current_verifier()
    result = []
    for item in document["challenges"]:
        identifier = item["id"]
        if not re.fullmatch(r"h[0-9]{2}", identifier):
            raise ValueError("Invalid independent challenge identifier")
        if item["category"] not in ("legal", "semantic_fault", "unsupported"):
            raise ValueError("Unknown independent challenge category")
        fields = {
            "orders": ("order_id", "total", "channel"),
            "lines": ("line_id", "order_id", "amount", "status", "settled_at"),
        }
        converted = {}
        for name, columns in fields.items():
            rows = []
            for row in item["data"][name]:
                if isinstance(row, dict):
                    if set(row) != set(columns):
                        raise ValueError("Unexpected source record fields")
                    rows.append(tuple(row[column] for column in columns))
                else:
                    rows.append(tuple(row))
            converted[name] = tuple(rows)
        task = verifier.runtime.ProbeTask(**item["task"])
        data = verifier.runtime.ProbeData(**converted)
        expected = item["expected_rows"]
        # Existing record oracle is a cross-check, not the label author. No
        # correct SQL is invented: this placeholder is never executed or sent.
        case = verifier.cases.Case(
            identifier,
            task,
            data,
            item["candidate_sql"],
            "EVALUATOR_REFERENCE_UNUSED",
            None if item["category"] == "legal" else item["category"],
            item["reason"],
        )
        if not verifier.cases.grade_rows(verifier.cases.reference_result(case), expected)[
            "correct"
        ]:
            raise ValueError("Independent target disagrees with task-record oracle")
        result.append(
            Challenge(
                identifier,
                item["category"],
                task,
                data,
                item["candidate_sql"],
                expected,
                item["reason"],
            )
        )
    if len(result) != len({item.identifier for item in result}):
        raise ValueError("Duplicate independent challenge IDs")
    selected = document["natural_probe_ids"]
    by_id = {item.identifier: item for item in result}
    if (
        len(selected) != 4
        or len(set(selected)) != 4
        or any(name not in by_id or by_id[name].category != "legal" for name in selected)
    ):
        raise ValueError("Exactly four preselected legal natural tasks are required")
    return tuple(result)


def score(challenge, *, verifier: Verifier | None = None):
    verifier = verifier or current_verifier()
    guard = verifier.guard(challenge.task, challenge.data, feedback_policy="capability_v2")
    guard.execute(challenge.sql, challenge.task.task_id)
    publication = guard.publish()
    accepted = publication["status"] == "verified"
    grade = verifier.cases.grade_rows(publication["rows"], challenge.expected) if accepted else None
    audit = guard.last_result
    probe = audit.get("probe_report") or {}
    semantic_wrong_release = accepted and not grade["correct"]
    scope_breach = accepted and challenge.category == "unsupported"
    refused_legal = not accepted and challenge.category == "legal"
    passed = (accepted and grade["correct"]) if challenge.category == "legal" else not accepted
    return {
        "id": challenge.identifier,
        "category": challenge.category,
        "passed": bool(passed),
        "accepted": accepted,
        "semantic_wrong_release": bool(semantic_wrong_release),
        "scope_breach": bool(scope_breach),
        "refused_legal": refused_legal,
        "expected_rows": challenge.expected,
        "grade": grade,
        "publication": publication,
        "static": probe.get("static"),
        "dynamic_status": probe.get("status"),
        "dynamic_new_detection": bool(probe.get("new_detection_beyond_static")),
        "source_output_check": audit.get("source_output_check"),
        "audit": audit,
    }


async def evaluate(
    output: Path,
    dataset: Path,
    verifier_manifest: Path | None = None,
    *,
    regression_from: Path | None = None,
):
    output, dataset = output.resolve(), dataset.resolve()
    if verifier_manifest is not None:
        verifier_manifest = verifier_manifest.resolve()
    readable_roots = ((ROOT / "reports").resolve(), (ROOT / "tests/fixtures").resolve())
    inputs = (dataset, verifier_manifest) if verifier_manifest is not None else (dataset,)
    if not output.is_relative_to((ROOT / "reports").resolve()) or any(
        not any(path.is_relative_to(base) for base in readable_roots) for path in inputs
    ):
        raise ValueError(
            "Inputs must remain inside reports or tests/fixtures; output must remain inside reports"
        )
    freeze = (
        read(verifier_manifest) if verifier_manifest is not None else capture_current_verifier()
    )
    unchanged = verify_source_manifest(freeze)
    if not all(unchanged.values()):
        raise ValueError("Frozen verifier changed before initial evaluation")
    author = read(dataset.parent / "author_manifest.json")
    label_audit = read(dataset.parent / "label_audit.json")
    dataset_hash = sha(dataset)
    if (
        author["challenge_sha256"] != dataset_hash
        or label_audit["challenge_sha256"] != dataset_hash
        or label_audit["all_labels_valid"] is not True
    ):
        raise ValueError("Challenge hash mismatch or unresolved independent label dispute")
    document = read(dataset)
    verifier = current_verifier()
    challenges = load_challenges(document, verifier=verifier)
    if len(challenges) != 24 or Counter(item.category for item in challenges) != {
        "legal": 8,
        "semantic_fault": 10,
        "unsupported": 6,
    }:
        raise ValueError("Independent suite must match its preregistered 24-case composition")
    if regression_from is not None:
        regression_from = regression_from.resolve()
        if not any(regression_from.is_relative_to(base) for base in readable_roots):
            raise ValueError("Original evaluation must be inside reports or tests/fixtures")
        prior_manifest = read(regression_from / "evaluation_manifest.json")
        if prior_manifest["challenge_sha256"] != dataset_hash:
            raise ValueError("Regression must reuse the exact initial challenges")
    output.mkdir(parents=True, exist_ok=False)
    if verifier_manifest is None:
        verifier_manifest = output / "verifier_manifest.json"
        write(verifier_manifest, freeze)
    evaluation_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.relative_to(ROOT)),
        "challenge_sha256": dataset_hash,
        "verifier_manifest": str(verifier_manifest.relative_to(ROOT)),
        "verifier_manifest_sha256": sha(verifier_manifest),
        "evaluator_sha256": sha(Path(__file__)),
        "author_manifest_sha256": sha(dataset.parent / "author_manifest.json"),
        "label_audit_sha256": sha(dataset.parent / "label_audit.json"),
        "phase": "converted_regression" if regression_from else "current_component_check",
        "scope": (
            "Post-fix regression on challenges already examined for fixes; NOT an untouched holdout score."
            if regression_from
            else "Component check on disclosed independently authored query challenges; not a new holdout, agent benchmark or production generalization claim."
        ),
        "natural_probe_ids": document["natural_probe_ids"],
        "feedback_policy": "capability_v2",
        "real_model_used": False,
        "budget_start": None,
        "verifier_unchanged_before": unchanged,
    }
    if regression_from:
        evaluation_manifest["original_evaluation"] = str(regression_from.relative_to(ROOT))
    write(output / "evaluation_manifest.json", evaluation_manifest)
    workflow = {
        "status": "running",
        "real_model_used": False,
        "report_dirs": [],
        "budget": None,
        "scope": evaluation_manifest["scope"],
    }
    write(output / "workflow.json", workflow)
    try:
        rows = []
        for item in challenges:
            try:
                rows.append(score(item, verifier=verifier))
            except Exception as exc:
                rows.append(
                    {
                        "id": item.identifier,
                        "category": item.category,
                        "passed": False,
                        "accepted": False,
                        "semantic_wrong_release": False,
                        "scope_breach": False,
                        "refused_legal": False,
                        "dynamic_new_detection": False,
                        "operational_error": type(exc).__name__,
                    }
                )
        summary = {
            "scheduled": len(rows),
            "completed": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "category_counts": dict(Counter(row["category"] for row in rows)),
            "semantic_wrong_releases": sum(row["semantic_wrong_release"] for row in rows),
            "scope_breaches": sum(row["scope_breach"] for row in rows),
            "refused_legal": sum(row["refused_legal"] for row in rows),
            "operational_errors": sum("operational_error" in row for row in rows),
            "dynamic_new_detections": sum(row["dynamic_new_detection"] for row in rows),
        }
        write(output / "initial_component_results.json", {"summary": summary, "rows": rows})
        workflow["initial_summary"] = summary
        write(output / "workflow.json", workflow)
        workflow.update(
            status="completed",
            verifier_unchanged_after=verify_source_manifest(freeze),
        )
        if not all(workflow["verifier_unchanged_after"].values()):
            workflow.update(status="invalid", error_code="VERIFIER_SOURCE_CHANGED")
        elif summary["operational_errors"]:
            workflow.update(status="error", error_code="COMPONENT_OPERATIONAL_ERROR")
    except BaseException as exc:
        workflow.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        write(output / "workflow.json", workflow)
    return workflow


def main():
    import asyncio

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    public = ROOT / "tests/fixtures/independent_holdout"
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            public / "frozen_challenges.json"
            if public.is_dir()
            else ROOT / "reports/independent_holdout_20261029/frozen_challenges.json"
        ),
    )
    parser.add_argument(
        "--verifier-manifest",
        type=Path,
        help="Optional current-source freeze. By default, capture this run's sources in the output directory; historical manifests are not rewritten.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("independent_eval_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    parser.add_argument(
        "--regression-from",
        type=Path,
        default=public / "initial_evaluation" if public.is_dir() else None,
    )
    args = parser.parse_args()
    result = asyncio.run(
        evaluate(
            args.output,
            args.dataset,
            args.verifier_manifest,
            regression_from=args.regression_from,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
