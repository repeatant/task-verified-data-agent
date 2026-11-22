"""Frozen development challenges with offline gates and real, bounded Agent runs.

Only this evaluator knows injected fault labels and scoring answers. Runtime
roles receive the ordinary task and, for controlled cases, first-turn programs.
No model calls occur during preflight or import.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import QwenReviewModel
from dataverify.agent.graph_checkpoint import JournaledBudget, read_budget
from dataverify.agent.graph_model import InjectedFirstGraphModel, QwenGraphModel
from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.evaluation.frozen_cases import (
    get_frozen_cases,
    grade_frozen_run,
    scoring_datasets,
)
from dataverify.runtime.discovery_budget import reviewed_truncation
from dataverify.runtime.experiment_budget import restore_budget
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget, validate_pricing
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.discovery_runtime import check_candidate, execute_readonly


def experiment_cases():
    """Sixteen controlled trajectories plus four separate natural-generation runs."""
    cases = get_frozen_cases()
    cases += [
        replace(
            case,
            case_id=case.family + "__natural",
            condition="natural",
            initial_sql=None,
            initial_plan=None,
        )
        for case in cases
        if case.condition == "correct"
    ]
    random.Random(20261107).shuffle(cases)
    return cases


def source_hashes(root: Path) -> dict[str, str]:
    """Hash runtime, prompts, evaluator and scoring code before any paid request."""
    files = list((root / "dataverify").rglob("*.py"))
    files += [root / "tools/evaluate_frozen.py", root / "docs/experiments/V1_FROZEN_PROTOCOL.md"]
    files += list((root / "requirements").glob("*.txt"))
    return {
        file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(files)
    }


def baseline_gates(case) -> dict:
    """Two no-repair gates; neither is a full no-auditor Agent ablation."""
    if case.condition == "natural":
        return {}
    try:
        initial_rows = execute_readonly(case.initial_sql, case.source)
        executable = True
    except ValueError:
        initial_rows = None
        executable = False
    agreement = check_candidate(
        case.initial_sql, case.initial_plan, case.initial_plan["fixtures"], source=case.source
    )["all_match"]
    initial = grade_frozen_run(
        case,
        {
            "status": "reviewed",
            "publication": {"sql": case.initial_sql, "rows": initial_rows},
            "candidate": {"sql": case.initial_sql},
            "plan": case.initial_plan,
            "audit_plan": case.initial_plan,
        },
    )
    correct = initial["correct_delivery"]
    return {
        "sql_executable_accept": executable,
        "single_reference_accept": agreement,
        "initial_sql_correct": correct,
        "initial_score": initial,
        "sql_executable_wrong_release": executable and not correct,
        "single_reference_wrong_release": agreement and not correct,
    }


def build_manifest(root: Path) -> dict:
    """Disclose the evaluation population without passing fault names to roles."""
    cases = experiment_cases()
    return {
        "format": "dataverify-frozen-development-v1",
        "scope": "4 public development intents; 16 controlled cases + 4 natural runs; one trial each",
        "independent_unseen_benchmark": False,
        "seed": 20261107,
        "parameters": {
            "temperature": 0,
            "max_output_tokens": 900,
            "max_attempts": 3,
            "max_reviews": 3,
            "max_reference_revisions": 1,
            "max_challenge_examples": 40,
        },
        "files": source_hashes(root),
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "condition": case.condition,
                "task_digest": digest_json(case.task.public_input()),
                "source_digest": digest_json(case.source.to_dict()),
                "initial_sql": case.initial_sql,
                "initial_plan_digest": digest_json(case.initial_plan),
                "scoring_digest": digest_json(
                    {name: data.to_dict() for name, data in scoring_datasets(case).items()}
                ),
                "baselines": baseline_gates(case),
            }
            for case in cases
        ],
    }


def create_top_up_budget(prior_path: Path, output: Path, top_up_cny: float):
    """Explicit additional tranche; retain prior ledger and sum BOTH allowances.

    This is not a reset of the old CNY 5 allowance. The caller must obtain user
    authorization for the added tranche, hold both file leases, and use one
    fresh output directory. A parent-side allocation prevents branching the
    same top-up twice. Previous reservations are not released or rewritten.
    """
    if not 0 < top_up_cny <= 3:
        raise ValueError("ADDITIONAL_ALLOWANCE_MUST_BE_IN_0_TO_3_CNY")
    prior = read_budget(prior_path)
    if not prior["reported_cost_complete"] or prior["stopped_reason"]:
        raise ValueError("PRIOR_USAGE_MUST_BE_COMPLETE_AND_UNSTOPPED")
    allocation = prior_path.with_suffix(".frozen-allocation.json")
    if allocation.exists():
        raise ValueError("PARENT_ALREADY_HAS_A_FROZEN_EXPERIMENT_ALLOCATION")
    if prior["money_limit_cny"] + top_up_cny > 8:
        raise ValueError("COMBINED_ALLOWANCE_EXCEEDS_CNY_8")
    budget = MonetaryRequestBudget(
        limit=150,
        max_input_chars=2_000_000,
        max_reported_tokens=350_000,
        money_limit_cny=top_up_cny,
    )
    journal = output / "budget.json"
    lineage = {
        "prior_journal": str(prior_path.resolve()),
        "prior_digest": digest_json(prior),
        "prior_snapshot": prior,
        "new_journal": str(journal.resolve()),
        "authorized_added_allowance_cny": top_up_cny,
        "combined_allowance_cny": prior["money_limit_cny"] + top_up_cny,
        "authorization": "Explicit user-approved extra allowance for frozen experiment",
    }
    write_json(allocation, lineage)
    write_json(output / "budget-lineage.json", lineage)
    return JournaledBudget(budget, journal), prior


def _safe_events(models) -> list[dict]:
    allowed = (
        "attempt",
        "model",
        "status",
        "usage",
        "usage_complete",
        "http_status",
        "error_code",
        "action_error_code",
        "elapsed_ms",
    )
    return sorted(
        [
            {"role": role, **{key: event[key] for key in allowed if key in event}}
            for role, model in models
            for event in model.http_events
        ],
        key=lambda event: event["attempt"],
    )


async def run_case(case, *, config, budget, output: Path) -> dict:
    """Run exactly once; unexpected failures remain visible and stop the cohort."""
    writer = QwenGraphModel(config, budget)
    reviewer = QwenReviewModel(config, budget)
    auditor = QwenReviewModel(config, budget)
    if case.condition != "natural":
        writer = InjectedFirstGraphModel(
            writer, {"task_id": case.task.task_id, "sql": case.initial_sql}
        )
        reviewer = InjectedFirstReviewModel(reviewer, case.initial_plan)
    models = [("writer", writer), ("reviewer", reviewer), ("auditor", auditor)]
    record, error = {}, None
    started = time.perf_counter()
    try:
        record = await run_discovery(
            case.task,
            case.source,
            writer,
            reviewer,
            auditor=auditor,
            checkpoint_path=output / (case.case_id + ".sqlite3"),
            thread_id="frozen",
        )
    except Exception as exc:
        # Never serialize a raw SDK exception or endpoint. A systemic failure
        # stops subsequent tasks; it is not silently converted to a failed answer.
        error = type(exc).__name__
        record = {"status": "runner_error", "error_type": error}
    finally:
        for _, model in models:
            await model.aclose()
    elapsed = time.perf_counter() - started
    events = _safe_events(models)
    record["http_events"] = events
    record["bound_source"] = case.source.to_dict()
    record["model_contexts"] = {name: model.requests for name, model in models}
    write_json(output / (case.case_id + ".json"), record)
    row = {
        "case_id": case.case_id,
        "family": case.family,
        "condition": case.condition,
        "status": record["status"],
        "runner_error": error,
        "elapsed_seconds": round(elapsed, 3),
        "score": grade_frozen_run(case, record),
        "reference_revisions": record.get("reference_revisions", 0),
        "sql_text_changed": case.initial_sql is not None
        and (record.get("candidate") or {}).get("sql") != case.initial_sql,
        "http_events": events,
        "usage": summarize_usage(events),
        "baseline": baseline_gates(case),
        "reason": record.get("publication", {}).get("reason"),
    }
    return row


def aggregate(rows: list[dict], *, planned: int = 20) -> dict:
    """Count by trajectory, keeping incomplete cohorts and natural runs separate."""
    groups = {}
    for condition in ("sql_wrong", "reference_wrong", "correlated_wrong", "correct", "natural"):
        selected = [row for row in rows if row["condition"] == condition]
        groups[condition] = {
            "planned": 4,
            "attempted": len(selected),
            **{
                metric: sum(bool(row["score"].get(metric)) for row in selected)
                for metric in (
                    "correct_delivery",
                    "joint_validated_delivery",
                    "wrong_release",
                    "correct_sql_damaged",
                )
            },
            "not_reviewed": sum(row["status"] != "reviewed" for row in selected),
        }
    events = [event for row in rows for event in row["http_events"]]
    return {
        "planned": planned,
        "attempted": len(rows),
        "complete": len(rows) == planned,
        "groups": groups,
        "usage": summarize_usage(events),
        "distinct_intents": 4,
        "independent_trials_per_case": 1,
    }


async def run_experiment(root, output, config, budget, prior, manifest):
    """Persist each task before continuing; no post-result prompt tuning/retries."""
    validate_pricing(config)
    rows = []
    continuations = []
    for case in experiment_cases():
        if source_hashes(root) != manifest["files"]:
            raise ValueError("FROZEN_SOURCE_CHANGED")
        start_budget = budget.snapshot()
        row = await run_case(case, config=config, budget=budget, output=output)
        rows.append(row)
        snapshot = budget.snapshot()
        if snapshot["stopped_reason"] == "PROVIDER_ERROR":
            # Terminate THIS task, retaining its failure. Only a fully accounted
            # truncation may unlock later tasks; never replace a failed trial.
            truncation_report = {
                "budget_start": start_budget,
                "budget_end": snapshot,
                "http_events": row["http_events"],
            }
            try:
                checked = reviewed_truncation(snapshot, truncation_report)
            except ValueError:
                pass
            else:
                restored = restore_budget(
                    checked, max_extra_requests=1, money_limit_cny=checked["money_limit_cny"]
                )
                restored.limit = snapshot["http_attempt_limit"]
                next_path = output / ("budget-after-" + case.case_id + ".json")
                if next_path.exists():
                    raise ValueError("TRUNCATION_CONTINUATION_ALREADY_EXISTS")
                write_json(output / (case.case_id + "-truncation.json"), truncation_report)
                continuations.append(
                    {
                        "failed_case": case.case_id,
                        "previous_journal": budget.path.name,
                        "previous_digest": digest_json(snapshot),
                        "next_journal": next_path.name,
                        "task_retried": False,
                    }
                )
                budget = JournaledBudget(restored, next_path)
                snapshot = budget.snapshot()
        report = {
            "manifest_digest": digest_json(manifest),
            "model": config.model,
            "scope": manifest["scope"],
            "rows": rows,
            **aggregate(rows),
            "combined_accounting": {
                "http_calls": prior["http_attempts"] + snapshot["http_attempts"],
                "reported_tokens": prior["reported_total_tokens"]
                + snapshot["reported_total_tokens"],
                "estimated_cny": round(
                    prior["estimated_reported_cny"] + snapshot["estimated_reported_cny"], 9
                ),
                "reserved_cny": round(
                    prior["reserved_upper_cny"] + snapshot["reserved_upper_cny"], 9
                ),
                "allowance_cny": prior["money_limit_cny"] + snapshot["money_limit_cny"],
            },
            "new_budget": {
                key: value for key, value in snapshot.items() if key != "reservation_ledger"
            },
            "latest_journal": budget.path.name,
            "accounted_truncation_continuations": continuations,
            "source_integrity_ok": source_hashes(root) == manifest["files"],
        }
        write_json(output / "summary.json", report)
        print(
            json.dumps({key: row[key] for key in ("case_id", "status", "score", "usage")}),
            flush=True,
        )
        if (
            row["runner_error"]
            or snapshot["stopped_reason"]
            or not snapshot["reported_cost_complete"]
        ):
            break
    return report
