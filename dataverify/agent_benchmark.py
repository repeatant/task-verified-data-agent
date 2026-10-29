"""Frozen, bounded three-arm development comparison in the native DB-GPT loop.

Offline scripts test the evaluation plumbing; only explicit live mode calls the
ordinary model API. All groups share tasks/fixtures/model/attempt allowance.
Counterexamples and expected answers stay out of initial model inputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_grader import grade_record
from .benchmark_tasks import TASKS, candidate_expectation, manifest, write_fixture
from .contracts import describe_contracts
from .database import database_sha256
from .dbgpt_bridge import PROJECT_ROOT, UPSTREAM_COMMIT, run_native_case, sql_candidate
from .dbgpt_qwen import InjectedFirstCandidateClient, QwenConfig, QwenLLMClient, RequestBudget
from .guarded_task import ConfirmedMetricTask

MODES = ("original", "code", "evidence")


def write_json(path: Path, value):
    partial = path.with_suffix(".part.json")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    partial.replace(path)


def canonical_candidate(metric_id: str):
    return sql_candidate(
        "SELECT SUM(total) AS revenue FROM orders"
        if metric_id == "order_revenue"
        else "SELECT SUM(amount) AS revenue FROM order_lines",
        metric_id,
    )


def summarize_rows(rows: list[dict], scheduled: int) -> dict:
    """Keep failures, unscored SQL and unrun cells visible, with denominators."""
    result = {}
    for mode in MODES:
        for track in ("natural", "controlled", "all"):
            subset = [
                r for r in rows if r["mode"] == mode and (track == "all" or r["track"] == track)
            ]
            if not subset:
                continue
            completed = [r for r in subset if r["status"] == "completed"]
            scored = [r for r in completed if not r["grade"]["unscorable"]]
            result[mode + ":" + track] = {
                "scheduled": len(subset),
                "completed": len(completed),
                "not_run": sum(r["status"] == "not_run" for r in subset),
                "operational_errors": sum(
                    r["status"] == "operational_error"
                    or r.get("grade", {}).get("operational_failure", False)
                    for r in subset
                ),
                "scorable": len(scored),
                "accepted": sum(r["grade"]["accepted"] for r in completed),
                "correct_answers": sum(r["grade"]["correct_answer"] for r in completed),
                "correct_refusals": sum(r["grade"]["correct_refusal"] for r in completed),
                "known_wrong_releases": sum(r["grade"]["wrong_release"] for r in completed),
                "unscorable": len(completed) - len(scored),
                "defined_targets": sum(r["grade"]["target_defined"] for r in completed),
                "defined_targets_not_answered_correctly": sum(
                    r["grade"]["target_defined"] and not r["grade"]["correct_answer"]
                    for r in completed
                ),
                "metric_drifts": sum(r["grade"]["metric_drift"] for r in completed),
                "valid_initial_candidates_rejected": sum(
                    r.get("initial_candidate_should_accept") is True
                    and r.get("first_verification", {}).get("status") == "rejected"
                    for r in completed
                    if r.get("first_verification")
                ),
                "actual_http_attempts": sum(r["http_attempts"] for r in subset),
                "native_turns": sum(r["native_turns"] for r in subset),
                "agent_seconds_total": round(sum(r["seconds"] for r in completed), 4),
                # Use the scheduled denominator: operational gaps cannot improve it.
                "target_outcomes_over_scheduled": {
                    "numerator": sum(
                        r["grade"]["correct_answer"] or r["grade"]["correct_refusal"]
                        for r in completed
                    ),
                    "denominator": len(subset),
                },
            }
    return {"scheduled_cells": scheduled, "groups": result}


def paired_feedback_results(rows: list[dict]):
    by_task: dict[str, dict] = defaultdict(dict)
    for row in rows:
        if row["status"] == "completed" and row["track"] == "controlled":
            by_task[row["task_id"]][row["mode"]] = row
    pairs = []
    for task_id, modes in by_task.items():
        if not all(mode in modes for mode in ("code", "evidence")):
            continue
        left, right = modes["code"], modes["evidence"]
        pairs.append(
            {
                "task_id": task_id,
                "code_correct": left["grade"]["correct_answer"],
                "evidence_correct": right["grade"]["correct_answer"],
                "code_first_decision": left["first_verification"],
                "evidence_first_decision": right["first_verification"],
                "same_first_decision": left["first_verification"] == right["first_verification"],
                "eligible_for_repair": left["grade"]["target_defined"]
                and left["first_verification"] is not None
                and left["first_verification"]["status"] == "rejected",
                "code_http_attempts": left["http_attempts"],
                "evidence_http_attempts": right["http_attempts"],
            }
        )
    return pairs


async def run_benchmark(
    output: Path,
    *,
    tasks=None,
    live: bool = False,
    config: QwenConfig | None = None,
    max_requests: int = 18,
    max_attempts: int = 2,
    client_factory=None,
) -> dict[str, Any]:
    """Use an exclusive run folder; never overwrite an earlier experiment.

    client_factory is an offline-test dependency seam, rejected in live mode.
    Natural tasks start with model-generated SQL in live mode. Controlled tasks
    inject exactly the same first candidate in every group, without an API call;
    only a subsequent model repair consumes API requests.
    """
    if live and client_factory is not None:
        raise ValueError("A live benchmark cannot use an injected test client factory")
    if not 1 <= max_requests <= 36 or not 1 <= max_attempts <= 3:
        raise ValueError("Request or attempt allowance out of bounded range")
    tasks = tuple(TASKS if tasks is None else tasks)
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Select distinct frozen tasks")
    if live and config is None:
        raise ValueError("Live mode requires explicit Qwen configuration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    database_dir = output / "databases"
    database_dir.mkdir()
    trace_dir = output / "traces"
    trace_dir.mkdir()
    budget = RequestBudget(max_requests, max_input_chars=150_000, max_reported_tokens=30_000)
    frozen = manifest(tasks)
    settings = {
        "upstream_commit": UPSTREAM_COMMIT,
        "model": config.model if config else "offline-scripted",
        "temperature": 0,
        "per_request_max_tokens": 900,
        "max_native_attempts": max_attempts,
        "http_attempt_limit": max_requests if live else 0,
        "trials_per_task_per_mode": 1,
        "split": "development_not_unseen_holdout",
        "arm_order": "rotate original/code/evidence by task index",
        "metric_confirmation": "caller_confirmed_not_inferred",
        "supports": describe_contracts()["supported_sql"],
    }
    freeze_document = {"tasks": frozen, "settings": settings}
    freeze_bytes = json.dumps(freeze_document, ensure_ascii=False, sort_keys=True).encode("utf-8")
    freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    write_json(output / "frozen_manifest.json", {**freeze_document, "sha256": freeze_sha256})
    own_files = (
        "dataverify/agent_benchmark.py",
        "dataverify/benchmark_tasks.py",
        "dataverify/benchmark_grader.py",
        "dataverify/dbgpt_qwen.py",
        "dataverify/dbgpt_bridge.py",
    )
    write_json(
        output / "code_fingerprints.json",
        {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in own_files
        },
    )
    # Predeclare the full grid. Interrupted runs must not shrink per-arm
    # denominators just because later cells were never appended.
    rows = [
        {
            "task_id": task.task_id,
            "track": task.track,
            "pair_id": task.pair_id,
            "mode": mode,
            "status": "not_run",
            "stop_reason": "NOT_STARTED",
            "http_attempts": 0,
            "native_turns": 0,
            "seconds": 0,
        }
        for task in tasks
        for mode in MODES
    ]
    row_indices = {(row["task_id"], row["mode"]): index for index, row in enumerate(rows)}
    all_http_events = []
    report = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_model_used": live,
        "type": "bounded-development-agent-comparison",
        "manifest_sha256": freeze_sha256,
        "settings": settings,
        "rows": rows,
        "limitation": "Frozen synthetic development tasks, one trial per task/arm, explicit metric confirmation, limited SQL grader. Injected first candidates are not natural model errors. Offline scripted outcomes test the harness only. A/C changes the entire policy; B/C isolates the diagnostic feedback. No production/general benchmark claim.",
    }

    def checkpoint(status="running"):
        report.update(
            status=status, budget=budget.snapshot(), summary=summarize_rows(rows, len(tasks) * 3)
        )
        write_json(output / "summary.json", report)

    checkpoint()
    try:
        for index, task in enumerate(tasks):
            path = database_dir / (task.task_id + ".sqlite3")
            write_fixture(path, task.fixture)
            source_digest = database_sha256(path)
            # Same fixture/task/prompt and allowance; rotate ordering to reduce
            # confounding every evidence run with a later wall-clock position.
            order = MODES[index % 3 :] + MODES[: index % 3]
            for mode in order:
                cell_index = row_indices[(task.task_id, mode)]
                base_row = {
                    "task_id": task.task_id,
                    "track": task.track,
                    "pair_id": task.pair_id,
                    "mode": mode,
                    "status": "not_run",
                    "source_sha256": source_digest,
                    "http_attempts": 0,
                    "native_turns": 0,
                    "seconds": 0,
                    "initial_candidate_should_accept": candidate_expectation(task)["should_accept"],
                }
                if live and (
                    budget.stopped_reason
                    or budget.used >= budget.limit
                    or budget.reported_tokens >= budget.max_reported_tokens
                ):
                    base_row["stop_reason"] = budget.stopped_reason or "GLOBAL_BUDGET_EXHAUSTED"
                    rows[cell_index] = base_row
                    checkpoint()
                    continue
                started = time.perf_counter()
                try:
                    kwargs = {}
                    if client_factory is not None:
                        kwargs["llm_client"] = client_factory(task, mode, budget)
                    elif live:
                        client = QwenLLMClient(config, budget)
                        if task.initial_candidate is not None:
                            client = InjectedFirstCandidateClient(
                                client, dict(task.initial_candidate)
                            )
                        kwargs["llm_client"] = client
                    else:
                        fixed = canonical_candidate(task.confirmed_metric_id)
                        kwargs["responses"] = (
                            [dict(task.initial_candidate)]
                            if task.initial_candidate is not None
                            else []
                        ) + [fixed] * max_attempts
                    public_input = task.public_input()
                    record = await run_native_case(
                        path,
                        ConfirmedMetricTask(
                            public_input["question"], public_input["confirmed_metric_id"]
                        ),
                        mode=mode,
                        max_attempts=max_attempts,
                        output_dir=output / "runtime",
                        **kwargs,
                    )
                    grade = grade_record(task, record, mode=mode)
                    source_unchanged = source_digest == database_sha256(path)
                    if not source_unchanged:
                        raise RuntimeError("Source changed in benchmark")
                    events = record["http_events"]
                    if any(event["status"] != "completed" for event in events):
                        grade.update(correct_refusal=False, operational_failure=True)
                    all_http_events.extend(deepcopy(events))
                    write_json(trace_dir / (task.task_id + "__" + mode + ".json"), record)
                    first = record["actions"][0] if record["actions"] else None
                    rows[cell_index] = {
                        **base_row,
                        "status": "completed",
                        "grade": grade,
                        "source_unchanged": source_unchanged,
                        "http_attempts": len(events),
                        "native_turns": record["llm_calls"],
                        "seconds": round(time.perf_counter() - started, 4),
                        "first_verification": {
                            "status": first.get("verification_status"),
                            "error_code": first.get("error_code"),
                        }
                        if first
                        else None,
                        "trace_file": "traces/" + task.task_id + "__" + mode + ".json",
                    }
                except Exception as error:
                    # No raw exception/credentials in the artifact. Keep the
                    # failed scheduled cell visible rather than dropping it.
                    rows[cell_index] = {
                        **base_row,
                        "status": "operational_error",
                        "error_type": type(error).__name__,
                        "seconds": round(time.perf_counter() - started, 4),
                    }
                checkpoint()
        report["paired_feedback"] = paired_feedback_results(rows)
        report["http_events_recorded"] = len(all_http_events)
        report["http_ledger_complete"] = len(all_http_events) == budget.used
        report["usage"] = {
            field: sum(e["usage"][field] for e in all_http_events)
            if report["http_ledger_complete"]
            and all_http_events
            and all(e["usage"][field] is not None for e in all_http_events)
            else None
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        report["all_cells_completed"] = len(rows) == len(tasks) * 3 and all(
            row["status"] == "completed" for row in rows
        )
        report["all_completed_sources_unchanged"] = all(
            row.get("source_unchanged", True) for row in rows if row["status"] == "completed"
        )
        # A wrong answer is an experimental result, not a crashed experiment.
        checkpoint("completed" if report["all_cells_completed"] else "incomplete")
        return report
    except BaseException:
        checkpoint("interrupted")
        raise
