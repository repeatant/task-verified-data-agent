"""Frozen full/compact feedback comparison in the native DB-GPT loop.

Controlled first candidates are injected test inputs, never model predictions.
Only explicit live mode calls Qwen. Exact targets and unseen-by-runtime boundary
fixtures remain in this evaluator; runtime modules must not import this module.
These are disclosed development tasks, not an unseen benchmark or production.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import time
from typing import Any

from .dbgpt_bridge import UPSTREAM_COMMIT
from .dbgpt_qwen import InjectedFirstCandidateClient, QwenConfig, QwenLLMClient
from .probe_agent_bridge import probe_sql_candidate, run_probe_native_case
from .probe_cases import Case, get_cases, grade_rows, reference_result
from .probe_cost_budget import MonetaryRequestBudget, validate_pricing
from .probe_runtime import ProbeData, TABLE_SCHEMA
from .probe_testing import probe_candidate
from .probe_capabilities import validate_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("full", "compact")
NATURAL_CASE_IDS = ("d01", "d03", "d04")


def _write(path: Path, document: Any) -> None:
    partial = path.with_suffix(".part.json")
    partial.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(path)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def boundary_fixture(case: Case) -> ProbeData:
    """Additional evaluator-only rows cover boundaries/refunds/zero/grouping.

    This is an exact-target regression fixture, not an unseen problem template.
    It is never passed to the agent, runtime properties or feedback generator.
    """
    task, source = case.task, case.source
    parent = max((row[0] for row in source.orders), default=0) + 10
    child = max((row[0] for row in source.lines), default=0) + 10
    start = datetime.fromisoformat(task.start)
    end = datetime.fromisoformat(task.end)
    interior = (start + (end - start) / 2).replace(microsecond=0).isoformat()
    extra_orders = ((parent, 37, None), (parent + 1, 11, task.other_label), (parent + 2, 0, "zero-only"))
    pending = "pending" if task.status != "pending" else "cancelled"
    cancelled = "cancelled" if task.status != "cancelled" else "pending"
    extra_lines = (
        (child, parent, 11, task.status, task.start),
        (child + 1, parent, -7, task.status, interior),
        (child + 2, parent, 23, task.status, task.end),
        (child + 3, parent, 31, pending, interior),
        (child + 4, parent + 1, 11, task.status, interior),
        (child + 5, parent + 1, 19, cancelled, interior),
        (child + 6, parent + 2, 47, pending, interior),
    )
    return ProbeData(source.orders + extra_orders, source.lines + extra_lines)


def independent_query(sql: str, data: ProbeData) -> dict:
    """Execute published SQL for grading without consulting the runtime checker.

    SQLite authorizer, a VM/time limit and row cap enforce an independent read
    boundary. The target itself is computed from Python records, not this SQL.
    """
    if type(sql) is not str or not sql.strip() or len(sql) > 16_000:
        return {"status": "error", "code": "INVALID_PUBLISHED_SQL", "rows": []}
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total INTEGER NOT NULL,channel TEXT)")
        conn.execute("CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER REFERENCES orders(order_id),amount INTEGER,status TEXT,settled_at TEXT)")
        conn.executemany("INSERT INTO orders VALUES (?,?,?)", data.orders)
        conn.executemany("INSERT INTO order_lines VALUES (?,?,?,?,?)", data.lines)
        conn.commit()
        conn.execute("PRAGMA query_only=ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        deadline = time.monotonic() + 1.0
        callbacks = 0

        def progress():
            nonlocal callbacks
            callbacks += 1
            return int(callbacks > 100 or time.monotonic() > deadline)

        def authorizer(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and table in TABLE_SCHEMA:
                if (database == "main" and column in TABLE_SCHEMA[table]) or (database in {None, "main"} and column == ""):
                    return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (column or "").casefold() in {"sum", "coalesce"}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        conn.set_authorizer(authorizer)
        conn.set_progress_handler(progress, 1000)
        rows = conn.execute(sql).fetchmany(1001)
        if len(rows) > 1000:
            return {"status": "error", "code": "GRADING_ROW_BUDGET", "rows": []}
        return {"status": "ok", "code": None, "rows": [list(row) for row in rows]}
    except (sqlite3.Error, ValueError, OverflowError):
        return {"status": "error", "code": "GRADING_SQL_ERROR", "rows": []}
    finally:
        conn.close()


def grade_probe_record(case: Case, record: dict) -> dict:
    """Grade source answer, published-SQL consistency and boundary behaviour."""
    publication = record.get("publication") or {}
    accepted = publication.get("status") == "verified"
    task_preserved = publication.get("task_id") == case.task.task_id
    http_errors = any(event.get("status") != "completed" for event in record.get("http_events", []))
    operational = bool(record.get("operational_error") or http_errors or record.get("source_unchanged") is False)
    result = {
        "accepted": accepted, "task_preserved": task_preserved,
        "operational_failure": operational, "correct_task": False,
        "source_correct": False, "boundary_correct": False,
        "publication_matches_sql": False, "wrong_release": False,
        "unscorable": False, "scope": "source and additional disclosed-family boundary fixture, not arbitrary SQL proof",
    }
    if not accepted:
        return result
    sql = publication.get("verified_sql")
    expected = reference_result(case)
    source_grade = grade_rows(publication.get("rows"), expected)
    source_exec = independent_query(sql, case.source)
    hidden = boundary_fixture(case)
    hidden_exec = independent_query(sql, hidden)
    hidden_expected = reference_result(replace(case, source=hidden))
    result.update(source_grade=source_grade, source_correct=source_grade["correct"])
    if source_exec["status"] != "ok" or hidden_exec["status"] != "ok":
        result.update(unscorable=True, source_sql_status=source_exec["status"], boundary_sql_status=hidden_exec["status"])
    else:
        published_consistency = grade_rows(publication.get("rows"), source_exec["rows"])
        boundary_grade = grade_rows(hidden_exec["rows"], hidden_expected)
        result.update(publication_matches_sql=published_consistency["correct"],
                      boundary_grade=boundary_grade, boundary_correct=boundary_grade["correct"])
    semantically_correct = bool(task_preserved and result["source_correct"] and result["publication_matches_sql"] and result["boundary_correct"])
    result["correct_task"] = semantically_correct and not operational
    result["wrong_release"] = not semantically_correct
    return result


def _usage(events: list[dict]) -> dict:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    sums = {name + "_known": 0 for name in fields}
    unknown = 0
    for event in events:
        values = event.get("usage") or {}
        complete = all(type(values.get(name)) is int and values[name] >= 0 for name in fields)
        if not complete or values.get("prompt_tokens", 0) + values.get("completion_tokens", 0) != values.get("total_tokens"):
            unknown += 1
        for name in fields:
            if type(values.get(name)) is int and values[name] >= 0:
                sums[name + "_known"] += values[name]
    return {**sums, "event_count": len(events), "unknown_usage_events": unknown, "usage_complete": unknown == 0}


def summarize(rows: list[dict], modes: tuple[str, ...]) -> dict:
    groups = {}
    for mode in modes:
        for track in ("controlled", "natural"):
            selected = [row for row in rows if row["mode"] == mode and row["track"] == track]
            if not selected:
                continue
            completed = [row for row in selected if row["status"] == "completed"]
            groups[mode + ":" + track] = {
                "scheduled": len(selected), "completed": len(completed),
                "not_run": sum(row["status"] == "not_run" for row in selected),
                "operational_errors": sum(row["status"] == "operational_error" or row.get("grade", {}).get("operational_failure", False) for row in selected),
                "accepted": sum(row["grade"]["accepted"] for row in completed),
                "correct_tasks": sum(row["grade"]["correct_task"] for row in completed),
                "source_correct": sum(row["grade"]["source_correct"] for row in completed),
                "boundary_correct": sum(row["grade"]["boundary_correct"] for row in completed),
                "wrong_releases": sum(row["grade"]["wrong_release"] for row in completed),
                "unscorable": sum(row["grade"]["unscorable"] for row in completed),
                "actual_http_attempts": sum(row["http_attempts"] for row in selected),
                "native_turns": sum(row.get("native_turns", 0) for row in selected),
                "usage": _usage([event for row in selected for event in row.get("http_events", [])]),
                "correct_over_scheduled": {"numerator": sum(row["grade"]["correct_task"] for row in completed), "denominator": len(selected)},
            }
    return {"scheduled_cells": len(rows), "groups": groups}


def paired_results(rows: list[dict]) -> list[dict]:
    indexed = {(row["case_id"], row["seed"], row["mode"]): row for row in rows if row["track"] == "controlled"}
    pairs = []
    for case_id, seed in sorted({(row["case_id"], row["seed"]) for row in rows if row["track"] == "controlled"}):
        full, compact = indexed.get((case_id, seed, "full")), indexed.get((case_id, seed, "compact"))
        if full is None or compact is None:
            continue
        entry = {"case_id": case_id, "seed": seed,
                 "eligible_initial_counterexample": bool(full["eligible_initial_counterexample"]),
                 "full_status": full["status"], "compact_status": compact["status"],
                 "both_completed": full["status"] == compact["status"] == "completed"}
        for mode, row in (("full", full), ("compact", compact)):
            entry[mode + "_correct"] = row.get("grade", {}).get("correct_task", False)
            entry[mode + "_http_attempts"] = row["http_attempts"]
            entry[mode + "_usage"] = _usage(row.get("http_events", []))
            entry[mode + "_seconds"] = row.get("seconds")
            entry[mode + "_initial_evidence_matches_frozen"] = row.get("initial_evidence_matches_frozen")
        pairs.append(entry)
    return pairs


async def run_probe_benchmark(
    output: Path, *, live: bool = False, config: QwenConfig | None = None,
    seeds: tuple[int, ...] = (17, 31), include_natural: bool = True,
    include_controlled: bool = True, case_ids: tuple[str, ...] | None = None,
    modes: tuple[str, ...] = MODES, max_requests: int = 80, max_attempts: int = 2,
    money_limit_cny: float = 5, budget: MonetaryRequestBudget | None = None,
    client_factory=None,
    feedback_policy: str = "capability_v2",
    evaluation_cases: tuple[Case, ...] | None = None,
    natural_case_ids: tuple[str, ...] | None = None,
) -> dict:
    """Run an exclusive bounded experiment; supplied budget spans several runs.

    client_factory(case, mode, seed, track, budget) is for offline tests only.
    Natural tasks use the first seed once; controlled candidates use all seeds.
    """
    if live and client_factory is not None:
        raise ValueError("Live mode cannot use an injected client factory")
    validate_policy(feedback_policy)
    if live and config is None:
        raise ValueError("Live mode requires explicit ordinary Qwen configuration")
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > 3 or any(type(value) is not int for value in seeds):
        raise ValueError("Select 1..3 distinct integer seeds")
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Select distinct full/compact modes")
    if not include_natural and not include_controlled:
        raise ValueError("Select at least one track")
    if not 1 <= max_attempts <= 3 or not 1 <= max_requests <= 100:
        raise ValueError("Attempts/request allowance exceeds bounded experiment")
    pool = get_cases() if evaluation_cases is None else evaluation_cases
    if (not isinstance(pool, tuple) or not pool or any(not isinstance(case, Case) for case in pool)
            or len({case.case_id for case in pool}) != len(pool)
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', case.case_id) for case in pool)):
        raise ValueError('Evaluation requires distinct frozen Case records')
    by_id = {case.case_id: case for case in pool}
    selected_ids = tuple(by_id) if case_ids is None else tuple(case_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids) or any(name not in by_id for name in selected_ids):
        raise ValueError("Select distinct development case IDs")
    cases = tuple(by_id[name] for name in selected_ids)
    natural_ids = NATURAL_CASE_IDS if natural_case_ids is None else natural_case_ids
    if natural_case_ids is not None and (len(set(natural_ids)) != len(natural_ids) or any(name not in by_id for name in natural_ids)):
        raise ValueError('Natural task selection must refer to distinct evaluation cases')
    natural = tuple(by_id[name] for name in natural_ids if name in selected_ids) if include_natural else ()
    controlled = cases if include_controlled else ()
    if not natural and not controlled:
        raise ValueError("No tasks match the requested track")
    pricing = validate_pricing(config) if live else None
    budget = budget if budget is not None else MonetaryRequestBudget(
        limit=max_requests, max_input_chars=1_000_000, max_reported_tokens=250_000,
        money_limit_cny=money_limit_cny, max_output_tokens=900,
    )
    if live and budget.max_output_tokens != 900:
        raise ValueError("Live client output cap must match the 900-token reservation")
    budget_start = deepcopy(budget.snapshot())
    output = Path(output).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "reports").resolve()):
        raise ValueError("Experiment output must remain inside project reports")
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "runtime").mkdir()
    initial_probes = {}
    for case in controlled:
        for seed in seeds:
            initial_probes[(case.case_id, seed)] = probe_candidate(case.candidate_sql, case.task, seed=seed)
    rows = []
    for track, task_cases, trial_seeds in (("controlled", controlled, seeds), ("natural", natural, (seeds[0],))):
        for case in task_cases:
            for seed in trial_seeds:
                initial = initial_probes.get((case.case_id, seed)) if track == "controlled" else None
                eligible = bool(initial and initial.get("counterexample"))
                digest = _digest(initial["counterexample"]) if eligible else None
                for mode in modes:
                    rows.append({"case_id": case.case_id, "task_id": case.task.task_id,
                                 "track": track, "seed": seed, "mode": mode,
                                 "status": "not_run", "stop_reason": "NOT_STARTED",
                                 "eligible_initial_counterexample": eligible,
                                 "initial_counterexample_sha256": digest,
                                 "http_attempts": 0, "http_events": [], "native_turns": 0})
    own_files = ("probe_live_benchmark.py", "probe_agent_bridge.py", "probe_guard.py", "probe_cost_budget.py", "probe_testing.py", "probe_runtime.py", "probe_cases.py", "dbgpt_qwen.py", "probe_capabilities.py", "capability_cases.py")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "disclosed_development_not_unseen_holdout", "upstream_commit": UPSTREAM_COMMIT,
        "real_model_used": live, "requested_model": config.model if live else "offline-scripted",
        "seeds": list(seeds), "modes": list(modes), "max_native_attempts": max_attempts,
        "per_request_max_tokens": 900, "pricing": pricing, "budget_start": budget_start,
        "feedback_policy": feedback_policy,
        "cases": [asdict(case) for case in cases],
        "boundary_fixtures": {case.case_id: asdict(boundary_fixture(case)) for case in cases},
        "initial_probes": {f"{case_id}:{seed}": report for (case_id, seed), report in initial_probes.items()},
        "planned_grid": deepcopy(rows),
        "code_sha256": {"dataverify/" + name: hashlib.sha256((PROJECT_ROOT / "dataverify" / name).read_bytes()).hexdigest() for name in own_files},
        "note": "Only confirmed task/schema goes to the model. Reference SQL and targets in this evaluator never enter runtime. Controlled first turns are injected, all offline repairs are scripts.",
    }
    _write(output / "frozen_manifest.json", manifest)
    report = {
        "status": "running", "created_at_utc": manifest["created_at_utc"],
        "real_model_used": live, "manifest_sha256": _digest(manifest),
        "feedback_policy": feedback_policy,
        "split": manifest["split"], "rows": rows,
        "candidate_cases": len(controlled), "task_specifications": len({case.task.task_id for case in cases}),
        "natural_questions": len(natural), "budget_start": budget_start,
        "limitation": "Known development candidates, two feedback variants and fixed seeds. Boundary fixtures test related data behaviour, not unseen problem generalization. No new-detector or production claim.",
    }
    all_events = []

    def checkpoint(status="running"):
        report.update(status=status, budget_end=deepcopy(budget.snapshot()),
                      summary=summarize(rows, modes), pairs=paired_results(rows), usage=_usage(all_events))
        report["this_run_http_attempts"] = report["budget_end"]["http_attempts"] - budget_start["http_attempts"]
        report["audit_event_count"] = len(all_events)
        report["audit_ledger_complete"] = len(all_events) == report["this_run_http_attempts"]
        _write(output / "summary.json", report)
        _write(output / "request_audit.json", all_events)

    checkpoint()
    index = {(row["case_id"], row["track"], row["seed"], row["mode"]): row for row in rows}
    try:
        groups = list(dict.fromkeys((row["case_id"], row["track"], row["seed"]) for row in rows))
        for position, (case_id, track, seed) in enumerate(groups):
            case = by_id[case_id]
            order = modes[position % len(modes):] + modes[:position % len(modes)]
            for mode in order:
                row = index[(case_id, track, seed, mode)]
                current = budget.snapshot()
                if live and (current.get("stopped_reason") or current["http_attempts"] >= current["http_attempt_limit"] or current["reported_total_tokens"] >= current["reported_token_stop_threshold"]):
                    row["stop_reason"] = current.get("stopped_reason") or "GLOBAL_BUDGET_EXHAUSTED"
                    checkpoint()
                    continue
                started = time.perf_counter()
                before_requests = budget.snapshot()["http_attempts"]
                source_digest = _digest(asdict(case.source))
                record = None
                try:
                    kwargs = {}
                    if client_factory is not None:
                        kwargs["llm_client"] = client_factory(case, mode, seed, track, budget)
                    elif live:
                        client = QwenLLMClient(config, budget, max_tokens=900)
                        kwargs["llm_client"] = InjectedFirstCandidateClient(client, probe_sql_candidate(case.candidate_sql, case.task.task_id)) if track == "controlled" else client
                    else:
                        reference = probe_sql_candidate(case.reference_sql, case.task.task_id)
                        kwargs["responses"] = ([probe_sql_candidate(case.candidate_sql, case.task.task_id)] if track == "controlled" else []) + [reference] * max_attempts
                    record = await run_probe_native_case(
                        case.task, case.source, mode=mode, seed=seed, max_attempts=max_attempts,
                        output_dir=output / "runtime" / f"{case_id}_{track}_{seed}_{mode}", **kwargs,
                        feedback_policy=feedback_policy,
                    )
                    source_unchanged = source_digest == _digest(asdict(case.source))
                    if not source_unchanged:
                        raise RuntimeError("Immutable source fixture changed")
                    grade = grade_probe_record(case, record)
                    events = deepcopy(record.get("http_events", []))
                    all_events.extend(events)
                    delta = budget.snapshot()["http_attempts"] - before_requests
                    ledger_complete = len(events) == delta
                    if not ledger_complete:
                        budget.stop("AUDIT_LEDGER_INCOMPLETE")
                        grade.update(operational_failure=True, correct_task=False)
                    if events and not _usage(events)["usage_complete"]:
                        grade.update(operational_failure=True, correct_task=False)
                    first_action = (record.get("actions") or [{}])[0]
                    first_probe = first_action.get("probe_report") or first_action.get("probe") or {}
                    first_witness = first_probe.get("counterexample")
                    matches = (_digest(first_witness) == row["initial_counterexample_sha256"]) if row["eligible_initial_counterexample"] and first_witness else None
                    row.update(status="completed", stop_reason=None, grade=grade,
                               source_sha256=source_digest, source_unchanged=source_unchanged,
                               http_attempts=delta, http_events=events, audit_ledger_complete=ledger_complete,
                               native_turns=record.get("llm_calls", len(record.get("llm_requests", []))),
                               initial_evidence_matches_frozen=matches,
                               publication=deepcopy(record.get("publication")))
                    trace = {"cell": {k: v for k, v in row.items() if k != "http_events"}, "record": record}
                    _write(output / "traces" / f"{case_id}_{track}_{seed}_{mode}.json", trace)
                except Exception as exc:
                    delta = budget.snapshot()["http_attempts"] - before_requests
                    if delta:
                        budget.stop("RUNNER_ERROR_USAGE_OR_LEDGER_UNKNOWN")
                    row.update(status="operational_error", stop_reason="RUNNER_" + type(exc).__name__,
                               http_attempts=delta, audit_ledger_complete=delta == 0)
                    _write(output / "traces" / f"{case_id}_{track}_{seed}_{mode}.json", {"cell": row})
                finally:
                    row["seconds"] = round(time.perf_counter() - started, 6)
                    checkpoint()
    except BaseException:
        checkpoint("interrupted")
        raise
    report["all_cells_completed"] = all(row["status"] == "completed" for row in rows)
    checkpoint("completed" if report["all_cells_completed"] else "incomplete")
    return report
