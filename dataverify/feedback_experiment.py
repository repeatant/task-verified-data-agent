"""Development-only feedback ablation and state-adaptive repair experiments.

The evaluator owns answers and grading. Runtime modules must not import it.
Offline scripts exercise interfaces, not a model's ability to repair queries.
Live experiments continue one complete monetary ledger, with a separate bounded
allowance for new calls and reservations. No credentials are read on import.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import experiment_budget
from .dbgpt_bridge import UPSTREAM_COMMIT
from .dbgpt_qwen import InjectedFirstCandidateClient, QwenConfig, QwenLLMClient
from .experiment_budget import IncrementalAllowance as IncrementalAllowance
from .experiment_budget import restore_budget as restore_budget
from .probe_agent_bridge import probe_sql_candidate, run_probe_native_case
from .probe_cases import Case, get_case
from .probe_cost_budget import MonetaryRequestBudget, validate_pricing
from .probe_live_benchmark import boundary_fixture, grade_probe_record
from .reporting import digest_json, summarize_usage, write_json

JsonObject = dict[str, Any]
Arm = tuple[str, str, str]
ClientFactory = Callable[[Case, str, int, str, IncrementalAllowance], Any]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "ablation": (
        ("neither", "compact", "legacy_v1"),
        ("capability_only", "compact", "capability_only"),
        ("guidance_only", "compact", "guidance_only"),
        ("both", "compact", "capability_v2"),
    ),
    "strategy": (
        ("full_only", "full", "capability_v2"),
        ("compact_only", "compact", "capability_v2"),
        ("adaptive", "adaptive", "capability_v2"),
    ),
}
SOURCE_FILES = (
    "dataverify/feedback_experiment.py",
    "dataverify/probe_agent_bridge.py",
    "dataverify/probe_guard.py",
    "dataverify/probe_capabilities.py",
    "dataverify/probe_runtime.py",
    "dataverify/probe_testing.py",
    "dataverify/probe_cases.py",
    "dataverify/probe_live_benchmark.py",
    "dataverify/probe_cost_budget.py",
    "dataverify/dbgpt_qwen.py",
    "dataverify/experiment_budget.py",
    "dataverify/reporting.py",
    "dataverify/feedback_strategy.py",
)


def source_fingerprints() -> dict[str, str]:
    """Fingerprint every implementation file that can affect this experiment."""
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def _review_accounted_truncation(document: JsonObject, path: Path, raw: bytes) -> JsonObject:
    """Compatibility entry point; the budget module owns the actual review."""
    return experiment_budget.review_accounted_truncation(
        document, path, raw, project_root=PROJECT_ROOT
    )


def load_prior_budget(
    path: Path,
    *,
    max_extra_requests: int = 24,
    money_limit_cny: float = 5,
    allow_truncated_resume: bool = False,
) -> MonetaryRequestBudget:
    """Continue a checked report within this runner's project directory."""
    return experiment_budget.load_prior_budget(
        path,
        project_root=PROJECT_ROOT,
        max_extra_requests=max_extra_requests,
        money_limit_cny=money_limit_cny,
        allow_truncated_resume=allow_truncated_resume,
    )


def summarize(rows: list[JsonObject], stage: str, *, live: bool) -> JsonObject:
    groups = {}
    for arm, _, _ in ARMS[stage]:
        for track in ("controlled", "natural"):
            selected = [r for r in rows if r["arm"] == arm and r["track"] == track]
            if not selected:
                continue
            completed = [r for r in selected if r["status"] == "completed"]
            correct = sum(r["grade"]["correct_task"] for r in completed)
            groups[arm + ":" + track] = {
                "scheduled": len(selected),
                "completed": len(completed),
                "not_run": sum(r["status"] == "not_run" for r in selected),
                "operational_errors": sum(
                    r["status"] == "operational_error"
                    or r.get("grade", {}).get("operational_failure", False)
                    for r in selected
                ),
                "correct_tasks": correct,
                "real_model_correct_tasks": correct if live else None,
                "offline_contract_correct": correct if not live else None,
                "correct_over_scheduled": {"numerator": correct, "denominator": len(selected)},
                "wrong_releases": sum(r["grade"]["wrong_release"] for r in completed),
                "actual_http_attempts": sum(r["http_attempts"] for r in selected),
                "injected_initial_turns": sum(r.get("injected_initial_turns", 0) for r in selected),
                "native_turns": sum(r.get("native_turns", 0) for r in selected),
                "escalation_transitions": sum(r.get("escalation_transitions", 0) for r in selected),
                "escalated_feedbacks": sum(r.get("escalated_feedbacks", 0) for r in selected),
                "consumed_escalated_feedbacks": sum(
                    r.get("consumed_escalated_feedbacks", 0) for r in selected
                ),
                "full_evidence_feedbacks": sum(
                    r.get("full_evidence_feedbacks", 0) for r in selected
                ),
                "consumed_full_evidence_feedbacks": sum(
                    r.get("consumed_full_evidence_feedbacks", 0) for r in selected
                ),
                "diagnostic_queries": sum(r.get("diagnostic_queries", 0) for r in selected),
                "regression_queries": sum(r.get("regression_queries", 0) for r in selected),
                "elapsed_seconds": round(sum(r.get("seconds", 0) for r in selected), 6),
                "usage": summarize_usage([e for r in selected for e in r.get("http_events", [])]),
            }
    return {
        "scheduled_cells": len(rows),
        "groups": groups,
        "interpretation": "real-model development evaluation"
        if live
        else "scripted interface/regression checks only; not model repair evidence",
    }


def paired_results(rows: list[JsonObject], stage: str) -> list[JsonObject]:
    pairs = []
    groups = list(dict.fromkeys((r["case_id"], r["track"], r["seed"]) for r in rows))
    for case_id, track, seed in groups:
        selected = [
            r for r in rows if (r["case_id"], r["track"], r["seed"]) == (case_id, track, seed)
        ]
        pairs.append(
            {
                "case_id": case_id,
                "track": track,
                "seed": seed,
                "all_arms_completed": all(r["status"] == "completed" for r in selected),
                "arms": {
                    r["arm"]: {
                        "status": r["status"],
                        "correct_task": r.get("grade", {}).get("correct_task", False),
                        "http_attempts": r["http_attempts"],
                        "usage": summarize_usage(r.get("http_events", [])),
                        "escalated_feedbacks": r.get("escalated_feedbacks", 0),
                    }
                    for r in selected
                },
            }
        )
    return pairs


def feedback_selection_events(record: JsonObject) -> list[JsonObject]:
    """Distinguish produced feedback from feedback sent to a later model call.

    A final failed turn may select full evidence with no retry left. Such a
    selection must not be counted as evidence that a model used that feedback.
    Native request existence and actual HTTP attempts are recorded separately.
    """
    selections = []
    requests = record.get("llm_requests", [])
    origins = record.get("turn_origins", [])
    posted = len(record.get("http_events", []))
    previous_mode = "compact"
    for action_index, action in enumerate(record.get("actions", [])):
        feedback = action.get("feedback", {})
        strategy = feedback.get("feedback_strategy")
        if not strategy:
            continue
        next_index = action_index + 1
        forwarded = next_index < len(requests)
        if origins:
            real_ordinal = sum(origin == "real_qwen_http" for origin in origins[: next_index + 1])
            real_next = (
                forwarded
                and next_index < len(origins)
                and origins[next_index] == "real_qwen_http"
                and posted >= real_ordinal
            )
        else:
            real_next = forwarded and next_index < posted
        failed_full = (
            feedback.get("status") != "verified" and strategy.get("selected_mode") == "full"
        )
        selections.append(
            {
                **deepcopy(strategy),
                "after_action_index": action_index,
                "new_escalation_transition": bool(failed_full and previous_mode != "full"),
                "failed_full_feedback": bool(failed_full),
                "full_evidence_emitted": bool(
                    failed_full
                    and strategy.get("evidence_available")
                    and feedback.get("diagnostic_evidence")
                ),
                "sent_in_later_native_request": bool(forwarded),
                "sent_in_later_http_attempt": bool(real_next),
            }
        )
        previous_mode = strategy.get("selected_mode", previous_mode)
    return selections


@dataclass(frozen=True)
class _ExperimentSettings:
    """Caller options, kept separate from mutable run state and report JSON."""

    stage: str
    live: bool
    config: QwenConfig | None
    controlled_case_ids: tuple[str, ...]
    natural_case_ids: tuple[str, ...]
    arm_ids: tuple[str, ...] | None
    seeds: tuple[int, ...]
    max_attempts: int
    max_extra_requests: int
    extra_reservation_cny: float
    money_limit_cny: float
    client_factory: ClientFactory | None


@dataclass(frozen=True)
class _ExperimentPlan:
    """Validated inputs shared by every arm of the comparison."""

    arms: tuple[Arm, ...]
    controlled: tuple[Case, ...]
    natural: tuple[Case, ...]
    cases_by_id: dict[str, Case]
    budget: MonetaryRequestBudget
    pricing: JsonObject | None


@dataclass
class _ExperimentRun:
    """Own the mutable report; runtime Agent code never sees this object."""

    output: Path
    settings: _ExperimentSettings
    plan: _ExperimentPlan
    allowance: IncrementalAllowance
    rows: list[JsonObject]
    hashes: dict[str, str]
    report: JsonObject
    events: list[JsonObject] = field(default_factory=list)

    def checkpoint(self, status: str = "running") -> None:
        """Persist partial outcomes even when a later trial cannot run."""
        end = deepcopy(self.plan.budget.snapshot())
        start = self.report["budget_start"]
        self.report.update(
            status=status,
            budget_end=end,
            summary=summarize(self.rows, self.settings.stage, live=self.settings.live),
            pairs=paired_results(self.rows, self.settings.stage),
            usage=summarize_usage(self.events),
            this_run_http_attempts=end["http_attempts"] - start["http_attempts"],
            this_run_reserved_upper_cny=float(
                Decimal(str(end["reserved_upper_cny"])) - Decimal(str(start["reserved_upper_cny"]))
            ),
            audit_event_count=len(self.events),
            local_allowance_stop=self.allowance.stop_reason(),
            source_integrity_ok=source_fingerprints() == self.hashes,
        )
        self.report["audit_ledger_complete"] = (
            len(self.events) == self.report["this_run_http_attempts"]
        )
        write_json(self.output / "summary.json", self.report)
        write_json(self.output / "request_audit.json", self.events)


def _select_arms(stage: str, arm_ids: tuple[str, ...] | None) -> tuple[Arm, ...]:
    if stage not in ARMS:
        raise ValueError("Stage must be ablation or strategy")
    known_arms = {arm for arm, _, _ in ARMS[stage]}
    if arm_ids is not None and (
        not arm_ids
        or len(set(arm_ids)) != len(arm_ids)
        or any(arm not in known_arms for arm in arm_ids)
    ):
        raise ValueError("Select distinct arms registered for this stage")
    return tuple(item for item in ARMS[stage] if arm_ids is None or item[0] in arm_ids)


def _validate_run_options(
    settings: _ExperimentSettings, budget: MonetaryRequestBudget | None
) -> None:
    if type(settings.max_attempts) is not int or not 1 <= settings.max_attempts <= 3:
        raise ValueError("Select 1..3 native attempts")
    seeds = settings.seeds
    if (
        not seeds
        or len(seeds) > 3
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError("Select 1..3 distinct integer seeds")
    if settings.live and (settings.config is None or settings.client_factory is not None):
        raise ValueError("Live mode requires explicit config and forbids a test client factory")
    if settings.live and budget is None:
        raise ValueError("Live mode requires a supplied complete cumulative budget")


def _prepare_plan(
    settings: _ExperimentSettings, budget: MonetaryRequestBudget | None
) -> _ExperimentPlan:
    arms = _select_arms(settings.stage, settings.arm_ids)
    _validate_run_options(settings, budget)
    controlled = tuple(get_case(name) for name in settings.controlled_case_ids)
    natural = tuple(get_case(name) for name in settings.natural_case_ids)
    if not controlled and not natural:
        raise ValueError("Select at least one task")
    if len(set(settings.controlled_case_ids)) != len(controlled) or len(
        set(settings.natural_case_ids)
    ) != len(natural):
        raise ValueError("Select distinct case IDs within each track")
    pricing = validate_pricing(settings.config) if settings.live else None
    budget = (
        budget
        if budget is not None
        else MonetaryRequestBudget(
            limit=settings.max_extra_requests,
            money_limit_cny=settings.money_limit_cny,
            max_output_tokens=900,
        )
    )
    if not isinstance(budget, MonetaryRequestBudget) or budget.max_output_tokens != 900:
        raise ValueError("A 900-token monetary budget is required")
    if settings.live and (
        not budget.snapshot()["reported_cost_complete"] or budget.snapshot()["stopped_reason"]
    ):
        raise ValueError("Live continuation requires a complete unstopped ledger")
    return _ExperimentPlan(
        arms=arms,
        controlled=controlled,
        natural=natural,
        cases_by_id={case.case_id: case for case in controlled + natural},
        budget=budget,
        pricing=pricing,
    )


def _planned_grid(settings: _ExperimentSettings, plan: _ExperimentPlan) -> list[JsonObject]:
    rows = []
    tracks = (
        ("controlled", plan.controlled, settings.seeds),
        ("natural", plan.natural, (settings.seeds[0],)),
    )
    for track, cases, trial_seeds in tracks:
        for case in cases:
            for seed in trial_seeds:
                for arm, mode, policy in plan.arms:
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "task_id": case.task.task_id,
                            "track": track,
                            "seed": seed,
                            "arm": arm,
                            "mode": mode,
                            "feedback_policy": policy,
                            "status": "not_run",
                            "stop_reason": "NOT_STARTED",
                            "http_attempts": 0,
                            "http_events": [],
                            "native_turns": 0,
                        }
                    )
    return rows


def _frozen_manifest(run: _ExperimentRun) -> JsonObject:
    settings, plan = run.settings, run.plan
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": settings.stage,
        "split": "disclosed_development_not_unseen_holdout",
        "real_model_used": settings.live,
        "requested_model": settings.config.model if settings.live else "offline-scripted",
        "upstream_commit": UPSTREAM_COMMIT,
        "pricing": plan.pricing,
        "seeds": list(settings.seeds),
        "max_native_attempts": settings.max_attempts,
        "per_request_max_tokens": 900,
        "arms": plan.arms,
        "budget_start": deepcopy(plan.budget.snapshot()),
        "continuation_review": deepcopy(getattr(plan.budget, "_continuation_review", [])),
        "ablation_factors": {
            "capability_description": "Additional full execution_contract; every arm retains the same basic supported_sql description.",
            "classified_guidance": "Classified repair_guidance added to failed Action feedback.",
        },
        "max_extra_requests": settings.max_extra_requests,
        "extra_reservation_cny": settings.extra_reservation_cny,
        "cases": [asdict(case) for case in plan.cases_by_id.values()],
        "boundary_fixtures": {
            key: asdict(boundary_fixture(case)) for key, case in plan.cases_by_id.items()
        },
        "planned_grid": deepcopy(run.rows),
        "code_sha256": run.hashes,
        "ordering": "Arm order rotates by paired task/track/seed; no result-based selection.",
        "offline_script": "Controlled: injected bad, repeat bad, evaluator reference; natural: evaluator reference. Interface checks only.",
        "note": "Evaluator labels, references and expected rows do not enter live model context. Controlled first native turns are injected and not billed model outputs.",
    }


def _start_run(
    output: Path, settings: _ExperimentSettings, plan: _ExperimentPlan
) -> _ExperimentRun:
    # 预先分配全部单元，预算中止也不会缩小成功率分母。
    allowance = IncrementalAllowance(
        plan.budget, settings.max_extra_requests, settings.extra_reservation_cny
    )
    output = Path(output).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "reports").resolve()):
        raise ValueError("Experiment output must remain inside project reports")
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "runtime").mkdir()
    run = _ExperimentRun(
        output=output,
        settings=settings,
        plan=plan,
        allowance=allowance,
        rows=_planned_grid(settings, plan),
        hashes=source_fingerprints(),
        report={},
    )
    manifest = _frozen_manifest(run)
    write_json(output / "frozen_manifest.json", manifest)
    run.report = {
        "status": "running",
        "stage": settings.stage,
        "split": manifest["split"],
        "created_at_utc": manifest["created_at_utc"],
        "real_model_used": settings.live,
        "manifest_sha256": digest_json(manifest),
        "rows": run.rows,
        "budget_start": manifest["budget_start"],
        "continuation_review": deepcopy(manifest["continuation_review"]),
        "max_extra_requests": settings.max_extra_requests,
        "extra_reservation_cny": settings.extra_reservation_cny,
        "limitation": "Known fixed-schema development tasks; repetitions and converted regressions are not independent unseen tasks. Offline results are scripted interface checks, never model effectiveness.",
    }
    run.checkpoint()
    return run


def _client_arguments(run: _ExperimentRun, case: Case, row: JsonObject) -> JsonObject:
    settings = run.settings
    if settings.client_factory is not None:
        return {
            "llm_client": settings.client_factory(
                case, row["arm"], row["seed"], row["track"], run.allowance
            )
        }
    if settings.live:
        client = QwenLLMClient(settings.config, run.allowance, max_tokens=900)
        if row["track"] == "controlled":
            client = InjectedFirstCandidateClient(
                client, probe_sql_candidate(case.candidate_sql, case.task.task_id)
            )
        return {"llm_client": client}
    bad = probe_sql_candidate(case.candidate_sql, case.task.task_id)
    good = probe_sql_candidate(case.reference_sql, case.task.task_id)
    responses = (
        [bad, bad, good][: settings.max_attempts]
        if row["track"] == "controlled"
        else [good] * settings.max_attempts
    )
    return {"responses": responses}


def _complete_trial(
    run: _ExperimentRun,
    case: Case,
    row: JsonObject,
    record: JsonObject,
    before_requests: int,
    source_hash: str,
) -> None:
    events = deepcopy(record.get("http_events", []))
    run.events.extend(events)
    grade = grade_probe_record(case, record)
    delta = run.plan.budget.snapshot()["http_attempts"] - before_requests
    complete = len(events) == delta
    if not complete:
        run.plan.budget.stop("AUDIT_LEDGER_INCOMPLETE")
    if not complete or (events and not summarize_usage(events)["usage_complete"]):
        grade.update(operational_failure=True, correct_task=False)
    actions = record.get("actions", [])
    selections = feedback_selection_events(record)
    origins = record.get("turn_origins", [])
    row.update(
        status="completed",
        stop_reason=None,
        grade=grade,
        http_attempts=delta,
        http_events=events,
        audit_ledger_complete=complete,
        source_sha256=source_hash,
        source_unchanged=True,
        native_turns=record.get("llm_calls", 0),
        injected_initial_turns=sum(value == "injected_test_fault_no_api_call" for value in origins),
        offline_scripted=not run.settings.live,
        feedback_selections=selections,
        escalation_transitions=sum(item["new_escalation_transition"] for item in selections),
        escalated_feedbacks=sum(item["failed_full_feedback"] for item in selections),
        consumed_escalated_feedbacks=sum(
            item["failed_full_feedback"] and item["sent_in_later_http_attempt"]
            for item in selections
        ),
        full_evidence_feedbacks=sum(item["full_evidence_emitted"] for item in selections),
        consumed_full_evidence_feedbacks=sum(
            item["full_evidence_emitted"] and item["sent_in_later_http_attempt"]
            for item in selections
        ),
        diagnostic_queries=sum(
            (action.get("probe_report") or {}).get("queries", 0) for action in actions
        ),
        regression_queries=sum(action.get("regression_query_attempts", 0) for action in actions),
        publication=deepcopy(record.get("publication")),
    )


def _record_trial_error(
    run: _ExperimentRun, row: JsonObject, error: Exception, before_requests: int
) -> None:
    delta = run.plan.budget.snapshot()["http_attempts"] - before_requests
    if delta:
        run.plan.budget.stop("RUNNER_ERROR_USAGE_OR_LEDGER_UNKNOWN")
    row.update(
        status="operational_error",
        stop_reason="RUNNER_" + type(error).__name__,
        http_attempts=delta,
        audit_ledger_complete=delta == 0,
    )


async def _run_trial(run: _ExperimentRun, row: JsonObject) -> None:
    case = run.plan.cases_by_id[row["case_id"]]
    name = f"{row['case_id']}_{row['track']}_{row['seed']}_{row['arm']}"
    trace = run.output / "traces" / f"{name}.json"
    started = time.perf_counter()
    before_requests = run.plan.budget.snapshot()["http_attempts"]
    source_hash = digest_json(asdict(case.source))
    try:
        record = await run_probe_native_case(
            case.task,
            case.source,
            mode=row["mode"],
            feedback_policy=row["feedback_policy"],
            seed=row["seed"],
            max_attempts=run.settings.max_attempts,
            output_dir=run.output / "runtime" / name,
            **_client_arguments(run, case, row),
        )
        if digest_json(asdict(case.source)) != source_hash:
            raise RuntimeError("SOURCE_FIXTURE_CHANGED")
        _complete_trial(run, case, row, record, before_requests, source_hash)
        write_json(
            trace,
            {
                "cell": {key: value for key, value in row.items() if key != "http_events"},
                "record": record,
            },
        )
    except Exception as error:
        _record_trial_error(run, row, error, before_requests)
        write_json(trace, {"cell": row})
    finally:
        row["seconds"] = round(time.perf_counter() - started, 6)
        run.checkpoint()


def _skip_trial(run: _ExperimentRun, row: JsonObject) -> bool:
    if source_fingerprints() != run.hashes:
        row["stop_reason"] = "SOURCE_CHANGED_DURING_EXPERIMENT"
        run.checkpoint()
        return True
    reason = run.allowance.stop_reason() if run.settings.live else None
    if reason:
        row["stop_reason"] = reason
        run.checkpoint()
        return True
    return False


async def _run_grid(run: _ExperimentRun) -> None:
    groups = list(dict.fromkeys((row["case_id"], row["track"], row["seed"]) for row in run.rows))
    index = {(row["case_id"], row["track"], row["seed"], row["arm"]): row for row in run.rows}
    for position, (case_id, track, seed) in enumerate(groups):
        arms = run.plan.arms
        offset = position % len(arms)
        # 按任务轮换分组顺序，不能根据已看到的结果挑选下一组。
        for arm, _, _ in arms[offset:] + arms[:offset]:
            row = index[(case_id, track, seed, arm)]
            if not _skip_trial(run, row):
                await _run_trial(run, row)


async def run_feedback_experiment(
    output: Path,
    *,
    stage: str = "ablation",
    live: bool = False,
    config: QwenConfig | None = None,
    budget: MonetaryRequestBudget | None = None,
    controlled_case_ids: tuple[str, ...] = ("d06", "d10", "d13"),
    natural_case_ids: tuple[str, ...] = ("d01", "d03", "d04"),
    arm_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (53,),
    max_attempts: int = 3,
    max_extra_requests: int = 24,
    extra_reservation_cny: float = 0.8,
    money_limit_cny: float = 5,
    client_factory: ClientFactory | None = None,
) -> JsonObject:
    """Validate, freeze, execute and checkpoint one paired development study.

    Natural tasks use the first seed; seed replicates are not new task templates.
    A client factory is offline-only, and an existing budget is never reset.
    """
    settings = _ExperimentSettings(
        stage=stage,
        live=live,
        config=config,
        controlled_case_ids=controlled_case_ids,
        natural_case_ids=natural_case_ids,
        arm_ids=arm_ids,
        seeds=seeds,
        max_attempts=max_attempts,
        max_extra_requests=max_extra_requests,
        extra_reservation_cny=extra_reservation_cny,
        money_limit_cny=money_limit_cny,
        client_factory=client_factory,
    )
    plan = _prepare_plan(settings, budget)
    run = _start_run(output, settings, plan)
    try:
        await _run_grid(run)
    except BaseException:
        run.checkpoint("interrupted")
        raise
    run.report["all_cells_completed"] = all(row["status"] == "completed" for row in run.rows)
    complete = run.report["all_cells_completed"] and run.report["source_integrity_ok"]
    run.checkpoint("completed" if complete else "incomplete")
    return run.report
