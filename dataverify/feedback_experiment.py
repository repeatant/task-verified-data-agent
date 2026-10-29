"""Development-only feedback ablation and state-adaptive repair experiments.

The evaluator owns answers and grading. Runtime modules must not import it.
Offline scripts exercise interfaces, not a model's ability to repair queries.
Live experiments continue one complete monetary ledger, with a separate bounded
allowance for new calls and reservations. No credentials are read on import.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import time

from .dbgpt_bridge import UPSTREAM_COMMIT
from .dbgpt_qwen import InjectedFirstCandidateClient, QwenClientError, QwenConfig, QwenLLMClient
from .probe_agent_bridge import probe_sql_candidate, run_probe_native_case
from .probe_cases import get_case
from .probe_cost_budget import MonetaryRequestBudget, validate_pricing
from .probe_live_benchmark import _usage, _write, boundary_fixture, grade_probe_record


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
    "dataverify/feedback_experiment.py", "dataverify/probe_agent_bridge.py",
    "dataverify/probe_guard.py", "dataverify/probe_capabilities.py",
    "dataverify/probe_runtime.py", "dataverify/probe_testing.py",
    "dataverify/probe_cases.py", "dataverify/probe_live_benchmark.py",
    "dataverify/probe_cost_budget.py", "dataverify/dbgpt_qwen.py",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def source_fingerprints() -> dict:
    return {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def restore_budget(snapshot: dict, *, max_extra_requests: int = 24,
                   money_limit_cny: float = 5) -> MonetaryRequestBudget:
    """Replay completed attempts exactly; raise only the technical request cap.

    Existing money, input and Token limits are retained. Unknown usage, changed
    pricing, inconsistent totals and failed/in-flight ledgers cannot be resumed.
    """
    if type(max_extra_requests) is not int or not 1 <= max_extra_requests <= 36:
        raise ValueError("Select 1..36 additional HTTP attempts")
    if (snapshot.get("reported_cost_complete") is not True or snapshot.get("stopped_reason")
            or snapshot.get("unknown_count") != 0 or snapshot.get("in_flight_count") != 0
            or snapshot.get("unmatched_usage_events") != 0):
        raise ValueError("Only complete, unstopped ledgers can be continued")
    prices = {"model": "qwen3.8-flash", "region": "cn-beijing", "currency": "CNY",
              "input_cny_per_million_tokens": .8, "output_cny_per_million_tokens": 2.7}
    if any(snapshot.get("pricing", {}).get(key) != value for key, value in prices.items()):
        raise ValueError("Prior ledger pricing differs")
    if snapshot.get("money_limit_cny") != money_limit_cny:
        raise ValueError("Continuation must preserve the prior money allowance")
    used = snapshot.get("http_attempts")
    if type(used) is not int or used < 0:
        raise ValueError("Invalid prior attempt count")
    budget = MonetaryRequestBudget(
        limit=used + max_extra_requests,
        max_input_chars=snapshot["input_char_limit"],
        max_reported_tokens=snapshot["reported_token_stop_threshold"],
        money_limit_cny=money_limit_cny, max_output_tokens=snapshot["max_output_tokens"],
    )
    for entry in snapshot.get("reservation_ledger", []):
        if entry.get("status") != "usage_complete":
            raise ValueError("Prior reservation has incomplete usage")
        if budget.reserve(entry["prompt_chars"]) != entry["attempt"]:
            raise ValueError("Prior ledger attempts are not sequential")
        budget.record_tokens({name: entry[name] for name in
                              ("prompt_tokens", "completion_tokens", "total_tokens")})
    restored = budget.snapshot()
    for key in ("http_attempts", "input_chars", "reported_total_tokens", "complete_count",
                "reported_prompt_tokens", "reported_completion_tokens", "reserved_upper_cny",
                "estimated_reported_cny", "max_output_tokens", "money_limit_cny", "reservation_ledger"):
        if restored[key] != snapshot.get(key):
            raise ValueError("Prior ledger cannot be reproduced exactly")
    return budget


def _review_accounted_truncation(document: dict, path: Path, raw: bytes) -> dict:
    """Review a complete report, not just a stopped monetary snapshot.

    Only the final posted response may be a truncated HTTP-200 response. All
    usage must already be accounted for in the immutable monetary ledger.
    Transport failures, missing usage and unrelated provider errors remain
    non-resumable. This review grants no refund or extra output allowance.
    """
    if any(document.get(key) is not True for key in
           ("real_model_used", "source_integrity_ok", "audit_ledger_complete")):
        raise ValueError("Truncation continuation requires complete source and audit integrity")
    start, end = document.get("budget_start"), document.get("budget_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ValueError("Truncation continuation requires both budget snapshots")
    if (end.get("stopped_reason") != "PROVIDER_ERROR" or end.get("reported_cost_complete") is not True
            or end.get("unknown_count") != 0 or end.get("in_flight_count") != 0
            or end.get("unmatched_usage_events") != 0 or end.get("max_output_tokens") != 900):
        raise ValueError("Only fully accounted provider truncation can be reviewed")
    if (start.get("reported_cost_complete") is not True or start.get("stopped_reason")
            or start.get("unknown_count") != 0 or start.get("in_flight_count") != 0
            or start.get("unmatched_usage_events") != 0):
        raise ValueError("Initial budget was not a complete unstopped ledger")
    begin, finish = start.get("http_attempts"), end.get("http_attempts")
    if (type(begin) is not int or type(finish) is not int or not 0 <= begin < finish
            or document.get("this_run_http_attempts") != finish - begin):
        raise ValueError("Truncation report attempt counts disagree")
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Truncation report contains no auditable rows")
    events = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("http_events"), list):
            raise ValueError("Missing per-cell HTTP audit")
        posted = row["http_events"]
        if type(row.get("http_attempts")) is not int or row["http_attempts"] != len(posted):
            raise ValueError("Per-cell attempt counts disagree")
        events.extend(posted)
    attempts = [event.get("attempt") for event in events if isinstance(event, dict)]
    if (len(events) != finish - begin or len(attempts) != len(events)
            or any(type(attempt) is not int for attempt in attempts)
            or sorted(attempts) != list(range(begin + 1, finish + 1))
            or document.get("audit_event_count") != len(events)):
        raise ValueError("New HTTP events do not cover the ledger exactly")
    ledger = end.get("reservation_ledger")
    if (not isinstance(ledger, list) or len(ledger) != finish
            or ledger[:begin] != start.get("reservation_ledger")):
        raise ValueError("Historical reservations changed or are missing")
    failures = []
    totals = {name: 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for event in events:
        usage = event.get("usage")
        if (event.get("usage_complete") is not True or event.get("http_status") != 200
                or not isinstance(usage, dict)
                or any(type(usage.get(key)) is not int or usage[key] < 0 for key in totals)
                or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]):
            raise ValueError("Truncation audit contains incomplete HTTP usage")
        entry = ledger[event["attempt"] - 1]
        if (entry.get("attempt") != event["attempt"] or entry.get("status") != "usage_complete"
                or any(entry.get(key) != usage[key] for key in totals)):
            raise ValueError("HTTP event usage differs from the monetary ledger")
        for key in totals:
            totals[key] += usage[key]
        if event.get("status") == "completed":
            if event.get("error_code"):
                raise ValueError("Completed HTTP event contains an error")
        elif (event.get("status") == "failed" and event.get("error_code") == "QWEN_RESPONSE_TRUNCATED"
              and event["attempt"] == finish and usage["completion_tokens"] == 900):
            failures.append(event)
        else:
            raise ValueError("Only a final accounted HTTP-200 truncation can be resumed")
    if len(failures) != 1:
        raise ValueError("Exactly one final truncation must explain the provider stop")
    for token_field, snapshot_field in (("prompt_tokens", "reported_prompt_tokens"),
            ("completion_tokens", "reported_completion_tokens"), ("total_tokens", "reported_total_tokens")):
        if end.get(snapshot_field, -1) - start.get(snapshot_field, -1) != totals[token_field]:
            raise ValueError("Report usage totals disagree with the cumulative ledger")
    return {
        "source_report": path.relative_to(PROJECT_ROOT).as_posix(),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "original_stop_reason": "PROVIDER_ERROR",
        "review_reason": "Explicit continuation after one final HTTP-200 QWEN_RESPONSE_TRUNCATED response with complete matching usage and unchanged source/audit records.",
        "reviewed_failed_attempt": finish,
        "cumulative_http_attempts_preserved": finish,
        "cumulative_reported_tokens_preserved": end["reported_total_tokens"],
        "cumulative_reserved_upper_cny_preserved": end["reserved_upper_cny"],
        "refund_applied": False, "max_output_tokens_preserved": 900,
        "original_failed_cell_retained": True,
    }


def load_prior_budget(path: Path, *, max_extra_requests: int = 24,
                      money_limit_cny: float = 5,
                      allow_truncated_resume: bool = False) -> MonetaryRequestBudget:
    if type(allow_truncated_resume) is not bool:
        raise ValueError("Truncation continuation requires an explicit Boolean choice")
    path = Path(path).resolve()
    if not path.is_relative_to((PROJECT_ROOT / "reports").resolve()):
        raise ValueError("Prior budget must be an audited report inside reports")
    if path.stat().st_size > 5_000_000:
        raise ValueError("Prior budget report is too large")
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if document.get("real_model_used") is not True:
        raise ValueError("Prior report was not a live experiment")
    snapshot = document.get("budget_end", document.get("budget"))
    if not isinstance(snapshot, dict):
        raise ValueError("No complete budget found in prior report")
    reviews = deepcopy(document.get("continuation_review", []))
    if not isinstance(reviews, list) or any(not isinstance(review, dict) for review in reviews):
        raise ValueError("Invalid continuation review lineage")
    if allow_truncated_resume and snapshot.get("stopped_reason"):
        review = _review_accounted_truncation(document, path, raw)
        snapshot = deepcopy(snapshot)
        snapshot["stopped_reason"] = None
        reviews.append(review)
    budget = restore_budget(snapshot, max_extra_requests=max_extra_requests,
                            money_limit_cny=money_limit_cny)
    budget._continuation_review = reviews
    return budget


class IncrementalAllowance:
    """Serialized wrapper; never refunds or resets the shared monetary budget.

    Additional reservation is checked before delegating to reserve(), using the
    same fixed list-price rule as MonetaryRequestBudget. Exhausting this local
    allowance leaves a complete global ledger available for a reviewed next run.
    """

    def __init__(self, budget: MonetaryRequestBudget, max_requests: int, max_cny: float):
        if type(max_requests) is not int or not 1 <= max_requests <= 36:
            raise ValueError("Select 1..36 additional HTTP attempts")
        if (type(max_cny) not in (int, float) or not math.isfinite(max_cny)
                or not Decimal(0) < Decimal(str(max_cny)) <= Decimal("0.8")):
            raise ValueError("Additional reservation must be positive and at most CNY 0.8")
        self.budget, self.start = budget, budget.snapshot()
        self.max_requests, self.max_cny = max_requests, Decimal(str(max_cny))
        self.local_stop_reason = None

    def stop_reason(self):
        current = self.budget.snapshot()
        if current.get("stopped_reason"):
            return current["stopped_reason"]
        if self.local_stop_reason:
            return self.local_stop_reason
        if current["http_attempts"] - self.start["http_attempts"] >= self.max_requests:
            return "ADDITIONAL_REQUEST_LIMIT"
        if current["http_attempts"] >= current["http_attempt_limit"]:
            return "GLOBAL_REQUEST_LIMIT"
        if current["reported_total_tokens"] >= current["reported_token_stop_threshold"]:
            return "GLOBAL_TOKEN_LIMIT"
        return None

    def reserve(self, prompt_chars: int):
        if type(prompt_chars) is not int or prompt_chars < 0:
            raise ValueError("prompt_chars must be a nonnegative integer")
        reason = self.stop_reason()
        if reason:
            raise QwenClientError("LIVE_" + reason)
        current = self.budget.snapshot()
        extra = Decimal(str(current["reserved_upper_cny"])) - Decimal(str(self.start["reserved_upper_cny"]))
        next_nano = (4 * prompt_chars + 4096) * 800 + current["max_output_tokens"] * 2700
        if extra + Decimal(next_nano) / 1_000_000_000 > self.max_cny:
            self.local_stop_reason = "ADDITIONAL_RESERVATION_LIMIT"
            raise QwenClientError("LIVE_ADDITIONAL_RESERVATION_LIMIT")
        return self.budget.reserve(prompt_chars)

    def record_tokens(self, usage):
        return self.budget.record_tokens(usage)

    def stop(self, reason):
        return self.budget.stop(reason)


def summarize(rows: list[dict], stage: str, *, live: bool) -> dict:
    groups = {}
    for arm, _, _ in ARMS[stage]:
        for track in ("controlled", "natural"):
            selected = [r for r in rows if r["arm"] == arm and r["track"] == track]
            if not selected:
                continue
            completed = [r for r in selected if r["status"] == "completed"]
            correct = sum(r["grade"]["correct_task"] for r in completed)
            groups[arm + ":" + track] = {
                "scheduled": len(selected), "completed": len(completed),
                "not_run": sum(r["status"] == "not_run" for r in selected),
                "operational_errors": sum(r["status"] == "operational_error"
                                          or r.get("grade", {}).get("operational_failure", False) for r in selected),
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
                "consumed_escalated_feedbacks": sum(r.get("consumed_escalated_feedbacks", 0) for r in selected),
                "full_evidence_feedbacks": sum(r.get("full_evidence_feedbacks", 0) for r in selected),
                "consumed_full_evidence_feedbacks": sum(r.get("consumed_full_evidence_feedbacks", 0) for r in selected),
                "diagnostic_queries": sum(r.get("diagnostic_queries", 0) for r in selected),
                "regression_queries": sum(r.get("regression_queries", 0) for r in selected),
                "elapsed_seconds": round(sum(r.get("seconds", 0) for r in selected), 6),
                "usage": _usage([e for r in selected for e in r.get("http_events", [])]),
            }
    return {"scheduled_cells": len(rows), "groups": groups,
            "interpretation": "real-model development evaluation" if live else
            "scripted interface/regression checks only; not model repair evidence"}


def paired_results(rows: list[dict], stage: str) -> list[dict]:
    pairs = []
    groups = list(dict.fromkeys((r["case_id"], r["track"], r["seed"]) for r in rows))
    for case_id, track, seed in groups:
        selected = [r for r in rows if (r["case_id"], r["track"], r["seed"]) == (case_id, track, seed)]
        pairs.append({"case_id": case_id, "track": track, "seed": seed,
                      "all_arms_completed": all(r["status"] == "completed" for r in selected),
                      "arms": {r["arm"]: {"status": r["status"],
                                             "correct_task": r.get("grade", {}).get("correct_task", False),
                                             "http_attempts": r["http_attempts"],
                                             "usage": _usage(r.get("http_events", [])),
                                             "escalated_feedbacks": r.get("escalated_feedbacks", 0)}
                               for r in selected}})
    return pairs


def feedback_selection_events(record: dict) -> list[dict]:
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
            real_ordinal = sum(origin == "real_qwen_http" for origin in origins[:next_index + 1])
            real_next = (forwarded and next_index < len(origins)
                         and origins[next_index] == "real_qwen_http" and posted >= real_ordinal)
        else:
            real_next = forwarded and next_index < posted
        failed_full = feedback.get("status") != "verified" and strategy.get("selected_mode") == "full"
        selections.append({**deepcopy(strategy), "after_action_index": action_index,
                           "new_escalation_transition": bool(failed_full and previous_mode != "full"),
                           "failed_full_feedback": bool(failed_full),
                           "full_evidence_emitted": bool(failed_full and strategy.get("evidence_available")
                                                         and feedback.get("diagnostic_evidence")),
                           "sent_in_later_native_request": bool(forwarded),
                           "sent_in_later_http_attempt": bool(real_next)})
        previous_mode = strategy.get("selected_mode", previous_mode)
    return selections


async def run_feedback_experiment(
    output: Path, *, stage: str = "ablation", live: bool = False,
    config: QwenConfig | None = None, budget: MonetaryRequestBudget | None = None,
    controlled_case_ids: tuple[str, ...] = ("d06", "d10", "d13"),
    natural_case_ids: tuple[str, ...] = ("d01", "d03", "d04"),
    arm_ids: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (53,), max_attempts: int = 3,
    max_extra_requests: int = 24, extra_reservation_cny: float = .8,
    money_limit_cny: float = 5, client_factory=None,
) -> dict:
    """Freeze a crossed plan, then run serialized paired native Agent trials.

    client_factory(case, arm, seed, track, budget_wrapper) is offline-test-only.
    Natural tasks use the first seed; seed replicates are not new task templates.
    A shared budget is never replaced, replenished or refunded by this function.
    """
    if stage not in ARMS:
        raise ValueError("Stage must be ablation or strategy")
    known_arms = {arm for arm, _, _ in ARMS[stage]}
    if arm_ids is not None and (not arm_ids or len(set(arm_ids)) != len(arm_ids)
                               or any(arm not in known_arms for arm in arm_ids)):
        raise ValueError("Select distinct arms registered for this stage")
    selected_arms = tuple(item for item in ARMS[stage] if arm_ids is None or item[0] in arm_ids)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("Select 1..3 native attempts")
    if not seeds or len(seeds) > 3 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Select 1..3 distinct integer seeds")
    if live and (config is None or client_factory is not None):
        raise ValueError("Live mode requires explicit config and forbids a test client factory")
    if live and budget is None:
        raise ValueError("Live mode requires a supplied complete cumulative budget")
    controlled = tuple(get_case(name) for name in controlled_case_ids)
    natural = tuple(get_case(name) for name in natural_case_ids)
    if not controlled and not natural:
        raise ValueError("Select at least one task")
    if len(set(controlled_case_ids)) != len(controlled) or len(set(natural_case_ids)) != len(natural):
        raise ValueError("Select distinct case IDs within each track")
    pricing = validate_pricing(config) if live else None
    budget = budget if budget is not None else MonetaryRequestBudget(
        limit=max_extra_requests, money_limit_cny=money_limit_cny, max_output_tokens=900)
    if not isinstance(budget, MonetaryRequestBudget) or budget.max_output_tokens != 900:
        raise ValueError("A 900-token monetary budget is required")
    if live and (not budget.snapshot()["reported_cost_complete"] or budget.snapshot()["stopped_reason"]):
        raise ValueError("Live continuation requires a complete unstopped ledger")
    allowance = IncrementalAllowance(budget, max_extra_requests, extra_reservation_cny)
    start_budget = deepcopy(budget.snapshot())
    output = Path(output).resolve()
    if not output.is_relative_to((PROJECT_ROOT / "reports").resolve()):
        raise ValueError("Experiment output must remain inside project reports")
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "runtime").mkdir()
    rows = []
    for track, cases, trial_seeds in (("controlled", controlled, seeds), ("natural", natural, (seeds[0],))):
        for case in cases:
            for seed in trial_seeds:
                for arm, mode, policy in selected_arms:
                    rows.append({"case_id": case.case_id, "task_id": case.task.task_id,
                                 "track": track, "seed": seed, "arm": arm, "mode": mode,
                                 "feedback_policy": policy, "status": "not_run", "stop_reason": "NOT_STARTED",
                                 "http_attempts": 0, "http_events": [], "native_turns": 0})
    cases_by_id = {case.case_id: case for case in controlled + natural}
    hashes = source_fingerprints()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
        "split": "disclosed_development_not_unseen_holdout", "real_model_used": live,
        "requested_model": config.model if live else "offline-scripted",
        "upstream_commit": UPSTREAM_COMMIT, "pricing": pricing,
        "seeds": list(seeds), "max_native_attempts": max_attempts,
        "per_request_max_tokens": 900, "arms": selected_arms, "budget_start": start_budget,
        "continuation_review": deepcopy(getattr(budget, "_continuation_review", [])),
        "ablation_factors": {"capability_description": "Additional full execution_contract; every arm retains the same basic supported_sql description.",
                             "classified_guidance": "Classified repair_guidance added to failed Action feedback."},
        "max_extra_requests": max_extra_requests, "extra_reservation_cny": extra_reservation_cny,
        "cases": [asdict(case) for case in cases_by_id.values()],
        "boundary_fixtures": {key: asdict(boundary_fixture(case)) for key, case in cases_by_id.items()},
        "planned_grid": deepcopy(rows), "code_sha256": hashes,
        "ordering": "Arm order rotates by paired task/track/seed; no result-based selection.",
        "offline_script": "Controlled: injected bad, repeat bad, evaluator reference; natural: evaluator reference. Interface checks only.",
        "note": "Evaluator labels, references and expected rows do not enter live model context. Controlled first native turns are injected and not billed model outputs.",
    }
    _write(output / "frozen_manifest.json", manifest)
    report = {"status": "running", "stage": stage, "split": manifest["split"],
              "created_at_utc": manifest["created_at_utc"], "real_model_used": live,
              "manifest_sha256": _digest(manifest), "rows": rows, "budget_start": start_budget,
              "continuation_review": deepcopy(manifest["continuation_review"]),
              "max_extra_requests": max_extra_requests, "extra_reservation_cny": extra_reservation_cny,
              "limitation": "Known fixed-schema development tasks; repetitions and converted regressions are not independent unseen tasks. Offline results are scripted interface checks, never model effectiveness."}
    all_events = []

    def checkpoint(status="running"):
        end = deepcopy(budget.snapshot())
        report.update(status=status, budget_end=end, summary=summarize(rows, stage, live=live),
                      pairs=paired_results(rows, stage), usage=_usage(all_events),
                      this_run_http_attempts=end["http_attempts"] - start_budget["http_attempts"],
                      this_run_reserved_upper_cny=float(Decimal(str(end["reserved_upper_cny"])) - Decimal(str(start_budget["reserved_upper_cny"]))),
                      audit_event_count=len(all_events), local_allowance_stop=allowance.stop_reason(),
                      source_integrity_ok=source_fingerprints() == hashes)
        report["audit_ledger_complete"] = len(all_events) == report["this_run_http_attempts"]
        _write(output / "summary.json", report)
        _write(output / "request_audit.json", all_events)

    checkpoint()
    pairs = list(dict.fromkeys((r["case_id"], r["track"], r["seed"]) for r in rows))
    index = {(r["case_id"], r["track"], r["seed"], r["arm"]): r for r in rows}
    try:
        for position, (case_id, track, seed) in enumerate(pairs):
            case = cases_by_id[case_id]
            arms = selected_arms
            order = arms[position % len(arms):] + arms[:position % len(arms)]
            for arm, mode, policy in order:
                row = index[(case_id, track, seed, arm)]
                if source_fingerprints() != hashes:
                    row["stop_reason"] = "SOURCE_CHANGED_DURING_EXPERIMENT"
                    checkpoint()
                    continue
                reason = allowance.stop_reason() if live else None
                if reason:
                    row["stop_reason"] = reason
                    checkpoint()
                    continue
                started = time.perf_counter()
                before_requests = budget.snapshot()["http_attempts"]
                source_hash = _digest(asdict(case.source))
                try:
                    kwargs = {}
                    if client_factory is not None:
                        kwargs["llm_client"] = client_factory(case, arm, seed, track, allowance)
                    elif live:
                        client = QwenLLMClient(config, allowance, max_tokens=900)
                        kwargs["llm_client"] = InjectedFirstCandidateClient(
                            client, probe_sql_candidate(case.candidate_sql, case.task.task_id)) if track == "controlled" else client
                    else:
                        bad = probe_sql_candidate(case.candidate_sql, case.task.task_id)
                        good = probe_sql_candidate(case.reference_sql, case.task.task_id)
                        kwargs["responses"] = [bad, bad, good][:max_attempts] if track == "controlled" else [good] * max_attempts
                    record = await run_probe_native_case(
                        case.task, case.source, mode=mode, feedback_policy=policy, seed=seed,
                        max_attempts=max_attempts, output_dir=output / "runtime" / f"{case_id}_{track}_{seed}_{arm}",
                        **kwargs)
                    if _digest(asdict(case.source)) != source_hash:
                        raise RuntimeError("SOURCE_FIXTURE_CHANGED")
                    events = deepcopy(record.get("http_events", []))
                    all_events.extend(events)
                    grade = grade_probe_record(case, record)
                    delta = budget.snapshot()["http_attempts"] - before_requests
                    complete = len(events) == delta
                    if not complete:
                        budget.stop("AUDIT_LEDGER_INCOMPLETE")
                    if not complete or (events and not _usage(events)["usage_complete"]):
                        grade.update(operational_failure=True, correct_task=False)
                    actions = record.get("actions", [])
                    selections = feedback_selection_events(record)
                    origins = record.get("turn_origins", [])
                    row.update(status="completed", stop_reason=None, grade=grade,
                               http_attempts=delta, http_events=events, audit_ledger_complete=complete,
                               source_sha256=source_hash, source_unchanged=True,
                               native_turns=record.get("llm_calls", 0),
                               injected_initial_turns=sum(v == "injected_test_fault_no_api_call" for v in origins),
                               offline_scripted=not live,
                               feedback_selections=selections,
                               escalation_transitions=sum(s["new_escalation_transition"] for s in selections),
                               escalated_feedbacks=sum(s["failed_full_feedback"] for s in selections),
                               consumed_escalated_feedbacks=sum(s["failed_full_feedback"] and s["sent_in_later_http_attempt"] for s in selections),
                               full_evidence_feedbacks=sum(s["full_evidence_emitted"] for s in selections),
                               consumed_full_evidence_feedbacks=sum(s["full_evidence_emitted"] and s["sent_in_later_http_attempt"] for s in selections),
                               diagnostic_queries=sum((a.get("probe_report") or {}).get("queries", 0) for a in actions),
                               regression_queries=sum(a.get("regression_query_attempts", 0) for a in actions),
                               publication=deepcopy(record.get("publication")))
                    _write(output / "traces" / f"{case_id}_{track}_{seed}_{arm}.json",
                           {"cell": {k: v for k, v in row.items() if k != "http_events"}, "record": record})
                except Exception as error:
                    delta = budget.snapshot()["http_attempts"] - before_requests
                    if delta:
                        budget.stop("RUNNER_ERROR_USAGE_OR_LEDGER_UNKNOWN")
                    row.update(status="operational_error", stop_reason="RUNNER_" + type(error).__name__,
                               http_attempts=delta, audit_ledger_complete=delta == 0)
                    _write(output / "traces" / f"{case_id}_{track}_{seed}_{arm}.json", {"cell": row})
                finally:
                    row["seconds"] = round(time.perf_counter() - started, 6)
                    checkpoint()
    except BaseException:
        checkpoint("interrupted")
        raise
    report["all_cells_completed"] = all(r["status"] == "completed" for r in rows)
    checkpoint("completed" if report["all_cells_completed"] and report["source_integrity_ok"] else "incomplete")
    return report
