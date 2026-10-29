"""Continue audited experiment spending without resetting the monetary ledger.

This module owns continuation validation and the allowance for additional calls.
It performs no model requests and never reads credentials. The experiment runner
passes its project root explicitly so paths can be checked at the boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

from .dbgpt_qwen import QwenClientError
from .probe_cost_budget import MonetaryRequestBudget

JsonObject = dict[str, Any]
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_RESTORED_FIELDS = (
    "http_attempts",
    "input_chars",
    "reported_total_tokens",
    "complete_count",
    "reported_prompt_tokens",
    "reported_completion_tokens",
    "reserved_upper_cny",
    "estimated_reported_cny",
    "max_output_tokens",
    "money_limit_cny",
    "reservation_ledger",
)


def restore_budget(
    snapshot: JsonObject, *, max_extra_requests: int = 24, money_limit_cny: float = 5
) -> MonetaryRequestBudget:
    """Replay every paid attempt and increase only the technical request cap."""
    if type(max_extra_requests) is not int or not 1 <= max_extra_requests <= 36:
        raise ValueError("Select 1..36 additional HTTP attempts")
    if (
        snapshot.get("reported_cost_complete") is not True
        or snapshot.get("stopped_reason")
        or snapshot.get("unknown_count") != 0
        or snapshot.get("in_flight_count") != 0
        or snapshot.get("unmatched_usage_events") != 0
    ):
        raise ValueError("Only complete, unstopped ledgers can be continued")
    prices = {
        "model": "qwen3.8-flash",
        "region": "cn-beijing",
        "currency": "CNY",
        "input_cny_per_million_tokens": 0.8,
        "output_cny_per_million_tokens": 2.7,
    }
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
        money_limit_cny=money_limit_cny,
        max_output_tokens=snapshot["max_output_tokens"],
    )
    # 逐条重放，而不是只复制总额；重放后每个收费字段必须与原账本相同。
    _replay_reservations(snapshot, budget)
    restored = budget.snapshot()
    if any(restored[key] != snapshot.get(key) for key in _RESTORED_FIELDS):
        raise ValueError("Prior ledger cannot be reproduced exactly")
    return budget


def _replay_reservations(snapshot: JsonObject, budget: MonetaryRequestBudget) -> None:
    for entry in snapshot.get("reservation_ledger", []):
        if entry.get("status") != "usage_complete":
            raise ValueError("Prior reservation has incomplete usage")
        if budget.reserve(entry["prompt_chars"]) != entry["attempt"]:
            raise ValueError("Prior ledger attempts are not sequential")
        budget.record_tokens({name: entry[name] for name in _TOKEN_FIELDS})


def _review_snapshots(document: JsonObject) -> tuple[JsonObject, JsonObject, int, int]:
    """Validate report-level integrity before inspecting individual HTTP events."""
    if any(
        document.get(key) is not True
        for key in ("real_model_used", "source_integrity_ok", "audit_ledger_complete")
    ):
        raise ValueError("Truncation continuation requires complete source and audit integrity")
    start, end = document.get("budget_start"), document.get("budget_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ValueError("Truncation continuation requires both budget snapshots")
    if (
        end.get("stopped_reason") != "PROVIDER_ERROR"
        or end.get("reported_cost_complete") is not True
        or end.get("unknown_count") != 0
        or end.get("in_flight_count") != 0
        or end.get("unmatched_usage_events") != 0
        or end.get("max_output_tokens") != 900
    ):
        raise ValueError("Only fully accounted provider truncation can be reviewed")
    if (
        start.get("reported_cost_complete") is not True
        or start.get("stopped_reason")
        or start.get("unknown_count") != 0
        or start.get("in_flight_count") != 0
        or start.get("unmatched_usage_events") != 0
    ):
        raise ValueError("Initial budget was not a complete unstopped ledger")
    begin, finish = start.get("http_attempts"), end.get("http_attempts")
    if (
        type(begin) is not int
        or type(finish) is not int
        or not 0 <= begin < finish
        or document.get("this_run_http_attempts") != finish - begin
    ):
        raise ValueError("Truncation report attempt counts disagree")
    return start, end, begin, finish


def _review_events(document: JsonObject, begin: int, finish: int) -> list[JsonObject]:
    """Require each new attempt to appear exactly once in the per-cell audit."""
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
    if (
        len(events) != finish - begin
        or len(attempts) != len(events)
        or any(type(attempt) is not int for attempt in attempts)
        or sorted(attempts) != list(range(begin + 1, finish + 1))
        or document.get("audit_event_count") != len(events)
    ):
        raise ValueError("New HTTP events do not cover the ledger exactly")
    return events


def _review_ledger(start: JsonObject, end: JsonObject, begin: int, finish: int) -> list[JsonObject]:
    ledger = end.get("reservation_ledger")
    if (
        not isinstance(ledger, list)
        or len(ledger) != finish
        or ledger[:begin] != start.get("reservation_ledger")
    ):
        raise ValueError("Historical reservations changed or are missing")
    return ledger


def _review_event_usage(
    events: list[JsonObject], ledger: list[JsonObject], finish: int
) -> dict[str, int]:
    """Match usage to paid reservations and allow only the final truncation."""
    failures = 0
    totals = {name: 0 for name in _TOKEN_FIELDS}
    for event in events:
        usage = event.get("usage")
        if (
            event.get("usage_complete") is not True
            or event.get("http_status") != 200
            or not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in totals)
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
        ):
            raise ValueError("Truncation audit contains incomplete HTTP usage")
        entry = ledger[event["attempt"] - 1]
        if (
            entry.get("attempt") != event["attempt"]
            or entry.get("status") != "usage_complete"
            or any(entry.get(key) != usage[key] for key in totals)
        ):
            raise ValueError("HTTP event usage differs from the monetary ledger")
        for key in totals:
            totals[key] += usage[key]
        if event.get("status") == "completed":
            if event.get("error_code"):
                raise ValueError("Completed HTTP event contains an error")
        elif (
            event.get("status") == "failed"
            and event.get("error_code") == "QWEN_RESPONSE_TRUNCATED"
            and event["attempt"] == finish
            and usage["completion_tokens"] == 900
        ):
            failures += 1
        else:
            raise ValueError("Only a final accounted HTTP-200 truncation can be resumed")
    if failures != 1:
        raise ValueError("Exactly one final truncation must explain the provider stop")
    return totals


def review_accounted_truncation(
    document: JsonObject, path: Path, raw: bytes, *, project_root: Path
) -> JsonObject:
    """Approve only a completely accounted final HTTP-200 truncated response.

    This narrow exception grants no refund or extra output allowance. Transport
    errors, missing usage and unrelated provider errors remain non-resumable.
    """
    start, end, begin, finish = _review_snapshots(document)
    events = _review_events(document, begin, finish)
    ledger = _review_ledger(start, end, begin, finish)
    totals = _review_event_usage(events, ledger, finish)
    for token_field, snapshot_field in (
        ("prompt_tokens", "reported_prompt_tokens"),
        ("completion_tokens", "reported_completion_tokens"),
        ("total_tokens", "reported_total_tokens"),
    ):
        if end.get(snapshot_field, -1) - start.get(snapshot_field, -1) != totals[token_field]:
            raise ValueError("Report usage totals disagree with the cumulative ledger")
    return {
        "source_report": path.relative_to(project_root).as_posix(),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "original_stop_reason": "PROVIDER_ERROR",
        "review_reason": "Explicit continuation after one final HTTP-200 QWEN_RESPONSE_TRUNCATED response with complete matching usage and unchanged source/audit records.",
        "reviewed_failed_attempt": finish,
        "cumulative_http_attempts_preserved": finish,
        "cumulative_reported_tokens_preserved": end["reported_total_tokens"],
        "cumulative_reserved_upper_cny_preserved": end["reserved_upper_cny"],
        "refund_applied": False,
        "max_output_tokens_preserved": 900,
        "original_failed_cell_retained": True,
    }


def load_prior_budget(
    path: Path,
    *,
    project_root: Path,
    max_extra_requests: int = 24,
    money_limit_cny: float = 5,
    allow_truncated_resume: bool = False,
) -> MonetaryRequestBudget:
    """Read and validate one local report, preserving its continuation lineage."""
    if type(allow_truncated_resume) is not bool:
        raise ValueError("Truncation continuation requires an explicit Boolean choice")
    path = Path(path).resolve()
    if not path.is_relative_to((project_root / "reports").resolve()):
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
        review = review_accounted_truncation(document, path, raw, project_root=project_root)
        snapshot = deepcopy(snapshot)
        snapshot["stopped_reason"] = None
        reviews.append(review)
    budget = restore_budget(
        snapshot, max_extra_requests=max_extra_requests, money_limit_cny=money_limit_cny
    )
    budget._continuation_review = reviews
    return budget


class IncrementalAllowance:
    """Reserve additional calls before delegating to the cumulative ledger."""

    def __init__(self, budget: MonetaryRequestBudget, max_requests: int, max_cny: float) -> None:
        if type(max_requests) is not int or not 1 <= max_requests <= 36:
            raise ValueError("Select 1..36 additional HTTP attempts")
        if (
            type(max_cny) not in (int, float)
            or not math.isfinite(max_cny)
            or not Decimal(0) < Decimal(str(max_cny)) <= Decimal("0.8")
        ):
            raise ValueError("Additional reservation must be positive and at most CNY 0.8")
        self.budget = budget
        self.start = budget.snapshot()
        self.max_requests = max_requests
        self.max_cny = Decimal(str(max_cny))
        self.local_stop_reason: str | None = None

    def stop_reason(self) -> str | None:
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

    def reserve(self, prompt_chars: int) -> int:
        if type(prompt_chars) is not int or prompt_chars < 0:
            raise ValueError("prompt_chars must be a nonnegative integer")
        reason = self.stop_reason()
        if reason:
            raise QwenClientError("LIVE_" + reason)
        current = self.budget.snapshot()
        extra = Decimal(str(current["reserved_upper_cny"])) - Decimal(
            str(self.start["reserved_upper_cny"])
        )
        next_nano = (4 * prompt_chars + 4096) * 800 + current["max_output_tokens"] * 2700
        if extra + Decimal(next_nano) / 1_000_000_000 > self.max_cny:
            self.local_stop_reason = "ADDITIONAL_RESERVATION_LIMIT"
            raise QwenClientError("LIVE_ADDITIONAL_RESERVATION_LIMIT")
        return self.budget.reserve(prompt_chars)

    def record_tokens(self, usage: JsonObject) -> None:
        self.budget.record_tokens(usage)

    def stop(self, reason: str) -> None:
        self.budget.stop(reason)
