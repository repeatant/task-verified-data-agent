"""Explicitly authorized larger campaign, without resetting earlier spending.

Four independent serialized request lanes allocate at most CNY 52 in total.
The prior CNY 8 allocation remains reserved and is linked by immutable digests.
No refund of old reservations, hidden retry or environment access occurs here.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from dataverify.agent.graph_checkpoint import JournaledBudget, exclusive_run, read_budget
from dataverify.integrations.qwen_config import QwenClientError
from dataverify.runtime.discovery_budget import reviewed_truncation
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget
from dataverify.runtime.reporting import digest_json, write_json


class CampaignBudget(MonetaryRequestBudget):
    _allowance_ceiling = Decimal("52")
    _output_ceiling = 1800


def authorize_campaign(output: Path, prior_paths: list[Path], *, workers=4):
    """One allocation, bound to the user's explicit cumulative CNY 60 approval."""
    path = output / "authorization.json"
    prior_paths = [p.resolve() for p in prior_paths]
    if len(set(prior_paths)) != 2 or type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError("CAMPAIGN_PRIOR_ALLOWANCE_OR_WORKERS")
    prior = [read_budget(p) for p in prior_paths]
    if any(not p["reported_cost_complete"] or p["stopped_reason"] for p in prior):
        raise ValueError("CAMPAIGN_UNRESOLVED_PRIOR_USAGE")
    if [p["money_limit_cny"] for p in prior] != [5, 3]:
        raise ValueError("CAMPAIGN_PRIOR_ALLOWANCE_OR_WORKERS")
    receipt = {
        "format": "expanded-campaign-authorization-v1",
        "cumulative_limit_cny": 60,
        "prior_allowance_cny": 8,
        "new_allowance_cny": 52,
        "workers": workers,
        "user_authorization": "允许累计上限 ¥60，完成完整实验",
        "date": "2026-11-07",
        "prior": [
            {"path": str(p.resolve()), "digest": digest_json(s), "snapshot": s}
            for p, s in zip(prior_paths, prior)
        ],
    }
    # Prevent allocating the same explicit top-up to two output directories.
    marker = prior_paths[-1].with_suffix(".expanded-allocation.json")
    allocation = {"output": str(output.resolve()), "receipt": receipt}
    with exclusive_run(marker):
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != allocation:
            raise ValueError("CAMPAIGN_ALLOWANCE_ALREADY_ALLOCATED_OR_CHANGED")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("CAMPAIGN_AUTHORIZATION_CHANGED")
        output.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            write_json(marker, allocation)
        if not path.exists():
            write_json(path, receipt)
    return receipt


def _validate_receipt(output, receipt):
    paths = [Path(item["path"]) for item in receipt["prior"]]
    checked = authorize_campaign(output, paths, workers=receipt["workers"])
    if checked != receipt:
        raise ValueError("CAMPAIGN_AUTHORIZATION_CHANGED")


def _new_budget(receipt):
    return CampaignBudget(
        limit=5000,
        max_input_chars=30_000_000,
        max_reported_tokens=15_000_000,
        money_limit_cny=52 // receipt["workers"],
        max_output_tokens=1800,
    )


def _replay(old, receipt):
    budget = _new_budget(receipt)
    if not old["reported_cost_complete"] or old["stopped_reason"]:
        raise ValueError("CAMPAIGN_LANE_STOPPED")
    for item in old["reservation_ledger"]:
        budget.reserve(item["prompt_chars"])
        budget.record_tokens(
            {key: item[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        )
    if budget.snapshot() != old:
        raise ValueError("CAMPAIGN_LEDGER_REPLAY_MISMATCH")
    return budget


def _lane_path(output, lane):
    base = output / f"budget-lane-{lane}.json"
    pointer = output / f"budget-lane-{lane}.current.json"
    if not pointer.exists():
        return base, []
    saved = json.loads(pointer.read_text(encoding="utf-8"))
    history = saved.get("history", [])
    if (
        not history
        or saved.get("current") != f"budget-lane-{lane}.continuation-{len(history)}.json"
    ):
        raise ValueError("CAMPAIGN_CONTINUATION_POINTER")
    prior = base
    for index, entry in enumerate(history, 1):
        incident_path = output / f"budget-lane-{lane}.incident-{index}.json"
        incident = json.loads(incident_path.read_text(encoding="utf-8"))
        if entry != {"incident": incident_path.name, "digest": digest_json(incident)}:
            raise ValueError("CAMPAIGN_CONTINUATION_HISTORY_CHANGED")
        snapshot = read_budget(prior)
        if digest_json(snapshot) != incident["prior_digest"] or incident["prior"] != prior.name:
            raise ValueError("CAMPAIGN_CONTINUATION_PARENT_CHANGED")
        normalized = reviewed_truncation(snapshot, incident["report"])
        following = output / f"budget-lane-{lane}.continuation-{index}.json"
        later = read_budget(following)
        prefix = normalized["reservation_ledger"]
        if later["reservation_ledger"][: len(prefix)] != prefix:
            raise ValueError("CAMPAIGN_CONTINUATION_HISTORY_RESET")
        for key in (
            "money_limit_cny",
            "http_attempt_limit",
            "input_char_limit",
            "reported_token_stop_threshold",
            "max_output_tokens",
        ):
            if later[key] != normalized[key]:
                raise ValueError("CAMPAIGN_CONTINUATION_ALLOWANCE_CHANGED")
        prior = following
    return prior, history


def lane_budget(output: Path, lane: int, receipt: dict):
    """Caller must hold exclusive_run for this lane's entire paid session."""
    _validate_receipt(output, receipt)
    workers = receipt["workers"]
    if type(lane) is not int or not 0 <= lane < workers:
        raise ValueError("CAMPAIGN_LANE")
    path, _ = _lane_path(output, lane)
    budget = _replay(read_budget(path), receipt) if path.exists() else _new_budget(receipt)
    return JournaledBudget(budget, path)


def _immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("CAMPAIGN_IMMUTABLE_RECORD_CHANGED")
        return
    write_json(path, value)


def continue_accounted_truncation(output: Path, lane: int, receipt: dict, budget, report: dict):
    """Continue NEXT task with full history after accounted truncation, never retry.

    Unknown outcomes and other provider errors remain stopped. Required report
    fields: budget_start, budget_end, http_events in chronological attempt order.
    The caller holds the lane lease throughout continuation and later calls.
    """
    _validate_receipt(output, receipt)
    path, history = _lane_path(output, lane)
    if budget.path != path.resolve() or read_budget(path) != budget.snapshot():
        raise ValueError("CAMPAIGN_CONTINUATION_JOURNAL_MISMATCH")
    old = budget.snapshot()
    normalized = reviewed_truncation(old, report)
    rebuilt = _replay(normalized, receipt)
    index = len(history) + 1
    incident_path = output / f"budget-lane-{lane}.incident-{index}.json"
    safe_report = {key: report[key] for key in ("budget_start", "budget_end", "http_events")}
    incident = {
        "prior": path.name,
        "prior_digest": digest_json(old),
        "report": safe_report,
        "action": "continue_next_task_without_retry_or_refund",
    }
    _immutable_json(incident_path, incident)
    next_path = output / f"budget-lane-{lane}.continuation-{index}.json"
    result = JournaledBudget(rebuilt, next_path)
    history += [{"incident": incident_path.name, "digest": digest_json(incident)}]
    pointer = output / f"budget-lane-{lane}.current.json"
    partial = pointer.with_suffix(".partial.json")
    write_json(partial, {"current": next_path.name, "history": history})
    partial.replace(pointer)
    return result


class CellBudget:
    """All roles of a trajectory share a cap; the durable lane owns accounting."""

    def __init__(self, lane, limit=12):
        if type(limit) is not int or limit < 1:
            raise ValueError("CAMPAIGN_CELL_REQUEST_LIMIT")
        self.lane, self.limit = lane, limit
        self.start = lane.snapshot()["http_attempts"]

    def reserve(self, chars):
        if self.snapshot()["http_attempts"] - self.start >= self.limit:
            raise QwenClientError("CELL_HTTP_REQUEST_LIMIT")
        return self.lane.reserve(chars)

    def record_tokens(self, usage):
        self.lane.record_tokens(usage)

    def stop(self, reason):
        self.lane.stop(reason)

    def snapshot(self):
        return self.lane.snapshot()
