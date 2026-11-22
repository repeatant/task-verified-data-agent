"""Budget-only continuation of a frozen experiment; dry-run is the default.

No frozen runtime file is changed. Completed quality outcomes are never rerun.
Only explicit, fully accounted money-reservation interruptions are eligible for
an infrastructure retry, with the prior record retained and linked. Execute
only after the original campaign exits and the user approves cumulative CNY 120.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.discovery_model import QwenReviewModel, plan_prompt, plan_validation_feedback
from dataverify.agent.graph_checkpoint import JournaledBudget, exclusive_run, read_budget
from dataverify.agent.graph_model import GraphModelError, QwenGraphModel, parse_graph_action
from dataverify.evaluation.expanded_experiment import (
    ABLATIONS,
    BASE_METHODS,
    SEED,
    SQL_PROMPT,
    _events,
    controlled_bundles,
    message,
    prepare_first,
    public_bundles,
    run_cell,
    source_hashes,
)
from dataverify.evaluation.expanded_scoring import aggregate_runs
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import CampaignBudget, CellBudget, _lane_path
from dataverify.runtime.discovery_budget import reviewed_truncation
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.discovery_runtime import validate_plan

MONEY_STOP = "MONEY_RESERVATION_LIMIT"
MONEY_CODES = {"LIVE_MONEY_RESERVATION_LIMIT", "LIVE_BUDGET_STOPPED_MONEY_RESERVATION_LIMIT"}
TRANSPORT_CODES = {"GRAPH_QWEN_TRANSPORT_READTIMEOUT", "GRAPH_QWEN_TRANSPORT_CONNECTERROR"}


class ExtendedCampaignBudget(CampaignBudget):
    _allowance_ceiling = Decimal("112")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("CONTINUATION_IMMUTABLE_RECORD_CHANGED")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def replay(snapshot, *, extend=False, quarantined=()):
    """Replay every historical charge; only an explicit money stop may clear."""
    allowed = (
        {None, MONEY_STOP, "TRANSPORT_ERROR"}
        if extend and quarantined
        else ({None, MONEY_STOP} if extend else {None})
    )
    unknown = [
        item["attempt"]
        for item in snapshot["reservation_ledger"]
        if item["status"] == "usage_unknown"
    ]
    if (
        snapshot.get("stopped_reason") not in allowed
        or not set(unknown).issubset(quarantined)
        or snapshot.get("in_flight_count")
        or snapshot.get("unmatched_usage_events")
        or snapshot["unknown_count"] != len(unknown)
    ):
        raise ValueError("CONTINUATION_UNRESOLVED_OR_NON_MONEY_FAILURE")
    if snapshot["money_limit_cny"] != (13 if extend else 28):
        raise ValueError("CONTINUATION_UNEXPECTED_LANE_ALLOWANCE")
    budget = ExtendedCampaignBudget(
        limit=5000,
        max_input_chars=30_000_000,
        max_reported_tokens=15_000_000,
        money_limit_cny=28,
        max_output_tokens=1800,
    )
    for item in snapshot["reservation_ledger"]:
        budget.reserve(item["prompt_chars"])
        if item["status"] == "usage_unknown":
            if item.get("stop_reason") != "TRANSPORT_ERROR":
                raise ValueError("QUARANTINE_ONLY_REVIEWED_TRANSPORT_FAILURES")
            budget.stop(item["stop_reason"])
            # Authorized quarantine retains this ledger entry and its complete
            # conservative reservation; only permits a different later task.
            budget.stopped_reason = None
        else:
            budget.record_tokens(
                {key: item[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
            )
    expected = deepcopy(snapshot)
    expected.update(money_limit_cny=28, stopped_reason=None)
    if budget.snapshot() != expected:
        raise ValueError("CONTINUATION_LEDGER_REPLAY_MISMATCH")
    return budget


def quarantine_evidence(base_run, lane, snapshot):
    """Match each old unknown to an immutable, reviewed transport failure event."""
    unknown = [item for item in snapshot["reservation_ledger"] if item["status"] == "usage_unknown"]
    if not unknown:
        return []
    evidence = []
    for item in unknown:
        matches = []
        for folder in (base_run / "records", base_run / "initial"):
            for path in folder.glob("*.json"):
                record = read(path)
                for event in record.get("http_events", []):
                    prefix = event.get("event_id", "").split(":")[0].split(".")[0]
                    if prefix == f"budget-lane-{lane}" and event.get("attempt") == item["attempt"]:
                        matches.append((path, record, event))
        if len(matches) != 1:
            raise ValueError("QUARANTINE_REQUIRES_UNIQUE_ORIGINAL_EVENT")
        path, record, event = matches[0]
        if (
            event.get("error_code") not in TRANSPORT_CODES
            or event.get("status") != "failed"
            or event.get("usage_complete") is not False
            or any(value is not None for value in event.get("usage", {}).values())
        ):
            raise ValueError("QUARANTINE_EVENT_NOT_REVIEWED_TRANSPORT_FAILURE")
        evidence.append(
            {
                "attempt": item["attempt"],
                "reserved_upper_cny": item["reserved_upper_cny"],
                "source": str(path.resolve()),
                "source_digest": digest_json(record),
                "event": event,
            }
        )
    return evidence


def allocate_extension(campaign, base_run, *, approved_limit):
    """Caller holds all four original lane leases and the allocation lease."""
    if approved_limit != 120:
        raise ValueError("EXPLICIT_CUMULATIVE_CNY120_APPROVAL_REQUIRED")
    old_authorization = read(campaign / "authorization.json")
    if old_authorization["cumulative_limit_cny"] != 60 or old_authorization["workers"] != 4:
        raise ValueError("CONTINUATION_PRIOR_AUTHORIZATION")
    for old in old_authorization["prior"]:
        if read_budget(Path(old["path"])) != old["snapshot"]:
            raise ValueError("CONTINUATION_EARLIER_ACCOUNTING_CHANGED")
    directory = campaign / "budget120"
    receipt_path = directory / "authorization.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        marker = campaign / "budget120-allocation.json"
        if (
            receipt.get("cumulative_limit_cny") != 120
            or receipt.get("per_lane_limit_cny") != 28
            or receipt.get("workers") != 4
            or receipt.get("old_authorization_digest") != digest_json(old_authorization)
            or receipt.get("frozen_source_hashes")
            != read(base_run / "protocol.json")["source_hashes"]
            or read(marker) != {"directory": str(directory.resolve()), "receipt": receipt}
        ):
            raise ValueError("CONTINUATION_AUTHORIZATION_CHANGED")
        for parent in receipt["parents"]:
            if read_budget(Path(parent["path"])) != parent["snapshot"]:
                raise ValueError("RETIRED_ORIGINAL_LANE_CHANGED")
            for incident in parent["quarantined_unknowns"]:
                if digest_json(read(Path(incident["source"]))) != incident["source_digest"]:
                    raise ValueError("QUARANTINE_EVIDENCE_CHANGED")
        return directory, receipt
    parents = []
    for lane in range(4):
        path, _ = _lane_path(campaign, lane)
        snapshot = read_budget(path)
        quarantines = quarantine_evidence(base_run, lane, snapshot)
        replay(snapshot, extend=True, quarantined={item["attempt"] for item in quarantines})
        parents.append(
            {
                "lane": lane,
                "path": str(path.resolve()),
                "digest": digest_json(snapshot),
                "snapshot": snapshot,
                "quarantined_unknowns": quarantines,
            }
        )
    receipt = {
        "format": "frozen-campaign-budget-extension-v1",
        "user_authorization": "允许累计保守上限 ¥120",
        "cumulative_limit_cny": 120,
        "historical_allowance_cny": 8,
        "replacement_campaign_allowance_cny": 112,
        "additional_allowance_cny": 60,
        "per_lane_limit_cny": 28,
        "workers": 4,
        "base_protocol_digest": digest_json(read(base_run / "protocol.json")),
        "frozen_source_hashes": read(base_run / "protocol.json")["source_hashes"],
        "old_authorization_digest": digest_json(old_authorization),
        "parents": parents,
        "old_runner_retired": True,
        "quarantine_review": "Explicit engineering review: retain original ReadTimeout/ConnectError failures and unknown reserved usage; continue only different tasks; do not retry failed cells or invent usage.",
        "instruction": "Do not resume the CNY60 runner after this allocation; all future spending uses these continued lanes.",
    }
    marker = campaign / "budget120-allocation.json"
    immutable(marker, {"directory": str(directory.resolve()), "receipt": receipt})
    immutable(receipt_path, receipt)
    for item in parents:
        path = directory / f"budget-lane-{item['lane']}.json"
        if not path.exists():
            JournaledBudget(
                replay(
                    item["snapshot"],
                    extend=True,
                    quarantined={entry["attempt"] for entry in item["quarantined_unknowns"]},
                ),
                path,
            )
    return directory, receipt


def load_lane(directory, lane, receipt):
    for item in receipt["parents"]:
        if digest_json(read_budget(Path(item["path"]))) != item["digest"]:
            raise ValueError("RETIRED_ORIGINAL_LANE_CHANGED")
    quarantined = lane_quarantines(directory, lane, receipt)
    path, _ = extended_lane_path(directory, lane, quarantined)
    return JournaledBudget(replay(read_budget(path), quarantined=quarantined), path)


def lane_quarantines(directory, lane, receipt):
    """Return only transport attempts authorized by immutable reviewed incidents."""
    quarantined = {entry["attempt"] for entry in receipt["parents"][lane]["quarantined_unknowns"]}
    _, history = extended_lane_path(directory, lane, quarantined)
    for entry in history:
        incident = read(directory / entry["incident"])
        if incident.get("kind") == "manual_transport_review":
            quarantined.update(item["attempt"] for item in incident["reviewed_events"])
    return quarantined


def validate_transport_review(snapshot, incident, quarantined):
    """Normalize exactly reviewed unknowns, preserving every charge and result."""
    if (
        incident.get("kind") != "manual_transport_review"
        or incident.get("action") != "continue_different_cells_without_retry_or_refund"
        or snapshot.get("stopped_reason") != "TRANSPORT_ERROR"
        or snapshot.get("in_flight_count")
        or snapshot.get("unmatched_usage_events")
        or incident.get("snapshot_digest") != digest_json(snapshot)
    ):
        raise ValueError("TRANSPORT_REVIEW_SNAPSHOT_MISMATCH")
    request_path = Path(incident["review_request"])
    request = read(request_path)
    if digest_json(request) != incident["review_request_digest"]:
        raise ValueError("TRANSPORT_REVIEW_REQUEST_CHANGED")
    unknowns = {
        item["attempt"]: item
        for item in snapshot["reservation_ledger"]
        if item["status"] == "usage_unknown"
    }
    reviewed = incident["reviewed_events"]
    if not reviewed or len({item["attempt"] for item in reviewed}) != len(reviewed):
        raise ValueError("TRANSPORT_REVIEW_EMPTY_OR_DUPLICATE")
    reviewed_attempts = {item["attempt"] for item in reviewed}
    if (
        not reviewed_attempts.issubset(unknowns)
        or set(unknowns) - set(quarantined) - reviewed_attempts
    ):
        raise ValueError("TRANSPORT_REVIEW_UNAPPROVED_UNKNOWN")
    for item in reviewed:
        if item not in request.get("events", []):
            raise ValueError("TRANSPORT_EVENT_NOT_EXPLICITLY_APPROVED")
        record = read(Path(item["record_path"]))
        if digest_json(record) != item["record_digest"] or record.get("budget_end") != snapshot:
            raise ValueError("TRANSPORT_REVIEW_RECORD_CHANGED")
        matches = [
            event
            for event in record.get("http_events", [])
            if event.get("event_id") == item["event_id"]
        ]
        if len(matches) != 1:
            raise ValueError("TRANSPORT_REVIEW_EVENT_NOT_UNIQUE")
        event = matches[0]
        entry = unknowns[item["attempt"]]
        prefix = item["event_id"].split(":")[0].split(".")[0]
        if (
            prefix != f"budget-lane-{item['lane']}"
            or event.get("attempt") != item["attempt"]
            or event.get("error_code") not in TRANSPORT_CODES
            or event.get("status") != "failed"
            or event.get("usage_complete") is not False
            or set(event.get("usage", {})) != {"prompt_tokens", "completion_tokens", "total_tokens"}
            or any(value is not None for value in event["usage"].values())
            or entry.get("stop_reason") != "TRANSPORT_ERROR"
            or item["journal_digest"] != digest_json(snapshot)
            or Path(item["journal_path"]).name != incident["prior"]
            or item["reserved_upper_cny"] != entry["reserved_upper_cny"]
        ):
            raise ValueError("TRANSPORT_REVIEW_EVENT_MISMATCH")
    normalized = deepcopy(snapshot)
    normalized["stopped_reason"] = None
    replay(normalized, quarantined=set(quarantined) | reviewed_attempts)
    return normalized


def validate_accounted_client_review(snapshot, incident, quarantined):
    """Accept one reviewed client failure only after every new token is charged.

    The raw HTTP event stays failed/incomplete. This validates accounting from
    its numeric usage and the complete ledger, without reclassifying its result.
    """
    request = read(Path(incident["review_request"]))
    entries = incident.get("reviewed_events", [])
    if (
        incident.get("kind") != "manual_accounted_client_review"
        or incident.get("action") != "continue_different_cells_without_retry_or_refund"
        or snapshot.get("stopped_reason") != "CLIENT_ERROR"
        or snapshot.get("in_flight_count")
        or snapshot.get("unmatched_usage_events")
        or incident.get("snapshot_digest") != digest_json(snapshot)
        or digest_json(request) != incident["review_request_digest"]
        or len(entries) != 1
    ):
        raise ValueError("CLIENT_REVIEW_SNAPSHOT_MISMATCH")
    item = entries[0]
    if item not in request.get("events", []):
        raise ValueError("CLIENT_EVENT_NOT_EXPLICITLY_APPROVED")
    record = read(Path(item["record_path"]))
    if (
        digest_json(record) != item["record_digest"]
        or record.get("budget_end") != snapshot
        or item["journal_digest"] != digest_json(snapshot)
        or Path(item["journal_path"]).name != incident["prior"]
        or item["lane"] != incident["lane"]
    ):
        raise ValueError("CLIENT_REVIEW_RECORD_OR_JOURNAL_CHANGED")
    start = record["budget_start"]
    prefix = start["reservation_ledger"]
    new = snapshot["reservation_ledger"][len(prefix) :]
    before_unknown = [entry for entry in prefix if entry["status"] == "usage_unknown"]
    after_unknown = [
        entry for entry in snapshot["reservation_ledger"] if entry["status"] == "usage_unknown"
    ]
    if (
        snapshot["reservation_ledger"][: len(prefix)] != prefix
        or before_unknown != after_unknown
        or start["unknown_count"] != snapshot["unknown_count"]
        or not {entry["attempt"] for entry in after_unknown}.issubset(quarantined)
        or start.get("stopped_reason")
        or start.get("in_flight_count")
        or not new
        or any(entry["status"] != "usage_complete" for entry in new)
    ):
        raise ValueError("CLIENT_REVIEW_NEW_UNKNOWN_OR_CHANGED_HISTORY")
    events = record.get("http_events", [])
    if len(events) != len(new) or [event["attempt"] for event in events] != [
        entry["attempt"] for entry in new
    ]:
        raise ValueError("CLIENT_REVIEW_EVENT_LEDGER_COVERAGE")
    for index, (event, entry) in enumerate(zip(events, new)):
        usage = event.get("usage", {})
        is_final = index == len(new) - 1
        if (
            set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
            or usage != {key: entry[key] for key in usage}
            or event.get("http_status") != 200
            or event.get("event_id", "").split(":")[0].split(".")[0]
            != f"budget-lane-{item['lane']}"
            or (
                not is_final
                and (event.get("usage_complete") is not True or event.get("status") != "completed")
            )
            or (
                is_final
                and (
                    event.get("event_id") != item["event_id"]
                    or event.get("attempt") != item["attempt"]
                    or event.get("attempt") != snapshot["http_attempts"]
                    or event.get("status") != "failed"
                    or event.get("error_code") != "GRAPH_UNEXPECTED_CLIENT_ERROR"
                    or event.get("usage_complete") is not False
                    or item["reserved_upper_cny"] != entry["reserved_upper_cny"]
                )
            )
        ):
            raise ValueError("CLIENT_REVIEW_NUMERIC_USAGE_NOT_EXACTLY_ACCOUNTED")
    normalized = deepcopy(snapshot)
    normalized["stopped_reason"] = None
    replay(normalized, quarantined=quarantined)
    return normalized


def validate_truncation(snapshot, report, quarantined):
    """Validate new fully-accounted events without erasing an unknown old prefix."""
    if snapshot != report.get("budget_end"):
        raise ValueError("TRUNCATION_REPORT_JOURNAL_MISMATCH")
    start = report["budget_start"]
    unknown_before = [
        entry for entry in start["reservation_ledger"] if entry["status"] == "usage_unknown"
    ]
    unknown_after = [
        entry for entry in snapshot["reservation_ledger"] if entry["status"] == "usage_unknown"
    ]
    if (
        unknown_before != unknown_after
        or not {item["attempt"] for item in unknown_after}.issubset(quarantined)
        or start["unknown_count"] != snapshot["unknown_count"]
    ):
        raise ValueError("NEW_UNKNOWN_USAGE_CANNOT_AUTO_CONTINUE")
    # The frozen validator inspects the contiguous NEW event range. Suppress
    # only its aggregate completeness flag in this temporary validation copy.
    probe = deepcopy(report)
    for name in ("budget_start", "budget_end"):
        probe[name]["unknown_count"] = 0
        probe[name]["reported_cost_complete"] = True
    reviewed_truncation(probe["budget_end"], probe)
    normalized = deepcopy(snapshot)
    normalized["stopped_reason"] = None
    return normalized


def extended_lane_path(directory, lane, quarantined):
    quarantined = set(quarantined)
    path = directory / f"budget-lane-{lane}.json"
    pointer = directory / f"budget-lane-{lane}.current.json"
    if not pointer.exists():
        return path, []
    saved = read(pointer)
    history = saved["history"]
    if saved["current"] != f"budget-lane-{lane}.continuation-{len(history)}.json":
        raise ValueError("CONTINUATION_POINTER_CHANGED")
    for index, entry in enumerate(history, 1):
        incident_path = directory / f"budget-lane-{lane}.incident-{index}.json"
        incident = read(incident_path)
        if entry != {"incident": incident_path.name, "digest": digest_json(incident)}:
            raise ValueError("CONTINUATION_INCIDENT_CHANGED")
        old = read_budget(path)
        if incident["prior"] != path.name or incident["prior_digest"] != digest_json(old):
            raise ValueError("CONTINUATION_PARENT_CHANGED")
        if incident.get("kind") == "manual_transport_review":
            if incident.get("lane") != lane:
                raise ValueError("TRANSPORT_REVIEW_LANE_CHANGED")
            normalized = validate_transport_review(old, incident, quarantined)
            quarantined.update(item["attempt"] for item in incident["reviewed_events"])
        elif incident.get("kind") == "manual_accounted_client_review":
            if incident.get("lane") != lane:
                raise ValueError("CLIENT_REVIEW_LANE_CHANGED")
            normalized = validate_accounted_client_review(old, incident, quarantined)
        else:
            normalized = validate_truncation(old, incident["report"], quarantined)
        path = directory / f"budget-lane-{lane}.continuation-{index}.json"
        following = read_budget(path)
        prefix = normalized["reservation_ledger"]
        if (
            following["reservation_ledger"][: len(prefix)] != prefix
            or following["money_limit_cny"] != 28
        ):
            raise ValueError("CONTINUATION_HISTORY_OR_ALLOWANCE_CHANGED")
    return path, history


def apply_transport_review(directory, receipt, request_path, *, apply=False):
    """Apply an exact reviewed accounting incident, never a blanket exception.

    Dry validation is read-only. The caller holds all campaign leases on apply.
    Each old journal stays immutable; a new journal retains its complete ledger.
    """
    request_path = request_path.resolve()
    request = read(request_path)
    is_client = request.get("format") == "manual-accounted-client-review-v1"
    if (
        request.get("format")
        not in {"manual-transport-quarantine-v1", "manual-accounted-client-review-v1"}
        or request.get("budget_authorization_digest") != digest_json(receipt)
        or not request.get("reviewed_by")
        or request.get("decision")
        != (
            "preserve_failed_cells_and_accounted_usage_continue_other_cells"
            if is_client
            else "preserve_failed_cells_and_full_unknown_reservations_continue_other_cells"
        )
        or not request.get("events")
    ):
        raise ValueError("EXPLICIT_TRANSPORT_REVIEW_REQUIRED")
    marker = (
        directory
        / f"{'client' if is_client else 'transport'}-review-{digest_json(request)[:20]}.json"
    )
    if marker.exists():
        if read(marker) != {"request": str(request_path), "digest": digest_json(request)}:
            raise ValueError("TRANSPORT_REVIEW_MARKER_CHANGED")
        for lane in {item["lane"] for item in request["events"]}:
            lane_quarantines(directory, lane, receipt)
        return
    proposals = []
    for lane in sorted({item["lane"] for item in request["events"]}):
        if type(lane) is not int or lane not in range(4):
            raise ValueError("TRANSPORT_REVIEW_BAD_LANE")
        original = {entry["attempt"] for entry in receipt["parents"][lane]["quarantined_unknowns"]}
        path, history = extended_lane_path(directory, lane, original)
        quarantined = lane_quarantines(directory, lane, receipt)
        events = [item for item in request["events"] if item["lane"] == lane]
        # An interrupted application may have written this lane's incident
        # before another lane; verify and reuse it, never allocate a second time.
        matched = [read(directory / item["incident"]) for item in history]
        if any(item.get("review_request_digest") == digest_json(request) for item in matched):
            continue
        if not is_client and all(item["attempt"] in quarantined for item in events):
            if not any(
                item.get("review_request_digest") == digest_json(request) for item in matched
            ):
                raise ValueError("TRANSPORT_REVIEW_ALREADY_APPROVED_DIFFERENTLY")
            continue
        snapshot = read_budget(path)
        if any(Path(item["journal_path"]).resolve() != path.resolve() for item in events):
            raise ValueError("TRANSPORT_REVIEW_JOURNAL_CHANGED")
        incident = {
            "kind": "manual_accounted_client_review" if is_client else "manual_transport_review",
            "lane": lane,
            "prior": path.name,
            "prior_digest": digest_json(snapshot),
            "snapshot_digest": digest_json(snapshot),
            "review_request": str(request_path),
            "review_request_digest": digest_json(request),
            "reviewed_events": events,
            "action": "continue_different_cells_without_retry_or_refund",
        }
        validator = validate_accounted_client_review if is_client else validate_transport_review
        normalized = validator(snapshot, incident, quarantined)
        authorized = (
            quarantined if is_client else quarantined | {item["attempt"] for item in events}
        )
        proposals.append((lane, history, incident, normalized, authorized))
    if not apply:
        return
    for lane, history, incident, normalized, quarantined in proposals:
        index = len(history) + 1
        incident_path = directory / f"budget-lane-{lane}.incident-{index}.json"
        immutable(incident_path, incident)
        following = directory / f"budget-lane-{lane}.continuation-{index}.json"
        if following.exists():
            if read_budget(following) != normalized:
                raise ValueError("TRANSPORT_REVIEW_NEXT_JOURNAL_CHANGED")
        else:
            JournaledBudget(replay(normalized, quarantined=quarantined), following)
        write_json(
            directory / f"budget-lane-{lane}.current.json",
            {
                "current": following.name,
                "history": history
                + [{"incident": incident_path.name, "digest": digest_json(incident)}],
            },
        )
    immutable(marker, {"request": str(request_path), "digest": digest_json(request)})


def apply_accounted_client_review(directory, receipt, request_path, *, apply=False):
    """Explicitly selected route for fully ledger-accounted client errors."""
    if read(request_path).get("format") != "manual-accounted-client-review-v1":
        raise ValueError("EXPLICIT_ACCOUNTED_CLIENT_REVIEW_REQUIRED")
    return apply_transport_review(directory, receipt, request_path, apply=apply)


def rollover_truncation(directory, lane, budget, report, quarantined=()):
    """Retain a failed truncated cell; continue only the next cell without refund."""
    path, history = extended_lane_path(directory, lane, quarantined)
    if path.resolve() != budget.path or read_budget(path) != budget.snapshot():
        raise ValueError("CONTINUATION_JOURNAL_CHANGED")
    snapshot = budget.snapshot()
    normalized = validate_truncation(snapshot, report, quarantined)
    index = len(history) + 1
    incident_path = directory / f"budget-lane-{lane}.incident-{index}.json"
    incident = {
        "prior": path.name,
        "prior_digest": digest_json(snapshot),
        "report": {key: report[key] for key in ("budget_start", "budget_end", "http_events")},
        "action": "continue_next_cell_without_retry_or_refund",
    }
    immutable(incident_path, incident)
    following = directory / f"budget-lane-{lane}.continuation-{index}.json"
    result = JournaledBudget(replay(normalized, quarantined=quarantined), following)
    history += [{"incident": incident_path.name, "digest": digest_json(incident)}]
    write_json(
        directory / f"budget-lane-{lane}.current.json",
        {"current": following.name, "history": history},
    )
    return result


def money_interrupted(record):
    budget = record.get("budget_end", {})
    if (
        budget.get("stopped_reason") != MONEY_STOP
        or budget.get("reported_cost_complete") is not True
    ):
        return False
    result = record.get("record", {})
    codes = {
        record.get("sql_error"),
        record.get("plan_error"),
        result.get("reason"),
        result.get("publication", {}).get("reason"),
    }
    return bool(codes & MONEY_CODES)


def bundles_and_methods(manifest, protocol):
    bundles = public_bundles(manifest, "test") + controlled_bundles()
    if protocol["phase"] == "stability":
        bundles = [
            replace(bundle, case_id=bundle.case_id + f"--rep{rep}")
            for bundle in bundles
            if bundle.case_id in protocol["stability_case_ids"]
            for rep in (2, 3)
        ]
    random.Random(SEED).shuffle(bundles)
    if protocol.get("limit") is not None:
        bundles = bundles[: protocol["limit"]]
    result = []
    for index, bundle in enumerate(bundles):
        methods = ["full"] if protocol["phase"] == "stability" else list(BASE_METHODS)
        if protocol["phase"] == "test" and bundle.case_id in protocol["ablation_case_ids"]:
            methods += ABLATIONS
        random.Random(SEED + int(digest_json(bundle.case_id)[:8], 16)).shuffle(methods)
        result.append((index % 4, bundle, methods))
    return result


def work_plan(manifest, protocol, base_run, output):
    decisions = []
    for lane, bundle, methods in bundles_and_methods(manifest, protocol):
        for method in methods:
            name = f"{bundle.case_id}--{method}"
            prior, current = (
                base_run / "records" / (name + ".json"),
                output / "records" / (name + ".json"),
            )
            if current.exists():
                decision = "continued_record_exists"
            elif prior.exists():
                decision = (
                    "retry_infrastructure_money_limit"
                    if money_interrupted(read(prior))
                    else "keep_completed_outcome"
                )
            else:
                checkpoint = base_run / "checkpoints" / (name + ".sqlite3")
                if checkpoint.exists():
                    decision = "blocked_unrecorded_original_checkpoint"
                elif (output / "checkpoints" / (name + ".sqlite3")).exists():
                    decision = "blocked_unrecorded_continuation_checkpoint"
                else:
                    decision = "not_previously_run"
            decisions.append(
                {
                    "name": name,
                    "lane": lane,
                    "decision": decision,
                    "prior_digest": digest_json(read(prior)) if prior.exists() else None,
                }
            )
    return decisions


async def complete_initial(bundle, config, lane, old, path):
    """Keep successful initial SQL/plan and resume only unpaid money-blocked work."""
    if path.exists():
        return read(path)
    if old is None:
        return await prepare_first(bundle, config, lane, path)
    if old["task_digest"] != digest_json(bundle.task.public_input()):
        raise ValueError("CONTINUATION_INITIAL_TASK_CHANGED")
    if not money_interrupted(old):
        immutable(path, old)
        return old
    original_events = old.get("http_events", [])
    attempts = sum(event["role"] == "initial_reference" for event in original_events)
    writer_attempts = sum(event["role"] == "initial_writer" for event in original_events)
    budget = CellBudget(lane, limit=max(1, 4 - len(original_events)))
    start = lane.snapshot()
    writer = QwenGraphModel(config, budget, max_tokens=1800)
    reviewer = QwenReviewModel(config, budget, max_tokens=1800)
    models = [("initial_writer", writer), ("initial_reference", reviewer)]
    value = deepcopy(old)
    try:
        if not value["sql"] and value.get("sql_error") in MONEY_CODES and writer_attempts == 0:
            try:
                action = parse_graph_action(
                    await writer.generate(
                        [message("system", SQL_PROMPT), message("user", bundle.task.public_input())]
                    )
                )
                if action["task_id"] != bundle.task.task_id:
                    raise ValueError("TASK_ID_MISMATCH")
                value["sql"] = action["sql"]
            except (ValueError, GraphModelError) as error:
                value["sql_error"] = str(error)
        if not value["plan"] and not lane.snapshot()["stopped_reason"] and attempts < 3:
            requests = old.get("contexts", {}).get("initial_reference", [])
            messages = (
                deepcopy(requests[-1])
                if requests
                else [
                    message("system", plan_prompt(bundle.task)),
                    message("user", bundle.task.public_input()),
                ]
            )
            for _ in range(3 - attempts):
                action = None
                try:
                    action = await reviewer.generate(messages)
                    value["plan"] = validate_plan(action, bundle.task)
                    break
                except (ValueError, GraphModelError) as error:
                    value["plan_error"] = str(error)
                    if lane.snapshot()["stopped_reason"]:
                        break
                    messages += [
                        message("assistant", action or {"error": str(error)}),
                        message("user", plan_validation_feedback(action, str(error), bundle.task)),
                    ]
    finally:
        await writer.aclose()
        await reviewer.aclose()
    events = _events(models, lane)
    value["http_events"] = original_events + events
    value["usage"] = summarize_usage(value["http_events"])
    value["contexts"] = {
        name: old.get("contexts", {}).get(name, []) + model.requests for name, model in models
    }
    value["infrastructure_continuation"] = {
        "original_digest": digest_json(old),
        "preserved_sql": old["sql"],
        "budget_start": start,
        "budget_end": lane.snapshot(),
        "http_events": events,
    }
    # Top-level accounting is the new contiguous tranche; merged initial usage
    # retains all old events separately for cold-start/actual-cost reporting.
    value.update(budget_start=start, budget_end=lane.snapshot())
    write_json(path, value)
    return value


def merged_summary(base_run, output, plan, protocol):
    rows, provenance, initial, superseded_usage = [], [], [], []
    for item in plan:
        original = base_run / "records" / (item["name"] + ".json")
        continued = output / "records" / (item["name"] + ".json")
        if continued.exists():
            if original.exists() and not money_interrupted(read(original)):
                raise ValueError("CONTINUATION_WOULD_REPLACE_QUALITY_OUTCOME")
            chosen = continued
            if original.exists():
                superseded_usage.append({"http_events": read(original).get("http_events", [])})
        elif original.exists():
            chosen = original
        else:
            continue
        document = read(chosen)
        rows.append(document["row"])
        provenance.append(
            {
                "name": item["name"],
                "selected": str(chosen.resolve()),
                "selected_digest": digest_json(document),
                "superseded": str(original.resolve())
                if chosen == continued and original.exists()
                else None,
                "reason": "retry_infrastructure_money_limit"
                if chosen == continued and original.exists()
                else "original_or_first_attempt",
            }
        )
    names = {
        p.name
        for folder in (base_run / "initial", output / "initial")
        for p in folder.glob("*.json")
    }
    for name in names:
        current = output / "initial" / name
        initial.append(read(current if current.exists() else base_run / "initial" / name))
    analysis = aggregate_runs(
        rows, bootstrap_repetitions=2000, initial_records=initial + superseded_usage
    )
    result = {
        "base_protocol_digest": digest_json(protocol),
        "expected_records": len(plan),
        "available_records": len(rows),
        "missing_records": len(plan) - len(rows),
        "rows": rows,
        "provenance": provenance,
        "analysis": analysis,
    }
    write_json(output / "merged-summary.json", result)
    return result


async def continue_run(args, manifest, protocol, plan, directory, receipt):
    config = QwenConfig.from_local_environment()
    validate_pricing(config)
    entries = bundles_and_methods(manifest, protocol)
    decisions = {item["name"]: item for item in plan}
    stopped = asyncio.Event()

    async def worker(index):
        try:
            budget = load_lane(directory, index, receipt)
            quarantined = lane_quarantines(directory, index, receipt)
            for lane, bundle, methods in entries:
                if lane != index or stopped.is_set():
                    continue
                needed = [
                    m
                    for m in methods
                    if decisions[f"{bundle.case_id}--{m}"]["decision"]
                    in {"retry_infrastructure_money_limit", "not_previously_run"}
                ]
                if not needed:
                    continue
                if source_hashes(ROOT) != protocol["source_hashes"]:
                    raise ValueError("FROZEN_SOURCE_CHANGED")
                prior_path = args.base_run / "initial" / (bundle.case_id + ".json")
                old = read(prior_path) if prior_path.exists() else None
                initial = await complete_initial(
                    bundle,
                    config,
                    budget,
                    old,
                    args.output / "initial" / (bundle.case_id + ".json"),
                )
                if budget.snapshot()["stopped_reason"]:
                    report = initial.get("infrastructure_continuation", initial)
                    budget = rollover_truncation(directory, index, budget, report, quarantined)
                for method in needed:
                    if stopped.is_set():
                        break
                    await run_cell(bundle, method, initial, config, budget, args.output)
                    if budget.snapshot()["stopped_reason"]:
                        record = read(args.output / "records" / f"{bundle.case_id}--{method}.json")
                        budget = rollover_truncation(directory, index, budget, record, quarantined)
        except BaseException:
            stopped.set()
            raise

    outcomes = await asyncio.gather(*(worker(i) for i in range(4)), return_exceptions=True)
    errors = [
        {"type": type(value).__name__, "code": str(value)[:180]}
        for value in outcomes
        if isinstance(value, BaseException)
    ]
    result = merged_summary(args.base_run, args.output, plan, protocol)
    result["run_errors"] = errors
    write_json(
        args.output / "continuation-status.json",
        {
            "missing_records": result["missing_records"],
            "errors": errors,
            "stopped": stopped.is_set(),
        },
    )
    print(
        json.dumps(
            {
                "expected": result["expected_records"],
                "available": result["available_records"],
                "errors": errors,
            }
        ),
        flush=True,
    )


def bind_driver_revision(output, binding, review_path=None, *, apply=False):
    """Append driver history without changing prior protocol or outcome provenance."""
    protocol_path = output / "continuation-protocol.json"
    current = binding["driver_sha256"]
    source = Path(__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != current:
        raise ValueError("CONTINUATION_DRIVER_CHANGED")
    if not protocol_path.exists():
        if apply:
            immutable(protocol_path, binding)
            (output / "drivers").mkdir(exist_ok=True)
            (output / "drivers" / (current + ".py")).write_bytes(source)
        return
    original = read(protocol_path)
    if {key: value for key, value in original.items() if key != "driver_sha256"} != {
        key: value for key, value in binding.items() if key != "driver_sha256"
    }:
        raise ValueError("CONTINUATION_PROTOCOL_CHANGED")
    pointer = output / "driver-history.json"
    history = read(pointer)["history"] if pointer.exists() else []
    previous = original["driver_sha256"]
    for number, entry in enumerate(history, 1):
        path = output / f"driver-revision-{number}.json"
        revision = read(path)
        if (
            entry != {"path": path.name, "digest": digest_json(revision)}
            or revision["previous_sha256"] != previous
        ):
            raise ValueError("CONTINUATION_DRIVER_HISTORY_CHANGED")
        if digest_json(read(Path(revision["review_request"]))) != revision["review_request_digest"]:
            raise ValueError("CONTINUATION_DRIVER_REVIEW_CHANGED")
        for saved, checksum in revision["preserved_records"].items():
            if file_digest(output / saved) != checksum:
                raise ValueError("CONTINUATION_PRIOR_RESULT_CHANGED")
        previous = revision["driver_sha256"]
        if file_digest(output / "drivers" / (previous + ".py")) != previous:
            raise ValueError("CONTINUATION_DRIVER_ARCHIVE_CHANGED")
    if previous == current:
        return
    if review_path is None:
        raise ValueError("CONTINUATION_NEW_DRIVER_REQUIRES_EXPLICIT_REVIEW")
    archive = output / "drivers" / (previous + ".py")
    if not archive.exists() or file_digest(archive) != previous:
        raise ValueError("CONTINUATION_PRIOR_DRIVER_ARCHIVE_REQUIRED")
    revision = {
        "previous_sha256": previous,
        "driver_sha256": current,
        "original_protocol_digest": digest_json(original),
        "review_request": str(review_path.resolve()),
        "review_request_digest": digest_json(read(review_path)),
        "scope": "Explicit transport accounting continuation only; frozen model/runtime unchanged; no failed cell retry.",
        "preserved_records": {
            str(path.relative_to(output)): file_digest(path)
            for folder in (output / "records", output / "initial")
            for path in sorted(folder.glob("*.json"))
        },
    }
    if apply:
        path = output / f"driver-revision-{len(history) + 1}.json"
        immutable(path, revision)
        (output / "drivers" / (current + ".py")).write_bytes(source)
        write_json(
            pointer, {"history": history + [{"path": path.name, "digest": digest_json(revision)}]}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-manifest", type=Path, default=ROOT / "reports/public-data/bird-manifest-v2.json"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-cumulative-limit", type=int)
    parser.add_argument(
        "--review-transport",
        type=Path,
        help="Explicit reviewed events; never a blanket permission for future errors.",
    )
    parser.add_argument(
        "--review-accounted-client",
        type=Path,
        help="Explicit failed client event whose numeric usage is already exactly charged.",
    )
    args = parser.parse_args()
    if args.review_transport and args.review_accounted_client:
        raise ValueError("ONE_EXPLICIT_REVIEW_PER_INVOCATION")
    review_path = args.review_transport or args.review_accounted_client
    review_handler = (
        apply_accounted_client_review if args.review_accounted_client else apply_transport_review
    )
    if (
        args.review_transport
        and read(args.review_transport).get("format") != "manual-transport-quarantine-v1"
    ):
        raise ValueError("EXPLICIT_TRANSPORT_REVIEW_REQUIRED")
    args.base_run, args.output = args.base_run.resolve(), args.output.resolve()
    if args.base_run == args.output:
        raise ValueError("CONTINUATION_REQUIRES_SEPARATE_OUTPUT")
    campaign = args.base_run.parent.parent
    protocol, manifest = read(args.base_run / "protocol.json"), read(args.public_manifest)
    if protocol["phase"] not in {"test", "stability"} or protocol["source_hashes"] != source_hashes(
        ROOT
    ):
        raise ValueError("CONTINUATION_REQUIRES_UNCHANGED_TEST_RUNTIME")
    if digest_json(manifest) != protocol["public_manifest_digest"]:
        raise ValueError("PUBLIC_MANIFEST_CHANGED")
    for db in manifest["databases"].values():
        if file_digest(Path(db["path"])) != db["sha256"]:
            raise ValueError("ORIGINAL_DATABASE_CHANGED")
    plan = work_plan(manifest, protocol, args.base_run, args.output)
    counts = {
        decision: sum(item["decision"] == decision for item in plan)
        for decision in sorted({item["decision"] for item in plan})
    }
    print(
        json.dumps(
            {"dry_run": not args.execute, "expected": len(plan), "decisions": counts},
            ensure_ascii=False,
        )
    )
    directory = campaign / "budget120"
    receipt_path = directory / "authorization.json"
    if review_path:
        # Dry-run validates immutable event IDs, failed records, and full ledger
        # snapshots; it must not create or change any budget journal.
        review_handler(directory, read(receipt_path), review_path)
    if receipt_path.exists():
        bind_driver_revision(
            args.output,
            {
                "base_protocol_digest": digest_json(protocol),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "budget_authorization_digest": digest_json(read(receipt_path)),
                "scope": "Budget-only continuation; frozen core, model, inputs and limits unchanged; only explicit quota interruptions may be superseded.",
            },
            review_path,
        )
    if not args.execute:
        return
    if any(item["decision"].startswith("blocked_") for item in plan):
        raise ValueError("UNRECORDED_CHECKPOINT_REQUIRES_REVIEW")
    # Holding every original lease guarantees no original worker can still pay.
    with ExitStack() as stack:
        stack.enter_context(exclusive_run(campaign / "allocation"))
        for lane in range(4):
            stack.enter_context(exclusive_run(campaign / f"lane-{lane}-execution"))
        directory, receipt = allocate_extension(
            campaign, args.base_run, approved_limit=args.approved_cumulative_limit
        )
        immutable(
            directory / f"execution-{digest_json(str(args.base_run))[:12]}.json",
            {
                "base_run": str(args.base_run),
                "output": str(args.output),
                "base_protocol_digest": digest_json(protocol),
            },
        )
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ("initial", "records", "checkpoints"):
            (args.output / name).mkdir(exist_ok=True)
        bind_driver_revision(
            args.output,
            {
                "base_protocol_digest": digest_json(protocol),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "budget_authorization_digest": digest_json(receipt),
                "scope": "Budget-only continuation; frozen core, model, inputs and limits unchanged; only explicit quota interruptions may be superseded.",
            },
            review_path,
            apply=True,
        )
        if review_path:
            review_handler(directory, receipt, review_path, apply=True)
        write_json(args.output / "work-plan.json", plan)
        asyncio.run(continue_run(args, manifest, protocol, plan, directory, receipt))


if __name__ == "__main__":
    main()
