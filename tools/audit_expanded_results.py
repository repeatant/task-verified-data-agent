"""Read-only audit of frozen main results and budget-only continuation.

No model clients, API requests, retries, or report overwrites are performed.
The output distinguishes incomplete coverage from final performance. Expensive
offline rescoring is optional and never replaces persisted outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.evaluation.expanded_scoring import aggregate_runs, score_controlled, score_public
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.runtime.reporting import digest_json, summarize_usage

MONEY_CODES = {"LIVE_MONEY_RESERVATION_LIMIT", "LIVE_BUDGET_STOPPED_MONEY_RESERVATION_LIMIT"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def money_interruption(document):
    budget = document.get("budget_end", {})
    result = document.get("record", {})
    codes = {
        document.get("sql_error"),
        document.get("plan_error"),
        result.get("reason"),
        result.get("publication", {}).get("reason"),
    }
    return (
        budget.get("stopped_reason") == "MONEY_RESERVATION_LIMIT"
        and budget.get("reported_cost_complete") is True
        and bool(codes & MONEY_CODES)
    )


def expected_cells(protocol, *, stability=False):
    if stability:
        return {
            f"{case}--rep{rep}--full"
            for case in protocol["stability_case_ids"]
            for rep in range(2, 2 + protocol["stability_additional_repetitions"])
        }
    ablations = set(protocol["ablation_case_ids"])
    return {
        f"{case['case_id']}--{method}"
        for case in protocol["cases"]
        for method in protocol["base_methods"]
        + (protocol["ablation_methods"] if case["case_id"] in ablations else [])
    }


def merge_records(expected, base, continuation=None):
    """Quality/transport failures are immutable; only named quota stops may yield."""
    selected, provenance, errors, all_documents = {}, [], [], []
    for directory in (base, continuation):
        if directory is not None:
            for path in sorted((directory / "records").glob("*.json")):
                if path.stem not in expected:
                    errors.append(f"UNPLANNED_RECORD:{path.name}")
                all_documents.append(read(path))
    for name in sorted(expected):
        old_path = base / "records" / (name + ".json")
        new_path = continuation / "records" / (name + ".json") if continuation else None
        old = read(old_path) if old_path.exists() else None
        new = read(new_path) if new_path and new_path.exists() else None
        if old is not None and new is not None and not money_interruption(old):
            errors.append(f"QUALITY_OUTCOME_REPLACED:{name}")
            chosen, path = old, old_path
        elif new is not None:
            chosen, path = new, new_path
        elif old is not None:
            chosen, path = old, old_path
        else:
            continue
        selected[name] = (chosen, path)
        provenance.append(
            {
                "name": name,
                "selected": str(path),
                "selected_digest": digest_json(chosen),
                "superseded_digest": digest_json(old)
                if new is not None and old is not None and path == new_path
                else None,
                "reason": "invalid_replacement_preserved_original"
                if new is not None and old is not None and path == old_path
                else "money_interruption_only"
                if new is not None and old is not None and path == new_path
                else "original_or_first_attempt",
            }
        )
    return selected, provenance, errors, all_documents


def initial_caches(base, continuation):
    caches, errors, all_documents = {}, [], []
    names = {
        path.name
        for directory in (base, continuation)
        if directory
        for path in (directory / "initial").glob("*.json")
    }
    for name in sorted(names):
        old_path = base / "initial" / name
        new_path = continuation / "initial" / name if continuation else None
        old = read(old_path) if old_path.exists() else None
        new = read(new_path) if new_path and new_path.exists() else None
        all_documents += [item for item in (old, new) if item is not None]
        if old is not None and new is not None and old != new:
            if not money_interruption(old):
                errors.append(f"INITIAL_QUALITY_CACHE_REPLACED:{name}")
            for field in ("task_digest", "sql", "plan"):
                if old.get(field) is not None and old.get(field) != new.get(field):
                    errors.append(f"INITIAL_VALUE_CHANGED:{name}:{field}")
            previous = old.get("http_events", [])
            if new.get("http_events", [])[: len(previous)] != previous:
                errors.append(f"INITIAL_EVENT_PREFIX_CHANGED:{name}")
            if new.get("infrastructure_continuation", {}).get("original_digest") != digest_json(
                old
            ):
                errors.append(f"INITIAL_LINEAGE_MISMATCH:{name}")
        caches[(str(base), name)] = old
        if continuation:
            caches[(str(continuation), name)] = new if new is not None else old
    return caches, errors, all_documents


def unique_events(documents):
    events, errors = {}, []
    for document in documents:
        for event in document.get("http_events", []):
            key = event.get("event_id")
            if not key:
                errors.append("EVENT_ID_MISSING")
                continue
            if key in events and any(
                events[key].get(field) != event.get(field)
                for field in ("usage", "status", "error_code", "model")
            ):
                errors.append(f"EVENT_ID_CONFLICT:{key}")
            events[key] = event
    return events, errors


def verify_sources(root, base, protocol, manifest):
    errors = []
    with zipfile.ZipFile(base / "frozen-source.zip") as archive:
        for name, expected in protocol["source_hashes"].items():
            try:
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    errors.append(f"FROZEN_ARCHIVE_CHANGED:{name}")
            except KeyError:
                errors.append(f"FROZEN_ARCHIVE_FILE_MISSING:{name}")
            path = root / name
            if not path.exists() or file_digest(path) != expected:
                errors.append(f"CURRENT_SOURCE_DIFFERS:{name}")
    if digest_json(manifest) != protocol["public_manifest_digest"]:
        errors.append("PUBLIC_MANIFEST_CHANGED")
    checked = {}
    for name, database in manifest["databases"].items():
        path = Path(database["path"])
        actual = file_digest(path) if path.exists() else None
        checked[name] = actual
        if actual != database["sha256"]:
            errors.append(f"ORIGINAL_DATABASE_CHANGED:{name}")
    return {"errors": errors, "database_digests": checked, "protocol_digest": digest_json(protocol)}


def _latest_ledger(directory, lane):
    pointer = directory / f"budget-lane-{lane}.current.json"
    path = directory / (
        read(pointer)["current"] if pointer.exists() else f"budget-lane-{lane}.json"
    )
    if path.resolve().parent != directory.resolve():
        raise ValueError("LEDGER_POINTER_OUTSIDE_DIRECTORY")
    document = read(path)
    return path, document.get("budget", document)


def _inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("AUDIT_PATH_OUTSIDE_DIRECTORY")
    return path


def driver_history_audit(root, continuation, protocol):
    """Bind explicitly reviewed driver revisions without rewriting old bindings."""
    original = read(continuation / "continuation-protocol.json")
    errors, versions = [], [original["driver_sha256"]]
    if original["base_protocol_digest"] != digest_json(protocol):
        errors.append("CONTINUATION_PROTOCOL_BINDING")
    pointer = continuation / "driver-history.json"
    history = read(pointer)["history"] if pointer.exists() else []
    for index, entry in enumerate(history, 1):
        path = _inside(continuation, entry["path"])
        revision = read(path)
        if path.name != f"driver-revision-{index}.json" or digest_json(revision) != entry["digest"]:
            errors.append(f"DRIVER_HISTORY_ENTRY:{index}")
        if revision["previous_sha256"] != versions[-1] or revision[
            "original_protocol_digest"
        ] != digest_json(original):
            errors.append(f"DRIVER_HISTORY_CHAIN:{index}")
        if digest_json(read(revision["review_request"])) != revision["review_request_digest"]:
            errors.append(f"DRIVER_REVIEW_CHANGED:{index}")
        for relative, expected in revision["preserved_records"].items():
            saved = _inside(continuation, relative)
            if saved.parent.name not in {"records", "initial"} or file_digest(saved) != expected:
                errors.append(f"DRIVER_PRIOR_RECORD_CHANGED:{index}:{relative}")
        versions.append(revision["driver_sha256"])
    for version in versions:
        path = continuation / "drivers" / (version + ".py")
        # Legacy single-version runs did not require an archive. Once revised,
        # every executed version must be retained, including the original.
        if history or path.exists():
            if not path.exists() or file_digest(path) != version:
                errors.append(f"DRIVER_ARCHIVE_CHANGED:{version}")
    if file_digest(root / "tools/continue_expanded.py") != versions[-1]:
        errors.append("CONTINUATION_DRIVER_CHANGED")
    return {"errors": errors, "versions": versions, "reviewed_revisions": len(history)}


def accounted_client_review_audit(directory, lane, prior_path, prior, incident):
    """A failed event can have known charges; never relabel its outcome."""
    errors, reviewed = [], []
    request = read(incident["review_request"])
    if (
        incident.get("lane") != lane
        or incident.get("action") != "continue_different_cells_without_retry_or_refund"
        or incident.get("snapshot_digest") != digest_json(prior)
        or digest_json(request) != incident["review_request_digest"]
        or request.get("format") != "manual-accounted-client-review-v1"
        or request.get("budget_authorization_digest")
        != digest_json(read(directory / "authorization.json"))
        or request.get("decision")
        != "preserve_failed_cells_and_accounted_usage_continue_other_cells"
        or not request.get("reviewed_by")
        or prior.get("stopped_reason") != "CLIENT_ERROR"
        or len(incident.get("reviewed_events", [])) != 1
    ):
        errors.append(f"ACCOUNTED_CLIENT_REVIEW_BINDING:{lane}")
    for item in incident.get("reviewed_events", []):
        document = read(item["record_path"])
        if (
            item not in request.get("events", [])
            or digest_json(document) != item["record_digest"]
            or document.get("budget_end") != prior
            or item["journal_digest"] != digest_json(prior)
            or Path(item["journal_path"]).resolve() != prior_path.resolve()
            or item["lane"] != lane
        ):
            errors.append(f"ACCOUNTED_CLIENT_REVIEW_EVIDENCE:{lane}:{item['attempt']}")
            continue
        start = document["budget_start"]
        prefix = start["reservation_ledger"]
        suffix = prior["reservation_ledger"][len(prefix) :]
        if (
            prior["reservation_ledger"][: len(prefix)] != prefix
            or [entry for entry in prefix if entry["status"] == "usage_unknown"]
            != [
                entry for entry in prior["reservation_ledger"] if entry["status"] == "usage_unknown"
            ]
            or start["unknown_count"] != prior["unknown_count"]
            or not suffix
            or any(entry["status"] != "usage_complete" for entry in suffix)
        ):
            errors.append(f"ACCOUNTED_CLIENT_CHANGED_HISTORY:{lane}:{item['attempt']}")
        events = document.get("http_events", [])
        if len(events) != len(suffix) or [event["attempt"] for event in events] != [
            entry["attempt"] for entry in suffix
        ]:
            errors.append(f"ACCOUNTED_CLIENT_EVENT_COVERAGE:{lane}:{item['attempt']}")
            continue
        for index, (event, charge) in enumerate(zip(events, suffix, strict=True)):
            usage = event.get("usage", {})
            numeric = set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"} and all(
                type(value) is int and value >= 0 for value in usage.values()
            )
            if (
                not numeric
                or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                != usage.get("total_tokens")
                or usage != {key: charge.get(key) for key in usage}
                or event.get("http_status") != 200
                or event.get("event_id", "").split(":")[0].split(".")[0] != f"budget-lane-{lane}"
            ):
                errors.append(f"ACCOUNTED_CLIENT_NUMERIC_USAGE:{lane}:{event['attempt']}")
            if index == len(events) - 1:
                if (
                    event.get("status") != "failed"
                    or event.get("error_code") != "GRAPH_UNEXPECTED_CLIENT_ERROR"
                    or event.get("usage_complete") is not False
                    or event.get("event_id") != item["event_id"]
                    or event.get("attempt") != item["attempt"]
                    or event.get("attempt") != prior["http_attempts"]
                    or item["reserved_upper_cny"] != charge["reserved_upper_cny"]
                ):
                    errors.append(f"ACCOUNTED_CLIENT_FAILURE_CHANGED:{lane}:{item['attempt']}")
            elif event.get("status") != "completed" or event.get("usage_complete") is not True:
                errors.append(f"ACCOUNTED_CLIENT_PREVIOUS_EVENT_FAILED:{lane}:{event['attempt']}")
        reviewed.append(
            {
                "event_id": item["event_id"],
                "lane": lane,
                "attempt": item["attempt"],
                "classification": "failed_response_with_known_accounted_tokens",
            }
        )
    return {"errors": errors, "reviewed_known_failures": reviewed}


def ledger_history_audit(directory, lane):
    """Check append-only journal lineage and explicit unknown-usage reviews."""
    errors, reviews, known_failures = [], [], []
    pointer = directory / f"budget-lane-{lane}.current.json"
    if not pointer.exists():
        return {
            "errors": errors,
            "reviewed_unknowns": reviews,
            "reviewed_known_failures": known_failures,
            "incidents": 0,
        }
    saved = read(pointer)
    history = saved["history"]
    if saved["current"] != f"budget-lane-{lane}.continuation-{len(history)}.json":
        errors.append(f"LEDGER_CURRENT_POINTER:{lane}")
    prior_path = directory / f"budget-lane-{lane}.json"
    for index, entry in enumerate(history, 1):
        incident_path = _inside(directory, entry["incident"])
        incident = read(incident_path)
        prior_document = read(prior_path)
        prior = prior_document.get("budget", prior_document)
        if (
            incident_path.name != f"budget-lane-{lane}.incident-{index}.json"
            or digest_json(incident) != entry["digest"]
        ):
            errors.append(f"LEDGER_INCIDENT_CHANGED:{lane}:{index}")
        if incident["prior"] != prior_path.name or incident["prior_digest"] != digest_json(prior):
            errors.append(f"LEDGER_INCIDENT_PARENT:{lane}:{index}")
        if incident.get("kind") == "manual_transport_review":
            request = read(incident["review_request"])
            if (
                incident.get("lane") != lane
                or incident.get("action") != "continue_different_cells_without_retry_or_refund"
                or incident.get("snapshot_digest") != digest_json(prior)
                or digest_json(request) != incident["review_request_digest"]
                or request.get("budget_authorization_digest")
                != digest_json(read(directory / "authorization.json"))
                or request.get("decision")
                != "preserve_failed_cells_and_full_unknown_reservations_continue_other_cells"
                or not request.get("reviewed_by")
                or prior.get("stopped_reason") != "TRANSPORT_ERROR"
            ):
                errors.append(f"TRANSPORT_REVIEW_BINDING:{lane}:{index}")
            for item in incident["reviewed_events"]:
                document = read(item["record_path"])
                matches = [
                    event
                    for event in document.get("http_events", [])
                    if event.get("event_id") == item["event_id"]
                ]
                entries = [
                    charge
                    for charge in prior["reservation_ledger"]
                    if charge["attempt"] == item["attempt"]
                ]
                if (
                    item not in request.get("events", [])
                    or digest_json(document) != item["record_digest"]
                    or document.get("budget_end") != prior
                    or item["journal_digest"] != digest_json(prior)
                    or Path(item["journal_path"]).resolve() != prior_path.resolve()
                    or len(matches) != 1
                    or len(entries) != 1
                ):
                    errors.append(f"TRANSPORT_REVIEW_EVIDENCE:{lane}:{index}:{item['attempt']}")
                    continue
                event, charge = matches[0], entries[0]
                if (
                    event.get("error_code")
                    not in {"GRAPH_QWEN_TRANSPORT_READTIMEOUT", "GRAPH_QWEN_TRANSPORT_CONNECTERROR"}
                    or event.get("status") != "failed"
                    or event.get("usage_complete") is not False
                    or any(value is not None for value in event.get("usage", {}).values())
                    or charge["status"] != "usage_unknown"
                    or item["reserved_upper_cny"] != charge["reserved_upper_cny"]
                    or item["lane"] != lane
                ):
                    errors.append(f"TRANSPORT_UNKNOWN_CHANGED:{lane}:{index}:{item['attempt']}")
                reviews.append(
                    {
                        "lane": lane,
                        "attempt": item["attempt"],
                        "event_id": item["event_id"],
                        "reserved_upper_cny": item["reserved_upper_cny"],
                    }
                )
        elif incident.get("kind") == "manual_accounted_client_review":
            client = accounted_client_review_audit(directory, lane, prior_path, prior, incident)
            errors += client["errors"]
            known_failures += client["reviewed_known_failures"]
        elif incident.get("report", {}).get("budget_end") != prior:
            errors.append(f"TRUNCATION_REPORT_BINDING:{lane}:{index}")
        following_path = directory / f"budget-lane-{lane}.continuation-{index}.json"
        document = read(following_path)
        following = document.get("budget", document)
        old_ledger = prior["reservation_ledger"]
        if following["reservation_ledger"][: len(old_ledger)] != old_ledger:
            errors.append(f"LEDGER_INCIDENT_PREFIX_CHANGED:{lane}:{index}")
        if following["money_limit_cny"] != prior["money_limit_cny"]:
            errors.append(f"LEDGER_INCIDENT_ALLOWANCE_CHANGED:{lane}:{index}")
        prior_path = following_path
    return {
        "errors": errors,
        "reviewed_unknowns": reviews,
        "reviewed_known_failures": known_failures,
        "incidents": len(history),
    }


def ledger_audit(campaign, events):
    """Use latest continued lanes, never sum historical snapshots twice."""
    extended = campaign / "budget120"
    directory = extended if (extended / "authorization.json").exists() else campaign
    authorization = read(directory / "authorization.json")
    originals = read(campaign / "authorization.json")
    errors, ledgers, entries, histories = [], [], {}, []
    accounted_failed_events = []
    for lane in range(4):
        for history_directory in (campaign, extended) if directory == extended else (campaign,):
            history = ledger_history_audit(history_directory, lane)
            errors += history["errors"]
            histories.append({"directory": str(history_directory), "lane": lane, **history})
        path, snapshot = _latest_ledger(directory, lane)
        ledger = snapshot["reservation_ledger"]
        if len(ledger) != snapshot["http_attempts"]:
            errors.append(f"LEDGER_ATTEMPT_COUNT:{lane}")
        if (
            abs(sum(item["reserved_upper_cny"] for item in ledger) - snapshot["reserved_upper_cny"])
            > 1e-6
        ):
            errors.append(f"LEDGER_RESERVATION_SUM:{lane}")
        complete = [item for item in ledger if item["status"] == "usage_complete"]
        unknown = [item for item in ledger if item["status"] == "usage_unknown"]
        if len(complete) != snapshot["complete_count"] or len(unknown) != snapshot["unknown_count"]:
            errors.append(f"LEDGER_STATUS_COUNTS:{lane}")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if sum(item.get(field, 0) for item in complete) != snapshot["reported_" + field]:
                errors.append(f"LEDGER_TOKEN_SUM:{lane}:{field}")
        if (
            abs(
                sum(item.get("estimated_reported_cny", 0) for item in complete)
                - snapshot["estimated_reported_cny"]
            )
            > 1e-6
        ):
            errors.append(f"LEDGER_KNOWN_COST_SUM:{lane}")
        if directory == extended:
            parent = authorization["parents"][lane]
            prior = read(parent["path"])
            prior = prior.get("budget", prior)
            if digest_json(prior) != parent["digest"]:
                errors.append(f"RETIRED_LEDGER_CHANGED:{lane}")
            prefix = prior["reservation_ledger"]
            if ledger[: len(prefix)] != prefix:
                errors.append(f"LEDGER_PREFIX_CHANGED:{lane}")
        for index, item in enumerate(ledger, 1):
            if item["attempt"] != index:
                errors.append(f"LEDGER_ATTEMPT_SEQUENCE:{lane}:{index}")
            entries[f"budget-lane-{lane}:{index}"] = item
        ledgers.append(
            {
                "path": str(path),
                "snapshot_digest": digest_json(snapshot),
                **{
                    key: snapshot[key]
                    for key in (
                        "http_attempts",
                        "complete_count",
                        "unknown_count",
                        "in_flight_count",
                        "reserved_upper_cny",
                        "estimated_reported_cny",
                        "reported_cost_complete",
                        "reported_prompt_tokens",
                        "reported_completion_tokens",
                        "stopped_reason",
                    )
                },
            }
        )
    for key, event in events.items():
        entry = entries.get(key)
        if entry is None:
            errors.append(f"EVENT_WITHOUT_LEDGER:{key}")
            continue
        usage = event.get("usage", {})
        if entry["status"] == "usage_complete":
            if any(
                usage.get(field) != entry.get(field)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                errors.append(f"EVENT_LEDGER_USAGE_MISMATCH:{key}")
            if event.get("status") != "completed" or event.get("usage_complete") is not True:
                accounted_failed_events.append(
                    {
                        "event_id": key,
                        "event_status": event.get("status"),
                        "event_usage_complete_flag": event.get("usage_complete"),
                        "error_code": event.get("error_code"),
                        "ledger_status": entry["status"],
                        "reported_usage": usage,
                        "estimated_reported_cny": entry.get("estimated_reported_cny"),
                        "interpretation": "Known accounted tokens do not imply a successful model response or task. Raw event flags and failed outcome are retained.",
                    }
                )
        elif event.get("usage_complete") is True:
            errors.append(f"UNKNOWN_LEDGER_HAS_COMPLETE_EVENT:{key}")
    unknown = [
        {
            "event_id": key,
            "reserved_upper_cny": item["reserved_upper_cny"],
            "status": item["status"],
            "in_selected_run_records": key in events,
        }
        for key, item in entries.items()
        if item["status"] != "usage_complete"
    ]
    earlier = [item["snapshot"] for item in originals["prior"]]
    for item in originals["prior"]:
        current = read(item["path"])
        if current.get("budget", current) != item["snapshot"]:
            errors.append("PRE_CAMPAIGN_LEDGER_CHANGED")
    return {
        "errors": errors,
        "lanes": ledgers,
        "histories": histories,
        "unknown_reservations": unknown,
        "accounted_failed_events": accounted_failed_events,
        "scope": "All campaign runs including development plus preserved earlier allowances; not only main-test cost.",
        "cumulative_authorized_cny": authorization["cumulative_limit_cny"],
        "cumulative_reserved_cny": sum(item["reserved_upper_cny"] for item in ledgers + earlier),
        "cumulative_known_estimated_cny": sum(
            item["estimated_reported_cny"] for item in ledgers + earlier
        ),
        "known_cost_is_incomplete": bool(unknown)
        or any(not item["reported_cost_complete"] for item in earlier),
    }


def distribution(values):
    """Describe observed latencies; no claim that repeated variants are independent."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "total": 0, "median": None, "p95": None}

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "total": sum(ordered),
        "median": quantile(0.5),
        "p95": quantile(0.95),
    }


def outcome_summary(rows):
    """Small explicit ratios complement the clustered paired analysis."""
    grouped = {}
    for kind in sorted({row["kind"] for row in rows}):
        grouped[kind] = {}
        for method in sorted({row["method"] for row in rows if row["kind"] == kind}):
            items = [row for row in rows if row["kind"] == kind and row["method"] == method]
            delivered = sum(row["score"]["delivered"] for row in items)
            wrong = sum(row["score"]["wrong_release"] for row in items)
            grouped[kind][method] = {
                "available": len(items),
                "correct_delivery": sum(row["score"]["correct_delivery"] for row in items),
                "delivered": delivered,
                "wrong_release": wrong,
                "wrong_fraction_of_all": wrong / len(items),
                "wrong_fraction_of_deliveries": wrong / delivered if delivered else None,
                "statuses": dict(Counter(row["status"] for row in items)),
                "distinct_clusters": len({row["cluster_id"] for row in items}),
                "post_initial_elapsed_seconds": sum(row["elapsed_seconds"] for row in items),
                "cold_start_effective_elapsed_seconds": sum(
                    row.get("cold_start_elapsed_seconds", row["elapsed_seconds"]) for row in items
                ),
                "latency_seconds": {
                    "post_initial": distribution([row["elapsed_seconds"] for row in items]),
                    "cold_start_effective": distribution(
                        [
                            row.get("cold_start_elapsed_seconds", row["elapsed_seconds"])
                            for row in items
                        ]
                    ),
                },
            }
    return grouped


def inspect_stability(protocol, main_selected, base, continuation, *, scoring_context=None):
    """Keep repeat outputs separate, while binding each to its original task."""
    expected = expected_cells(protocol, stability=True)
    selected, provenance, errors, documents = merge_records(expected, base, continuation)
    caches, cache_errors, initials = initial_caches(base, continuation)
    errors += cache_errors
    bindings = {case["case_id"]: case for case in protocol["cases"]}
    comparisons = []
    rows, mismatched_scores = [], []
    for original in protocol["stability_case_ids"]:
        observations = []
        first = main_selected.get(original + "--full")
        if first is not None:
            observations.append((1, first[0]))
        for repetition in range(2, 2 + protocol["stability_additional_repetitions"]):
            name = f"{original}--rep{repetition}--full"
            if name not in selected:
                continue
            document, path = selected[name]
            row, record = document["row"], document["record"]
            binding = bindings[original]
            if name != f"{row['case_id']}--{row['method']}" or row["method"] != "full":
                errors.append(f"STABILITY_NAME_BINDING:{name}")
            if any(
                document.get(key) != binding.get(key) for key in ("task_digest", "source_digest")
            ):
                errors.append(f"STABILITY_TASK_SOURCE_BINDING:{name}")
            if row.get("cluster_id") != binding.get("cluster_id"):
                errors.append(f"STABILITY_CLUSTER_BINDING:{name}")
            initial = caches.get((str(path.parent.parent), row["case_id"] + ".json"))
            if initial is None or initial.get("task_digest") != binding["task_digest"]:
                errors.append(f"STABILITY_INITIAL_CACHE:{name}")
            elif row.get("shared_initial_http_events") != initial.get("http_events", []):
                errors.append(f"STABILITY_SHARED_INITIAL_EVENTS:{name}")
            elif row.get("shared_initial_usage") != summarize_usage(initial.get("http_events", [])):
                errors.append(f"STABILITY_SHARED_INITIAL_USAGE:{name}")
            if row.get("http_events") != document.get("http_events") or row.get(
                "usage"
            ) != summarize_usage(document.get("http_events", [])):
                errors.append(f"STABILITY_USAGE:{name}")
            if row["status"] != record["status"]:
                errors.append(f"STABILITY_STATUS:{name}")
            if scoring_context is not None and initial is not None:
                manifest, gold, controls = scoring_context
                if row["kind"] == "controlled":
                    rescored = score_controlled(controls[original], record)
                else:
                    target = gold[original]
                    rescored = score_public(
                        {
                            "db_path": manifest["databases"][target["db_id"]]["path"],
                            "gold_rows": target["gold_rows"],
                            "initial_sql": initial.get("sql"),
                        },
                        record,
                    )
                if rescored != row["score"]:
                    mismatched_scores.append(
                        {"name": name, "saved": row["score"], "recomputed": rescored}
                    )
            observations.append((repetition, document))
            rows.append(row)
        comparisons.append(
            {
                "original_case_id": original,
                "observed_runs": len(observations),
                "status_varies": len({doc["row"]["status"] for _, doc in observations}) > 1,
                "delivery_correctness_varies": len(
                    {doc["row"]["score"]["correct_delivery"] for _, doc in observations}
                )
                > 1,
                "published_answer_varies": len(
                    {
                        digest_json(doc["record"].get("publication", {}).get("rows"))
                        for _, doc in observations
                    }
                )
                > 1,
                "runs": [
                    {
                        "repetition": rep,
                        "status": doc["row"]["status"],
                        "correct_delivery": doc["row"]["score"]["correct_delivery"],
                    }
                    for rep, doc in observations
                ],
            }
        )
    events, event_errors = unique_events(documents + initials)
    errors += event_errors
    if mismatched_scores:
        errors.append("STABILITY_OFFLINE_RESCORE_MISMATCH")
    return {
        "expected": len(expected),
        "available": len(selected),
        "status": "audit_failed"
        if errors
        else "complete"
        if len(selected) == len(expected)
        else "incomplete",
        "errors": errors,
        "missing": sorted(expected - set(selected)),
        "provenance": provenance,
        "outcomes": outcome_summary(rows),
        "per_original_task": comparisons,
        "actual_repeat_usage": summarize_usage(list(events.values())),
        "rescore": {"requested": scoring_context is not None, "mismatches": mismatched_scores},
    }, documents + initials


def audit_run(
    root,
    base,
    continuation,
    manifest_path,
    campaign,
    *,
    rescore=False,
    repetitions=2000,
    stability_base=None,
    stability_continuation=None,
):
    protocol, manifest = read(base / "protocol.json"), read(manifest_path)
    source = verify_sources(root, base, protocol, manifest)
    expected = expected_cells(protocol)
    selected, provenance, errors, documents = merge_records(expected, base, continuation)
    caches, initial_errors, initials = initial_caches(base, continuation)
    errors += source["errors"] + initial_errors
    rows, failure_codes, mismatched_scores = [], Counter(), []
    bindings = {case["case_id"]: case for case in protocol["cases"]}
    gold = {
        f"bird-{item['question_id']}": item
        for item in manifest["eligibility"]
        if not item["reasons"]
    }
    if rescore:
        from dataverify.evaluation.expanded_cases import expanded_cases

        controls = {case.case_id: case for case in expanded_cases()}
    for name, (document, path) in selected.items():
        row = document["row"]
        if name != f"{row['case_id']}--{row['method']}":
            errors.append(f"RECORD_NAME_BINDING:{name}")
        binding = bindings.get(row["case_id"], {})
        if any(document.get(key) != binding.get(key) for key in ("task_digest", "source_digest")):
            errors.append(f"RECORD_TASK_SOURCE_BINDING:{name}")
        if row.get("cluster_id") != binding.get("cluster_id"):
            errors.append(f"RECORD_CLUSTER_BINDING:{name}")
        initial = caches.get((str(path.parent.parent), row["case_id"] + ".json"))
        if initial is None:
            errors.append(f"INITIAL_CACHE_MISSING:{name}")
        else:
            allowed = (
                {"initial_writer"}
                if row["method"] in ("direct", "self_repair")
                else {"initial_writer", "initial_reference"}
            )
            initial_events = [
                event for event in initial.get("http_events", []) if event["role"] in allowed
            ]
            if row.get("shared_initial_http_events") != initial_events:
                errors.append(f"INITIAL_SHARED_EVENTS_MISMATCH:{name}")
            if row.get("shared_initial_usage") != summarize_usage(initial_events):
                errors.append(f"INITIAL_SHARED_USAGE_MISMATCH:{name}")
            if initial.get("task_digest") != binding.get("task_digest"):
                errors.append(f"INITIAL_TASK_BINDING:{name}")
        if row.get("http_events") != document.get("http_events") or row.get(
            "usage"
        ) != summarize_usage(document.get("http_events", [])):
            errors.append(f"RECORD_USAGE_MISMATCH:{name}")
        record, score = document["record"], row["score"]
        if row["status"] != record["status"] or bool(score["delivered"]) != (
            record["status"] == "reviewed" and bool(record.get("publication", {}).get("sql"))
        ):
            errors.append(f"RECORD_DELIVERY_STATUS:{name}")
        if score["correct_delivery"] and (score["wrong_release"] or not score["delivered"]):
            errors.append(f"CONTRADICTORY_SCORE:{name}")
        reason = (
            record.get("reason")
            or record.get("error_type")
            or record.get("publication", {}).get("reason")
            or row["status"]
        )
        if not score["correct_delivery"]:
            failure_codes[str(reason)[:180]] += 1
        if rescore and initial is not None and not source["errors"]:
            if row["kind"] == "controlled":
                actual = score_controlled(controls[row["case_id"]], record)
            else:
                target = gold[row["case_id"]]
                actual = score_public(
                    {
                        "db_path": manifest["databases"][target["db_id"]]["path"],
                        "gold_rows": target["gold_rows"],
                        "initial_sql": initial.get("sql"),
                    },
                    record,
                )
            if actual != score:
                mismatched_scores.append({"name": name, "saved": score, "recomputed": actual})
        rows.append(row)
    if mismatched_scores:
        errors.append("OFFLINE_RESCORE_MISMATCH")
    events, event_errors = unique_events(documents + initials)
    errors += event_errors
    driver = None
    if continuation and (continuation / "continuation-protocol.json").exists():
        driver = driver_history_audit(root, continuation, protocol)
        errors += driver["errors"]
    if continuation and (continuation / "work-plan.json").exists():
        for item in read(continuation / "work-plan.json"):
            old = base / "records" / (item["name"] + ".json")
            if item.get("prior_digest") and (
                not old.exists() or digest_json(read(old)) != item["prior_digest"]
            ):
                errors.append(f"ORIGINAL_RECORD_CHANGED:{item['name']}")
    missing = sorted(expected - set(selected))
    stability = {
        "expected": len(expected_cells(protocol, stability=True)),
        "available": 0,
        "status": "not_supplied",
    }
    if stability_base:
        stability, stability_documents = inspect_stability(
            protocol,
            selected,
            stability_base,
            stability_continuation,
            scoring_context=(manifest, gold, controls)
            if rescore and not source["errors"]
            else None,
        )
        errors += stability["errors"]
        all_events, all_event_errors = unique_events(documents + initials + stability_documents)
        errors += all_event_errors
    else:
        all_events = events
    ledger = ledger_audit(campaign, all_events)
    errors += ledger["errors"]
    status = (
        "audit_failed"
        if errors
        else "incomplete"
        if missing
        else "main_complete_stability_pending"
        if stability["status"] != "complete"
        else "complete"
    )
    return {
        "format": "expanded-independent-audit-v1",
        "status": status,
        "coverage": {"expected": len(expected), "available": len(selected), "missing": missing},
        "population": {
            "public_questions": protocol["public_cases"],
            "controlled_trajectories": protocol["controlled_trajectories"],
            "controlled_parent_fault_pairs": protocol["controlled_parent_fault_pairs"],
            "controlled_distinct_requirements": protocol["controlled_distinct_requirements"],
        },
        "source_audit": source,
        "driver_history": driver,
        "errors": errors,
        "provenance": provenance,
        "rescore": {"requested": rescore, "mismatches": mismatched_scores},
        "observed_outcomes": outcome_summary(rows),
        "comparison_scopes": {
            "primary_baselines": list(protocol["base_methods"]),
            "ablations": list(protocol["ablation_methods"]),
            "ablation_case_ids": list(protocol["ablation_case_ids"]),
            "paired_comparison_rule": "Only identical task IDs are compared; full baseline is restricted to the ablation subset for those differences.",
        },
        "failure_reasons": dict(failure_codes),
        "analysis": aggregate_runs(
            rows, bootstrap_repetitions=repetitions, initial_records=initials + documents
        ),
        "actual_main_usage": summarize_usage(list(events.values())),
        "budget_audit": ledger,
        "stability": stability,
        "interpretation": "Observed rates are not final until planned coverage and audits pass. Refusal is non-delivery; controlled variants and stability repeats are not new independent tasks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--stability-run", type=Path)
    parser.add_argument("--stability-continuation", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an audit; choose a versioned output path.")
    result = audit_run(
        ROOT,
        args.base_run,
        args.continuation,
        args.manifest,
        args.campaign,
        rescore=args.rescore,
        repetitions=args.bootstrap_repetitions,
        stability_base=args.stability_run,
        stability_continuation=args.stability_continuation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "expected": result["coverage"]["expected"],
                "available": result["coverage"]["available"],
                "errors": result["errors"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
