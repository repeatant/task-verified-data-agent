"""Independent audit cannot hide failed cells or replace quality outcomes."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/audit_expanded_results.py"
SPEC = importlib.util.spec_from_file_location("audit_expanded_results_test", TOOL)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def stopped(*, money=True):
    return {
        "budget_end": {
            "stopped_reason": "MONEY_RESERVATION_LIMIT" if money else "TRANSPORT_ERROR",
            "reported_cost_complete": money,
        },
        "record": {
            "status": "inconclusive",
            "reason": "LIVE_MONEY_RESERVATION_LIMIT"
            if money
            else "GRAPH_QWEN_TRANSPORT_READTIMEOUT",
        },
        "http_events": [],
    }


def test_frozen_grid_is_980_and_stability_is_separate():
    protocol = {
        "cases": [{"case_id": str(index)} for index in range(200)],
        "base_methods": ["direct", "self_repair", "two_way", "full"],
        "ablation_methods": ["no_search", "visible_audit", "model_authorization"],
        "ablation_case_ids": list(map(str, range(60))),
        "stability_case_ids": list(map(str, range(10))),
        "stability_additional_repetitions": 2,
    }
    main, stable = audit.expected_cells(protocol), audit.expected_cells(protocol, stability=True)
    assert len(main) == 980 and len(stable) == 20
    assert not main & stable


def test_missing_cells_stay_missing(tmp_path):
    base = tmp_path / "base"
    save(base / "records/a--full.json", {"record": {"status": "inconclusive"}})
    selected, _, errors, _ = audit.merge_records({"a--full", "b--full"}, base)
    assert set(selected) == {"a--full"}
    assert errors == []


@pytest.mark.parametrize(
    "original",
    [
        stopped(money=False),
        {"record": {"status": "reviewed"}},
        {"record": {"status": "needs_clarification"}},
    ],
)
def test_quality_and_unknown_transport_outcomes_cannot_be_replaced(tmp_path, original):
    base, continuation = tmp_path / "base", tmp_path / "cont"
    save(base / "records/a--full.json", original)
    save(continuation / "records/a--full.json", {"record": {"status": "reviewed"}})
    selected, _, errors, _ = audit.merge_records({"a--full"}, base, continuation)
    assert selected["a--full"][0] == original
    assert errors == ["QUALITY_OUTCOME_REPLACED:a--full"]


def test_only_explicit_accounted_money_limit_can_continue(tmp_path):
    base, continuation = tmp_path / "base", tmp_path / "cont"
    old, new = stopped(), {"record": {"status": "reviewed"}}
    save(base / "records/a--full.json", old)
    save(continuation / "records/a--full.json", new)
    selected, provenance, errors, documents = audit.merge_records({"a--full"}, base, continuation)
    assert errors == [] and selected["a--full"][0] == new
    assert provenance[0]["superseded_digest"] == audit.digest_json(old)
    assert len(documents) == 2


def test_unplanned_record_is_not_silently_scored(tmp_path):
    base = tmp_path / "base"
    save(base / "records/unplanned--full.json", {})
    selected, _, errors, _ = audit.merge_records({"a--full"}, base)
    assert selected == {}
    assert errors == ["UNPLANNED_RECORD:unplanned--full.json"]


def test_initial_continuation_keeps_existing_program_and_prefix(tmp_path):
    base, continuation = tmp_path / "base", tmp_path / "cont"
    old = {
        **stopped(),
        "task_digest": "task",
        "sql": "SELECT 1",
        "plan": None,
        "plan_error": "LIVE_MONEY_RESERVATION_LIMIT",
        "http_events": [{"event_id": "lane:1"}],
    }
    new = {
        **old,
        "plan": {"reference": "[[1]]"},
        "http_events": old["http_events"] + [{"event_id": "lane:2"}],
        "infrastructure_continuation": {"original_digest": audit.digest_json(old)},
    }
    save(base / "initial/a.json", old)
    save(continuation / "initial/a.json", new)
    _, errors, _ = audit.initial_caches(base, continuation)
    assert not errors
    new["sql"] = "SELECT 2"
    save(continuation / "initial/a.json", new)
    _, errors, _ = audit.initial_caches(base, continuation)
    assert "INITIAL_VALUE_CHANGED:a.json:sql" in errors


def test_successful_initial_plan_cannot_be_replaced(tmp_path):
    base, continuation = tmp_path / "base", tmp_path / "cont"
    old = {
        "task_digest": "task",
        "sql": "SELECT 1",
        "plan": {"reference": "[[1]]"},
        "http_events": [],
    }
    new = {**old, "plan": {"reference": "[[2]]"}}
    save(base / "initial/a.json", old)
    save(continuation / "initial/a.json", new)
    _, errors, _ = audit.initial_caches(base, continuation)
    assert "INITIAL_QUALITY_CACHE_REPLACED:a.json" in errors
    assert "INITIAL_VALUE_CHANGED:a.json:plan" in errors


def test_duplicate_events_are_once_but_conflicts_rejected():
    event = {"event_id": "budget-lane-0:1", "usage": {"total_tokens": 12}, "status": "completed"}
    events, errors = audit.unique_events([{"http_events": [event]}, {"http_events": [event]}])
    assert len(events) == 1 and not errors
    _, errors = audit.unique_events(
        [{"http_events": [event, {**event, "usage": {"total_tokens": 13}}]}]
    )
    assert errors == ["EVENT_ID_CONFLICT:budget-lane-0:1"]


def test_current_code_and_frozen_archive_are_both_checked(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    source = tmp_path / "frozen.py"
    source.write_text("unchanged", encoding="utf-8")
    database = tmp_path / "test.sqlite"
    database.write_bytes(b"sample database bytes")
    with zipfile.ZipFile(base / "frozen-source.zip", "w") as archive:
        archive.write(source, "frozen.py")
    manifest = {
        "databases": {"test": {"path": str(database), "sha256": audit.file_digest(database)}}
    }
    protocol = {
        "source_hashes": {"frozen.py": audit.file_digest(source)},
        "public_manifest_digest": audit.digest_json(manifest),
    }
    assert not audit.verify_sources(tmp_path, base, protocol, manifest)["errors"]
    source.write_text("changed", encoding="utf-8")
    database.write_bytes(b"changed")
    errors = audit.verify_sources(tmp_path, base, protocol, manifest)["errors"]
    assert "CURRENT_SOURCE_DIFFERS:frozen.py" in errors
    assert "ORIGINAL_DATABASE_CHANGED:test" in errors


def test_zero_deliveries_is_not_perfect_conditional_accuracy():
    row = {
        "kind": "public",
        "method": "full",
        "status": "inconclusive",
        "cluster_id": "task1",
        "elapsed_seconds": 1,
        "score": {"delivered": False, "correct_delivery": False, "wrong_release": False},
    }
    summary = audit.outcome_summary([row])["public"]["full"]
    assert summary["correct_delivery"] == 0
    assert summary["wrong_fraction_of_deliveries"] is None
    assert summary["wrong_fraction_of_all"] == 0


def test_ledger_pointer_cannot_leave_directory(tmp_path):
    save(tmp_path / "budget-lane-0.current.json", {"current": "../elsewhere.json"})
    with pytest.raises(ValueError, match="POINTER_OUTSIDE"):
        audit._latest_ledger(tmp_path, 0)


def test_latency_distribution_keeps_empty_distinct_from_fast():
    assert audit.distribution([])["median"] is None
    assert audit.distribution([1, 3])["median"] == 2
    assert audit.distribution([1, 3])["p95"] == pytest.approx(2.9)


def test_stability_repeats_are_bound_and_compared_without_expanding_main(tmp_path):
    protocol = {
        "stability_case_ids": ["task-a"],
        "stability_additional_repetitions": 2,
        "cases": [
            {
                "case_id": "task-a",
                "task_digest": "task",
                "source_digest": "source",
                "cluster_id": "task-a",
            }
        ],
    }

    def document(case_id, correct):
        return {
            "task_digest": "task",
            "source_digest": "source",
            "http_events": [],
            "record": {"status": "reviewed", "publication": {"rows": [[1 if correct else 2]]}},
            "row": {
                "case_id": case_id,
                "method": "full",
                "kind": "public",
                "cluster_id": "task-a",
                "status": "reviewed",
                "elapsed_seconds": 2,
                "usage": audit.summarize_usage([]),
                "http_events": [],
                "shared_initial_http_events": [],
                "shared_initial_usage": audit.summarize_usage([]),
                "score": {
                    "correct_delivery": correct,
                    "delivered": True,
                    "wrong_release": not correct,
                },
            },
        }

    first = document("task-a", True)
    for rep in (2, 3):
        name = f"task-a--rep{rep}"
        save(tmp_path / "records" / (name + "--full.json"), document(name, rep == 3))
        save(tmp_path / "initial" / (name + ".json"), {"task_digest": "task", "http_events": []})
    summary, _ = audit.inspect_stability(
        protocol, {"task-a--full": (first, Path("main"))}, tmp_path, None
    )
    assert summary["status"] == "complete"
    assert summary["available"] == 2
    assert summary["per_original_task"][0]["observed_runs"] == 3
    assert summary["per_original_task"][0]["delivery_correctness_varies"] is True
    assert summary["outcomes"]["public"]["full"]["available"] == 2
    changed = document("task-a--rep2", False)
    changed["task_digest"] = "another-task"
    save(tmp_path / "records/task-a--rep2--full.json", changed)
    summary, _ = audit.inspect_stability(protocol, {}, tmp_path, None)
    assert "STABILITY_TASK_SOURCE_BINDING:task-a--rep2--full" in summary["errors"]


def test_driver_revision_preserves_old_protocol_code_and_outputs(tmp_path):
    output = tmp_path / "run"
    driver = tmp_path / "tools/continue_expanded.py"
    driver.parent.mkdir()
    driver.write_text("new explicitly reviewed driver", encoding="utf-8")
    new_hash = audit.file_digest(driver)
    old_hash = audit.hashlib.sha256(b"old driver").hexdigest()
    protocol = {"frozen": "protocol"}
    original = {"base_protocol_digest": audit.digest_json(protocol), "driver_sha256": old_hash}
    save(output / "continuation-protocol.json", original)
    (output / "drivers").mkdir()
    (output / "drivers" / (old_hash + ".py")).write_bytes(b"old driver")
    (output / "drivers" / (new_hash + ".py")).write_bytes(driver.read_bytes())
    record_path = output / "records/a--full.json"
    save(record_path, {"quality_failure": "preserved"})
    request = tmp_path / "manual-review.json"
    save(request, {"reviewed_by": "operator"})
    revision = {
        "previous_sha256": old_hash,
        "driver_sha256": new_hash,
        "original_protocol_digest": audit.digest_json(original),
        "review_request": str(request),
        "review_request_digest": audit.digest_json(audit.read(request)),
        "preserved_records": {"records/a--full.json": audit.file_digest(record_path)},
    }
    save(output / "driver-revision-1.json", revision)
    save(
        output / "driver-history.json",
        {"history": [{"path": "driver-revision-1.json", "digest": audit.digest_json(revision)}]},
    )
    result = audit.driver_history_audit(tmp_path, output, protocol)
    assert not result["errors"] and result["reviewed_revisions"] == 1
    assert audit.read(output / "continuation-protocol.json") == original
    save(record_path, {"quality_failure": "replaced"})
    assert (
        "DRIVER_PRIOR_RECORD_CHANGED:1:records/a--full.json"
        in audit.driver_history_audit(tmp_path, output, protocol)["errors"]
    )


def test_manual_transport_review_cannot_refund_unknown_or_change_failure(tmp_path):
    directory = tmp_path / "budget"
    authorization = {"cumulative_limit_cny": 120}
    save(directory / "authorization.json", authorization)
    prior = {
        "money_limit_cny": 28,
        "stopped_reason": "TRANSPORT_ERROR",
        "reservation_ledger": [
            {"attempt": 1, "status": "usage_unknown", "reserved_upper_cny": 0.025}
        ],
    }
    prior_path = directory / "budget-lane-0.json"
    save(prior_path, {"budget": prior})
    event = {
        "event_id": "budget-lane-0:1",
        "attempt": 1,
        "status": "failed",
        "error_code": "GRAPH_QWEN_TRANSPORT_READTIMEOUT",
        "usage_complete": False,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }
    record = {"budget_end": prior, "http_events": [event]}
    record_path = tmp_path / "records/a--full.json"
    save(record_path, record)
    item = {
        "lane": 0,
        "attempt": 1,
        "event_id": event["event_id"],
        "record_path": str(record_path),
        "record_digest": audit.digest_json(record),
        "journal_path": str(prior_path),
        "journal_digest": audit.digest_json(prior),
        "reserved_upper_cny": 0.025,
    }
    request = {
        "budget_authorization_digest": audit.digest_json(authorization),
        "reviewed_by": "operator",
        "decision": "preserve_failed_cells_and_full_unknown_reservations_continue_other_cells",
        "events": [item],
    }
    request_path = tmp_path / "review.json"
    save(request_path, request)
    incident = {
        "kind": "manual_transport_review",
        "lane": 0,
        "prior": prior_path.name,
        "prior_digest": audit.digest_json(prior),
        "snapshot_digest": audit.digest_json(prior),
        "review_request": str(request_path),
        "review_request_digest": audit.digest_json(request),
        "reviewed_events": [item],
        "action": "continue_different_cells_without_retry_or_refund",
    }
    save(directory / "budget-lane-0.incident-1.json", incident)
    save(
        directory / "budget-lane-0.current.json",
        {
            "current": "budget-lane-0.continuation-1.json",
            "history": [
                {"incident": "budget-lane-0.incident-1.json", "digest": audit.digest_json(incident)}
            ],
        },
    )
    following = {**prior, "stopped_reason": None}
    save(directory / "budget-lane-0.continuation-1.json", {"budget": following})
    result = audit.ledger_history_audit(directory, 0)
    assert not result["errors"] and len(result["reviewed_unknowns"]) == 1
    following["reservation_ledger"] = []
    save(directory / "budget-lane-0.continuation-1.json", {"budget": following})
    assert (
        "LEDGER_INCIDENT_PREFIX_CHANGED:0:1" in audit.ledger_history_audit(directory, 0)["errors"]
    )


def test_known_tokens_on_failed_event_are_not_unknown_or_success(tmp_path):
    save(tmp_path / "authorization.json", {"cumulative_limit_cny": 60, "prior": []})
    for lane in range(4):
        active = lane == 0
        entry = {
            "attempt": 1,
            "status": "usage_complete",
            "reserved_upper_cny": 0.02,
            "estimated_reported_cny": 0.0035727,
            "prompt_tokens": 2532,
            "completion_tokens": 573,
            "total_tokens": 3105,
        }
        snapshot = {
            "http_attempts": int(active),
            "complete_count": int(active),
            "unknown_count": 0,
            "in_flight_count": 0,
            "reserved_upper_cny": 0.02 if active else 0,
            "estimated_reported_cny": 0.0035727 if active else 0,
            "reported_cost_complete": True,
            "reported_prompt_tokens": 2532 if active else 0,
            "reported_completion_tokens": 573 if active else 0,
            "reported_total_tokens": 3105 if active else 0,
            "stopped_reason": "CLIENT_ERROR" if active else None,
            "reservation_ledger": [entry] if active else [],
        }
        save(tmp_path / f"budget-lane-{lane}.json", {"budget": snapshot})
    event = {
        "event_id": "budget-lane-0:1",
        "status": "failed",
        "usage_complete": False,
        "error_code": "GRAPH_UNEXPECTED_CLIENT_ERROR",
        "usage": {"prompt_tokens": 2532, "completion_tokens": 573, "total_tokens": 3105},
    }
    result = audit.ledger_audit(tmp_path, {event["event_id"]: event})
    assert not result["errors"] and not result["unknown_reservations"]
    assert result["known_cost_is_incomplete"] is False
    known_failure = result["accounted_failed_events"][0]
    assert known_failure["event_status"] == "failed"
    assert known_failure["event_usage_complete_flag"] is False
    assert known_failure["estimated_reported_cny"] == pytest.approx(0.0035727)


@pytest.mark.parametrize("tamper", [None, "flag", "usage", "refund"])
def test_accounted_client_review_preserves_failure_and_exact_tokens(tmp_path, tamper):
    directory = tmp_path / "budget"
    authorization = {"cumulative_limit_cny": 120}
    save(directory / "authorization.json", authorization)
    usage = {"prompt_tokens": 2532, "completion_tokens": 573, "total_tokens": 3105}
    charge = {"attempt": 1, "status": "usage_complete", "reserved_upper_cny": 0.02, **usage}
    start = {"reservation_ledger": [], "unknown_count": 0}
    prior = {
        "stopped_reason": "CLIENT_ERROR",
        "http_attempts": 1,
        "reservation_ledger": [charge],
        "unknown_count": 0,
    }
    prior_path = directory / "budget-lane-0.json"
    save(prior_path, {"budget": prior})
    event = {
        "event_id": "budget-lane-0:1",
        "attempt": 1,
        "http_status": 200,
        "status": "failed",
        "error_code": "GRAPH_UNEXPECTED_CLIENT_ERROR",
        "usage_complete": False,
        "usage": dict(usage),
    }
    if tamper == "flag":
        event["usage_complete"] = True
    elif tamper == "usage":
        event["usage"]["total_tokens"] = 3106
    elif tamper == "refund":
        charge["status"] = "usage_unknown"
    document = {"budget_start": start, "budget_end": prior, "http_events": [event]}
    record_path = tmp_path / "failure.json"
    save(record_path, document)
    item = {
        "lane": 0,
        "attempt": 1,
        "event_id": event["event_id"],
        "record_path": str(record_path),
        "record_digest": audit.digest_json(document),
        "journal_path": str(prior_path),
        "journal_digest": audit.digest_json(prior),
        "reserved_upper_cny": 0.02,
    }
    request = {
        "format": "manual-accounted-client-review-v1",
        "budget_authorization_digest": audit.digest_json(authorization),
        "decision": "preserve_failed_cells_and_accounted_usage_continue_other_cells",
        "reviewed_by": "operator",
        "events": [item],
    }
    request_path = tmp_path / "review.json"
    save(request_path, request)
    incident = {
        "kind": "manual_accounted_client_review",
        "lane": 0,
        "action": "continue_different_cells_without_retry_or_refund",
        "snapshot_digest": audit.digest_json(prior),
        "review_request": str(request_path),
        "review_request_digest": audit.digest_json(request),
        "reviewed_events": [item],
    }
    result = audit.accounted_client_review_audit(directory, 0, prior_path, prior, incident)
    if tamper:
        assert result["errors"]
    else:
        assert not result["errors"]
        assert (
            result["reviewed_known_failures"][0]["classification"]
            == "failed_response_with_known_accounted_tokens"
        )
