"""Continuation accounting tests use temporary offline ledgers, never live APIs."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget
from dataverify.runtime.campaign_budget import CampaignBudget
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget
from dataverify.runtime.reporting import write_json


def module():
    path = Path(__file__).resolve().parents[1] / "tools/continue_expanded.py"
    spec = importlib.util.spec_from_file_location("continue_driver_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def account(budget, completion=10):
    attempt = budget.reserve(100)
    usage = {"prompt_tokens": 10, "completion_tokens": completion, "total_tokens": 10 + completion}
    budget.record_tokens(usage)
    return attempt, usage


def old_unknown():
    budget = CampaignBudget(
        limit=5000,
        max_input_chars=30_000_000,
        max_reported_tokens=15_000_000,
        money_limit_cny=13,
        max_output_tokens=1800,
    )
    account(budget)
    budget.reserve(200)
    budget.stop("TRANSPORT_ERROR")
    return budget.snapshot()


def test_unknown_extension_preserves_reservation_and_unknown_status():
    driver = module()
    old = old_unknown()
    with pytest.raises(ValueError):
        driver.replay(old, extend=True)
    rebuilt = driver.replay(old, extend=True, quarantined={2})
    new = rebuilt.snapshot()
    assert new["money_limit_cny"] == 28 and new["stopped_reason"] is None
    assert new["unknown_count"] == 1 and new["reported_cost_complete"] is False
    assert new["reserved_upper_cny"] == old["reserved_upper_cny"]
    assert new["reservation_ledger"] == old["reservation_ledger"]
    account(rebuilt)
    assert rebuilt.snapshot()["http_attempts"] == 3
    assert rebuilt.snapshot()["unknown_count"] == 1
    assert driver.replay(rebuilt.snapshot(), quarantined={2}).snapshot() == rebuilt.snapshot()


def test_new_unknown_is_not_auto_quarantined():
    driver = module()
    rebuilt = driver.replay(old_unknown(), extend=True, quarantined={2})
    rebuilt.reserve(200)
    rebuilt.stop("TRANSPORT_ERROR")
    with pytest.raises(ValueError):
        driver.replay(rebuilt.snapshot(), quarantined={2})


def test_accounted_truncation_after_unknown_prefix_retains_all_costs(tmp_path):
    driver = module()
    budget = JournaledBudget(
        driver.replay(old_unknown(), extend=True, quarantined={2}), tmp_path / "budget-lane-0.json"
    )
    start = budget.snapshot()
    attempt, usage = account(budget, 1800)
    budget.stop("PROVIDER_ERROR")
    report = {
        "budget_start": start,
        "budget_end": budget.snapshot(),
        "http_events": [
            {
                "attempt": attempt,
                "http_status": 200,
                "usage_complete": True,
                "usage": usage,
                "status": "failed",
                "error_code": "GRAPH_QWEN_RESPONSE_TRUNCATED",
            }
        ],
    }
    old_bytes = budget.path.read_bytes()
    later = driver.rollover_truncation(tmp_path, 0, budget, report, {2})
    assert budget.path.read_bytes() == old_bytes
    assert later.snapshot()["unknown_count"] == 1
    assert later.snapshot()["reservation_ledger"] == budget.snapshot()["reservation_ledger"]
    assert later.snapshot()["reserved_upper_cny"] == budget.snapshot()["reserved_upper_cny"]
    assert driver.extended_lane_path(tmp_path, 0, {2})[0] == later.path


def test_only_exact_transport_evidence_can_be_quarantined(tmp_path):
    driver = module()
    snapshot = old_unknown()
    folder = tmp_path / "records"
    folder.mkdir()
    event = {
        "event_id": "budget-lane-0:2",
        "attempt": 2,
        "status": "failed",
        "usage_complete": False,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "error_code": "GRAPH_QWEN_TRANSPORT_READTIMEOUT",
    }
    write_json(folder / "failed.json", {"http_events": [event]})
    evidence = driver.quarantine_evidence(tmp_path, 0, snapshot)
    assert len(evidence) == 1 and evidence[0]["reserved_upper_cny"] > 0
    event["error_code"] = "GRAPH_QWEN_HTTP_500"
    write_json(folder / "failed.json", {"http_events": [event]})
    with pytest.raises(ValueError, match="NOT_REVIEWED_TRANSPORT"):
        driver.quarantine_evidence(tmp_path, 0, snapshot)


def test_transport_failure_never_qualifies_as_quality_retry():
    driver = module()
    record = {"budget_end": old_unknown(), "record": {"reason": "GRAPH_QWEN_TRANSPORT_READTIMEOUT"}}
    assert driver.money_interrupted(record) is False


def test_explicit_money_stop_qualifies_but_ordinary_quality_failure_does_not():
    driver = module()
    record = {
        "budget_end": {"stopped_reason": "MONEY_RESERVATION_LIMIT", "reported_cost_complete": True},
        "record": {"reason": "LIVE_MONEY_RESERVATION_LIMIT"},
    }
    assert driver.money_interrupted(record)
    record["record"]["reason"] = "REFERENCE_EVALUATION"
    assert not driver.money_interrupted(record)


def test_unknown_history_and_caps_cannot_be_changed():
    driver = module()
    old = old_unknown()
    bad = deepcopy(old)
    bad["reserved_upper_cny"] = 0
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        driver.replay(bad, extend=True, quarantined={2})
    bad = deepcopy(old)
    bad["max_output_tokens"] = 1801
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        driver.replay(bad, extend=True, quarantined={2})


def test_allocation_is_once_preserves_all_old_files_and_supports_stability(tmp_path):
    driver = module()
    campaign = tmp_path / "campaign"
    base = campaign / "runs/main"
    (base / "records").mkdir(parents=True)
    protocol = {"phase": "test", "source_hashes": {"fixed.py": "known-hash"}}
    write_json(base / "protocol.json", protocol)
    previous = []
    for allowance in (5, 3):
        path = tmp_path / f"prior{allowance}.json"
        journal = JournaledBudget(MonetaryRequestBudget(money_limit_cny=allowance), path)
        previous.append({"path": str(path), "snapshot": journal.snapshot()})
    write_json(
        campaign / "authorization.json",
        {"cumulative_limit_cny": 60, "workers": 4, "prior": previous},
    )
    old_files = {}
    for lane in range(4):
        path = campaign / f"budget-lane-{lane}.json"
        budget = CampaignBudget(
            limit=5000,
            max_input_chars=30_000_000,
            max_reported_tokens=15_000_000,
            money_limit_cny=13,
            max_output_tokens=1800,
        )
        journal = JournaledBudget(budget, path)
        account(journal)
        if lane == 0:
            journal.reserve(100)
            journal.stop("TRANSPORT_ERROR")
            write_json(
                base / "records/failed.json",
                {
                    "http_events": [
                        {
                            "event_id": "budget-lane-0:2",
                            "attempt": 2,
                            "status": "failed",
                            "usage_complete": False,
                            "usage": {
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None,
                            },
                            "error_code": "GRAPH_QWEN_TRANSPORT_CONNECTERROR",
                        }
                    ]
                },
            )
        old_files[path] = path.read_bytes()
    with pytest.raises(ValueError, match="APPROVAL"):
        driver.allocate_extension(campaign, base, approved_limit=60)
    directory, receipt = driver.allocate_extension(campaign, base, approved_limit=120)
    assert (
        sum(
            driver.read_budget(directory / f"budget-lane-{lane}.json")["money_limit_cny"]
            for lane in range(4)
        )
        + 8
        == 120
    )
    assert all(path.read_bytes() == content for path, content in old_files.items())
    assert driver.load_lane(directory, 0, receipt).snapshot()["unknown_count"] == 1
    stability = campaign / "runs/stability"
    stability.mkdir()
    write_json(stability / "protocol.json", {**protocol, "phase": "stability"})
    directory2, receipt2 = driver.allocate_extension(campaign, stability, approved_limit=120)
    assert directory2 == directory and receipt2 == receipt


def test_stability_plan_only_runs_two_full_repetitions(monkeypatch):
    driver = module()
    bundle = driver.controlled_bundles()[0]
    monkeypatch.setattr(driver, "public_bundles", lambda *args: [])
    monkeypatch.setattr(driver, "controlled_bundles", lambda: [bundle])
    protocol = {
        "phase": "stability",
        "stability_case_ids": [bundle.case_id],
        "ablation_case_ids": [bundle.case_id],
    }
    entries = driver.bundles_and_methods({}, protocol)
    assert len(entries) == 2
    assert {entry[1].case_id for entry in entries} == {
        bundle.case_id + "--rep2",
        bundle.case_id + "--rep3",
    }
    assert all(entry[2] == ["full"] for entry in entries)


def review_fixture(tmp_path):
    driver = module()
    directory = tmp_path / "budgets"
    directory.mkdir()
    budget = JournaledBudget(
        driver.replay(old_unknown(), extend=True, quarantined={2}), directory / "budget-lane-0.json"
    )
    budget.reserve(300)
    budget.stop("TRANSPORT_ERROR")
    event = {
        "event_id": "budget-lane-0:3",
        "attempt": 3,
        "status": "failed",
        "usage_complete": False,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "error_code": "GRAPH_QWEN_TRANSPORT_READTIMEOUT",
    }
    record = {
        "budget_end": budget.snapshot(),
        "http_events": [event],
        "row": {"status": "inconclusive"},
    }
    record_path = tmp_path / "record.json"
    write_json(record_path, record)
    receipt = {
        "parents": [
            {"lane": i, "quarantined_unknowns": [{"attempt": 2}] if i == 0 else []}
            for i in range(4)
        ]
    }
    request = {
        "format": "manual-transport-quarantine-v1",
        "reviewed_by": "explicit test operator",
        "budget_authorization_digest": driver.digest_json(receipt),
        "decision": "preserve_failed_cells_and_full_unknown_reservations_continue_other_cells",
        "events": [
            {
                "lane": 0,
                "attempt": 3,
                "event_id": "budget-lane-0:3",
                "record_path": str(record_path.resolve()),
                "record_digest": driver.digest_json(record),
                "journal_path": str(budget.path),
                "journal_digest": driver.digest_json(budget.snapshot()),
                "reserved_upper_cny": budget.snapshot()["reservation_ledger"][-1][
                    "reserved_upper_cny"
                ],
            }
        ],
    }
    request_path = tmp_path / "review.json"
    write_json(request_path, request)
    return driver, directory, budget, receipt, request_path, record_path


def test_explicit_review_preserves_all_history_and_is_idempotent(tmp_path):
    driver, directory, old, receipt, request, record = review_fixture(tmp_path)
    original = {p: p.read_bytes() for p in (old.path, record, request)}
    driver.apply_transport_review(directory, receipt, request)
    assert not (directory / "budget-lane-0.current.json").exists()
    driver.apply_transport_review(directory, receipt, request, apply=True)
    allowed = driver.lane_quarantines(directory, 0, receipt)
    assert allowed == {2, 3}
    path, history = driver.extended_lane_path(directory, 0, {2})
    snapshot = driver.read_budget(path)
    assert snapshot["unknown_count"] == 2 and snapshot["reported_cost_complete"] is False
    assert snapshot["reservation_ledger"] == old.snapshot()["reservation_ledger"]
    assert snapshot["reserved_upper_cny"] == old.snapshot()["reserved_upper_cny"]
    assert snapshot["stopped_reason"] is None and len(history) == 1
    driver.apply_transport_review(directory, receipt, request, apply=True)
    assert driver.extended_lane_path(directory, 0, {2})[1] == history
    assert all(path.read_bytes() == data for path, data in original.items())
    next_budget = driver.replay(snapshot, quarantined=allowed)
    next_budget.reserve(400)
    next_budget.stop("TRANSPORT_ERROR")
    with pytest.raises(ValueError):
        driver.replay(next_budget.snapshot(), quarantined=allowed)


@pytest.mark.parametrize(
    "field", ["record_digest", "journal_digest", "event_id", "reserved_upper_cny"]
)
def test_review_rejects_changed_bindings(tmp_path, field):
    driver, directory, old, receipt, path, _ = review_fixture(tmp_path)
    request = driver.read(path)
    request["events"][0][field] = "changed"
    write_json(path, request)
    with pytest.raises(ValueError):
        driver.apply_transport_review(directory, receipt, path, apply=True)
    assert not (directory / "budget-lane-0.current.json").exists()


def test_review_rejects_new_unknown_not_in_request(tmp_path):
    driver, directory, old, receipt, path, _ = review_fixture(tmp_path)
    request = driver.read(path)
    request["events"][0]["attempt"] = 2
    write_json(path, request)
    with pytest.raises(ValueError):
        driver.apply_transport_review(directory, receipt, path, apply=True)


def test_driver_revision_is_append_only_and_freezes_existing_outcomes(tmp_path):
    driver = module()
    output = tmp_path / "output"
    (output / "drivers").mkdir(parents=True)
    (output / "records").mkdir()
    old_source = b"previous driver"
    old_hash = driver.hashlib.sha256(old_source).hexdigest()
    current_hash = driver.hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest()
    (output / "drivers" / (old_hash + ".py")).write_bytes(old_source)
    binding = {"driver_sha256": current_hash, "fixed": "same"}
    old_protocol = {**binding, "driver_sha256": old_hash}
    write_json(output / "continuation-protocol.json", old_protocol)
    write_json(output / "records/completed.json", {"result": "inconclusive"})
    request = tmp_path / "review.json"
    write_json(request, {"approved": "only exact transport events"})
    with pytest.raises(ValueError, match="EXPLICIT_REVIEW"):
        driver.bind_driver_revision(output, binding)
    driver.bind_driver_revision(output, binding, request, apply=True)
    assert driver.read(output / "continuation-protocol.json") == old_protocol
    driver.bind_driver_revision(output, binding)
    write_json(output / "records/completed.json", {"result": "succeeded"})
    with pytest.raises(ValueError, match="PRIOR_RESULT_CHANGED"):
        driver.bind_driver_revision(output, binding)


def client_review_fixture(tmp_path):
    driver = module()
    directory = tmp_path / "budgets"
    directory.mkdir()
    budget = JournaledBudget(
        driver.replay(old_unknown(), extend=True, quarantined={2}), directory / "budget-lane-0.json"
    )
    start = budget.snapshot()
    attempt = budget.reserve(3000)
    usage = {"prompt_tokens": 2532, "completion_tokens": 573, "total_tokens": 3105}
    budget.record_tokens(usage)
    budget.stop("CLIENT_ERROR")
    event = {
        "event_id": f"budget-lane-0:{attempt}",
        "attempt": attempt,
        "status": "failed",
        "usage_complete": False,
        "usage": usage,
        "http_status": 200,
        "error_code": "GRAPH_UNEXPECTED_CLIENT_ERROR",
    }
    record = {
        "budget_start": start,
        "budget_end": budget.snapshot(),
        "http_events": [event],
        "row": {"status": "inconclusive"},
    }
    record_path = tmp_path / "record.json"
    write_json(record_path, record)
    receipt = {
        "parents": [
            {"lane": i, "quarantined_unknowns": [{"attempt": 2}] if i == 0 else []}
            for i in range(4)
        ]
    }
    request = {
        "format": "manual-accounted-client-review-v1",
        "reviewed_by": "explicit test operator",
        "budget_authorization_digest": driver.digest_json(receipt),
        "decision": "preserve_failed_cells_and_accounted_usage_continue_other_cells",
        "events": [
            {
                "lane": 0,
                "attempt": attempt,
                "event_id": event["event_id"],
                "record_path": str(record_path.resolve()),
                "record_digest": driver.digest_json(record),
                "journal_path": str(budget.path),
                "journal_digest": driver.digest_json(budget.snapshot()),
                "reserved_upper_cny": budget.snapshot()["reservation_ledger"][-1][
                    "reserved_upper_cny"
                ],
            }
        ],
    }
    request_path = tmp_path / "review.json"
    write_json(request_path, request)
    return driver, directory, budget, receipt, request_path, record_path


def test_accounted_client_review_keeps_raw_failure_and_every_charge(tmp_path):
    driver, directory, old, receipt, request, record = client_review_fixture(tmp_path)
    preserved = {path: path.read_bytes() for path in (old.path, request, record)}
    driver.apply_accounted_client_review(directory, receipt, request)
    assert not (directory / "budget-lane-0.current.json").exists()
    driver.apply_accounted_client_review(directory, receipt, request, apply=True)
    path, history = driver.extended_lane_path(directory, 0, {2})
    assert driver.lane_quarantines(directory, 0, receipt) == {2}
    snapshot = driver.read_budget(path)
    assert snapshot == {**old.snapshot(), "stopped_reason": None}
    assert snapshot["unknown_count"] == 1 and snapshot["reported_cost_complete"] is False
    assert snapshot["reservation_ledger"][-1]["estimated_reported_cny"] == 0.0035727
    driver.apply_accounted_client_review(directory, receipt, request, apply=True)
    assert driver.extended_lane_path(directory, 0, {2})[1] == history
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize(
    "change",
    [
        "wrong_usage",
        "missing_usage",
        "http_failure",
        "wrong_code",
        "missing_event",
        "changed_prefix",
    ],
)
def test_client_review_rejects_incomplete_or_mismatched_accounting(tmp_path, change):
    driver, directory, old, receipt, request_path, record_path = client_review_fixture(tmp_path)
    record = driver.read(record_path)
    if change == "wrong_usage":
        record["http_events"][0]["usage"]["prompt_tokens"] = 2533
    elif change == "missing_usage":
        record["http_events"][0]["usage"]["prompt_tokens"] = None
    elif change == "http_failure":
        record["http_events"][0]["http_status"] = 500
    elif change == "wrong_code":
        record["http_events"][0]["error_code"] = "GRAPH_QWEN_HTTP_500"
    elif change == "missing_event":
        record["http_events"] = []
    else:
        record["budget_start"]["reservation_ledger"][0]["prompt_chars"] += 1
    write_json(record_path, record)
    request = driver.read(request_path)
    request["events"][0]["record_digest"] = driver.digest_json(record)
    write_json(request_path, request)
    with pytest.raises(ValueError):
        driver.apply_accounted_client_review(directory, receipt, request_path, apply=True)
    assert not (directory / "budget-lane-0.current.json").exists()
