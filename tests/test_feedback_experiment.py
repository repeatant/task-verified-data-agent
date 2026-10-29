"""No-network checks for crossed experiments and incremental budget safety."""

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)
pytest.importorskip("hypothesis")

from dataverify import feedback_experiment as experiment
from dataverify.probe_cases import reference_result
from dataverify.probe_cost_budget import MonetaryRequestBudget


def completed_budget():
    budget = MonetaryRequestBudget()
    budget.reserve(100)
    budget.record_tokens({"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140})
    return budget


def test_continuation_preserves_every_spending_counter_and_only_changes_call_cap():
    snapshot = completed_budget().snapshot()
    restored = experiment.restore_budget(snapshot, max_extra_requests=24).snapshot()
    assert restored["http_attempt_limit"] == 25
    for key, value in snapshot.items():
        if key != "http_attempt_limit":
            assert restored[key] == value


@pytest.mark.parametrize("field,value", [("reported_cost_complete", False), ("stopped_reason", "ERROR"),
    ("unknown_count", 1), ("in_flight_count", 1), ("unmatched_usage_events", 1),
    ("reserved_upper_cny", 0), ("reported_total_tokens", 0), ("http_attempts", 0)])
def test_bad_prior_ledgers_cannot_reset_the_allowance(field, value):
    snapshot = completed_budget().snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError):
        experiment.restore_budget(snapshot)


def test_new_request_cap_is_enforced_without_erasing_complete_global_usage():
    budget = completed_budget()
    wrapper = experiment.IncrementalAllowance(budget, 1, .8)
    wrapper.reserve(100)
    wrapper.record_tokens({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
    with pytest.raises(Exception, match="ADDITIONAL_REQUEST_LIMIT"):
        wrapper.reserve(100)
    assert budget.snapshot()["http_attempts"] == 2
    assert budget.snapshot()["reported_cost_complete"]
    assert not budget.snapshot()["stopped_reason"]


def test_additional_money_reservation_blocks_before_the_post():
    budget = completed_budget()
    before = budget.snapshot()
    wrapper = experiment.IncrementalAllowance(budget, 24, .001)
    with pytest.raises(Exception, match="ADDITIONAL_RESERVATION_LIMIT"):
        wrapper.reserve(100)
    assert budget.snapshot() == before


@pytest.mark.parametrize("value", [0, 37, True, 1.5])
def test_invalid_extra_attempt_limit_rejected(value):
    with pytest.raises(ValueError):
        experiment.IncrementalAllowance(completed_budget(), value, .8)


@pytest.fixture
def report_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "reports").mkdir(parents=True)
    monkeypatch.setattr(experiment, "PROJECT_ROOT", root)
    monkeypatch.setattr(experiment, "source_fingerprints", lambda: {"frozen.py": "abc"})
    return root / "reports"


def fake_record(case):
    return {"publication": {"status": "verified", "task_id": case.task.task_id,
                             "verified_sql": case.reference_sql, "rows": reference_result(case)},
            "http_events": [], "actions": [], "source_unchanged": True, "llm_calls": 1}


def test_ablation_grid_crosses_independent_features_and_separates_tracks(report_root, monkeypatch):
    seen = []

    async def run(task, data, **kwargs):
        seen.append(kwargs)
        case = experiment.get_case("d10" if task.output_kind == "orders" else "d04")
        return fake_record(case)

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "ablation",
        controlled_case_ids=("d10",), natural_case_ids=("d04",), seeds=(53, 71)))
    assert len(result["rows"]) == 12
    assert len(result["pairs"]) == 3
    assert {item["feedback_policy"] for item in seen} == {"legacy_v1", "capability_only", "guidance_only", "capability_v2"}
    assert all(item["mode"] == "compact" and item["max_attempts"] == 3 for item in seen)
    assert {r["seed"] for r in result["rows"] if r["track"] == "natural"} == {53}
    assert result["all_cells_completed"] and result["audit_ledger_complete"]
    assert not result["real_model_used"]
    assert all(g["real_model_correct_tasks"] is None for g in result["summary"]["groups"].values())


def test_strategy_changes_only_mode_and_preserves_max_attempts(report_root, monkeypatch):
    seen = []

    async def run(task, data, **kwargs):
        seen.append(kwargs)
        return fake_record(experiment.get_case("d10"))

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "strategy", stage="strategy",
        controlled_case_ids=("d10",), natural_case_ids=()))
    assert len(result["rows"]) == 3
    assert {x["mode"] for x in seen} == {"full", "compact", "adaptive"}
    assert all(x["feedback_policy"] == "capability_v2" and x["max_attempts"] == 3 for x in seen)
    assert result["all_cells_completed"]


def test_failed_cells_remain_in_scheduled_denominator(report_root, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("Never persist this arbitrary raw exception")

    monkeypatch.setattr(experiment, "run_probe_native_case", broken)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "broken",
        controlled_case_ids=("d10",), natural_case_ids=()))
    assert not result["all_cells_completed"]
    assert all(r["status"] == "operational_error" for r in result["rows"])
    assert all(g["correct_over_scheduled"] == {"numerator": 0, "denominator": 1} for g in result["summary"]["groups"].values())
    assert "arbitrary raw exception" not in (report_root / "broken/summary.json").read_text()


def test_output_is_exclusive_and_cannot_escape_project_reports(report_root):
    with pytest.raises(ValueError, match="inside"):
        asyncio.run(experiment.run_feedback_experiment(report_root.parent / "outside"))
    existing = report_root / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        asyncio.run(experiment.run_feedback_experiment(existing))


def test_live_rejects_test_client_factory_before_any_run(report_root):
    with pytest.raises(ValueError, match="forbids"):
        asyncio.run(experiment.run_feedback_experiment(report_root / "live", live=True,
            config=object(), client_factory=lambda *args: None))


def test_shared_budget_is_not_reset_between_stages(report_root, monkeypatch):
    budget = completed_budget()

    async def run(task, data, **kwargs):
        return fake_record(experiment.get_case("d10"))

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    initial = deepcopy(budget.snapshot())
    for stage in ("ablation", "strategy"):
        result = asyncio.run(experiment.run_feedback_experiment(report_root / stage, stage=stage,
            controlled_case_ids=("d10",), natural_case_ids=(), budget=budget))
        assert result["budget_start"] == initial == result["budget_end"]
        assert result["this_run_http_attempts"] == 0


def test_final_feedback_without_a_retry_is_not_counted_as_consumed():
    strategy = {"selected_mode": "full", "escalation_reason": "consecutive_failures"}
    record = {"actions": [{"feedback": {}}, {"feedback": {"feedback_strategy": strategy}}],
              "llm_requests": [[], []], "http_events": [{}, {}]}
    events = experiment.feedback_selection_events(record)
    assert len(events) == 1
    assert not events[0]["sent_in_later_native_request"]
    assert not events[0]["sent_in_later_http_attempt"]


def test_controlled_injected_turn_is_not_miscounted_as_an_http_attempt():
    strategy = {"selected_mode": "full", "escalation_reason": "repeated_candidate"}
    record = {"actions": [{"feedback": {}}, {"feedback": {"feedback_strategy": strategy}}, {"feedback": {}}],
              "llm_requests": [[], [], []], "http_events": [{}, {}],
              "turn_origins": ["injected_test_fault_no_api_call", "real_qwen_http", "real_qwen_http"]}
    event = experiment.feedback_selection_events(record)[0]
    assert event["sent_in_later_native_request"]
    assert event["sent_in_later_http_attempt"]


def test_blocked_next_request_is_not_counted_as_posted():
    strategy = {"selected_mode": "full", "escalation_reason": "repeated_candidate"}
    record = {"actions": [{"feedback": {}}, {"feedback": {"feedback_strategy": strategy}}],
              "llm_requests": [[], [], []], "http_events": [{}],
              "turn_origins": ["injected_test_fault_no_api_call", "real_qwen_http", "real_qwen_http"]}
    event = experiment.feedback_selection_events(record)[0]
    assert event["sent_in_later_native_request"]
    assert not event["sent_in_later_http_attempt"]


def test_accepted_feedback_does_not_create_an_extra_escalation_or_evidence_exposure():
    strategy = {"selected_mode": "full", "escalation_reason": "repeated_candidate", "evidence_available": True}
    record = {"actions": [
        {"feedback": {"status": "unverified", "feedback_strategy": strategy, "diagnostic_evidence": "full"}},
        {"feedback": {"status": "unverified", "feedback_strategy": strategy, "diagnostic_evidence": "full"}},
        {"feedback": {"status": "verified", "feedback_strategy": {**strategy, "evidence_available": False}}}],
        "llm_requests": [[], [], []], "http_events": [{}, {}, {}]}
    events = experiment.feedback_selection_events(record)
    assert sum(s["new_escalation_transition"] for s in events) == 1
    assert sum(s["failed_full_feedback"] for s in events) == 2
    assert sum(s["full_evidence_emitted"] for s in events) == 2
    assert not events[-1]["failed_full_feedback"]


def test_invalid_action_with_no_probe_is_recorded_without_runner_error(report_root, monkeypatch):
    async def run(task, data, **kwargs):
        record = fake_record(experiment.get_case("d10"))
        record["actions"] = [{"probe_report": None, "feedback": {"status": "unverified", "error_code": "INVALID_ACTION_FORMAT"}}]
        return record

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "invalid_action",
        controlled_case_ids=("d10",), natural_case_ids=()))
    assert result["all_cells_completed"]
    assert all(r["diagnostic_queries"] == 0 for r in result["rows"])


def test_cli_live_requires_prior_ledger_before_reading_configuration(monkeypatch):
    spec = importlib.util.spec_from_file_location("feedback_cli", Path(__file__).resolve().parents[1] / "tools/evaluate_feedback_strategies.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["evaluate_feedback_strategies.py", "--live"])
    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", lambda: (_ for _ in ()).throw(AssertionError("must not read config")))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def truncated_document():
    budget = completed_budget()
    start = deepcopy(budget.snapshot())
    budget.reserve(1000)
    usage = {"prompt_tokens": 1000, "completion_tokens": 900, "total_tokens": 1900}
    budget.record_tokens(usage)
    budget.stop("PROVIDER_ERROR")
    return {"real_model_used": True, "source_integrity_ok": True, "audit_ledger_complete": True,
            "audit_event_count": 1, "this_run_http_attempts": 1,
            "budget_start": start, "budget_end": budget.snapshot(),
            "rows": [{"case_id": "d03", "track": "natural", "arm": "neither",
                      "status": "completed", "http_attempts": 1, "http_events": [
                          {"attempt": 2, "status": "failed", "http_status": 200,
                           "error_code": "QWEN_RESPONSE_TRUNCATED", "usage": usage,
                           "usage_complete": True}]}]}


def save_prior(report_root, document):
    path = report_root / "prior.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_explicit_truncation_review_preserves_source_and_all_paid_usage(report_root):
    original = truncated_document()
    path = save_prior(report_root, original)
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="complete, unstopped"):
        experiment.load_prior_budget(path)
    budget = experiment.load_prior_budget(path, allow_truncated_resume=True)
    snapshot = budget.snapshot()
    for key in ("http_attempts", "reported_total_tokens", "reserved_upper_cny", "estimated_reported_cny", "reservation_ledger"):
        assert snapshot[key] == original["budget_end"][key]
    assert snapshot["stopped_reason"] is None
    assert snapshot["max_output_tokens"] == 900
    assert path.read_bytes() == raw
    review = budget._continuation_review[0]
    assert review["reviewed_failed_attempt"] == 2
    assert review["original_stop_reason"] == "PROVIDER_ERROR"
    assert review["source_report"] == "reports/prior.json"
    assert not review["refund_applied"]
    assert review["source_report_sha256"] == experiment.hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", [
    "unknown_usage", "event_incomplete", "wrong_event_total", "wrong_error", "http_error",
    "wrong_stop", "event_count", "wrong_attempt", "duplicate_event", "ledger_usage",
    "audit_incomplete", "source_changed", "in_flight", "unmatched", "historical_ledger_changed",
])
def test_truncation_flag_cannot_bypass_unreviewed_failures(report_root, mutation):
    doc = truncated_document()
    event = doc["rows"][0]["http_events"][0]
    if mutation == "unknown_usage":
        doc["budget_end"]["unknown_count"] = 1
    elif mutation == "event_incomplete":
        event["usage_complete"] = False
    elif mutation == "wrong_event_total":
        event["usage"]["total_tokens"] += 1
    elif mutation == "wrong_error":
        event["error_code"] = "QWEN_INVALID_RESPONSE_ENVELOPE"
    elif mutation == "http_error":
        event["http_status"] = 500
    elif mutation == "wrong_stop":
        doc["budget_end"]["stopped_reason"] = "TRANSPORT_ERROR"
    elif mutation == "event_count":
        doc["this_run_http_attempts"] = 2
    elif mutation == "wrong_attempt":
        event["attempt"] = 1
    elif mutation == "duplicate_event":
        doc["rows"][0]["http_events"].append(deepcopy(event))
        doc["rows"][0]["http_attempts"] = 2
    elif mutation == "ledger_usage":
        doc["budget_end"]["reservation_ledger"][-1]["prompt_tokens"] += 1
    elif mutation == "audit_incomplete":
        doc["audit_ledger_complete"] = False
    elif mutation == "source_changed":
        doc["source_integrity_ok"] = False
    elif mutation == "in_flight":
        doc["budget_end"]["in_flight_count"] = 1
    elif mutation == "unmatched":
        doc["budget_end"]["unmatched_usage_events"] = 1
    elif mutation == "historical_ledger_changed":
        doc["budget_end"]["reservation_ledger"][0]["prompt_chars"] += 1
    with pytest.raises(ValueError):
        experiment.load_prior_budget(save_prior(report_root, doc), allow_truncated_resume=True)


def test_continuation_review_lineage_is_saved_in_the_next_manifest(report_root, monkeypatch):
    budget = experiment.load_prior_budget(save_prior(report_root, truncated_document()), allow_truncated_resume=True)

    async def run(task, data, **kwargs):
        return fake_record(experiment.get_case("d10"))

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "next_stage", stage="strategy",
        controlled_case_ids=("d10",), natural_case_ids=(), budget=budget))
    frozen = json.loads((report_root / "next_stage/frozen_manifest.json").read_text(encoding="utf-8"))
    assert result["continuation_review"] == frozen["continuation_review"] == budget._continuation_review
    assert result["budget_start"]["http_attempts"] == 2


def test_arm_subset_schedules_only_requested_registered_cell(report_root, monkeypatch):
    seen = []

    async def run(task, data, **kwargs):
        seen.append(kwargs)
        return fake_record(experiment.get_case("d03"))

    monkeypatch.setattr(experiment, "run_probe_native_case", run)
    result = asyncio.run(experiment.run_feedback_experiment(report_root / "supplement",
        controlled_case_ids=(), natural_case_ids=("d03",), arm_ids=("capability_only",)))
    assert len(result["rows"]) == 1
    assert result["rows"][0]["arm"] == "capability_only"
    assert seen[0]["feedback_policy"] == "capability_only"


@pytest.mark.parametrize("arms", [(), ("neither", "neither"), ("adaptive",), ("unknown",)])
def test_arm_subset_cannot_add_unknown_or_cross_stage_policies(report_root, arms):
    with pytest.raises(ValueError, match="registered"):
        asyncio.run(experiment.run_feedback_experiment(report_root / "bad_arms", arm_ids=arms))
