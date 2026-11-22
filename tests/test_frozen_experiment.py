"""Check experiment accounting and role boundaries without model API calls."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget, read_budget
from dataverify.agent.graph_model import InjectedFirstGraphModel
from dataverify.evaluation import frozen_experiment as experiment
from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.evaluation.frozen_cases import get_frozen_cases
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget

CASES = get_frozen_cases()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_baseline_gates_distinguish_wrong_sql_from_wrong_reference(case):
    gates = experiment.baseline_gates(case)
    correct_sql = case.condition in {"correct", "reference_wrong"}
    agreement = case.condition in {"correct", "correlated_wrong"}
    assert gates["sql_executable_accept"]
    assert gates["initial_sql_correct"] is correct_sql
    assert gates["single_reference_accept"] is agreement
    assert gates["sql_executable_wrong_release"] is (not correct_sql)
    assert gates["single_reference_wrong_release"] is (case.condition == "correlated_wrong")


def test_cohort_is_reproducible_and_natural_has_no_injected_programs():
    first, second = experiment.experiment_cases(), experiment.experiment_cases()
    assert first == second
    assert len(first) == len({case.case_id for case in first}) == 20
    natural = [case for case in first if case.condition == "natural"]
    assert len(natural) == 4
    for case in natural:
        assert case.initial_sql is None
        assert case.initial_plan is None
        assert experiment.baseline_gates(case) == {}
    assert {case.task.task_id for case in first} == {case.family for case in CASES}


def _row(condition, status="reviewed", **scores):
    return {
        "case_id": "test_" + condition,
        "condition": condition,
        "status": status,
        "score": scores,
        "http_events": [],
        "runner_error": None,
    }


def test_aggregation_preserves_planned_denominators_and_abstentions():
    summary = experiment.aggregate(
        [
            _row("sql_wrong", correct_delivery=True),
            _row("reference_wrong", "needs_clarification"),
            _row("correlated_wrong", wrong_release=True),
        ]
    )
    assert summary["planned"] == 20
    assert summary["attempted"] == 3
    assert not summary["complete"]
    assert summary["distinct_intents"] == 4
    assert summary["independent_trials_per_case"] == 1
    for group in summary["groups"].values():
        assert group["planned"] == 4
    assert summary["groups"]["natural"]["attempted"] == 0
    refusal = summary["groups"]["reference_wrong"]
    assert refusal["not_reviewed"] == 1
    assert refusal["correct_delivery"] == refusal["wrong_release"] == 0
    assert summary["groups"]["correlated_wrong"]["wrong_release"] == 1


def _prior_budget(tmp_path, *, pending=False, stopped=False):
    path = tmp_path / "prior.json"
    budget = JournaledBudget(MonetaryRequestBudget(money_limit_cny=5), path)
    budget.reserve(100)
    if not pending:
        budget.record_tokens({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    if stopped:
        budget.stop("TEST_STOP")
    return path, budget


def test_top_up_keeps_prior_ledger_and_prevents_branching_same_allowance(tmp_path):
    prior_path, original = _prior_budget(tmp_path)
    previous_bytes, previous = prior_path.read_bytes(), original.snapshot()
    output = tmp_path / "first"
    output.mkdir()
    budget, restored = experiment.create_top_up_budget(prior_path, output, 3)
    assert restored == previous
    assert prior_path.read_bytes() == previous_bytes
    assert budget.snapshot()["money_limit_cny"] == 3
    assert budget.snapshot()["http_attempts"] == 0
    lineage = json.loads((output / "budget-lineage.json").read_text(encoding="utf-8"))
    assert lineage["prior_snapshot"] == previous
    assert lineage["combined_allowance_cny"] == 8
    assert lineage["authorized_added_allowance_cny"] == 3
    assert read_budget(output / "budget.json") == budget.snapshot()
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="PARENT_ALREADY_HAS"):
        experiment.create_top_up_budget(prior_path, second, 3)
    assert not (second / "budget.json").exists()


@pytest.mark.parametrize("amount", [0, -1, 3.01, float("nan"), float("inf")])
def test_unauthorized_top_up_size_is_rejected_without_allocating(tmp_path, amount):
    path, _ = _prior_budget(tmp_path)
    output = tmp_path / "new"
    output.mkdir()
    with pytest.raises(ValueError, match="ADDITIONAL_ALLOWANCE"):
        experiment.create_top_up_budget(path, output, amount)
    assert not path.with_suffix(".frozen-allocation.json").exists()
    assert not (output / "budget.json").exists()


@pytest.mark.parametrize("state", ["pending", "stopped"])
def test_unresolved_or_stopped_prior_usage_cannot_be_hidden_by_top_up(tmp_path, state):
    path, _ = _prior_budget(tmp_path, pending=state == "pending", stopped=state == "stopped")
    output = tmp_path / "new"
    output.mkdir()
    with pytest.raises(ValueError, match="PRIOR_USAGE_MUST_BE_COMPLETE_AND_UNSTOPPED"):
        experiment.create_top_up_budget(path, output, 3)
    assert not path.with_suffix(".frozen-allocation.json").exists()


@pytest.mark.parametrize("condition", ["sql_wrong", "reference_wrong", "natural"])
def test_run_case_keeps_scoring_and_fault_labels_outside_runtime(tmp_path, monkeypatch, condition):
    case = next(case for case in experiment.experiment_cases() if case.condition == condition)
    finished = False
    models = []
    marker_config, marker_budget = object(), object()

    class FakeClient:
        def __init__(self, config, budget):
            assert config is marker_config
            assert budget is marker_budget
            self.budget = budget
            self.model_name = "offline-test"
            self.requests, self.http_events = [], []
            self.closed = False
            models.append(self)

        async def aclose(self):
            self.closed = True

        async def generate(self, messages):
            raise AssertionError("This boundary test must not request model answers")

    async def fake_graph(task, source, writer, reviewer, *, auditor, checkpoint_path, thread_id):
        nonlocal finished
        assert task is case.task
        assert source is case.source
        assert task.task_id == case.family
        public = json.dumps(task.public_input())
        for forbidden in (condition, case.case_id, "expected", "scoring", "gold_sql"):
            assert forbidden not in public
        assert len({id(writer), id(reviewer), id(auditor)}) == 3
        assert isinstance(auditor, FakeClient)
        if condition == "natural":
            assert isinstance(writer, FakeClient)
            assert isinstance(reviewer, FakeClient)
        else:
            assert isinstance(writer, InjectedFirstGraphModel)
            assert isinstance(reviewer, InjectedFirstReviewModel)
            messages = [{"role": "user", "content": "read task"}]
            assert await writer.generate(messages) == {
                "sql": case.initial_sql,
                "task_id": case.family,
                "thought": "",
            }
            assert await reviewer.generate(messages) == case.initial_plan
        assert checkpoint_path.parent == tmp_path
        assert thread_id == "frozen"
        finished = True
        return {"status": "needs_clarification"}

    original_grade = experiment.grade_frozen_run

    def score_after_runtime(*args, **kwargs):
        assert finished, "Scoring must not run while the model workflow is active"
        return original_grade(*args, **kwargs)

    monkeypatch.setattr(experiment, "QwenGraphModel", FakeClient)
    monkeypatch.setattr(experiment, "QwenReviewModel", FakeClient)
    monkeypatch.setattr(experiment, "run_discovery", fake_graph)
    monkeypatch.setattr(experiment, "grade_frozen_run", score_after_runtime)
    row = asyncio.run(
        experiment.run_case(case, config=marker_config, budget=marker_budget, output=tmp_path)
    )
    assert finished and all(model.closed for model in models)
    assert row["status"] == "needs_clarification"
    assert row["runner_error"] is None
    assert not row["score"]["correct_delivery"]
    assert row["http_events"] == []


def test_combined_report_adds_prior_usage_without_resetting_cost(tmp_path, monkeypatch):
    prior_path, original = _prior_budget(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    budget, prior = experiment.create_top_up_budget(prior_path, output, 3)
    source_hashes = {"frozen.py": "stable"}
    manifest = {"files": source_hashes, "scope": "test"}
    monkeypatch.setattr(experiment, "source_hashes", lambda root: deepcopy(source_hashes))
    monkeypatch.setattr(experiment, "experiment_cases", lambda: [object()])
    monkeypatch.setattr(experiment, "validate_pricing", lambda config: None)

    async def fake_case(case, **kwargs):
        budget.reserve(100)
        budget.record_tokens({"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24})
        row = _row("correct", correct_delivery=True)
        row["usage"] = {}
        return row

    monkeypatch.setattr(experiment, "run_case", fake_case)
    report = asyncio.run(
        experiment.run_experiment(
            tmp_path, output, SimpleNamespace(model="offline-test"), budget, prior, manifest
        )
    )
    combined = report["combined_accounting"]
    assert combined["http_calls"] == 2
    assert combined["reported_tokens"] == 36
    assert combined["allowance_cny"] == 8
    assert combined["estimated_cny"] == pytest.approx(
        prior["estimated_reported_cny"] + budget.snapshot()["estimated_reported_cny"]
    )
    assert combined["reserved_cny"] == pytest.approx(
        prior["reserved_upper_cny"] + budget.snapshot()["reserved_upper_cny"]
    )
    assert read_budget(prior_path) == original.snapshot() == prior
    assert report["attempted"] == 1 and not report["complete"]


@pytest.mark.parametrize("failure", ["accounted_truncation", "other_error", "unknown_usage"])
def test_only_accounted_truncation_advances_without_retry_or_allowance_reset(
    tmp_path, monkeypatch, failure
):
    prior_path, _ = _prior_budget(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    budget, prior = experiment.create_top_up_budget(prior_path, output, 3)
    hashes = {"frozen.py": "stable"}
    manifest = {"files": hashes, "scope": "test"}
    cases = [SimpleNamespace(case_id="first"), SimpleNamespace(case_id="second")]
    monkeypatch.setattr(experiment, "source_hashes", lambda root: deepcopy(hashes))
    monkeypatch.setattr(experiment, "experiment_cases", lambda: cases)
    monkeypatch.setattr(experiment, "validate_pricing", lambda config: None)
    visits = []

    async def fake_case(case, **kwargs):
        active = kwargs["budget"]
        visits.append(case.case_id)
        attempt = active.reserve(100)
        usage = {"prompt_tokens": 20, "completion_tokens": 900, "total_tokens": 920}
        event = {"attempt": attempt, "http_status": 200}
        row = _row("correct", "inconclusive" if case.case_id == "first" else "reviewed")
        row["case_id"] = case.case_id
        if case.case_id == "first":
            if failure != "unknown_usage":
                active.record_tokens(usage)
                event.update(usage=usage, usage_complete=True)
            active.stop("PROVIDER_ERROR")
            event.update(
                status="failed",
                error_code=(
                    "GRAPH_QWEN_RESPONSE_TRUNCATED"
                    if failure == "accounted_truncation"
                    else "GRAPH_QWEN_HTTP_500"
                ),
            )
        else:
            assert active.path.name == "budget-after-first.json"
            assert active.snapshot()["money_limit_cny"] == 3
            assert active.snapshot()["http_attempt_limit"] == 150
            assert active.snapshot()["http_attempts"] == 2
            active.record_tokens(usage)
            event.update(status="completed", usage=usage, usage_complete=True)
        row["http_events"] = [event]
        row["usage"] = {}
        return row

    monkeypatch.setattr(experiment, "run_case", fake_case)
    report = asyncio.run(
        experiment.run_experiment(
            tmp_path, output, SimpleNamespace(model="offline-test"), budget, prior, manifest
        )
    )
    original = read_budget(output / "budget.json")
    assert original["stopped_reason"] == "PROVIDER_ERROR"
    assert read_budget(prior_path) == prior
    if failure == "accounted_truncation":
        assert visits == ["first", "second"]
        assert report["rows"][0]["status"] == "inconclusive"
        assert report["latest_journal"] == "budget-after-first.json"
        assert len(report["accounted_truncation_continuations"]) == 1
        assert not report["accounted_truncation_continuations"][0]["task_retried"]
        restored = read_budget(output / report["latest_journal"])
        assert restored["reservation_ledger"][:1] == original["reservation_ledger"]
        assert restored["money_limit_cny"] == original["money_limit_cny"] == 3
        assert restored["reported_total_tokens"] == 1840
        assert report["combined_accounting"]["reported_tokens"] == 1852
        assert report["combined_accounting"]["http_calls"] == 3
        assert report["combined_accounting"]["allowance_cny"] == 8
    else:
        assert visits == ["first"]
        assert report["accounted_truncation_continuations"] == []
        assert report["latest_journal"] == "budget.json"
        assert not (output / "budget-after-first.json").exists()
