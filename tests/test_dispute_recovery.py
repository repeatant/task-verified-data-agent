"""Offline experiment contracts; scripted actions are not LLM success evidence."""

import asyncio
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataverify.agent import dispute_model
from dataverify.agent.graph_checkpoint import JournaledBudget
from dataverify.evaluation import dispute_recovery as experiment
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation.expanded_cases import scoring_data
from dataverify.integrations.qwen_config import QwenClientError, QwenConfig
from dataverify.runtime.campaign_budget import CampaignBudget
from dataverify.runtime.reporting import digest_json, write_json
from dataverify.verification.discovery_runtime import execute_readonly


def completed_record(case, *, correct=True, publication_source=None):
    sql = case.base.good_sql if correct else case.base.bad_sql
    reference = case.base.good_reference if correct else case.base.bad_reference
    source = publication_source or experiment.visible_diagnostic_fixture()
    return {
        "status": "reviewed",
        "candidate": {"sql": sql},
        "plan": {**case.reference_plan, "reference": reference},
        "audit_plan": {**case.auditor_plan, "reference": reference},
        "publication": {"sql": sql, "rows": execute_readonly(sql, source)},
    }


def test_frozen_cases_are_six_trajectories_five_old_requirements():
    cases = experiment.study_cases()
    assert len(cases) == 6
    assert len({case.base.task.task_id for case in cases}) == 5
    assert [case.condition for case in cases] == [
        "sql_wrong",
        "reference_wrong",
        "auditor_wrong",
        "correlated_wrong",
        "correct",
        "all_wrong",
    ]
    assert experiment.METHODS == ("self_review", "legacy_full", "dispute_resolver")
    assert bounded.max_reservation(len(cases) * (1 + 2 * experiment.MAX_ACTIONS)) == pytest.approx(
        2.3401872
    )


def test_common_visible_fixture_is_fixed_and_separate_from_grading_inputs():
    visible = digest_json(experiment.visible_diagnostic_fixture().to_dict())
    for case in experiment.study_cases():
        fixtures = scoring_data(case.base)
        assert len(fixtures) == 25
        assert visible not in {digest_json(data.to_dict()) for data in fixtures.values()}
        assert experiment.case_binding(case)["visible_diagnostic_digest"] == visible


@pytest.mark.parametrize("index,role", [(0, "candidate"), (1, "reference"), (2, "auditor")])
def test_each_single_fault_role_is_scored_independently(index, role):
    case = experiment.study_cases()[index]
    score = experiment.score_result(case, "dispute_resolver", completed_record(case))
    assert score["roles"][role]["initial_correct"] is False
    assert score["roles"][role]["repaired"] is True
    assert all(item["final_correct"] for item in score["roles"].values())
    assert score["joint_correct_delivery"] is True
    assert all(item["grade"]["total"] == 25 for item in score["roles"].values())


def test_a_good_sql_does_not_hide_a_bad_auditor():
    case = experiment.study_cases()[2]
    record = completed_record(case)
    record["audit_plan"] = case.auditor_plan
    score = experiment.score_result(case, "dispute_resolver", record)
    assert score["correct_delivery"] is True
    assert score["roles"]["candidate"]["final_correct"] is True
    assert score["roles"]["auditor"]["final_correct"] is False
    assert score["wrong_auditor_released"] is True
    assert score["joint_correct_delivery"] is False


def test_baseline_reference_and_auditor_are_na_not_correct():
    case = experiment.study_cases()[1]
    record = completed_record(case)
    record.pop("plan")
    record.pop("audit_plan")
    score = experiment.score_result(case, "self_review", record)
    for role in ("reference", "auditor"):
        assert score["roles"][role]["final_correct"] is None
        assert score["roles"][role]["repaired"] is None
        assert score["roles"][role]["grade"] is None
    assert score["joint_correct_delivery"] is None


def test_publication_rows_are_checked_against_each_methods_actual_source():
    case = experiment.study_cases()[0]
    visible_record = completed_record(case)
    original_record = completed_record(case, publication_source=case.base.source)
    assert visible_record["publication"]["rows"] != original_record["publication"]["rows"]
    assert experiment.score_result(case, "dispute_resolver", visible_record)["correct_delivery"]
    assert not experiment.score_result(case, "legacy_full", visible_record)["correct_delivery"]
    assert experiment.score_result(case, "legacy_full", original_record)["correct_delivery"]
    assert not experiment.score_result(case, "dispute_resolver", original_record)[
        "correct_delivery"
    ]


def test_shared_error_negative_control_is_not_declared_correct_by_agreement():
    case = experiment.study_cases()[-1]
    score = experiment.score_result(case, "dispute_resolver", completed_record(case, correct=False))
    assert score["delivered"] is True
    assert score["wrong_release"] is True
    assert score["joint_correct_delivery"] is False
    assert score["all_final_programs_correct"] is False
    assert all(
        item["initial_correct"] is False and item["final_correct"] is False
        for item in score["roles"].values()
    )


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    case = experiment.study_cases()[0]
    config = QwenConfig(
        "sk-test-not-real", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-flash"
    )
    lane = JournaledBudget(
        CampaignBudget(money_limit_cny=28, max_output_tokens=1800), tmp_path / "budget-lane-0.json"
    )
    models = []

    class FakeModel:
        client_kind = "offline_test_double"

        def __init__(self, config, budget, *, max_tokens):
            self.budget, self.model_name = budget, config.model
            self.requests, self.http_events, self.usage = [], [], []
            self.closed = False
            models.append(self)

        async def generate(self, messages):
            self.requests.append(deepcopy(messages))
            attempt = self.budget.reserve(sum(len(item["content"]) for item in messages))
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            self.budget.record_tokens(usage)
            self.http_events.append(
                {"attempt": attempt, "status": "completed", "usage": usage, "usage_complete": True}
            )
            return self.action(messages)

        async def aclose(self):
            self.closed = True

    class Writer(FakeModel):
        def action(self, messages):
            return {"sql": case.base.good_sql, "task_id": case.base.task.task_id}

    class Reviewer(FakeModel):
        def action(self, messages):
            return {"action": "clarify", "reason": "Offline transport contract only."}

    class Resolver(FakeModel):
        def action(self, messages):
            context = json.loads(messages[-1]["content"])
            evidence = [context["latest_replay"]["id"]]
            if len(self.requests) == 1:
                return {
                    "action": "revise",
                    "target": "candidate",
                    "program": case.base.good_sql,
                    "reason": "Authored test repair, not autonomous diagnosis.",
                    "requirement_quote": case.base.task.question,
                    "evidence_ids": evidence,
                }
            return {
                "action": "finish",
                "reason": "Inspect updated replay.",
                "evidence_ids": evidence,
            }

    monkeypatch.setattr(experiment, "QwenGraphModel", Writer)
    monkeypatch.setattr(experiment, "QwenReviewModel", Reviewer)
    monkeypatch.setattr(dispute_model, "QwenDisputeModel", Resolver)
    output = tmp_path / "run"
    for folder in ("records", "checkpoints"):
        (output / folder).mkdir(parents=True)
    return SimpleNamespace(case=case, config=config, lane=lane, models=models, output=output)


@pytest.mark.parametrize("method", ["self_review", "dispute_resolver"])
def test_live_driver_offline_double_keeps_gold_out_and_does_not_reroll(scenario, method):
    s = scenario
    result = asyncio.run(experiment.run_cell(s.case, method, s.config, s.lane, s.output, "freeze"))
    assert result["row"]["score"]["repaired_initial_error"]
    assert result["row"]["score"]["correct_delivery"]
    contexts = json.dumps(result["contexts"])
    assert "gold_rows" not in contexts
    assert "gold_sql" not in contexts
    assert "sql_wrong" not in contexts
    assert "seeded_" not in contexts
    assert all(model.closed for model in s.models)
    before = s.lane.snapshot()
    previous = asyncio.run(
        experiment.run_cell(s.case, method, s.config, s.lane, s.output, "freeze")
    )
    assert previous == result
    assert s.lane.snapshot() == before
    assert len(s.models) == 1


def test_matched_legacy_injects_three_initial_programs_and_shares_four_paid_calls(
    scenario, monkeypatch
):
    s = scenario

    async def legacy(task, source, writer, reviewer, *, auditor, **kwargs):
        assert writer.budget is reviewer.budget is auditor.budget
        assert writer.budget.limit == 4
        context = [{"role": "user", "content": json.dumps(task.public_input())}]
        sql = await writer.generate(context)
        reference = await reviewer.generate(context)
        audit = await auditor.generate(context)
        assert sql["sql"] == s.case.initial_sql
        assert reference["reference"] == s.case.reference_plan["reference"]
        assert audit["reference"] == s.case.auditor_plan["reference"]
        for plan in (reference, audit):
            assert plan["fixtures"][-1]["data"] == experiment.visible_diagnostic_fixture().to_dict()
        assert s.lane.snapshot()["http_attempts"] == 0
        for delegate in (writer, reviewer, auditor, reviewer):
            await delegate.generate(context)
        with pytest.raises(QwenClientError, match="CELL_HTTP_REQUEST_LIMIT"):
            await writer.generate(context)
        assert kwargs["max_challenge_examples"] == 40
        return {
            "status": "inconclusive",
            "candidate": sql,
            "plan": reference,
            "audit_plan": audit,
            "publication": {},
        }

    monkeypatch.setattr(experiment, "run_discovery", legacy)
    result = asyncio.run(
        experiment.run_cell(s.case, "legacy_full", s.config, s.lane, s.output, "freeze")
    )
    assert len(result["http_events"]) == 4
    assert s.lane.snapshot()["http_attempts"] == 4
    assert len(s.models) == 3 and all(model.closed for model in s.models)
    assert result["row"]["score"]["roles"]["reference"]["final_correct"] is True
    assert result["row"]["score"]["roles"]["auditor"]["final_correct"] is True


def test_incomplete_paid_record_and_unrecorded_checkpoint_require_review(scenario):
    s = scenario
    write_json(s.output / "records/dispute-01--self_review.json", {"completed": False})
    with pytest.raises(ValueError, match="REQUIRES_REVIEW"):
        asyncio.run(
            experiment.run_cell(s.case, "self_review", s.config, s.lane, s.output, "freeze")
        )
    (s.output / "checkpoints/dispute-01--legacy_full.sqlite3").write_bytes(b"partial")
    with pytest.raises(ValueError, match="CHECKPOINT_REQUIRES_REVIEW"):
        asyncio.run(
            experiment.run_cell(s.case, "legacy_full", s.config, s.lane, s.output, "freeze")
        )
    assert not s.models


def test_summary_uses_initially_wrong_and_initially_correct_denominators(tmp_path):
    cases = experiment.study_cases()
    output = tmp_path / "summary"
    (output / "records").mkdir(parents=True)
    for case in cases:
        record = completed_record(case)
        write_json(
            output / "records" / (case.case_id + ".json"),
            {
                "completed": True,
                "row": {
                    "case_id": case.case_id,
                    "task_id": case.base.task.task_id,
                    "condition": case.condition,
                    "method": "dispute_resolver",
                    "status": "reviewed",
                    "score": experiment.score_result(case, "dispute_resolver", record),
                    "http_events": [],
                },
            },
        )
    protocol = {
        "cases": [{"case_id": case.case_id} for case in cases],
        "distinct_original_requirements": 5,
        "limitations": [],
    }
    result = experiment.summarize(output, protocol)["methods"]["dispute_resolver"]
    assert result["repaired_initial_error"] == {"numerator": 3, "denominator": 3}
    assert result["correct_sql_damaged"] == {"numerator": 0, "denominator": 3}
    assert result["roles"]["auditor"]["repaired"] == {"numerator": 2, "denominator": 2}
    assert result["roles"]["auditor"]["damaged"] == {"numerator": 0, "denominator": 4}


def test_prepare_cli_freezes_without_reading_credentials_or_allocating_budget(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "dispute_cli_test", root / "tools/evaluate_dispute_recovery.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    main = tmp_path / "old_main"
    (main / "tools").mkdir(parents=True)
    (main / "tools/continue_expanded.py").write_text(
        "# frozen driver placeholder, never loaded\n", encoding="utf-8"
    )
    output = tmp_path / "offline_preparation"

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline prepare must not load credentials or accounting driver")

    monkeypatch.setattr(cli.QwenConfig, "from_local_environment", forbidden)
    monkeypatch.setattr(cli, "load_driver", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_dispute_recovery.py", "--main", str(main), "--output", str(output), "--prepare"],
    )
    cli.main()
    protocol = bounded.read_json(output / "protocol.json")
    assert protocol["methods"] == list(experiment.METHODS)
    assert protocol["cost_bound"]["maximum_new_calls"] == 54
    assert protocol["cost_bound"]["maximum_reserved_cny"] == pytest.approx(2.3401872)
    assert protocol["controlled_fault_trajectories"] == 6
    assert protocol["distinct_original_requirements"] == 5
    assert (output / "frozen-source.zip").exists()
    assert not (main / "reports").exists()
