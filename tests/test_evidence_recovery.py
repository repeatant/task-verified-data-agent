"""Offline contracts for the experimental runner; no credentials or API calls."""

import asyncio
import json
import sqlite3
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation import evidence_recovery as recovery
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.campaign_budget import CampaignBudget
from dataverify.runtime.reporting import write_json
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.relational_data import SQLiteSource


def test_selection_ignores_gold_and_preserves_disjointness():
    entries = [
        {"question_id": i, "db_id": f"db_{i % 3}", "reasons": [], "gold_sql": "secret"}
        for i in range(50)
    ]
    manifest = {"test_ids": [0, 1], "development_ids": list(range(2, 10)), "eligibility": entries}
    earlier = {"test_ids": [10, 11]}
    first = recovery.select_question_ids(manifest, earlier)
    changed = deepcopy(manifest)
    changed["eligibility"].reverse()
    for entry in changed["eligibility"]:
        entry["gold_sql"] = "different answer, never used"
    assert recovery.select_question_ids(changed, earlier) == first
    assert len(first["holdout_ids"]) == 20
    assert len(set(first["holdout_ids"])) == 20
    assert not set(first["holdout_ids"]) & set(range(12))
    assert first["development_ids"] == list(range(2, 8))


def test_insufficient_unused_questions_is_not_silently_reused():
    manifest = {"test_ids": [], "development_ids": [], "eligibility": []}
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        recovery.select_question_ids(manifest, {"test_ids": []})


def test_reservation_bound_keeps_existing_budget_and_rejects_input_before_payment():
    lane = CampaignBudget(money_limit_cny=28, max_output_tokens=1800)
    budget = recovery.BoundedCellBudget(lane, 3)
    start = lane.snapshot()
    with pytest.raises(GraphModelError, match="RECOVERY_INPUT_CHAR_CAP"):
        budget.reserve(11_001)
    assert lane.snapshot() == start
    assert recovery.forecast()["total_max_calls"] == 202
    assert recovery.forecast()["total_max_reserved_cny"] == pytest.approx(8.7540336)
    assert recovery.forecast()["total_max_reserved_cny"] < 9.457916


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t(amount INTEGER)")
        connection.executemany("INSERT INTO t VALUES (?)", [(2,), (3,)])
    source = SQLiteSource.from_path(path)
    task = DiscoveryTask(
        "recovery_test", "Sum amounts.", schema=source.schema, allow_source_only_plan=True
    )
    bundle = SimpleNamespace(
        case_id=task.task_id,
        family="fixture",
        task=task,
        source=source,
        truth={"db_path": path, "gold_rows": [[5]], "private_sentinel": "GOLD_NEVER_IN_CONTEXT"},
    )
    config = QwenConfig(
        "sk-test-not-real", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-flash"
    )
    lane = JournaledBudget(
        CampaignBudget(money_limit_cny=28, max_output_tokens=1800), tmp_path / "budget-lane-0.json"
    )
    models = []

    class FakeModel:
        actions = []

        def __init__(self, config, budget, *, max_tokens):
            self.budget = budget
            self.requests, self.http_events = [], []
            self.closed = False
            models.append(self)

        async def generate(self, messages):
            self.requests.append(deepcopy(messages))
            attempt = self.budget.reserve(sum(len(m["content"]) for m in messages))
            usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            self.budget.record_tokens(usage)
            self.http_events.append(
                {"attempt": attempt, "status": "completed", "usage": usage, "usage_complete": True}
            )
            return deepcopy(self.actions[len(self.requests) - 1])

        async def aclose(self):
            self.closed = True

    class Writer(FakeModel):
        actions = [{"sql": "SELECT SUM(amount) FROM t", "task_id": task.task_id}]

    class Investigator(FakeModel):
        actions = [
            {
                "action": "revise",
                "sql": "SELECT SUM(amount) FROM t",
                "reason": "Count does not sum amounts.",
                "requirement_quote": "Sum amounts.",
                "evidence_ids": ["e0001"],
            },
            {"action": "finish", "reason": "The sum executes.", "evidence_ids": ["e0002"]},
        ]

    monkeypatch.setattr(recovery, "QwenGraphModel", Writer)
    monkeypatch.setattr(recovery, "QwenEvidenceModel", Investigator)
    output = tmp_path / "run"
    for sub in ("initial", "records"):
        (output / sub).mkdir(parents=True)
    common = {"sql": "SELECT COUNT(*) FROM t", "http_events": []}
    return SimpleNamespace(
        bundle=bundle,
        config=config,
        lane=lane,
        models=models,
        output=output,
        common=common,
        Writer=Writer,
    )


@pytest.mark.parametrize("method", recovery.METHODS)
def test_paired_arms_share_initial_sql_and_original_db_but_never_gold(scenario, method):
    s = scenario
    result = asyncio.run(
        recovery.paired_cell(s.bundle, method, s.common, s.config, s.lane, s.output, "frozen")
    )
    row = result["row"]
    assert row["score"]["initial_sql_correct"] is False
    assert row["score"]["repaired_initial_error"] is True
    assert row["score"]["correct_delivery"] is True
    assert row["score"]["wrong_release"] is False
    assert row["score"]["unchanged_delivery"] is False
    assert result["record"]["observations"][0]["rows"] == [[2]]
    text = json.dumps(result["contexts"])
    assert "GOLD_NEVER_IN_CONTEXT" not in text
    assert "gold_rows" not in text
    assert s.models[0].closed
    before = s.lane.snapshot()
    rerun = asyncio.run(
        recovery.paired_cell(s.bundle, method, s.common, s.config, s.lane, s.output, "frozen")
    )
    assert rerun == result
    assert s.lane.snapshot() == before
    assert len(s.models) == 1


def test_unchanged_correct_delivery_is_not_counted_as_a_repair(scenario):
    s = scenario
    s.common["sql"] = "SELECT SUM(amount) FROM t"
    result = asyncio.run(
        recovery.paired_cell(
            s.bundle, "self_review", s.common, s.config, s.lane, s.output, "frozen"
        )
    )
    assert result["row"]["score"]["unchanged_delivery"] is True
    assert result["row"]["score"]["repaired_initial_error"] is False
    assert result["row"]["score"]["preserved_correct_sql"] is True


def test_incomplete_paid_record_cannot_reroll(scenario):
    s = scenario
    write_json(s.output / "records/recovery_test--self_review.json", {"completed": False})
    with pytest.raises(ValueError, match="REQUIRES_REVIEW"):
        asyncio.run(
            recovery.paired_cell(
                s.bundle, "self_review", s.common, s.config, s.lane, s.output, "frozen"
            )
        )
    assert not s.models


def test_gate_rejects_status_only_improvement():
    baseline = {"final_sql_correct": 2, "wrong_release": 1}
    evidence = {**baseline, "correct_sql_damaged": 0, "repaired_initial_error": 0, "query_calls": 4}
    summary = {
        "complete": True,
        "phase": "development",
        "methods": {"self_review": baseline, "no_tools": baseline, "evidence": evidence},
    }
    gate = recovery.development_gate(summary)
    assert not gate["passed"]
    assert "no_candidate_accuracy_gain_over_both_comparators" in gate["reasons"]
    assert "no_observed_tool_supported_sql_repair" in gate["reasons"]
