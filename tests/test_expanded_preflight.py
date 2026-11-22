"""Offline experiment wiring checks; fake model behavior is never quality evidence."""

import asyncio
import importlib.util
import json
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataverify.evaluation import expanded_experiment as experiment
from dataverify.runtime.campaign_budget import CampaignBudget
from dataverify.runtime.reporting import digest_json


def test_all_controlled_conditions_and_methods_route_same_cached_inputs(monkeypatch, tmp_path):
    output = tmp_path
    (output / "records").mkdir()
    (output / "initial").mkdir()
    holder, dispatch, scored = {}, [], []

    class FakeModel:
        client_kind = "offline_preflight_only"
        model_name = "no-model"

        def __init__(self, config, budget, **kwargs):
            self.budget, self.requests, self.http_events, self.usage = budget, [], [], []

        async def generate(self, messages):
            self.requests.append(deepcopy(messages))
            case = holder["bundle"].truth
            return {"task_id": case.task.task_id, "sql": case.initial_sql}

        async def aclose(self):
            pass

    class FakeReviewer(FakeModel):
        async def generate(self, messages):
            self.requests.append(deepcopy(messages))
            return holder["bundle"].truth.initial_plan(auditor=True)

    async def fake_graph(task, source, writer, reviewer, **kwargs):
        bundle = holder["bundle"]
        context = [{"role": "user", "content": json.dumps(task.public_input())}]
        candidate = await writer.generate(context)
        plan = await reviewer.generate(context)
        assert candidate["sql"] == holder["initial"]["sql"]
        assert plan == holder["initial"]["plan"]
        assert task is bundle.task and source is bundle.source
        assert bundle.case_id not in context[0]["content"]
        assert bundle.condition not in task.task_id and bundle.family not in task.task_id
        audit = await kwargs["auditor"].generate(context) if "auditor" in kwargs else None
        if bundle.condition == "auditor_wrong" and audit is not None:
            assert audit["reference"] == bundle.truth.bad_reference
        dispatch.append((holder["method"], kwargs.get("policy")))
        record = experiment.direct_record(bundle, candidate["sql"])
        record.update(plan=plan, audit_plan=audit)
        return record

    def score_spy(case, record):
        assert case is holder["bundle"].truth
        assert record["candidate"]["sql"] == holder["initial"]["sql"]
        scored.append((case.case_id, holder["method"]))
        return {"correct_delivery": case.condition not in {"sql_wrong", "correlated_wrong"}}

    monkeypatch.setattr(experiment, "QwenGraphModel", FakeModel)
    monkeypatch.setattr(experiment, "QwenReviewModel", FakeReviewer)
    monkeypatch.setattr(experiment, "run_discovery", fake_graph)
    monkeypatch.setattr(experiment, "run_two_way", fake_graph)
    monkeypatch.setattr(experiment, "score_controlled", score_spy)
    lane = CampaignBudget(money_limit_cny=13, max_output_tokens=1800)
    lane.path = tmp_path / "budget-lane-0.json"

    async def scenario():
        for bundle in experiment.controlled_bundles():
            holder["bundle"] = bundle
            initial_path = output / "initial" / (bundle.case_id + ".json")
            initial = await experiment.prepare_first(bundle, object(), lane, initial_path)
            assert initial["injected"] and initial["http_events"] == []
            assert initial["task_digest"] == digest_json(bundle.task.public_input())
            holder["initial"] = initial
            for method in experiment.BASE_METHODS + experiment.ABLATIONS:
                holder["method"] = method
                before = deepcopy(initial)
                row = await experiment.run_cell(bundle, method, initial, object(), lane, output)
                assert initial == before
                assert row["method"] == method and row["case_id"] == bundle.case_id
                assert row["cluster_id"] == bundle.task.task_id
                saved = json.loads(
                    (output / "records" / f"{bundle.case_id}--{method}.json").read_text(
                        encoding="utf-8"
                    )
                )
                assert saved["task_digest"] == initial["task_digest"]
                # The injected SQL/reference are legitimate controlled inputs;
                # fault labels and hidden scoring fixtures must not enter roles.
                context = str(saved["contexts"])
                assert bundle.case_id not in context
                assert "boundaries_and_duplicates" not in context
                assert saved["http_events"] == []

    asyncio.run(scenario())
    assert len(scored) == 700
    assert len(dispatch) == 500
    for method, policy in dispatch:
        if method == "two_way":
            assert policy is None
        else:
            assert policy.challenge_search is (method != "no_search")
            assert policy.blind_audit is (method != "visible_audit")
            assert policy.evidence_authorization is (method != "model_authorization")


def load_runner():
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_expanded.py"
    spec = importlib.util.spec_from_file_location("expanded_preflight_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "phase,expected_count", [("development", 4), ("test", 7), ("stability", 2)]
)
def test_phase_selection_and_stability_only_full(monkeypatch, tmp_path, phase, expected_count):
    runner = load_runner()
    bundle = experiment.controlled_bundles()[0]
    calls = []
    monkeypatch.setattr(runner.QwenConfig, "from_local_environment", lambda: object())
    monkeypatch.setattr(runner, "validate_pricing", lambda config: None)
    monkeypatch.setattr(runner, "exclusive_run", lambda path: nullcontext())
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "save_summary", lambda output, protocol: {})
    monkeypatch.setattr(
        runner,
        "lane_budget",
        lambda *args: SimpleNamespace(snapshot=lambda: {"stopped_reason": None}),
    )
    monkeypatch.setattr(runner, "public_bundles", lambda *args: [bundle])
    monkeypatch.setattr(runner, "controlled_bundles", lambda: [])

    async def initial(*args):
        return {}

    async def cell(bundle, method, *args):
        calls.append((bundle.case_id, method, bundle.task.task_id))

    monkeypatch.setattr(runner, "prepare_first", initial)
    monkeypatch.setattr(runner, "run_cell", cell)
    args = SimpleNamespace(phase=phase, limit=None, campaign=tmp_path)
    protocol = {
        "source_hashes": {},
        "ablation_case_ids": [bundle.case_id],
        "stability_case_ids": [bundle.case_id],
    }
    asyncio.run(runner.run(args, {}, protocol, tmp_path, {"workers": 1}))
    assert len(calls) == expected_count
    if phase == "stability":
        assert {c[0] for c in calls} == {bundle.case_id + "--rep2", bundle.case_id + "--rep3"}
        assert {c[1] for c in calls} == {"full"}
        assert {c[2] for c in calls} == {bundle.task.task_id}
    elif phase == "development":
        assert {c[1] for c in calls} == set(experiment.BASE_METHODS)
    else:
        assert {c[1] for c in calls} == set(experiment.BASE_METHODS + experiment.ABLATIONS)
