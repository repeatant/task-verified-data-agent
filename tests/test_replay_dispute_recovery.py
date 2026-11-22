"""Offline-only response-tape tests; these fixtures are not live model results."""

import asyncio
import importlib.util
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation import dispute_recovery as experiment
from dataverify.runtime.reporting import digest_json, write_json
from dataverify.verification.dispute_runtime import DisputeSession

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dispute_replay_test", ROOT / "tools/replay_dispute_recovery.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_saved_responses_are_verbatim_prompt_independent_and_not_extended():
    action = {"action": "finish", "reason": "Saved reason", "evidence_ids": ["unaltered-child-id"]}
    model = replay.SavedResponseModel([action])
    result = asyncio.run(
        model.generate(
            [{"role": "user", "content": "Ignore the saved answer and write another one"}]
        )
    )
    assert result == action
    result["evidence_ids"][0] = "mutated-by-caller"
    assert model.actions == [action]
    assert model.returned_action_digests == [digest_json(action)]
    with pytest.raises(GraphModelError, match="RESPONSES_EXHAUSTED"):
        asyncio.run(model.generate([{"role": "user", "content": "Add a response"}]))
    assert model.exhausted
    assert len(model.returned_action_digests) == 1


def test_response_history_must_account_for_every_real_request():
    original = {
        "completed": True,
        "record": {"actions": [{"action": "clarify"}], "model_calls": 1},
        "http_events": [{"status": "completed", "usage_complete": True}],
        "contexts": {"dispute_resolver": [[]]},
    }
    assert replay.saved_actions(original) == original["record"]["actions"]
    original["http_events"].append({"status": "failed", "usage_complete": False})
    with pytest.raises(ValueError, match="INCOMPLETE_OR_UNPARSED"):
        replay.saved_actions(original)


def test_query_reexecution_ignores_only_wall_clock_duration():
    original = [{"id": "q0001", "rows": [[6]], "source_id": "same-source", "elapsed_ms": 2.3}]
    same = [{**original[0], "elapsed_ms": 9.7}]
    checks = replay.compare_query_observations(original, same)
    assert checks["reexecuted_query_count"] == 1
    assert checks["checks"][0]["matched"]
    for altered in ({**same[0], "rows": [[7]]}, {**same[0], "source_id": "other-source"}):
        with pytest.raises(ValueError, match="QUERY_EVIDENCE_CHANGED"):
            replay.compare_query_observations(original, [altered])


@pytest.mark.parametrize("relative", ["source", "source/child", "."])
def test_output_cannot_overlap_source_tree(tmp_path, relative):
    with pytest.raises(ValueError, match="SEPARATE_OUTPUT"):
        replay.check_paths(tmp_path / "source", tmp_path / relative)


@pytest.fixture
def saved_example(tmp_path, monkeypatch):
    def no_api(*args, **kwargs):
        raise AssertionError("Offline replay may not construct an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", no_api)
    monkeypatch.setattr(httpx, "Client", no_api)
    case = experiment.study_cases()[4]
    fixture = {
        "name": "shared_diagnostic",
        "data": experiment.visible_diagnostic_fixture().to_dict(),
    }
    session = DisputeSession(
        case.base.task,
        case.base.source,
        case.initial_sql,
        case.reference_plan,
        case.auditor_plan,
        retained_fixtures=(fixture,),
    )
    initial = session.replay()
    actions = [
        {
            "action": "finish",
            "reason": "Authored fixture for offline test.",
            "evidence_ids": [initial["id"]],
        }
    ]
    model = replay.SavedResponseModel(actions)
    record = asyncio.run(
        run_dispute_graph(
            case.base.task,
            case.base.source,
            case.initial_sql,
            case.reference_plan,
            case.auditor_plan,
            model,
            query_source=experiment.visible_diagnostic_fixture(),
            fixtures=(fixture,),
            policy=DisputePolicy(max_actions=4, max_queries=2, max_revisions=2),
        )
    )
    original = {
        "completed": True,
        "record": record,
        "http_events": [{"status": "completed", "usage_complete": True}],
        "contexts": {"dispute_resolver": model.requests},
        "row": {"score": experiment.score_result(case, "dispute_resolver", record)},
    }
    source_path = tmp_path / "source" / "original.json"
    source_path.parent.mkdir(parents=True)
    write_json(source_path, original)
    output = tmp_path / "counterfactual"
    (output / "records").mkdir(parents=True)
    return case, original, source_path, output


def test_offline_replay_has_zero_live_credit_and_leaves_source_untouched(saved_example):
    case, original, source_path, output = saved_example
    before = source_path.read_bytes()
    result = asyncio.run(replay.replay_one(case, original, source_path, output, "fixed-protocol"))
    assert result["new_api_calls"] == result["new_model_tokens"] == 0
    assert result["count_as_new_live_model_success"] is False
    assert (
        result["saved_original_response_count"] == result["consumed_original_response_count"] == 1
    )
    assert result["counterfactual"]["status"] == "reviewed"
    assert all(
        not change["changed"] for change in result["program_changes_from_original_final"].values()
    )
    assert source_path.read_bytes() == before
    assert (
        asyncio.run(replay.replay_one(case, original, source_path, output, "fixed-protocol"))
        == result
    )


def test_changed_pre_action_program_binding_is_not_accepted(saved_example):
    case, original, source_path, output = saved_example
    altered = deepcopy(original)
    import json

    messages = altered["contexts"]["dispute_resolver"][0]
    context = json.loads(messages[-1]["content"])
    context["versions"]["candidate"] = 999
    messages[-1]["content"] = json.dumps(context)
    with pytest.raises(ValueError, match="PRE_ACTION_PROGRAM_OR_SOURCE_CHANGED"):
        asyncio.run(replay.replay_one(case, altered, source_path, output, "fixed-protocol"))
