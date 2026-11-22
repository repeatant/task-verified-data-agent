"""Ablations remove one mechanism, not legal execution or durable bindings."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from dataverify.agent.discovery_graph import _DiscoveryNodes, run_discovery
from dataverify.agent.discovery_model import ScriptedReviewModel
from dataverify.agent.discovery_policies import DiscoveryPolicy
from dataverify.agent.graph_checkpoint import GraphResumeError
from dataverify.agent.graph_model import ScriptedGraphModel
from dataverify.agent.two_way_graph import run_two_way
from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.verification.discovery_runtime import DiscoveryTask, check_candidate, validate_plan
from dataverify.verification.reference_audit import find_pairwise_fixture, inspect_dispute
from dataverify.verification.relational_data import RelationalData

FINISH = {"action": "finish", "reason": "Offline bounded review."}


def roles(case, sql=None):
    return (
        ScriptedGraphModel([{"task_id": case.name, "sql": sql or case.good_sql}]),
        ScriptedReviewModel([case.offline_plan(), FINISH]),
        ScriptedReviewModel([case.offline_plan()]),
    )


def run(case, tmp_path, models, **kwargs):
    writer, reviewer, auditor = models
    return asyncio.run(
        run_discovery(
            case.task,
            case.source,
            writer,
            reviewer,
            auditor=auditor,
            checkpoint_path=tmp_path / "state.sqlite",
            thread_id="policy",
            **kwargs,
        )
    )


def test_no_search_never_calls_hypothesis(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled search was called")

    monkeypatch.setattr("dataverify.agent.discovery_audit.find_distinguishing_fixture", forbidden)
    case = get_discovery_cases()[0]
    result = run(case, tmp_path, roles(case), policy=DiscoveryPolicy(challenge_search=False))
    assert result["status"] == "reviewed"
    assert result["challenged_pairs"] == []
    assert result["publication"]["final_audit"]
    assert result["binding"]["policy"]["challenge_search"] is False


@pytest.mark.parametrize("blind", [True, False])
def test_auditor_exposure_is_explicit_and_bounded(blind, tmp_path):
    case = get_discovery_cases()[0]
    models = roles(case)
    result = run(case, tmp_path, models, policy=DiscoveryPolicy(blind_audit=blind))
    context = json.loads(models[2].requests[0][-1]["content"])
    assert ("candidate_sql" in context) is (not blind)
    assert ("reference_program" in context) is (not blind)
    assert ("observations" in context) is (not blind)
    assert "writer_messages" not in context
    assert result["publication"]["final_replay"]["all_match"]


def test_changed_policy_cannot_resume_old_checkpoint(tmp_path):
    case = get_discovery_cases()[0]
    run(case, tmp_path, roles(case), pause_after="plan")
    with pytest.raises(GraphResumeError, match="BINDING_CHANGED"):
        run(case, tmp_path, roles(case), resume=True, policy=DiscoveryPolicy(blind_audit=False))


def test_model_directed_attribution_retains_quote_and_check_validation():
    case = get_discovery_cases()[0]
    writer, reviewer, auditor = roles(case)
    nodes = _DiscoveryNodes(
        case.task,
        case.source,
        writer,
        reviewer,
        auditor,
        3,
        3,
        (),
        1,
        1,
        DiscoveryPolicy(evidence_authorization=False),
    )
    plan = case.offline_plan()
    evidence = inspect_dispute(case.task, case.source, case.good_sql, plan, plan, plan["fixtures"])
    action = {
        "action": "attribute",
        "target": "candidate",
        "reason": "Test deliberate unsupported repair.",
        "basis": [{"source": "question", "quote": case.task.question}],
        "checks": [evidence["checks"][0]["check_id"]],
    }
    assert nodes.authorize_action(action, evidence)["target"] == "candidate"
    nodes.policy = DiscoveryPolicy()
    assert nodes.authorize_action(action, evidence)["target"] == "ambiguous"
    nodes.policy = DiscoveryPolicy(evidence_authorization=False)
    invalid = deepcopy(action)
    invalid["checks"] = ["made-up-check"]
    with pytest.raises(ValueError, match="UNKNOWN_CHECK"):
        nodes.authorize_action(invalid, evidence)
    invalid = deepcopy(action)
    invalid["basis"][0]["quote"] = "nonexistent sentence"
    with pytest.raises(ValueError, match="BASIS_NOT_IN_SOURCE"):
        nodes.authorize_action(invalid, evidence)


@pytest.mark.parametrize("correct", [True, False])
def test_two_way_is_a_complete_repair_protocol_without_auditor(tmp_path, correct):
    case = get_discovery_cases()[0]
    writer = ScriptedGraphModel(
        [
            {"task_id": case.name, "sql": case.good_sql if correct else case.bad_sql},
            {"task_id": case.name, "sql": case.good_sql},
        ]
    )
    reviewer = ScriptedReviewModel([case.offline_plan(), FINISH])
    result = asyncio.run(
        run_two_way(
            case.task,
            case.source,
            writer,
            reviewer,
            checkpoint_path=tmp_path / "two.sqlite",
            thread_id="two",
        )
    )
    assert result["status"] == "reviewed"
    assert result["attempts"] == (1 if correct else 2)
    assert result["audit_plan"] is None
    assert result["audit_attempts"] == 0
    assert not any(event["node"] in {"audit", "attribute"} for event in result["trace"])
    assert result["plan"] == case.offline_plan()
    assert result["publication"]["final_replay"]["all_match"]


def test_two_way_does_not_hide_wrong_reference_or_claim_correctness(tmp_path):
    case = get_discovery_cases()[0]
    wrong_plan = case.offline_plan()
    wrong_plan["reference"] = "[[123456]]"
    writer = ScriptedGraphModel([{"task_id": case.name, "sql": case.good_sql}] * 3)
    reviewer = ScriptedReviewModel([wrong_plan])
    result = asyncio.run(
        run_two_way(
            case.task,
            case.source,
            writer,
            reviewer,
            checkpoint_path=tmp_path / "two.sqlite",
            thread_id="two",
        )
    )
    assert result["status"] == "unresolved"
    assert result["publication"]["rows"] is None
    assert result["reference_revisions"] == 0
    assert len(writer.requests) == 3


def test_two_way_replays_without_new_calls(tmp_path):
    case = get_discovery_cases()[0]

    async def scenario():
        writer, reviewer, _ = roles(case)
        kwargs = {"checkpoint_path": tmp_path / "two.sqlite", "thread_id": "two"}
        first = await run_two_way(case.task, case.source, writer, reviewer, **kwargs)
        writer2, reviewer2, _ = roles(case)
        second = await run_two_way(
            case.task, case.source, writer2, reviewer2, resume=True, **kwargs
        )
        assert second["publication"] == first["publication"]
        assert not writer2.requests and not reviewer2.requests

    asyncio.run(scenario())


@pytest.mark.parametrize("flag", ["blind_audit", "challenge_search", "evidence_authorization"])
def test_policy_rejects_truthy_non_boolean(flag):
    with pytest.raises(ValueError, match="BOOLEANS"):
        DiscoveryPolicy(**{flag: "false"})


def generic_example():
    schema = {"metrics": ["id", "value"]}
    types = {"metrics": ["INTEGER", "INTEGER"]}
    keys = {"metrics": ["id"]}
    source = RelationalData(schema, {"metrics": [[1, 10], [2, -3]]}, types, keys)
    task = DiscoveryTask(
        "generic_policy",
        "Return the sum of all metric values.",
        "Include negative values. Return zero for no rows.",
        schema=schema,
        column_types=types,
        primary_keys=keys,
    )
    plan = {
        "action": "plan",
        "basis": [{"source": "question", "quote": task.question}],
        "assumptions": [],
        "reference": "[[sum(r['value'] for r in tables['metrics'])]]",
        "fixtures": [{"name": "sample", "data": {"metrics": [[1, 10], [2, -3]]}}],
    }
    sql = "SELECT COALESCE(SUM(value),0) FROM metrics"
    return task, source, plan, sql


def test_generic_pairwise_search_finds_witness_without_third_program():
    task, source, plan, sql = generic_example()
    plan = validate_plan(plan, task)
    wrong = sql + " WHERE value>=0"
    witness = find_pairwise_fixture(wrong, plan, max_examples=30, source=source)
    assert witness is not None
    assert not check_candidate(wrong, plan, [witness])["all_match"]
    assert check_candidate(sql, plan, [witness])["all_match"]


@pytest.mark.parametrize("two_way", [True, False])
def test_generic_graphs_preserve_context_boundaries_and_actual_schema(tmp_path, two_way):
    task, source, plan, sql = generic_example()
    writer = ScriptedGraphModel(
        [{"task_id": task.task_id, "sql": sql, "thought": "PRIVATE_WRITER_MARK"}]
    )
    reviewer = ScriptedReviewModel([plan, FINISH])
    auditor = ScriptedReviewModel([plan])
    kwargs = {"checkpoint_path": tmp_path / "generic.sqlite", "thread_id": "generic"}
    if two_way:
        result = asyncio.run(run_two_way(task, source, writer, reviewer, **kwargs))
        assert result["audit_plan"] is None
        assert auditor.requests == []
    else:
        result = asyncio.run(
            run_discovery(task, source, writer, reviewer, auditor=auditor, **kwargs)
        )
        context = json.loads(auditor.requests[0][-1]["content"])
        assert set(context) == {"task", "data_samples"}
        assert set(context["task"]["schema"]) == {"metrics"}
        assert sql not in str(auditor.requests)
        assert plan["reference"] not in str(auditor.requests)
    assert "PRIVATE_WRITER_MARK" not in str(reviewer.requests + auditor.requests)
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == [[7]]
    assert result["publication"]["final_replay"]["all_match"]


@pytest.mark.parametrize("two_way", [True, False])
def test_source_only_plan_still_executes_source_and_final_replay(tmp_path, two_way):
    task, source, plan, sql = generic_example()
    task = replace(task, allow_source_only_plan=True)
    plan["fixtures"] = []
    writer = ScriptedGraphModel([{"task_id": task.task_id, "sql": sql}])
    reviewer = ScriptedReviewModel([plan, FINISH])
    auditor = ScriptedReviewModel([plan])
    kwargs = {"checkpoint_path": tmp_path / "source_only.sqlite", "thread_id": "source_only"}
    if two_way:
        result = asyncio.run(run_two_way(task, source, writer, reviewer, **kwargs))
    else:
        result = asyncio.run(
            run_discovery(task, source, writer, reviewer, auditor=auditor, **kwargs)
        )
        assert "fixtures 必须为 []" in auditor.requests[0][0]["content"]
    assert "fixtures 必须为 []" in reviewer.requests[0][0]["content"]
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == [[7]]
    assert result["publication"]["final_replay"]["checks"][0]["fixture"] == "bound_source"


@pytest.mark.parametrize("two_way", [True, False])
def test_generic_reviewer_patch_is_expanded_and_replayed_without_mutating_source(tmp_path, two_way):
    task, source, plan, sql = generic_example()
    original = source.to_dict()
    task = replace(task, allow_source_only_plan=True)
    plan["fixtures"] = []
    patch = {
        "action": "probe",
        "reason": "Exercise another signed value.",
        "fixtures": [
            {
                "name": "edited_observation",
                "data": {
                    "base": "bound_source",
                    "changes": [{"table": "metrics", "row": 1, "values": {"value": -8}}],
                },
            }
        ],
    }
    writer = ScriptedGraphModel([{"task_id": task.task_id, "sql": sql}])
    reviewer = ScriptedReviewModel([plan, patch, FINISH])
    auditor = ScriptedReviewModel([plan])
    kwargs = {"checkpoint_path": tmp_path / "source_patch.sqlite", "thread_id": "patch"}
    if two_way:
        result = asyncio.run(run_two_way(task, source, writer, reviewer, **kwargs))
    else:
        result = asyncio.run(
            run_discovery(task, source, writer, reviewer, auditor=auditor, **kwargs)
        )
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == [[7]]
    assert source.to_dict() == original
    retained = next(f for f in result["fixtures"] if f["name"] == "edited_observation")
    assert retained["data"]["tables"]["metrics"][1][1] == -8
    replay = result["publication"]["final_replay"]
    checked = next(c for c in replay["checks"] if c["fixture"] == "edited_observation")
    assert checked["actual"] == [[2]] and checked["status"] == "match"
    assert "ZERO-BASED" in reviewer.requests[1][0]["content"]
