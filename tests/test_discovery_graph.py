"""End-to-end graph contracts with explicitly authored offline role responses."""

import asyncio
from copy import deepcopy

import pytest

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import ScriptedReviewModel
from dataverify.agent.graph_checkpoint import GraphResumeError
from dataverify.agent.graph_model import ScriptedGraphModel
from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.evaluation.discovery_roles import AuthoredReviewModel, authored_attribution
from dataverify.verification.discovery_runtime import rows_equal

FINISH = {"action": "finish", "reason": "Authored bounded review; no semantic proof."}


def models(case, correct=False):
    writer = ScriptedGraphModel(
        [
            {
                "task_id": case.name,
                "sql": case.good_sql if correct else case.bad_sql,
                "thought": "WRITER_PRIVATE_SENTINEL",
            },
            {"task_id": case.name, "sql": case.good_sql},
        ]
    )
    actions = [case.offline_plan()]
    if not correct:
        actions.append(authored_attribution(case.task, "candidate"))
    actions.extend([case.offline_probe(), FINISH])
    return writer, AuthoredReviewModel(actions), ScriptedReviewModel([case.offline_plan()])


def run(case, tmp_path, writer, reviewer, auditor, **kwargs):
    return asyncio.run(
        run_discovery(
            case.task,
            case.source,
            writer,
            reviewer,
            auditor=auditor,
            checkpoint_path=tmp_path / "state.db",
            thread_id="test",
            **kwargs,
        )
    )


@pytest.mark.parametrize("case", get_discovery_cases(), ids=lambda c: c.name)
@pytest.mark.parametrize("correct", [False, True])
def test_dynamic_probe_repairs_and_preserves_valid_sql(case, correct, tmp_path):
    writer, reviewer, auditor = models(case, correct)
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] == "reviewed"
    assert rows_equal(result["publication"]["rows"], case.expected)
    assert result["attempts"] == (1 if correct else 2)
    assert len(result["fixtures"]) >= 2
    assert "WRITER_PRIVATE_SENTINEL" not in str(reviewer.requests + auditor.requests)
    assert case.bad_sql not in str(reviewer.requests[0])
    assert case.bad_sql not in str(auditor.requests)
    assert case.good_sql not in str(auditor.requests)
    assert case.offline_reference not in str(auditor.requests)
    assert "reference" not in str(writer.requests[0])
    assert result["plan"] == case.offline_plan()
    assert result["reference_revisions"] == 0
    assert len(auditor.requests) == 1


def test_finish_cannot_override_observed_failure(tmp_path):
    case = get_discovery_cases()[0]
    writer, reviewer, auditor = models(case)
    reviewer.actions[1] = FINISH
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] != "reviewed"
    assert result["publication"]["rows"] is None
    assert len(writer.requests) == 1


def test_ambiguous_request_stops_before_writer(tmp_path):
    case = get_discovery_cases()[0]
    writer, _, auditor = models(case)
    reviewer = ScriptedReviewModel([{"action": "clarify", "reason": "Which denominator?"}])
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] == "needs_clarification"
    assert not writer.requests and not auditor.requests


def test_reviewer_cannot_rewrite_reference_to_hide_failure(tmp_path):
    case = get_discovery_cases()[0]
    writer, reviewer, auditor = models(case)
    reviewer.actions[1] = case.offline_plan()
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] != "reviewed"
    assert result["reference_revisions"] == 0
    assert len(writer.requests) == 1


def test_resume_keeps_tests_and_reexecutes_without_model_calls(tmp_path):
    case = get_discovery_cases()[0]
    path = tmp_path / "state.db"

    async def scenario():
        writer, reviewer, auditor = models(case)
        paused = await run_discovery(
            case.task,
            case.source,
            writer,
            reviewer,
            auditor=auditor,
            checkpoint_path=path,
            thread_id="test",
            pause_after="plan",
        )
        assert paused["paused"] and not writer.requests and not auditor.requests
        reviewer2 = AuthoredReviewModel(reviewer.actions[1:])
        result = await run_discovery(
            case.task,
            case.source,
            writer,
            reviewer2,
            auditor=auditor,
            checkpoint_path=path,
            thread_id="test",
            resume=True,
        )
        assert result["status"] == "reviewed"
        writer3, reviewer3, auditor3 = models(case)
        resumed = await run_discovery(
            case.task,
            case.source,
            writer3,
            reviewer3,
            auditor=auditor3,
            checkpoint_path=path,
            thread_id="test",
            resume=True,
        )
        assert not writer3.requests and not reviewer3.requests and not auditor3.requests
        assert resumed["publication"]["final_replay"]["all_match"]
        with pytest.raises(GraphResumeError, match="BINDING"):
            await run_discovery(
                case.task,
                case.distinguishing,
                writer3,
                reviewer3,
                auditor=auditor3,
                checkpoint_path=path,
                thread_id="test",
                resume=True,
            )

    asyncio.run(scenario())


def test_plan_syntax_repair_remains_blind_to_sql(tmp_path):
    case = get_discovery_cases()[0]
    writer, reviewer, auditor = models(case, correct=True)
    invalid = deepcopy(case.offline_plan())
    invalid["reference"] = "[[unclosed"
    reviewer.actions.insert(0, invalid)
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] == "reviewed"
    assert result["plan_attempts"] == 2
    assert case.good_sql not in str(reviewer.requests[:2])
    assert "REFERENCE_SYNTAX" in str(reviewer.requests[1])


def test_unknown_model_outcome_refuses_automatic_resume(tmp_path):
    case = get_discovery_cases()[0]
    path = tmp_path / "state.db"
    writer, _, auditor = models(case)

    class InterruptedReviewer(AuthoredReviewModel):
        async def generate(self, messages):
            raise RuntimeError("simulated interruption after request reservation")

    async def scenario():
        with pytest.raises(RuntimeError):
            await run_discovery(
                case.task,
                case.source,
                writer,
                InterruptedReviewer([]),
                auditor=auditor,
                checkpoint_path=path,
                thread_id="test",
            )
        _, reviewer, auditor2 = models(case)
        with pytest.raises(GraphResumeError, match="OUTCOME_UNKNOWN"):
            await run_discovery(
                case.task,
                case.source,
                writer,
                reviewer,
                auditor=auditor2,
                checkpoint_path=path,
                thread_id="test",
                resume=True,
            )
        assert not reviewer.requests and not auditor2.requests

    asyncio.run(scenario())


def test_invalid_probe_is_returned_as_tool_feedback(tmp_path):
    case = get_discovery_cases()[0]
    writer, reviewer, auditor = models(case, correct=True)
    invalid = deepcopy(case.offline_probe())
    invalid["fixtures"][0]["data"]["orders"] = []
    reviewer.actions.insert(1, invalid)
    result = run(case, tmp_path, writer, reviewer, auditor)
    assert result["status"] == "reviewed"
    assert "must reference an existing order" in str(reviewer.requests[2])
    assert len(result["fixtures"]) >= 2


def test_declared_legacy_guard_remains_a_hard_gate(tmp_path):
    from dataverify.verification.discovery_runtime import DiscoveryTask
    from dataverify.verification.probe_runtime import ProbeTask

    case = get_discovery_cases()[0]
    task = DiscoveryTask(
        case.name, case.task.question, legacy_task=ProbeTask(case.name, case.task.question, ())
    )
    writer, _, auditor = models(case, correct=True)
    reviewer = ScriptedReviewModel([case.offline_plan(), FINISH])
    result = asyncio.run(
        run_discovery(
            task,
            case.source,
            writer,
            reviewer,
            auditor=auditor,
            checkpoint_path=tmp_path / "state.db",
            thread_id="test",
        )
    )
    assert result["status"] != "reviewed"
    assert result["observation"]["legacy"]["status"] != "verified"
    assert result["publication"]["rows"] is None
