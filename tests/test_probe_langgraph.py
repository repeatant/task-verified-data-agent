"""Exercise graph behavior at real persistence and publication boundaries."""

import asyncio
from dataclasses import replace

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite.aio")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dataverify.graph_checkpoint import GraphResumeError
from dataverify.graph_model import GraphModelError, ScriptedGraphModel
from dataverify.probe_cases import get_case, get_cases, reference_result
from dataverify.probe_langgraph import _GraphNodes, run_graph_case


def action(case, sql=None):
    return {"task_id": case.task.task_id, "sql": sql or case.reference_sql}


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("case", get_cases(), ids=lambda item: item.case_id)
def test_reference_candidates_share_the_existing_guard(case, tmp_path):
    result = run(
        run_graph_case(
            case.task,
            case.source,
            ScriptedGraphModel([action(case)]),
            checkpoint_path=tmp_path / "graph.db",
            thread_id="check",
        )
    )
    assert result["publication"]["status"] == "verified"
    assert result["publication"]["rows"] == reference_result(case)
    assert result["source_unchanged"] is True
    assert result["model_attempts"] == 1
    assert result["publication_verification"]["source_execution"]["status"] == "ok"


def test_resume_retains_actual_witness_and_repair_feedback(tmp_path):
    case = get_case("d06")
    path = tmp_path / "graph.db"

    async def scenario():
        first = ScriptedGraphModel([action(case, case.candidate_sql)])
        paused = await run_graph_case(
            case.task,
            case.source,
            first,
            checkpoint_path=path,
            thread_id="resume",
            pause_after="verify",
        )
        assert paused["status"] == "paused"
        assert paused["publication"]["rows"] is None
        assert paused["next_nodes"] == ["feedback"]
        second = ScriptedGraphModel([action(case)])
        result = await run_graph_case(
            case.task, case.source, second, checkpoint_path=path, thread_id="resume", resume=True
        )
        assert result["model_attempts"] == 2
        assert len(first.requests) == len(second.requests) == 1
        assert result["actions"][-1]["regression_checks"]
        assert result["publication"]["rows"] == reference_result(case)
        assert "diagnostic_evidence" in second.requests[0][-1]["content"]
        assert "reference_sql" not in str(second.requests)
        return result

    run(scenario())


def test_adaptive_escalation_survives_restart(tmp_path):
    case = get_case("d06")
    path = tmp_path / "graph.db"

    async def scenario():
        await run_graph_case(
            case.task,
            case.source,
            ScriptedGraphModel([action(case, case.candidate_sql)]),
            checkpoint_path=path,
            thread_id="adaptive",
            pause_after="verify",
        )
        model = ScriptedGraphModel([action(case, case.candidate_sql), action(case)])
        result = await run_graph_case(
            case.task, case.source, model, checkpoint_path=path, thread_id="adaptive", resume=True
        )
        strategy = result["actions"][1]["feedback"]["feedback_strategy"]
        assert strategy["selected_mode"] == "full"
        assert strategy["escalation_reason"] == "repeated_failed_candidate"
        assert result["publication"]["status"] == "verified"

    run(scenario())


def test_completed_resume_reexecutes_instead_of_trusting_cached_rows(tmp_path, monkeypatch):
    case = get_case("d01")
    path = tmp_path / "graph.db"

    async def scenario():
        await run_graph_case(
            case.task,
            case.source,
            ScriptedGraphModel([action(case)]),
            checkpoint_path=path,
            thread_id="done",
        )
        model = ScriptedGraphModel([action(case)])
        calls = []
        original = _GraphNodes.publish

        def observed(self, state):
            state["publication"]["rows"] = [[-999999]]
            calls.append(True)
            return original(self, state)

        monkeypatch.setattr(_GraphNodes, "publish", observed)
        result = await run_graph_case(
            case.task, case.source, model, checkpoint_path=path, thread_id="done", resume=True
        )
        assert calls and model.requests == []
        assert result["publication"]["rows"] == reference_result(case)

    run(scenario())


@pytest.mark.parametrize("change", ["source", "task", "config", "runtime"])
def test_resume_refuses_changed_binding(change, tmp_path, monkeypatch):
    case = get_case("d01")
    path = tmp_path / "graph.db"

    async def scenario():
        await run_graph_case(
            case.task,
            case.source,
            ScriptedGraphModel([action(case)]),
            checkpoint_path=path,
            thread_id="bound",
        )
        task, data, options = case.task, case.source, {}
        if change == "source":
            data = replace(data, orders=((999, 1, None),), lines=())
        if change == "task":
            task = replace(task, question="Changed confirmed request")
        if change == "config":
            options["max_attempts"] = 2
        if change == "runtime":
            monkeypatch.setattr(
                "dataverify.probe_langgraph.runtime_fingerprint", lambda: "new-version"
            )
        model = ScriptedGraphModel([action(case)])
        with pytest.raises(GraphResumeError, match="TASK_SOURCE_RUNTIME_OR_CONFIG_CHANGED"):
            await run_graph_case(
                task, data, model, checkpoint_path=path, thread_id="bound", resume=True, **options
            )
        assert not model.requests

    run(scenario())


def test_interrupted_model_node_is_not_automatically_retried(tmp_path):
    case = get_case("d01")
    path = tmp_path / "graph.db"

    class CrashingModel(ScriptedGraphModel):
        async def generate(self, messages):
            raise RuntimeError("simulated death after possible HTTP side effect")

    async def scenario():
        with pytest.raises(RuntimeError):
            await run_graph_case(
                case.task,
                case.source,
                CrashingModel([action(case)]),
                checkpoint_path=path,
                thread_id="crash",
            )
        model = ScriptedGraphModel([action(case)])
        with pytest.raises(GraphResumeError, match="MODEL_CALL_OUTCOME_UNKNOWN"):
            await run_graph_case(
                case.task, case.source, model, checkpoint_path=path, thread_id="crash", resume=True
            )
        assert model.requests == []
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": "crash"}})
            assert saved.checkpoint["channel_values"]["call_pending"] is True

    run(scenario())


def test_operational_error_stops_and_never_publishes_model_values(tmp_path):
    case = get_case("d01")

    class UnavailableModel(ScriptedGraphModel):
        async def generate(self, messages):
            raise GraphModelError("GRAPH_QWEN_TRANSPORT_TIMEOUT")

    result = run(
        run_graph_case(
            case.task,
            case.source,
            UnavailableModel([action(case)]),
            checkpoint_path=tmp_path / "graph.db",
            thread_id="error",
        )
    )
    assert result["publication"]["status"] == "unverified"
    assert result["publication"]["rows"] is None
    assert result["model_attempts"] == 1


def test_crash_after_completed_generation_reuses_checkpointed_candidate(tmp_path, monkeypatch):
    case = get_case("d01")
    path = tmp_path / "graph.db"
    verify = _GraphNodes.verify

    def crashed_verification(self, state):
        raise RuntimeError("process stopped after generation checkpoint")

    async def scenario():
        first = ScriptedGraphModel([action(case)])
        monkeypatch.setattr(_GraphNodes, "verify", crashed_verification)
        with pytest.raises(RuntimeError):
            await run_graph_case(
                case.task, case.source, first, checkpoint_path=path, thread_id="known-candidate"
            )
        assert len(first.requests) == 1
        monkeypatch.setattr(_GraphNodes, "verify", verify)
        resumed = ScriptedGraphModel([action(case)])
        result = await run_graph_case(
            case.task,
            case.source,
            resumed,
            checkpoint_path=path,
            thread_id="known-candidate",
            resume=True,
        )
        assert resumed.requests == []
        assert result["model_attempts"] == 1
        assert result["publication"]["rows"] == reference_result(case)

    run(scenario())


def test_malformed_action_uses_same_bounded_repair_loop(tmp_path):
    case = get_case("d01")
    model = ScriptedGraphModel(['{"answer": 100}', action(case)])
    result = run(
        run_graph_case(
            case.task, case.source, model, checkpoint_path=tmp_path / "graph.db", thread_id="format"
        )
    )
    assert result["actions"][0]["feedback"]["error_code"] == "INVALID_ACTION_FORMAT"
    assert result["model_attempts"] == 2
    assert result["publication"]["status"] == "verified"


def test_failed_candidates_cannot_extend_attempt_limit_on_resume(tmp_path):
    case = get_case("d06")
    path = tmp_path / "graph.db"

    async def scenario():
        failed = await run_graph_case(
            case.task,
            case.source,
            ScriptedGraphModel([action(case, case.candidate_sql)]),
            checkpoint_path=path,
            thread_id="limit",
            max_attempts=1,
        )
        assert failed["publication"]["rows"] is None
        with pytest.raises(GraphResumeError):
            await run_graph_case(
                case.task,
                case.source,
                ScriptedGraphModel([action(case)]),
                checkpoint_path=path,
                thread_id="limit",
                resume=True,
                max_attempts=3,
            )

    run(scenario())
