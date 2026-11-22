import asyncio
import json

import pytest

from dataverify.workbench.live import TaskBudget
from dataverify.workbench.runner import run_task
from dataverify.workbench.sample_source import AMBIGUOUS_COUNT, COUNT_LINES, DATASET, NET_AMOUNT


def run(directory, question, definitions="", **kwargs):
    return asyncio.run(
        run_task(
            run_id="task",
            question=question,
            definitions=definitions,
            source_id=DATASET,
            mode="offline",
            directory=directory,
            **kwargs,
        )
    )


def test_repairs_and_replays_both_programs_on_bound_database(tmp_path):
    stages = []
    result = run(tmp_path, COUNT_LINES, on_stage=stages.append)
    assert stages == ["discovery", "dispute", "publication"]
    assert result["status"] == "reviewed"
    assert result["rows"] == [[4]]
    assert result["columns"] == ["settled_lines"]
    assert [r["target"] for r in result["revisions"]] == ["candidate", "reference"]
    assert result["evidence"][-1]["scope"] == "original_database"
    policy = next(item for item in result["evidence"] if "allow_queries" in item)
    assert policy["allow_queries"] is True and policy["query_count"] == 1
    assert result["usage"]["requests"] == 0
    assert "writer_messages" not in json.dumps(result)
    assert run(tmp_path, COUNT_LINES) == result


def test_zero_model_calls_net_and_clarification(tmp_path):
    net = run(tmp_path / "net", NET_AMOUNT)
    assert net["rows"] == [[265]]
    assert not net["revisions"]
    ambiguous = run(tmp_path / "ask", AMBIGUOUS_COUNT)
    assert ambiguous["status"] == "needs_clarification"
    assert "按订单数" in ambiguous["clarification"]
    assert ambiguous["rows"] is None
    assert run(tmp_path / "orders", AMBIGUOUS_COUNT, "按订单数")["rows"] == [[2]]
    assert run(tmp_path / "lines", AMBIGUOUS_COUNT, "按明细条数")["rows"] == [[4]]


def test_split_citations_are_published_only_after_a_checked_delivery(tmp_path):
    from dataverify.agent.graph_checkpoint import GraphResumeError

    path = tmp_path / "split"
    result = run(path, COUNT_LINES, split_finish_evidence=True)
    assert result["status"] == "reviewed" and result["rows"] == [[4]]
    cited = result["finish_citations"]
    assert cited["supporting_ids"] == ["q0001"]
    latest = next(e for e in reversed(result["evidence"]) if "versions" in e)
    assert cited["signoff_ids"] == [latest["id"]]
    assert result == run(path, COUNT_LINES, split_finish_evidence=True)
    with pytest.raises(GraphResumeError, match="BINDING_CHANGED"):
        run(path, COUNT_LINES, split_finish_evidence=False)
    ambiguous = run(tmp_path / "ask-split", AMBIGUOUS_COUNT, split_finish_evidence=True)
    assert ambiguous["finish_citations"] is None


def test_rejected_delivery_reason_is_visible_after_the_last_turn():
    from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
    from dataverify.agent.dispute_model import ScriptedDisputeModel
    from dataverify.evaluation.discovery_cases import get_discovery_cases
    from dataverify.workbench.runner import _events

    case = get_discovery_cases()[0]
    model = ScriptedDisputeModel(
        [
            {
                "action": "query",
                "sql": "SELECT COUNT(*) FROM orders",
                "hypothesis": "Count orders.",
            },
            {"action": "finish", "reason": "Cite the observation.", "evidence_ids": ["q0001"]},
        ]
    )
    dispute = asyncio.run(
        run_dispute_graph(
            case.task,
            case.source,
            case.good_sql,
            case.offline_plan(),
            case.offline_plan(),
            model,
            policy=DisputePolicy(max_actions=2, reserve_final_decision=True),
        )
    )
    assert dispute["status"] == "inconclusive"
    assert dispute["publication"]["sql"] is None
    evidence, _ = _events({"trace": []}, dispute)
    rejected = [item for item in evidence if item["title"] == "未通过的动作"]
    assert len(rejected) == 1
    assert "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK" in rejected[0]["detail"]


def test_unadvertised_offline_query_and_changed_definition_never_get_canned_answer(tmp_path):
    for text, definition in (("How much is tax?", ""), (NET_AMOUNT, "只计正值")):
        result = run(tmp_path, text, definition)
        assert result["status"] == "inconclusive"
        assert result["rows"] is None
        assert not result["evidence"]
    with pytest.raises(ValueError, match="UNKNOWN_REGISTERED_SOURCE"):
        asyncio.run(
            run_task(
                run_id="task",
                source_id="../elsewhere",
                mode="offline",
                question=COUNT_LINES,
                definitions="",
                directory=tmp_path,
            )
        )


def test_task_accounting_survives_restart_and_excludes_other_runs(tmp_path):
    from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget

    lane = MonetaryRequestBudget(limit=12, money_limit_cny=1, max_output_tokens=900)
    journal = tmp_path / "attempts.json"
    first = TaskBudget(lane, [], journal)
    first.reserve(100)
    usage = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    first.record_tokens(usage)
    lane.reserve(200)
    lane.record_tokens(usage)
    resumed = TaskBudget(lane, json.loads(journal.read_text()), journal)
    resumed.reserve(110)
    resumed.record_tokens(usage)
    assert [r["attempt"] for r in resumed.snapshot()["reservation_ledger"]] == [1, 3]
    assert resumed.snapshot()["reported_total_tokens"] == 24


def test_portable_account_is_explicit_durable_and_never_supersedes_campaign(tmp_path, monkeypatch):
    from dataverify.workbench import live_budget
    from dataverify.workbench.live import _local_lane

    monkeypatch.setattr(live_budget, "ROOT", tmp_path / "fresh-checkout")
    monkeypatch.delenv("DATAVERIFY_BUDGET_CAMPAIGN", raising=False)
    with pytest.raises(FileNotFoundError):
        live_budget.local_account()
    live_budget.initialize_local(1)
    with pytest.raises(ValueError, match="ALREADY_EXISTS"):
        live_budget.initialize_local(3)
    task_directory = tmp_path / "task"
    task_directory.mkdir()
    with _local_lane(task_directory) as lane:
        lane.reserve(95)
        lane.record_tokens({"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10})
    with _local_lane(task_directory) as restored:
        assert restored.snapshot()["http_attempts"] == 1
        assert restored.snapshot()["reported_total_tokens"] == 10
    monkeypatch.setenv("DATAVERIFY_BUDGET_CAMPAIGN", str(tmp_path / "missing-history"))
    with pytest.raises(ValueError, match="HISTORICAL_CAMPAIGN_TAKES_PRECEDENCE"):
        live_budget.initialize_local(1)


@pytest.fixture
def mock_qwen(tmp_path, monkeypatch):
    import httpx

    from dataverify.runtime.probe_cost_budget import PRICE_MODEL
    from dataverify.workbench import live_budget

    monkeypatch.setattr(live_budget, "ROOT", tmp_path / "fresh")
    monkeypatch.delenv("DATAVERIFY_BUDGET_CAMPAIGN", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-standalone-fixture-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", PRICE_MODEL)
    live_budget.initialize_local(2)

    def install(respond):
        payloads = []

        async def serve(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            reply = await respond(payload, len(payloads) - 1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": json.dumps(reply, ensure_ascii=False)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                },
            )

        monkeypatch.setattr(
            httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(serve)
        )
        return payloads

    return install


def test_free_form_question_uses_real_clients_with_in_memory_transport(tmp_path, mock_qwen):
    """Verify wiring of Qwen's full HTTP envelope and metering with zero network."""
    from dataverify.workbench import live_budget

    question = "库里一共有多少笔订单？不用筛选明细状态。"
    plan = {
        "action": "plan",
        "fixtures": [],
        "assumptions": [],
        "basis": [{"source": "question", "quote": "多少笔订单"}],
        "reference": "[[len(tables['orders'])]]",
    }
    answers = [
        plan,
        {"task_id": "free", "sql": "SELECT COUNT(*) AS n FROM orders"},
        plan,
        {"action": "finish", "reason": "Checked bounded rows."},
    ]

    async def respond(payload, index):
        return answers[index]

    payloads = mock_qwen(respond)
    params = dict(
        run_id="free",
        question=question,
        definitions="",
        source_id=DATASET,
        mode="qwen",
        directory=tmp_path / "new-task",
    )
    result = asyncio.run(run_task(**params))
    assert result["status"] == "reviewed"
    assert result["rows"] == [[3]]
    assert result["usage"] == {"requests": 4, "total_tokens": 100, "complete": True}
    assert result == asyncio.run(run_task(**params))
    assert len(payloads) == 4
    assert live_budget.local_account()[1]["http_attempts"] == 4


@pytest.mark.parametrize(
    ("enable_investigation", "reserve_final"), [(False, False), (True, False), (True, True)]
)
def test_live_dispute_policy_matches_prompt_execution_and_resume(
    tmp_path, mock_qwen, enable_investigation, reserve_final
):
    from types import SimpleNamespace

    from dataverify.agent.dispute_model import DISPUTE_PROMPT, FINAL_DECISION_PROMPT
    from dataverify.agent.dispute_no_query import prompt_without_queries
    from dataverify.agent.graph_checkpoint import GraphResumeError
    from dataverify.workbench.offline import offline_models

    task = SimpleNamespace(question=COUNT_LINES, task_id="paired")
    writer, planner, auditor, _ = offline_models(task, "")
    discovery_roles = [planner, writer, auditor, planner]
    checked_sql = "SELECT COUNT(*) AS records FROM order_lines WHERE status='settled'"

    async def respond(payload, index):
        messages = payload["messages"]
        if index < len(discovery_roles):
            return await discovery_roles[index].generate(messages)
        expected = DISPUTE_PROMPT if enable_investigation else prompt_without_queries()
        terminal_turn = reserve_final and index == len(discovery_roles) + 3
        if terminal_turn:
            expected = FINAL_DECISION_PROMPT
        assert messages[0]["content"] == expected
        context = json.loads(messages[1]["content"])
        assert context["remaining"]["tools_enabled"] is (enable_investigation and not terminal_turn)
        if enable_investigation and not context["query_observations"]:
            return {"action": "query", "sql": checked_sql, "hypothesis": "核对明细计数。"}
        if not enable_investigation:
            assert context["remaining"]["queries"] == 0
        programs = context["programs"]
        evidence_ids = [context["latest_replay"]["id"]]
        for role, replacement in (("candidate", checked_sql), ("reference", programs["auditor"])):
            if programs[role] != replacement:
                return {
                    "action": "revise",
                    "target": role,
                    "program": replacement,
                    "requirement_quote": "相同金额的不同明细分别计数",
                    "reason": "金额重复仍要逐行计数。",
                    "evidence_ids": evidence_ids,
                }
        return {"action": "finish", "reason": "重放结果一致。", "evidence_ids": evidence_ids}

    payloads = mock_qwen(respond)
    params = dict(
        run_id="paired",
        question=COUNT_LINES,
        definitions="",
        source_id=DATASET,
        mode="qwen",
        directory=tmp_path / "paired",
    )
    # The disabled branch deliberately omits the option to test the product default.
    option = {"allow_queries": True} if enable_investigation else {}
    if reserve_final:
        option["reserve_final_decision"] = True
    result = asyncio.run(run_task(**params, **option))
    assert result["rows"] == [[4]]
    evidence = result["evidence"]
    policy = next(item for item in evidence if "allow_queries" in item)
    assert policy["allow_queries"] is enable_investigation
    assert policy["reserve_final_decision"] is reserve_final
    assert policy["query_count"] == int(enable_investigation)
    assert any(item.get("scope") == "original_database" for item in evidence)
    used = len(payloads)
    assert result == asyncio.run(run_task(**params, **option))
    with pytest.raises(GraphResumeError, match="BINDING_CHANGED"):
        asyncio.run(run_task(**params, allow_queries=not enable_investigation))
    with pytest.raises(GraphResumeError, match="BINDING_CHANGED"):
        asyncio.run(
            run_task(
                **params,
                allow_queries=enable_investigation,
                reserve_final_decision=not reserve_final,
            )
        )
    assert len(payloads) == used
