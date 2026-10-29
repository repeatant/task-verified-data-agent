"""End-to-end graph regressions; no network or real language model is used."""

import json
import sqlite3

import httpx
import pytest

from dataverify.database import database_sha256, seed_demo_database
from dataverify.engine import AgentRunner
from dataverify.providers import Action, JOIN, OfflineScriptedPlanner, QwenPlanner


@pytest.mark.parametrize("case,wrong_value", [("join_fanout", 540), ("distinct_fix", 180)])
def test_rejected_value_cannot_escape_as_final_fact(tmp_path, case, wrong_value):
    database = seed_demo_database(tmp_path / "demo.sqlite3")
    before = database_sha256(database)
    result = AgentRunner(database, tmp_path / "traces").run(case=case)
    queries = [event for event in result["trace"] if event.get("tool") == "query_metric"]
    assert queries[0]["result"]["value"] == wrong_value
    assert queries[0]["result"]["accepted_value"] is None
    assert queries[0]["result"]["verification_status"] == "rejected"
    assert result["status"] == "verified" and result["value"] == 280
    assert result["failed_queries"] == 1
    assert result["provider"] == "offline-scripted" and result["real_model_used"] is False
    assert database_sha256(database) == before
    saved = json.loads(__import__("pathlib").Path(result["trace_path"]).read_text(encoding="utf-8"))
    assert saved["run_id"] == result["run_id"]
    assert queries[-1]["result"]["evidence_id"].startswith(result["run_id"])


def test_no_repair_budget_means_no_accepted_value(tmp_path):
    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database).run(case="join_fanout", max_repairs=0)
    assert result["status"] == "unverified" and result["value"] is None
    assert result["stop_reason"] == "repair_limit"


def test_clean_line_grain_is_not_mistaken_for_bad_join(tmp_path):
    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database).run(case="clean_line_total", max_repairs=0)
    assert result["status"] == "verified" and result["value"] == 280
    assert result["failed_queries"] == 0


@pytest.mark.parametrize("orphan_amount,expected", [(50, 330), (0, 280)])
def test_line_repair_preserves_requested_metric_when_join_omits_orphan(tmp_path, orphan_amount, expected):
    """Repair the missing line coverage, not the question's metric identity.

    A zero-valued orphan deliberately makes the line and order totals equal;
    that numeric coincidence must not hide a switch to a different metric.
    """
    database = seed_demo_database(tmp_path / "orphan-line.sqlite3")
    connection = sqlite3.connect(database)
    try:
        connection.execute("INSERT INTO order_lines VALUES(7,999,?)", (orphan_amount,))
        connection.commit()
    finally:
        connection.close()
    before = database_sha256(database)
    result = AgentRunner(database, tmp_path / "traces").run(
        question="统计明细总收入", case="clean_line_total", max_repairs=1,
    )
    queries = [event for event in result["trace"] if event.get("tool") == "query_metric"]
    assert queries[0]["result"]["error_code"] == "MISSING_GRAIN"
    assert all(event["arguments"]["metric_id"] == "line_revenue" for event in queries)
    assert queries[-1]["result"]["metric_id"] == "line_revenue"
    assert queries[-1]["result"]["accepted_value"] == expected
    assert result["status"] == "verified" and result["value"] == expected
    assert "line_revenue" in result["answer"] and "order_revenue" not in result["answer"]
    assert result["failed_queries"] == 1
    assert database_sha256(database) == before


def test_unsupported_query_cannot_be_promoted_to_verified(tmp_path):
    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database).run(case="unsupported")
    assert result["status"] == "unverified" and result["value"] is None


def test_a_model_claim_without_tool_evidence_is_not_accepted(tmp_path):
    class PrematurePlanner:
        name = "test-stub"
        real_model = False

        def choose(self, messages):
            return Action("final", text="The answer is definitely 999999.")

    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database, planner=PrematurePlanner()).run()
    assert result["status"] == "unverified" and "999999" not in result["answer"]


def test_stubborn_planner_is_stopped_after_bounded_repairs(tmp_path):
    class StubbornPlanner:
        name = "test-stub"
        real_model = False

        def choose(self, messages):
            return Action("tool", "query_metric", {"sql": f"SELECT SUM(o.total) {JOIN}", "metric_id": "order_revenue"}, f"q-{len(messages)}")

    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database, planner=StubbornPlanner()).run(max_repairs=2)
    assert result["status"] == "unverified"
    assert result["failed_queries"] == 3 and result["steps"] == 3


def test_unknown_tools_are_bounded(tmp_path):
    class UnknownToolPlanner:
        name = "test-stub"
        real_model = False

        def choose(self, messages):
            return Action("tool", "not_registered", {}, f"q-{len(messages)}")

    database = seed_demo_database(tmp_path / "demo.sqlite3")
    result = AgentRunner(database, planner=UnknownToolPlanner()).run()
    assert result["status"] == "unverified" and result["steps"] == 8
    assert result["stop_reason"] == "tool_step_limit"


def test_usage_is_counted_per_run_when_planner_is_reused(tmp_path):
    class MeteredStub(OfflineScriptedPlanner):
        def __init__(self):
            super().__init__("clean_order_total")
            self.model = "mock-model"
            self.usage = [{"prompt_tokens": 999, "completion_tokens": 999, "total_tokens": 1998}]

        def choose(self, messages):
            self.usage.append({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
            return super().choose(messages)

    planner = MeteredStub()
    runner = AgentRunner(seed_demo_database(tmp_path / "demo.sqlite3"), planner=planner)
    first, second = runner.run(), runner.run()
    assert first["model"] == "mock-model"
    assert first["model_usage"] == second["model_usage"] == {"responses_with_usage": 2, "prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}


def test_missing_usage_is_not_reported_as_measured_usage(tmp_path):
    planner = OfflineScriptedPlanner("clean_order_total")
    planner.usage = []
    original_choose = planner.choose

    def choose(messages):
        planner.usage.append({})
        return original_choose(messages)

    planner.choose = choose
    result = AgentRunner(seed_demo_database(tmp_path / "demo.sqlite3"), planner=planner).run()
    assert result["status"] == "verified"
    assert result["model_usage"] == {"responses_with_usage": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_qwen_adapter_receives_structured_counterexample_and_repairs_through_graph(tmp_path):
    """Exercise the actual HTTP adapter and graph with a network-free transport.

    This validates protocol and evidence flow, not real model intelligence: the
    mocked model deliberately supplies a bad candidate followed by a repair.
    """
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            name, arguments = "inspect_schema", {}
        elif len(bodies) == 2:
            schema_message = body["messages"][-1]
            assert schema_message["role"] == "tool" and schema_message["name"] == "inspect_schema"
            assert "contracts" in json.loads(schema_message["content"])
            name, arguments = "query_metric", {"sql": f"SELECT SUM(o.total) {JOIN}", "metric_id": "order_revenue"}
        else:
            assert len(bodies) == 3
            feedback_message = body["messages"][-1]
            feedback = json.loads(feedback_message["content"])
            assert feedback_message["tool_call_id"] == "mock-call-2"
            assert feedback["verification_status"] == "rejected"
            assert feedback["accepted_value"] is None
            assert feedback["evidence"]
            name, arguments = "query_metric", {"sql": "SELECT SUM(total) FROM orders", "metric_id": "order_revenue"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"mock-call-{len(bodies)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        })

    database = seed_demo_database(tmp_path / "mock-qwen.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        result = AgentRunner(database, tmp_path / "traces", planner=planner).run(provider="qwen")
    assert len(bodies) == result["steps"] == 3
    assert result["status"] == "verified" and result["value"] == 280
    assert result["failed_queries"] == 1
    assert result["stop_reason"] == "verified_result"
    assert result["model_usage"]["total_tokens"] == 75
    saved = json.loads(__import__("pathlib").Path(result["trace_path"]).read_text(encoding="utf-8"))
    assert saved["trace"][1]["result"]["value"] == 540
    assert saved["trace"][2]["result"]["accepted_value"] == 280


def test_malformed_model_response_stops_with_trace_instead_of_crashing(tmp_path):
    database = seed_demo_database(tmp_path / "malformed-qwen.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": None}]}))) as client:
        planner = QwenPlanner(api_key="test-key-only", client=client)
        result = AgentRunner(database, tmp_path / "traces", planner=planner).run(provider="qwen")
    assert result["status"] == "unverified" and result["value"] is None
    assert result["steps"] == 0
    assert "request/response failed" in result["stop_reason"]
    assert result["trace"][0]["event"] == "provider_error"
    saved = __import__("pathlib").Path(result["trace_path"]).read_text(encoding="utf-8")
    assert "test-key-only" not in saved
