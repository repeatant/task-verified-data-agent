"""Authored protocol checks; none of these tests measures LLM repair quality."""

import asyncio
import json
import sqlite3
from dataclasses import replace

import pytest

from dataverify.agent.evidence_graph import EvidencePolicy, run_evidence_graph
from dataverify.agent.evidence_model import (
    ScriptedEvidenceModel,
    evidence_schema,
    parse_evidence_action,
)
from dataverify.agent.graph_model import GraphModelError
from dataverify.verification.discovery_runtime import DiscoveryTask
from dataverify.verification.relational_data import RelationalData, SQLiteSource

QUESTION = "Return net revenue including refunds."
BAD_SQL = "SELECT SUM(amount) FROM transactions WHERE amount > 0"
GOOD_SQL = "SELECT SUM(amount) FROM transactions"
QUERY = {
    "action": "query",
    "sql": "SELECT COUNT(*), SUM(amount) FROM transactions WHERE amount < 0",
    "hypothesis": "Are any refunds excluded by the positive-amount predicate?",
}


def data():
    return RelationalData(
        schema={"transactions": ["id", "amount"]},
        tables={"transactions": [[1, 100], [2, -25]]},
        column_types={"transactions": ["INTEGER", "INTEGER"]},
        primary_keys={"transactions": ["id"]},
    )


def task(source=None):
    source = source or data()
    return DiscoveryTask(
        "net_revenue",
        QUESTION,
        "Amounts are integer minor units.",
        schema=source.schema,
        column_types=source.column_types,
        primary_keys=source.primary_keys,
        foreign_keys=source.foreign_keys,
    )


def revise(sql=GOOD_SQL, evidence_ids=None, quote="including refunds"):
    return {
        "action": "revise",
        "sql": sql,
        "reason": "The predicate excludes observed refunds required by the original question.",
        "requirement_quote": quote,
        "evidence_ids": evidence_ids or ["e0002"],
    }


def finish(identifier="e0001"):
    return {
        "action": "finish",
        "reason": "Finite original-database review, no semantic guarantee.",
        "evidence_ids": [identifier],
    }


def run(actions, initial=BAD_SQL, source=None, policy=None, original_task=None):
    source = source or data()
    model = ScriptedEvidenceModel(actions)
    result = asyncio.run(
        run_evidence_graph(
            original_task or task(source),
            source,
            initial,
            model,
            policy=policy or EvidencePolicy(max_actions=len(actions)),
        )
    )
    return result, model


def test_query_repair_replay_and_finish_are_distinct_steps():
    result, model = run([QUERY, revise(), finish("e0003")])
    assert result["status"] == "reviewed"
    assert result["publication"]["rows"] == [[75]]
    assert result["model_calls"] == 3
    assert result["query_calls"] == result["revision_attempts"] == 1
    assert result["versions"][0]["sql"] == BAD_SQL
    assert result["versions"][1]["evidence_ids"] == ["e0002"]
    assert result["versions"][1]["previous_version"] == 1
    assert result["observations"][1]["rows"] == [[1, -25]]
    assert result["observations"][2]["rows"] == [[75]]
    assert "not_semantically_verified" in result["publication"]["assurance"]
    context = json.loads(model.requests[2][1]["content"])
    assert context["last_feedback"]["adopted"] is True
    assert context["candidate"]["version"] == 2


def test_finish_preserves_suitable_sql_without_mandatory_tool_calls():
    result, _ = run([finish()], initial=GOOD_SQL)
    assert result["status"] == "reviewed"
    assert result["candidate"]["sql"] == GOOD_SQL
    assert result["query_calls"] == result["revision_attempts"] == 0


def test_wrong_model_judgment_is_not_relabelled_as_verified():
    result, _ = run([finish()])
    assert result["publication"]["rows"] == [[100]]  # Executable but semantically wrong.
    assert result["publication"]["assurance"] == (
        "executed_and_model_reviewed_not_semantically_verified"
    )
    assert "gold" not in str(result).lower()


def test_no_tools_ablation_gets_same_execution_and_can_revise():
    result, model = run(
        [revise(evidence_ids=["e0001"]), finish("e0002")],
        policy=EvidencePolicy(tools_enabled=False, max_actions=2),
    )
    assert result["publication"]["rows"] == [[75]]
    context = json.loads(model.requests[0][1]["content"])
    assert context["observations"][0]["rows"] == [[100]]
    assert context["remaining"]["tools_enabled"] is False
    assert result["query_calls"] == 0


def test_disabled_queries_do_not_execute_or_consume_query_budget():
    result, _ = run([QUERY, finish()], policy=EvidencePolicy(tools_enabled=False, max_actions=2))
    assert result["query_calls"] == 0
    assert len(result["observations"]) == 1
    assert result["trace"][2]["code"] == "EVIDENCE_TOOLS_DISABLED"


@pytest.mark.parametrize(
    ("action", "code"),
    [
        (revise(evidence_ids=["invented"]), "EVIDENCE_UNKNOWN_OBSERVATION"),
        (revise(evidence_ids=["e0001"], quote="made up"), "EVIDENCE_QUOTE_NOT_IN_REQUIREMENT"),
    ],
)
def test_provenance_gate_rejects_invented_support(action, code):
    result, _ = run([action, finish()])
    assert result["candidate"]["sql"] == BAD_SQL
    assert result["revision_attempts"] == 0
    assert any(event.get("code") == code for event in result["trace"])


def test_failed_revision_preserves_original_and_records_failed_sql():
    result, _ = run([revise("DELETE FROM transactions", ["e0001"]), finish()])
    assert result["candidate"]["sql"] == BAD_SQL
    assert len(result["versions"]) == 2
    assert result["versions"][1]["accepted"] is False
    assert result["observations"][1]["status"] == "rejected"
    assert result["publication"]["rows"] == [[100]]


def test_execution_error_is_usable_evidence_for_schema_repair():
    action = revise(GOOD_SQL, ["e0001"])
    action["reason"] = (
        "The real execution reports an unknown column; use the declared amount column."
    )
    result, _ = run([action, finish("e0002")], initial="SELECT SUM(missing) FROM transactions")
    assert result["observations"][0]["status"] == "error"
    assert result["publication"]["rows"] == [[75]]


def test_successful_revision_is_not_automatically_published_at_budget_limit():
    result, _ = run([revise(evidence_ids=["e0001"])])
    assert result["candidate"]["sql"] == GOOD_SQL
    assert result["status"] == "inconclusive"
    assert result["publication"]["sql"] is None
    assert result["reason"] == "EVIDENCE_MODEL_BUDGET_EXHAUSTED"


def test_finish_requires_current_execution_after_revision():
    result, _ = run([revise(evidence_ids=["e0001"]), finish("e0001")])
    assert result["status"] == "inconclusive"
    assert any(item.get("code") == "EVIDENCE_STALE_CANDIDATE_VERSION" for item in result["trace"])


def test_invalid_initial_execution_cannot_finish():
    result, _ = run([finish()], initial="DELETE FROM transactions")
    assert result["status"] == "inconclusive"
    assert result["publication"]["rows"] is None


def test_clarification_does_not_publish_candidate():
    result, _ = run(
        [{"action": "clarify", "reason": "Which currency conversion date is intended?"}]
    )
    assert result["status"] == "needs_clarification"
    assert result["candidate"]["sql"] == BAD_SQL
    assert result["publication"]["sql"] is None


def test_model_format_failure_is_bounded_and_reported():
    result, model = run([{"unexpected": True}, finish()])
    assert result["status"] == "reviewed"
    assert result["model_calls"] == 2
    assert "EVIDENCE_INVALID_ACTION" in model.requests[1][1]["content"]


def test_query_budget_is_not_reset_by_rejected_actions():
    result, _ = run([QUERY, QUERY, finish()], policy=EvidencePolicy(max_actions=3, max_queries=1))
    assert result["query_calls"] == 1
    assert len(result["observations"]) == 2
    assert any(item.get("code") == "EVIDENCE_QUERY_BUDGET" for item in result["trace"])


def test_original_database_not_preview_and_private_path_not_in_context(tmp_path):
    path = tmp_path / "PRIVATE_PATH_SENTINEL.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE transactions(id INTEGER PRIMARY KEY, amount INTEGER)")
        connection.executemany("INSERT INTO transactions VALUES(?,?)", [(1, 100), (2, -25)])
    source = SQLiteSource.from_path(path)
    result, model = run([QUERY, revise(), finish("e0003")], source=source)
    assert result["source"]["scope"] == "original_database"
    assert result["publication"]["rows"] == [[75]]
    assert "PRIVATE_PATH_SENTINEL" not in str(model.requests)
    assert "PRIVATE_PATH_SENTINEL" not in str(result)


def test_prompt_rows_are_explicitly_sampled_but_saved_evidence_is_complete():
    source = data()
    source = replace(source, tables={"transactions": [[i, i] for i in range(1, 101)]})
    result, model = run([finish()], initial="SELECT * FROM transactions", source=source)
    prompt = json.loads(model.requests[0][1]["content"])
    assert len(result["observations"][0]["rows"]) == 100
    assert len(prompt["observations"][0]["rows"]) == 12
    assert prompt["observations"][0]["context_rows_omitted"] == 88


def test_task_and_source_schema_must_match():
    wrong = DiscoveryTask("wrong", QUESTION, schema={"other": ["x"]})
    with pytest.raises(ValueError, match="TASK_SOURCE_SCHEMA_MISMATCH"):
        run([finish()], original_task=wrong)


def test_context_budget_fails_before_model_call_and_preserves_requirement():
    original_task = replace(task(), question="Requirement " + "x" * 5900)
    result, model = run(
        [finish()], original_task=original_task, policy=EvidencePolicy(max_context_chars=4000)
    )
    assert result["status"] == "inconclusive"
    assert result["reason"] == "EVIDENCE_CONTEXT_BUDGET"
    assert result["model_calls"] == 0
    assert model.requests == []


@pytest.mark.parametrize("invalid", [True, -1, 13])
def test_policy_rejects_unbounded_model_calls(invalid):
    with pytest.raises(ValueError):
        EvidencePolicy(max_actions=invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        {"action": "query", "sql": "select 1", "hypothesis": "x", "path": "secret"},
        {"action": "finish", "reason": "x", "evidence_ids": []},
        {"action": "finish", "reason": "x", "evidence_ids": ["a", "a"]},
        '{"action":"finish","action":"clarify","reason":"x"}',
    ],
)
def test_action_parser_rejects_extra_fields_duplicates_and_empty_evidence(invalid):
    with pytest.raises(GraphModelError):
        parse_evidence_action(invalid)


def test_schema_declares_closed_actions():
    for variant in evidence_schema()["anyOf"]:
        assert variant["additionalProperties"] is False
        assert set(variant["required"]) == set(variant["properties"])
