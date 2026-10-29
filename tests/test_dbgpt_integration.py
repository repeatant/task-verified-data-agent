"""Optional real upstream loop regressions; run in .venv-dbgpt, never a fake loop."""

import asyncio
import importlib.util
import json
import sqlite3
from contextlib import closing

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)

from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent

from dataverify.dbgpt_bridge import (
    BAD_ORDER_SUM,
    GOOD_LINE_JOIN,
    GOOD_ORDER_SUM,
    PROJECT_ROOT,
    GuardedDataScientistAgent,
    gated_publication,
    run_native_case,
    sql_candidate,
    verify_upstream_files,
)
from dataverify.guarded_task import ConfirmedMetricTask, GuardedTaskExecutor


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "orders.sqlite3"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total REAL);"
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER,amount REAL);"
            "INSERT INTO orders VALUES(1,100),(2,100),(3,80);"
            "INSERT INTO order_lines VALUES(1,1,40),(2,1,60),(3,2,100),(4,3,20),(5,3,30),(6,3,30);"
        )
    return path


def run(database, tmp_path, candidates, **kwargs):
    task = kwargs.pop("task", ConfirmedMetricTask("Total order revenue", "order_revenue"))
    return asyncio.run(
        run_native_case(
            database,
            task,
            candidates,
            output_dir=tmp_path / "runtime",
            **kwargs,
        )
    )


def test_uses_upstream_generate_reply_and_verify_unchanged():
    assert GuardedDataScientistAgent.generate_reply is DataScientistAgent.generate_reply
    assert GuardedDataScientistAgent.verify is DataScientistAgent.verify
    assert GuardedDataScientistAgent.thinking is DataScientistAgent.thinking
    assert GuardedDataScientistAgent.write_memories is DataScientistAgent.write_memories


def test_changed_upstream_file_manifest_fails_closed(tmp_path):
    manifest = json.loads(
        (PROJECT_ROOT / "docs/research/dbgpt_pin.json").read_text(encoding="utf-8")
    )
    key = next(iter(manifest["sha256"]))
    manifest["sha256"][key] = "0" * 64
    path = tmp_path / "incorrect_pin.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Pinned upstream source changed"):
        verify_upstream_files(path)


def test_original_upstream_really_accepts_wrong_fanout(database, tmp_path):
    record = run(database, tmp_path, [sql_candidate(BAD_ORDER_SUM)], mode="original")
    assert record["upstream_success"]
    assert record["llm_calls"] == 1
    content = json.loads(record["native_reply"]["action_report"]["content"])
    assert content["data"] == [{"revenue": 540}]
    assert len(record["raw_resource_queries"]) == 2  # Action + native correctness_check.


@pytest.mark.parametrize("mode", ["code", "evidence"])
def test_native_retry_receives_rejection_and_repairs(database, tmp_path, mode):
    record = run(
        database, tmp_path, [sql_candidate(BAD_ORDER_SUM), sql_candidate(GOOD_ORDER_SUM)], mode=mode
    )
    assert record["upstream_success"]
    assert record["llm_calls"] == 2
    assert record["publication"]["accepted_value"] == 280
    assert record["publication"]["metric_id"] == "order_revenue"
    assert record["raw_resource_queries"] == []
    assert record["actions"][0]["value"] == 540  # Private diagnostic only.
    assert record["actions"][0]["accepted_value"] is None
    second_request = json.dumps(record["llm_requests"][1], ensure_ascii=False)
    assert "GRAIN_MULTIPLICATION" in second_request
    for witness_fact in ("duplicate_grain", "max_multiplicity", "order_id=3"):
        assert (witness_fact in second_request) == (mode == "evidence")
    events = [
        e["payload"]["action_output"] for e in record["stream_events"] if e["event_type"] == "act"
    ]
    assert events[0]["view"] is None
    assert json.loads(events[0]["content"])["value"] is None
    assert json.loads(events[0]["content"])["accepted_value"] is None
    assert json.loads(record["native_reply"]["content"])["accepted_value"] == 280


def test_legal_detail_join_not_rejected_just_for_one_to_many(database, tmp_path):
    record = run(
        database,
        tmp_path,
        [sql_candidate(GOOD_LINE_JOIN, "line_revenue")],
        task=ConfirmedMetricTask("Total line revenue", "line_revenue"),
    )
    assert record["upstream_success"]
    assert record["llm_calls"] == 1
    assert record["publication"]["accepted_value"] == 280


def test_metric_drift_blocked_even_when_numeric_values_happen_to_match(database, tmp_path):
    record = run(
        database,
        tmp_path,
        [sql_candidate(GOOD_LINE_JOIN, "line_revenue"), sql_candidate(GOOD_ORDER_SUM)],
    )
    assert record["actions"][0]["error_code"] == "METRIC_DRIFT"
    assert record["actions"][0]["execution_status"] == "blocked"
    assert record["publication"]["metric_id"] == "order_revenue"
    assert record["publication"]["accepted_value"] == 280


@pytest.mark.parametrize(
    "responses",
    [
        [sql_candidate(BAD_ORDER_SUM), sql_candidate(BAD_ORDER_SUM)],
        ["not JSON", "not JSON"],
        [sql_candidate(BAD_ORDER_SUM), "not JSON"],
    ],
)
def test_no_unverified_value_after_final_attempt(database, tmp_path, responses):
    record = run(database, tmp_path, responses)
    assert not record["upstream_success"]
    assert record["llm_calls"] == 2
    assert record["publication"]["status"] == "unverified"
    assert record["publication"]["accepted_value"] is None
    assert record["native_reply"]["action_report"]["view"] is None
    assert json.loads(record["native_reply"]["content"])["accepted_value"] is None


def test_bad_format_is_observed_by_real_retry(database, tmp_path):
    record = run(database, tmp_path, ["not JSON", sql_candidate(GOOD_ORDER_SUM)])
    assert "INVALID_ACTION_FORMAT" in json.dumps(record["llm_requests"][1])
    assert record["publication"]["accepted_value"] == 280


def test_source_change_between_check_and_final_publication_is_refused(
    database, tmp_path, monkeypatch
):
    native_hook = GuardedDataScientistAgent.adjust_final_message

    async def change_source_then_finalize(self, success, reply):
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE orders SET total=101 WHERE order_id=1")
            connection.commit()
        return await native_hook(self, success, reply)

    monkeypatch.setattr(
        GuardedDataScientistAgent, "adjust_final_message", change_source_then_finalize
    )
    record = run(database, tmp_path, [sql_candidate(GOOD_ORDER_SUM)])
    assert not record["upstream_success"]
    assert record["publication"]["error_code"] == "SOURCE_CHANGED_DURING_TASK"
    assert record["publication"]["accepted_value"] is None
    assert record["native_reply"]["action_report"]["view"] is None


def test_source_change_before_action_view_is_terminal_and_feedback_consistent(
    database, tmp_path, monkeypatch
):
    execute = GuardedTaskExecutor.execute

    def change_after_execute(self, sql, metric_id):
        feedback = execute(self, sql, metric_id)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE orders SET total=101 WHERE order_id=1")
            connection.commit()
        return feedback

    monkeypatch.setattr(GuardedTaskExecutor, "execute", change_after_execute)
    record = run(database, tmp_path, [sql_candidate(GOOD_ORDER_SUM)])
    assert record["llm_calls"] == 1  # Do not retry a changed source as a SQL error.
    assert record["publication"]["accepted_value"] is None
    assert record["actions"][0]["verification_status"] == "rejected"
    assert record["actions"][0]["error_code"] == "SOURCE_CHANGED_DURING_TASK"
    first_act = next(
        e["payload"]["action_output"] for e in record["stream_events"] if e["event_type"] == "act"
    )
    assert first_act["view"] is None
    assert first_act["have_retry"] is False
    assert json.loads(first_act["content"])["verification_status"] == "rejected"
    assert json.loads(first_act["content"])["accepted_value"] is None


def test_outer_gate_refuses_accepted_result_when_upstream_did_not_finish(database):
    guard = GuardedTaskExecutor(database, ConfirmedMetricTask("Revenue", "order_revenue"))
    guard.execute(GOOD_ORDER_SUM, "order_revenue")
    assert guard.publish()["accepted_value"] == 280
    publication = gated_publication(guard, False)
    assert publication["status"] == "unverified"
    assert publication["value"] is None
    assert publication["verified_sql"] is None
    assert publication["error_code"] == "UPSTREAM_DID_NOT_FINISH"
