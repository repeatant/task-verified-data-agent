"""Offline native-loop integration; no model API or environment-key lookup."""

import asyncio
import importlib.util
import json

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)
pytest.importorskip("hypothesis", reason="Optional probe generation dependencies are required")

from dbgpt.agent.expand.data_scientist_agent import DataScientistAgent

from dataverify.dbgpt_bridge import ScriptedLLMClient
from dataverify.probe_agent_bridge import (
    ProbeChartAction,
    ProbeDataScientistAgent,
    ProbeResource,
    probe_data_sha256,
    probe_publication,
    probe_sql_candidate,
    run_probe_native_case,
)
from dataverify.probe_guard import ProbeGuard
from dataverify.probe_runtime import ProbeData, ProbeTask

GOOD = (
    "SELECT COALESCE(SUM(l.amount),0) AS revenue FROM order_lines l "
    "WHERE l.status='settled' AND l.settled_at >= '2026-09-01T00:00:00' "
    "AND l.settled_at < '2026-10-01T00:00:00'"
)
BAD_END = GOOD.replace("l.settled_at <", "l.settled_at <=")


def task():
    return ProbeTask(
        "native01",
        "计算九月已结算明细净收入，含合法退款。",
        ("exclude_end", "exclude_status", "include_start", "include_negative"),
    )


def data():
    return ProbeData(((1, 700, "web"),), ((1, 1, 100, "settled", "2026-09-15T12:00:00"),))


def run(tmp_path, responses, *, mode="compact", attempts=2):
    return asyncio.run(
        run_probe_native_case(
            task(),
            data(),
            mode=mode,
            max_attempts=attempts,
            output_dir=tmp_path,
            responses=responses,
        )
    )


def test_generate_reply_and_retry_are_inherited_not_reimplemented():
    assert ProbeDataScientistAgent.generate_reply is DataScientistAgent.generate_reply


def test_resource_exposes_new_schema_without_source_rows_or_old_metric_protocol():
    resource = ProbeResource(task())
    try:
        payload = json.loads(resource.get_schema_link(resource._db_name))
        assert payload["tables"]["order_lines"] == [
            "line_id",
            "order_id",
            "amount",
            "status",
            "settled_at",
        ]
        assert payload["confirmed_task"]["task_id"] == "native01"
        text = json.dumps(payload)
        assert "metric_id" not in text
        assert "reference_sql" not in text
        assert "candidate_sql" not in text
        assert '"rows"' not in text
        with pytest.raises(ValueError):
            resource.get_schema_link("another-task")
    finally:
        resource.close()


def test_raw_resource_query_cannot_bypass_guard():
    resource = ProbeResource(task())
    try:
        with pytest.raises(RuntimeError, match="RAW_PROBE_QUERY_DISABLED"):
            resource._sync_query(resource._db_name, GOOD)
        assert resource.raw_query_attempts == 1
    finally:
        resource.close()


@pytest.mark.parametrize("mode", ["full", "compact", "code"])
def test_native_correct_candidate_publishes_guard_rows(tmp_path, mode):
    record = run(tmp_path, [probe_sql_candidate(GOOD, task().task_id)], mode=mode)
    assert record["native_loop"] == "DataScientistAgent.generate_reply (inherited)"
    assert record["upstream_file_check"]["checked_files"] > 0
    assert record["publication"]["status"] == "verified"
    assert record["publication"]["rows"] == [[100]]
    assert record["publication"]["verified_sql"] == GOOD
    assert record["raw_resource_queries"] == 0
    assert record["source_unchanged"]
    assert (
        record["source_sha256_before"] == record["source_sha256_after"] == probe_data_sha256(data())
    )
    assert record["operational_error"] is None
    assert record["cleanup_errors"] == []
    assert record["llm_calls"] == 1
    assert record["http_events"] == []


@pytest.mark.parametrize("mode", ["full", "compact", "code"])
def test_native_rejects_then_repairs_without_replacing_loop(tmp_path, mode):
    record = run(
        tmp_path,
        [probe_sql_candidate(BAD_END, task().task_id), probe_sql_candidate(GOOD, task().task_id)],
        mode=mode,
    )
    assert record["publication"]["status"] == "verified"
    assert record["publication"]["rows"] == [[100]]
    assert len(record["actions"]) == 2
    assert record["actions"][0]["feedback"]["error_code"]
    assert record["llm_calls"] == 2
    assert record["raw_resource_queries"] == 0
    assert not record["http_events"]
    assert "deterministic_offline" in record["client_kind"]


def test_unrepaired_failure_has_no_public_value_or_view(tmp_path):
    record = run(tmp_path, [probe_sql_candidate(BAD_END, task().task_id)], attempts=1)
    assert record["publication"]["status"] == "unverified"
    assert record["publication"]["rows"] is None
    assert record["publication"]["verified_sql"] is None
    assert not record["upstream_success"]
    reply = record["native_reply"]
    assert not reply["success"]
    if reply.get("action_report"):
        assert not reply["action_report"].get("view")


def test_wrong_task_id_is_rejected_and_can_be_corrected(tmp_path):
    record = run(
        tmp_path,
        [probe_sql_candidate(GOOD, "changed-task"), probe_sql_candidate(GOOD, task().task_id)],
    )
    assert record["actions"][0]["feedback"]["error_code"] == "TASK_DRIFT"
    assert record["publication"]["status"] == "verified"


def test_malformed_json_is_invalidated_then_native_retry_recovers(tmp_path):
    record = run(tmp_path, ["not JSON", probe_sql_candidate(GOOD, task().task_id)])
    assert record["actions"][0]["feedback"]["error_code"] == "INVALID_ACTION_FORMAT"
    assert record["publication"]["status"] == "verified"


def test_write_attempt_is_blocked_and_source_remains_unchanged(tmp_path):
    record = run(
        tmp_path,
        [
            probe_sql_candidate("DELETE FROM order_lines", task().task_id),
            probe_sql_candidate(GOOD, task().task_id),
        ],
    )
    assert record["actions"][0]["feedback"]["error_code"]
    assert record["publication"]["status"] == "verified"
    assert record["source_unchanged"]
    assert record["raw_resource_queries"] == 0


def test_invalid_action_clears_previously_verified_state():
    async def scenario():
        guard = ProbeGuard(task(), data())
        guard.execute(GOOD, task().task_id)
        assert guard.publish()["status"] == "verified"
        action = ProbeChartAction()
        action.guard = guard
        output = await action.run("not JSON")
        assert not output.is_exe_success
        assert output.view is None
        assert guard.publish()["status"] == "unverified"

    asyncio.run(scenario())


def test_upstream_failure_cannot_publish_even_a_verified_guard():
    guard = ProbeGuard(task(), data())
    guard.execute(GOOD, task().task_id)
    result = probe_publication(guard, False)
    assert result["status"] == "unverified"
    assert result["rows"] is None
    assert result["verified_sql"] is None


@pytest.mark.parametrize("attempts", [0, 4, True])
def test_bad_attempt_allowance_rejected_before_runtime(tmp_path, attempts):
    with pytest.raises(ValueError):
        asyncio.run(
            run_probe_native_case(
                task(), data(), responses=[], max_attempts=attempts, output_dir=tmp_path
            )
        )


def test_explicit_clients_and_offline_scripts_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(run_probe_native_case(task(), data(), output_dir=tmp_path))
    with pytest.raises(ValueError):
        asyncio.run(
            run_probe_native_case(
                task(), data(), output_dir=tmp_path, responses=[], llm_client=ScriptedLLMClient([])
            )
        )


def test_response_failure_keeps_safe_partial_record_and_no_publication(tmp_path):
    record = run(tmp_path, [])
    assert record["publication"]["status"] == "unverified"
    assert record["publication"]["rows"] is None
    assert record["http_events"] == []


def test_initial_model_context_has_new_task_schema_no_gold(tmp_path):
    record = run(tmp_path, [probe_sql_candidate(GOOD, task().task_id)])
    first = json.dumps(record["llm_requests"][0], ensure_ascii=False)
    assert "task_id" in first
    assert "settled_at" in first
    assert "channel" in first
    assert "reference_sql" not in first
    assert "fault_label" not in first
    assert "metric_id" not in first
