"""Harness scoring/denominator/trace regressions; never call a real model."""

import asyncio
import importlib.util
import json
import sqlite3

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)

from dataverify.agent_benchmark import run_benchmark, summarize_rows
from dataverify.benchmark_tasks import get_task, write_fixture
from dataverify.dbgpt_bridge import ReadOnlyDemoResource, ScriptedLLMClient


def grade(**kwargs):
    return {
        "accepted": True,
        "target_defined": True,
        "correct_answer": True,
        "correct_refusal": False,
        "wrong_release": False,
        "unscorable": False,
        "metric_drift": False,
        **kwargs,
    }


def row(**kwargs):
    return {
        "task_id": "case",
        "mode": "evidence",
        "track": "natural",
        "status": "completed",
        "grade": grade(),
        "http_attempts": 1,
        "native_turns": 1,
        "seconds": 1,
        **kwargs,
    }


def test_scheduled_denominator_keeps_unrun_and_operational_errors():
    rows = [row(), row(status="not_run", http_attempts=0), row(status="operational_error")]
    group = summarize_rows(rows, 3)["groups"]["evidence:all"]
    assert group["target_outcomes_over_scheduled"] == {"numerator": 1, "denominator": 3}
    assert group["completed"] == group["not_run"] == group["operational_errors"] == 1


def test_unscorable_accepted_sql_is_visible_and_never_counted_as_correct():
    rows = [row(grade=grade(correct_answer=False, unscorable=True))]
    group = summarize_rows(rows, 1)["groups"]["evidence:all"]
    assert group["accepted"] == 1
    assert group["unscorable"] == 1
    assert group["correct_answers"] == group["known_wrong_releases"] == 0
    assert group["target_outcomes_over_scheduled"]["numerator"] == 0


def test_explicit_correct_refusal_separate_from_target_answer():
    rows = [
        row(
            grade=grade(
                accepted=False, target_defined=False, correct_answer=False, correct_refusal=True
            )
        )
    ]
    group = summarize_rows(rows, 1)["groups"]["evidence:all"]
    assert group["correct_refusals"] == 1 and group["correct_answers"] == 0
    assert group["target_outcomes_over_scheduled"] == {"numerator": 1, "denominator": 1}


def test_native_failed_run_is_counted_as_operational_error_after_grading():
    rows = [row(grade=grade(accepted=False, correct_answer=False, operational_failure=True))]
    group = summarize_rows(rows, 1)["groups"]["evidence:all"]
    assert group["operational_errors"] == 1
    assert group["target_outcomes_over_scheduled"] == {"numerator": 0, "denominator": 1}


def test_small_actual_native_harness_and_controlled_feedback_isolation(tmp_path):
    output = tmp_path / "run"
    report = asyncio.run(
        run_benchmark(output, tasks=(get_task("n01"), get_task("c02"), get_task("c05")))
    )
    assert report["status"] == "completed"
    assert report["real_model_used"] is False
    assert report["budget"]["http_attempts"] == 0
    assert report["all_completed_sources_unchanged"]
    assert len(report["rows"]) == 9
    assert report["summary"]["groups"]["original:controlled"]["known_wrong_releases"] == 1
    assert report["summary"]["groups"]["code:controlled"]["known_wrong_releases"] == 0
    assert report["summary"]["groups"]["evidence:controlled"]["known_wrong_releases"] == 0
    assert report["summary"]["groups"]["code:controlled"]["valid_initial_candidates_rejected"] == 0
    assert report["http_ledger_complete"]
    assert all(pair["same_first_decision"] for pair in report["paired_feedback"])
    manifest = json.loads((output / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == report["manifest_sha256"]
    for mode in ("code", "evidence"):
        trace = json.loads(
            (output / "traces" / ("c02__" + mode + ".json")).read_text(encoding="utf-8")
        )
        messages = json.dumps(trace["llm_requests"][1], ensure_ascii=False)
        assert ("duplicate_grain" in messages) == (mode == "evidence")
        assert "metric_value" not in messages
        assert "should_accept" not in messages
        assert "candidate_reason" not in messages


def test_existing_experiment_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        asyncio.run(run_benchmark(tmp_path, tasks=(get_task("n01"),)))


def test_harness_does_not_drop_a_crashed_scheduled_cell(tmp_path):
    def factory(task, mode, budget):
        if mode == "code":
            raise RuntimeError("A private injected diagnostic must not be copied")
        return ScriptedLLMClient(
            [
                {
                    "display_type": "response_table",
                    "sql": "SELECT SUM(total) FROM orders",
                    "metric_id": "order_revenue",
                    "thought": "candidate",
                }
            ]
        )

    report = asyncio.run(
        run_benchmark(tmp_path / "run", tasks=(get_task("n01"),), client_factory=factory)
    )
    assert report["status"] == "incomplete"
    assert len(report["rows"]) == 3
    error = next(r for r in report["rows"] if r["mode"] == "code")
    assert error["status"] == "operational_error"
    assert error["error_type"] == "RuntimeError"
    assert "private injected diagnostic" not in json.dumps(report)
    assert report["summary"]["groups"]["code:all"]["target_outcomes_over_scheduled"] == {
        "numerator": 0,
        "denominator": 1,
    }


def test_live_mode_cannot_be_disguised_as_mock_client(tmp_path):
    with pytest.raises(ValueError, match="live benchmark cannot"):
        asyncio.run(run_benchmark(tmp_path / "run", live=True, client_factory=lambda *_: None))


def test_interrupted_run_retains_full_scheduled_grid(tmp_path):
    def factory(task, mode, budget):
        if mode == "code":
            raise KeyboardInterrupt
        return ScriptedLLMClient(
            [
                {
                    "display_type": "response_table",
                    "sql": "SELECT SUM(total) FROM orders",
                    "metric_id": "order_revenue",
                    "thought": "candidate",
                }
            ]
        )

    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            run_benchmark(output, tasks=(get_task("n01"), get_task("n02")), client_factory=factory)
        )
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert len(report["rows"]) == report["summary"]["scheduled_cells"] == 6
    for mode in ("original", "code", "evidence"):
        assert (
            report["summary"]["groups"][mode + ":all"]["target_outcomes_over_scheduled"][
                "denominator"
            ]
            == 2
        )


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH DATABASE '{destination}' AS side",
        "PRAGMA query_only=OFF",
        "SELECT name FROM sqlite_master",
        "SELECT load_extension('not-an-extension')",
        "CREATE TABLE bad(x)",
    ],
)
def test_original_resource_blocks_nonquery_operations(tmp_path, sql):
    path = write_fixture(tmp_path / "source.sqlite3", get_task("n01").fixture)
    resource = ReadOnlyDemoResource(path)
    destination = tmp_path / "unexpected.sqlite3"
    try:
        with pytest.raises(sqlite3.DatabaseError):
            resource._sync_query(
                str(path.resolve()), sql.format(destination=destination.as_posix())
            )
        assert not destination.exists()
        assert resource._sync_query(str(path.resolve()), "SELECT SUM(total) FROM orders")[1] == [
            (300.0,)
        ]
    finally:
        resource.close()
