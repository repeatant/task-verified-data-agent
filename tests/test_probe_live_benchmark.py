"""Offline tests for scoring, frozen denominators and budget handoff."""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import replace

import pytest

if importlib.util.find_spec("dbgpt") is None:
    pytest.skip("Optional pinned DB-GPT runtime is not installed", allow_module_level=True)
pytest.importorskip("hypothesis")

from dataverify import probe_live_benchmark as bench
from dataverify.probe_cases import get_case, reference_result
from dataverify.probe_cost_budget import MonetaryRequestBudget


def record_for(case, *, sql=None, rows=None):
    return {
        "publication": {
            "status": "verified",
            "task_id": case.task.task_id,
            "verified_sql": sql or case.reference_sql,
            "rows": reference_result(case) if rows is None else rows,
        },
        "source_unchanged": True,
        "http_events": [],
        "actions": [],
        "llm_requests": [],
        "llm_calls": 1,
    }


@pytest.mark.parametrize("case_id", ["d01", "d03", "d04", "d16"])
def test_reference_passes_independent_source_and_boundary_grading(case_id):
    case = get_case(case_id)
    grade = bench.grade_probe_record(case, record_for(case))
    assert grade["correct_task"]
    assert grade["source_correct"] and grade["boundary_correct"]
    assert not grade["unscorable"]


@pytest.mark.parametrize("case_id", ["d05", "d06", "d07", "d10", "d11", "d12", "d13", "d14"])
def test_source_coincidence_is_not_counted_as_full_correctness(case_id):
    case = get_case(case_id)
    observed = bench.independent_query(case.candidate_sql, case.source)
    assert observed["status"] == "ok"
    grade = bench.grade_probe_record(
        case, record_for(case, sql=case.candidate_sql, rows=observed["rows"])
    )
    assert grade["source_correct"]
    assert not grade["boundary_correct"]
    assert not grade["correct_task"]
    assert grade["wrong_release"]


def test_false_published_rows_cannot_pass_by_hiding_behind_good_sql():
    case = get_case("d01")
    grade = bench.grade_probe_record(case, record_for(case, rows=[[1]]))
    assert grade["boundary_correct"]
    assert not grade["publication_matches_sql"]
    assert not grade["correct_task"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "ATTACH DATABASE 'x' AS y",
        "SELECT * FROM sqlite_master",
        "SELECT random()",
        "SELECT SUM(amount) FROM order_lines; DELETE FROM orders",
    ],
)
def test_independent_evaluator_has_own_read_boundary(sql):
    assert bench.independent_query(sql, get_case("d01").source)["status"] == "error"


def test_independent_grader_does_not_call_runtime_checker(monkeypatch):
    import dataverify.probe_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "static_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not grade itself")),
    )
    case = get_case("d01")
    assert bench.grade_probe_record(case, record_for(case))["correct_task"]


def test_operational_error_and_task_drift_are_not_correct_answers():
    case = get_case("d01")
    record = record_for(case)
    record["operational_error"] = "TIMEOUT"
    assert not bench.grade_probe_record(case, record)["correct_task"]
    record = record_for(case)
    record["publication"]["task_id"] = "other"
    assert bench.grade_probe_record(case, record)["wrong_release"]


def test_unknown_and_inconsistent_usage_are_not_zero_cost():
    assert not bench._usage([{"usage": {}}])["usage_complete"]
    assert not bench._usage(
        [{"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99}}]
    )["usage_complete"]


def test_summary_keeps_unrun_and_failed_cells_in_denominator():
    rows = [
        {"mode": "full", "track": "controlled", "status": "not_run", "http_attempts": 0},
        {"mode": "full", "track": "controlled", "status": "operational_error", "http_attempts": 1},
    ]
    group = bench.summarize(rows, ("full",))["groups"]["full:controlled"]
    assert group["scheduled"] == 2
    assert group["correct_over_scheduled"] == {"numerator": 0, "denominator": 2}


@pytest.fixture
def report_root(tmp_path, monkeypatch):
    # A fake project root allows exclusive-output tests without touching saved
    # reports. Minimal copies are for source fingerprinting only.
    root = tmp_path / "project"
    (root / "dataverify").mkdir(parents=True)
    (root / "reports").mkdir()
    for name in bench.FINGERPRINT_FILES:
        (root / "dataverify" / name).write_text("# test fingerprint\n", encoding="utf-8")
    monkeypatch.setattr(bench, "PROJECT_ROOT", root)
    return root / "reports"


def fake_probe(sql, task, seed=17):
    return {"status": "no_counterexample_found", "counterexample": None}


def test_run_preallocates_grid_and_preserves_runner_errors(report_root, monkeypatch):
    monkeypatch.setattr(bench, "probe_candidate", fake_probe)

    async def broken(*args, **kwargs):
        raise RuntimeError("do not persist this raw exception")

    monkeypatch.setattr(bench, "run_probe_native_case", broken)
    output = report_root / "broken"
    report = asyncio.run(
        bench.run_probe_benchmark(output, case_ids=("d01",), include_natural=False, seeds=(17, 31))
    )
    assert report["summary"]["scheduled_cells"] == 4
    assert len(report["rows"]) == 4
    assert all(row["status"] == "operational_error" for row in report["rows"])
    assert "raw exception" not in (output / "summary.json").read_text(encoding="utf-8")
    assert not report["all_cells_completed"]


def test_natural_smoke_selects_one_seed_and_does_not_inject_first_candidate(
    report_root, monkeypatch
):
    observed = []

    async def fake_run(task, data, **kwargs):
        observed.append(kwargs)
        case = next(
            case for case in (get_case("d01"), get_case("d03")) if case.task.task_id == task.task_id
        )
        return record_for(case)

    monkeypatch.setattr(bench, "run_probe_native_case", fake_run)
    report = asyncio.run(
        bench.run_probe_benchmark(
            report_root / "smoke",
            case_ids=("d01", "d03"),
            include_controlled=False,
            modes=("compact",),
            seeds=(17, 31),
        )
    )
    assert len(report["rows"]) == 2
    assert {row["seed"] for row in report["rows"]} == {17}
    assert all(row["track"] == "natural" for row in report["rows"])
    assert all(len(item["responses"]) == 2 for item in observed)
    assert report["summary"]["groups"]["compact:natural"]["correct_tasks"] == 2


def test_shared_budget_is_not_reset_between_smoke_and_full_runs(report_root, monkeypatch):
    monkeypatch.setattr(bench, "probe_candidate", fake_probe)
    shared = MonetaryRequestBudget(
        limit=80,
        max_input_chars=1_000_000,
        max_reported_tokens=250_000,
        money_limit_cny=5,
        max_output_tokens=900,
    )
    shared.reserve(100)
    shared.record_tokens({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})

    async def fake_run(task, data, **kwargs):
        return record_for(get_case("d01"))

    monkeypatch.setattr(bench, "run_probe_native_case", fake_run)
    report = asyncio.run(
        bench.run_probe_benchmark(
            report_root / "shared",
            case_ids=("d01",),
            include_natural=False,
            seeds=(17,),
            budget=shared,
        )
    )
    assert report["budget_start"]["http_attempts"] == 1
    assert report["budget_end"]["http_attempts"] == 1
    assert report["this_run_http_attempts"] == 0
    assert report["audit_ledger_complete"]


def test_run_output_is_exclusive_and_stays_in_reports(report_root, monkeypatch):
    monkeypatch.setattr(bench, "probe_candidate", fake_probe)
    with pytest.raises(ValueError, match="inside"):
        asyncio.run(bench.run_probe_benchmark(report_root.parent / "outside", case_ids=("d01",)))
    exists = report_root / "exists"
    exists.mkdir()
    with pytest.raises(FileExistsError):
        asyncio.run(bench.run_probe_benchmark(exists, case_ids=("d01",)))


def test_live_never_accepts_test_client_factory(report_root):
    with pytest.raises(ValueError, match="injected"):
        asyncio.run(
            bench.run_probe_benchmark(
                report_root / "live", live=True, client_factory=lambda *args: None
            )
        )


def test_live_cannot_underreserve_the_fixed_client_output_cap(report_root):
    from dataverify.dbgpt_qwen import QwenConfig

    config = QwenConfig("sk-test-never-real", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    budget = MonetaryRequestBudget(max_output_tokens=100)
    with pytest.raises(ValueError, match="output cap"):
        asyncio.run(
            bench.run_probe_benchmark(
                report_root / "mismatch", live=True, config=config, budget=budget, case_ids=("d01",)
            )
        )
    assert budget.snapshot()["http_attempts"] == 0
    assert not (report_root / "mismatch").exists()


def test_external_evaluation_case_id_cannot_escape_report_directory(report_root):
    unsafe = replace(get_case("d01"), case_id="../escape")
    with pytest.raises(ValueError, match="distinct frozen"):
        asyncio.run(bench.run_probe_benchmark(report_root / "unsafe", evaluation_cases=(unsafe,)))
    assert not (report_root / "unsafe").exists()


def test_external_natural_tasks_use_explicit_frozen_selection(report_root, monkeypatch):
    from dataverify.capability_cases import get_capability_cases

    case = get_capability_cases()[0].case

    async def fake_run(task, data, **kwargs):
        assert kwargs["feedback_policy"] == "capability_v2"
        return record_for(case)

    monkeypatch.setattr(bench, "run_probe_native_case", fake_run)
    report = asyncio.run(
        bench.run_probe_benchmark(
            report_root / "external",
            evaluation_cases=(case,),
            natural_case_ids=(case.case_id,),
            include_controlled=False,
            modes=("compact",),
        )
    )
    assert report["rows"][0]["case_id"] == "v01"
    assert report["rows"][0]["grade"]["correct_task"]


@pytest.mark.parametrize("seeds", [(), (17, 17), (True,), (1, 2, 3, 4)])
def test_invalid_seeds_fail_before_creating_output(report_root, seeds):
    with pytest.raises(ValueError):
        asyncio.run(bench.run_probe_benchmark(report_root / "bad-seeds", seeds=seeds))
