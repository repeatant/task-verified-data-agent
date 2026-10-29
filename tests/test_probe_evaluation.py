from pathlib import Path

import pytest

pytest.importorskip("hypothesis", reason="optional probe experiment requires requirements-probes.txt")
from tools.evaluate_probes import ROOT, run_evaluation
from tools.replay_probe import replay_report
from dataverify.probe_cases import get_case
from dataverify.probe_testing import probe_candidate


def test_duplicate_seed_cannot_overwrite_trace_or_inflate_denominator():
    with pytest.raises(ValueError, match="must not repeat"):
        run_evaluation(ROOT / "reports" / "invalid-duplicate-seed", seeds=(17, 17))


def test_report_destination_must_remain_inside_project_reports(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        run_evaluation(tmp_path / "report")


def test_existing_output_is_never_replaced():
    with pytest.raises(FileExistsError):
        run_evaluation(ROOT / "reports")


def test_invalid_budget_rejected_before_creating_report():
    with pytest.raises(ValueError):
        run_evaluation(ROOT / "reports" / "invalid-budget", max_queries=0)


def test_saved_witness_replays_without_private_evaluator_or_model():
    case = get_case("d06")
    report = probe_candidate(case.candidate_sql, case.task)
    assert replay_report({"report": report})["reproduced"]
    assert replay_report(report, compact=False)["reproduced"]


def test_replay_rejects_forged_change_instead_of_falsely_rejecting_correct_query():
    case = get_case("d06")
    report = probe_candidate(case.candidate_sql, case.task)
    report["counterexample"]["compact"]["added_lines"] = ((99, 4, 1, "settled", "2026-09-20T00:00:00"),)
    report["counterexample"]["compact"]["added_orders"] = ((4, 1, "web"),)
    assert not replay_report(report)["reproduced"]
