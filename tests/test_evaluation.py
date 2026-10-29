"""独立oracle与受控对照：金额相同也不能掩盖错误统计粒度。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from dataverify.evaluation import CANDIDATES, FAMILIES, expected_result, make_fixture, run_evaluation


def test_cancelled_error_is_invalid_even_when_total_matches():
    fixture = make_fixture("cancelled_error", 101)
    candidate = next(item for item in CANDIDATES if item.name == "order_join")
    expected = expected_result(fixture, candidate)
    joined_total = sum(order[1] for order in fixture.orders for line in fixture.lines if order[0] == line[1])
    assert joined_total == pytest.approx(sum(order[1] for order in fixture.orders))
    assert expected["metric_value"] == pytest.approx(joined_total)
    assert expected["should_accept"] is False


def test_line_grain_join_is_valid_when_order_grain_join_fans_out():
    fixture = make_fixture("fanout", 211)
    order_query = next(item for item in CANDIDATES if item.name == "order_join")
    line_query = next(item for item in CANDIDATES if item.name == "line_join")
    assert expected_result(fixture, order_query)["should_accept"] is False
    assert expected_result(fixture, line_query)["should_accept"] is True


@pytest.mark.parametrize("family", FAMILIES)
def test_fixture_families_match_independent_expectations(tmp_path, family):
    report = run_evaluation(seeds=(101,), families=(family,), work_dir=tmp_path)
    assert report["guarded_wrong_result_releases"] == 0, report["cases"]
    assert report["guarded_valid_query_rejections"] == 0, report["cases"]
    assert report["source_databases_unchanged"]
    assert report["real_model_used"] is False
    assert report["llm_accuracy_benchmark"] is False
    assert not list(tmp_path.iterdir()), "Temporary fixture databases must be cleaned up"


def test_comparison_measures_false_accepts_and_valid_query_rejections(tmp_path):
    report = run_evaluation(seeds=(101, 211), work_dir=tmp_path)
    assert report["dataset_count"] == 2 * len(FAMILIES)
    assert report["candidate_count"] == 2 * len(FAMILIES) * len(CANDIDATES)
    assert report["valid_candidate_count"] > 0
    assert report["invalid_candidate_count"] > 0
    assert report["baseline_wrong_result_releases"] > 0
    assert report["numeric_match_but_invalid_count"] > 0
    assert report["guarded_wrong_result_releases"] == report["guarded_valid_query_rejections"] == 0
    assert report["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("seeds,families", [((1, 1), FAMILIES), ((), FAMILIES), ((1,), ()), ((1,), ("unknown",))])
def test_invalid_configuration_is_rejected(tmp_path, seeds, families):
    with pytest.raises(ValueError):
        run_evaluation(seeds=seeds, families=families, work_dir=tmp_path)


def test_entry_point_writes_report_without_api(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evaluation.json"
    process = subprocess.run(
        [sys.executable, str(root / "tools/evaluate_checker.py"), "--seeds", "907", "--families", "fanout", "orphan_line", "--output", str(output)],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(process.stdout)
    assert report["passed"] and report["candidate_count"] == 12
    assert summary["report_path"] == str(output.resolve())
    assert "cases" not in summary
