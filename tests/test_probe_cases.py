"""Independent target/score tests for the disclosed probe development suite."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dataverify.probe_cases import get_case, get_cases, grade_rows, reference_result


def test_suite_has_explicit_controlled_development_provenance():
    cases = get_cases()
    assert len(cases) == 16
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.split == "controlled_development" for case in cases)
    assert sum(case.fault_label is None for case in cases) == 5
    assert {case.task.output_kind for case in cases} == {"scalar", "orders", "channels"}
    with pytest.raises(KeyError):
        get_case("unknown")
    with pytest.raises(ValueError, match="development"):
        replace(cases[0], split="held_out")
    with pytest.raises(FrozenInstanceError):
        cases[0].reference_sql = "SELECT 0"


def test_public_input_is_explicit_allowlist_and_owns_mutable_containers():
    case = get_case("d06")
    payload = case.public_input()
    assert set(payload) == {"question", "confirmed_task"}
    assert set(payload["confirmed_task"]) == {
        "properties",
        "output_kind",
        "start",
        "end",
        "status",
        "other_label",
    }
    text = json.dumps(payload, ensure_ascii=False)
    for private in (case.case_id, case.reference_sql, case.fault_label, case.developer_note):
        assert private not in text
    payload["confirmed_task"]["properties"].clear()
    assert case.task.properties


@pytest.mark.parametrize(
    "case_id,expected",
    [
        ("d01", [[35_000]]),
        ("d02", [[35_000]]),
        ("d03", [[1, 20_000], [2, 15_000]]),
        ("d04", [["其他", 20_000], ["线下", 15_000]]),
        ("d15", [[6_300]]),
        ("d16", [[6_300]]),
    ],
)
def test_reference_uses_records_with_half_open_dates_status_and_refunds(case_id, expected):
    assert reference_result(get_case(case_id)) == expected


def test_reference_handles_empty_group_population_and_zero_orders():
    case = get_case("d03")
    empty_lines = replace(case.source, lines=())
    assert reference_result(replace(case, source=empty_lines)) == [[1, 0], [2, 0]]
    channel = get_case("d04")
    assert reference_result(replace(channel, source=replace(channel.source, lines=()))) == []
    scalar = get_case("d01")
    assert reference_result(replace(scalar, source=replace(scalar.source, lines=()))) == [[0]]


def test_channel_oracle_merges_null_and_literal_other():
    case = get_case("d04")
    data = replace(
        case.source,
        orders=((1, 1, None), (2, 1, "其他"), (3, 1, "线下")),
        lines=(
            (11, 1, 100, "settled", "2026-09-15T12:00:00"),
            (21, 2, -30, "settled", "2026-09-15T12:00:00"),
        ),
    )
    assert reference_result(replace(case, source=data)) == [["其他", 70]]


def test_reference_is_independent_of_sql_runtime_and_parser():
    script = """
import sys
from types import SimpleNamespace
for module in ("sqlglot", "dataverify.probe_runtime", "dataverify.checker"):
    sys.modules[module] = None
from dataverify.probe_cases import reference_result, grade_rows
case = SimpleNamespace(
    task=SimpleNamespace(output_kind="scalar", status="settled", start="2026-09-01T00:00:00", end="2026-10-01T00:00:00"),
    source=SimpleNamespace(orders=((1, 1, None),), lines=((11, 1, -25, "settled", "2026-09-01T00:00:00"),)))
assert reference_result(case) == [[-25]]
assert grade_rows([[-25]], [[-25]])["correct"]
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize(
    "actual,expected,code",
    [
        ([[100.0]], [[100]], "MATCH"),
        ([[99]], [[100]], "VALUE_MISMATCH"),
        ([[None]], [[0]], "INVALID_MEASURE"),
        ([[True]], [[1]], "INVALID_MEASURE"),
        ([[float("nan")]], [[0]], "INVALID_MEASURE"),
        ([[float("inf")]], [[0]], "INVALID_MEASURE"),
        ([], [[0]], "SCALAR_SHAPE_MISMATCH"),
        ([[0], [0]], [[0]], "SCALAR_SHAPE_MISMATCH"),
        ([["b", 2], ["a", 1]], [["a", 1], ["b", 2]], "MATCH"),
        ([["a", 1], ["a", 1]], [["a", 2]], "DUPLICATE_GROUP_KEY"),
        ([[1, 2]], [[1, 2], [2, 0]], "GROUP_COVERAGE_MISMATCH"),
        ([[True, 2]], [[1, 2]], "INVALID_GROUP_VALUE"),
        ([["1", 2]], [[1, 2]], "GROUP_COVERAGE_MISMATCH"),
        ([[None, 2]], [["其他", 2]], "INVALID_GROUP_VALUE"),
        ([["a", None]], [["a", 0]], "INVALID_GROUP_VALUE"),
        ([], [], "MATCH"),
        ([["a", 1]], [], "GROUP_COVERAGE_MISMATCH"),
        ({"rows": [[0]]}, [[0]], "INVALID_ROWS"),
    ],
)
def test_grader_rejects_false_matches_and_preserves_group_coverage(actual, expected, code):
    result = grade_rows(actual, expected)
    assert result["code"] == code
    assert result["correct"] is (code == "MATCH")


@pytest.mark.parametrize(
    "expected", [None, [[None]], [[True]], [["a", 1], ["a", 2]], [[1, 2, 3]], [[1], [2]]]
)
def test_invalid_reference_is_never_silently_scored(expected):
    with pytest.raises(ValueError):
        grade_rows([[0]], expected)


def test_case_validates_required_fields():
    case = get_case("d01")
    for field in ("case_id", "candidate_sql", "reference_sql", "developer_note"):
        with pytest.raises(ValueError):
            replace(case, **{field: ""})
    with pytest.raises(ValueError):
        replace(case, fault_label="")


def test_reference_does_not_read_reference_sql_or_fault_label():
    case = get_case("d01")
    assert reference_result(replace(case, reference_sql="SELECT 123", fault_label="made-up")) == [
        [35_000]
    ]


@pytest.mark.parametrize("case_id", [f"d{number:02d}" for number in range(1, 17)])
def test_reference_sql_is_executable_and_agrees_with_record_oracle(case_id):
    # This is a consistency check of two independent reference paths, not a
    # score calculated from the runtime verifier's own correctness decision.
    from dataverify.probe_runtime import execute_sql

    case = get_case(case_id)
    result = execute_sql(case.reference_sql, case.source, case.task)
    assert result["status"] == "ok", result
    assert grade_rows(result["rows"], reference_result(case))["correct"]
