"""Frozen DB-GPT tasks keep target truth outside every model's prompt."""

import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dataverify import benchmark_tasks
from dataverify.benchmark_tasks import (
    TASKS,
    Fixture,
    candidate_expectation,
    get_task,
    get_tasks,
    manifest,
    oracle,
    write_fixture,
)


def test_suite_is_small_disclosed_development_data():
    assert len(TASKS) == 14
    assert len(get_tasks("natural")) == 4
    assert len(get_tasks("controlled")) == 10
    assert all(task.split == "development" for task in TASKS)
    assert len({task.task_id for task in TASKS}) == len(TASKS)
    assert (
        len(
            {
                (task.initial_candidate["sql"], task.initial_candidate["metric_id"])
                for task in get_tasks("controlled")
            }
        )
        == 6
    )
    with pytest.raises(ValueError, match="development"):
        replace(TASKS[0], split="held_out")
    with pytest.raises(ValueError):
        get_tasks("unknown")
    with pytest.raises(KeyError):
        get_task("unknown")


@pytest.mark.parametrize(
    "task_id,total,rows",
    [
        ("n01", 300.0, 3),
        ("n02", 300.0, 3),
        ("n03", 300.0, 3),
        ("n04", 355.0, 5),
        ("c02", 300.0, 3),
        ("c03", 300.0, 3),
        ("c05", 330.0, 4),
        ("c06", 355.0, 5),
        ("c08", 300.0, 3),
    ],
)
def test_record_oracle_keeps_all_target_grains(task_id, total, rows):
    expected = oracle(get_task(task_id))
    assert expected == {
        "target_defined": True,
        "metric_value": total,
        "source_rows": rows,
        "source_grains": rows,
        "reason": None,
    }


def test_oracle_is_importable_without_the_checker_parser_or_old_oracle():
    script = """
import sys
for name in ('sqlglot', 'dataverify.checker', 'dataverify.evaluation'):
    sys.modules[name] = None
from dataverify.benchmark_tasks import TASKS, oracle
assert oracle(TASKS[0])['metric_value'] == 300.0
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
    "task_id,reason",
    [
        ("c09", "invalid_source_grain"),
        ("c10", "invalid_measure_value"),
    ],
)
def test_damaged_frozen_data_has_no_target_answer(task_id, reason):
    task = get_task(task_id)
    expected = oracle(task)
    assert expected["target_defined"] is False
    assert expected["metric_value"] is None
    assert expected["reason"] == reason
    assert candidate_expectation(task)["should_accept"] is False


@pytest.mark.parametrize(
    "orders,reason",
    [
        ((), "empty_source"),
        (((None, 1.0),), "invalid_source_grain"),
        (((1, 1.0), (1, 2.0)), "invalid_source_grain"),
        (((1, None),), "invalid_measure_value"),
        (((1, "1.0"),), "invalid_measure_value"),
        (((1, True),), "invalid_measure_value"),
        (((1, float("nan")),), "invalid_measure_value"),
        (((1, float("inf")),), "invalid_measure_value"),
        (((1, 10**400),), "invalid_measure_value"),
        (((1, 1e308), (2, 1e308)), "nonfinite_total"),
    ],
)
def test_oracle_fails_closed_for_undefined_source(orders, reason):
    task = replace(TASKS[0], fixture=Fixture(orders, ()))
    expected = oracle(task)
    assert not expected["target_defined"]
    assert expected["metric_value"] is None
    assert expected["reason"] == reason


def test_target_oracle_does_not_require_matching_other_table_records():
    order_task = replace(get_task("n01"), fixture=Fixture(((1, 9.0),), ()))
    line_task = replace(get_task("n02"), fixture=Fixture((), ((5, 999, 12.0),)))
    assert oracle(order_task)["metric_value"] == 9.0
    assert oracle(line_task)["metric_value"] == 12.0


def test_fixture_and_candidates_are_immutable_and_own_their_inputs():
    rows = [[1, 10.0]]
    fixture = Fixture(rows, [[11, 1, 10.0]])
    rows[0][1] = 100.0
    assert fixture.orders == ((1, 10.0),)
    with pytest.raises(TypeError):
        fixture.orders[0][1] = 100.0
    with pytest.raises(FrozenInstanceError):
        fixture.orders = ()
    with pytest.raises(FrozenInstanceError):
        TASKS[0].task_id = "changed"
    with pytest.raises(TypeError):
        get_task("c01").initial_candidate["sql"] = "SELECT 1"
    with pytest.raises(ValueError):
        Fixture(((1, []),), ())
    with pytest.raises(ValueError):
        Fixture(((1,),), ())


def test_public_input_is_an_allowlist_without_answers_or_fixture_labels():
    for task in TASKS:
        public = task.public_input()
        assert public == {
            "question": task.question,
            "confirmed_metric_id": task.confirmed_metric_id,
        }
        serialized = json.dumps(public, ensure_ascii=False)
        for private in (
            "expected",
            "metric_value",
            "fixture",
            "pair_id",
            "initial_candidate",
            "task_id",
            "300",
            "355",
        ):
            assert private not in serialized
        for fault_or_repair in ("fanout", "DISTINCT", "SUM(", "孤儿", "损坏", "去重", "缺失"):
            assert fault_or_repair not in serialized
        public["question"] = "caller edit"
        assert task.public_input()["question"] == task.question
    thoughts = {task.initial_candidate["thought"] for task in get_tasks("controlled")}
    assert thoughts == {"根据问题和已确认指标提交候选查询。"}


def test_metamorphic_pairs_keep_candidate_and_question_fixed():
    pairs = defaultdict(list)
    for task in TASKS:
        if task.pair_id:
            pairs[task.pair_id].append(task)
    assert len(pairs) >= 2
    for pair in pairs.values():
        assert len(pair) >= 2
        assert len({(task.fixture.orders, task.fixture.lines) for task in pair}) == len(pair)
        assert all(task.initial_candidate == pair[0].initial_candidate for task in pair)
        assert all(task.public_input() == pair[0].public_input() for task in pair)
    # Cardinality and missing-relation changes flip the known candidate's truth.
    assert candidate_expectation(get_task("c01"))["should_accept"]
    assert not candidate_expectation(get_task("c02"))["should_accept"]
    assert not candidate_expectation(get_task("c08"))["should_accept"]
    assert candidate_expectation(get_task("c05"))["should_accept"]
    assert not candidate_expectation(get_task("c06"))["should_accept"]


def test_equal_totals_do_not_make_wrong_source_or_value_distinct_valid():
    source_swap = get_task("c07")
    assert sum(row[1] for row in source_swap.fixture.orders) == sum(
        row[2] for row in source_swap.fixture.lines
    )
    assert candidate_expectation(source_swap)["candidate_reason"] == "metric_source_mismatch"
    for task_id in ("c03", "c04"):
        assert (
            candidate_expectation(get_task(task_id))["candidate_reason"]
            == "value_distinct_not_grain"
        )
    distinct_totals = [
        sum({row[1] for row in get_task(task_id).fixture.orders}) for task_id in ("c03", "c04")
    ]
    assert distinct_totals == [100.0, 300.0]


def test_unknown_candidate_has_no_declared_expectation():
    candidate = dict(get_task("c01").initial_candidate)
    candidate["sql"] = "SELECT 123"
    with pytest.raises(ValueError, match="frozen candidate"):
        candidate_expectation(replace(get_task("c01"), initial_candidate=candidate))


def test_database_creation_preserves_damage_and_closes_connection(tmp_path, monkeypatch):
    connections = []
    connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(benchmark_tasks.sqlite3, "connect", tracking_connect)
    target = tmp_path / "fixture.sqlite3"
    task = get_task("c09")
    assert write_fixture(target, task.fixture) == target
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    connection = connect(target)
    try:
        assert tuple(connection.execute("SELECT * FROM orders")) == task.fixture.orders
        assert all(row[5] == 0 for row in connection.execute("PRAGMA table_info(orders)"))
        assert tuple(connection.execute("SELECT * FROM order_lines")) == task.fixture.lines
    finally:
        connection.close()


def test_database_creation_never_overwrites_existing_file(tmp_path):
    target = tmp_path / "existing.sqlite3"
    original = b"existing file must be preserved exactly"
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        write_fixture(target, TASKS[0].fixture)
    assert target.read_bytes() == original


def test_manifest_is_json_safe_private_data_and_a_detached_copy():
    private = manifest()
    restored = json.loads(json.dumps(private, allow_nan=False))
    assert restored["split"] == "development"
    assert restored["task_count"] == len(TASKS)
    assert restored["tasks"][0]["expected"]["metric_value"] == 300.0
    assert restored["tasks"][0]["expected"]["should_accept"] is None
    assert restored["tasks"][-1]["expected"]["target_defined"] is False
    private["tasks"][0]["fixture"]["orders"][0][1] = 12345
    private["tasks"][4]["initial_candidate"]["sql"] = "SELECT 1"
    assert oracle(TASKS[0])["metric_value"] == 300.0
    assert get_task("c01").initial_candidate["sql"] != "SELECT 1"
    with pytest.raises(ValueError, match="unique"):
        manifest((TASKS[0], TASKS[0]))
