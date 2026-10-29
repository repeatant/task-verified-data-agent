"""Task identity and final-output boundaries, using real read-only SQL checks."""

import sqlite3
from contextlib import closing
from dataclasses import FrozenInstanceError

import pytest

from dataverify.checker import CheckedQueryExecutor
from dataverify.database import database_sha256
from dataverify.guarded_task import ConfirmedMetricTask, GuardedTaskExecutor

ORDER_SQL = "SELECT SUM(total) FROM orders"
LINE_SQL = "SELECT SUM(amount) FROM order_lines"
ORDER_JOIN = "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"
LINE_JOIN = "SELECT SUM(l.amount) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "orders.sqlite3"
    # Explicit close matters on Windows; sqlite's context manager only commits.
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total REAL);"
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER,amount REAL);"
            "INSERT INTO orders VALUES(1,100),(2,100),(3,80);"
            "INSERT INTO order_lines VALUES(1,1,40),(2,1,60),(3,2,100),(4,3,20),(5,3,30),(6,3,30);"
        )
    return path


@pytest.mark.parametrize(
    "question,metric",
    [
        ("", "order_revenue"),
        ("  \n", "order_revenue"),
        (None, "order_revenue"),
        ("问题", None),
        ("问题", ["order_revenue"]),
        ("问题", "invented_revenue"),
        ("x" * 2001, "order_revenue"),
    ],
)
def test_confirmation_requires_explicit_valid_inputs(question, metric):
    with pytest.raises(ValueError):
        ConfirmedMetricTask(question, metric)


def test_confirmation_is_immutable_and_does_not_pretend_to_infer_intent():
    # Deliberately ambiguous text: callers, not a keyword heuristic, confirm it.
    task = ConfirmedMetricTask("看看收入", "line_revenue")
    with pytest.raises(FrozenInstanceError):
        task.metric_id = "order_revenue"
    assert task.metric_id == "line_revenue"


def test_constructor_rejects_unconfirmed_dict_and_unknown_feedback_mode(database):
    with pytest.raises(TypeError):
        GuardedTaskExecutor(database, {"question": "收入", "metric_id": "order_revenue"})
    with pytest.raises(ValueError):
        GuardedTaskExecutor(
            database, ConfirmedMetricTask("收入", "order_revenue"), feedback_mode="guess"
        )


def test_equal_totals_do_not_authorize_wrong_metric_or_execute_sql(database, monkeypatch):
    # Both totals are 280. Numeric comparison alone cannot catch this error.
    with sqlite3.connect(database) as conn:
        assert conn.execute(ORDER_SQL).fetchone() == conn.execute(LINE_SQL).fetchone()
    task = ConfirmedMetricTask("订单收入", "order_revenue")
    guarded = GuardedTaskExecutor(database, task)

    def must_not_run(*args, **kwargs):
        pytest.fail("metric drift must be rejected before the SQL executor is called")

    monkeypatch.setattr(CheckedQueryExecutor, "execute", must_not_run)
    result = guarded.execute(LINE_SQL, "line_revenue")
    assert result["error_code"] == "METRIC_DRIFT"
    assert result["execution_status"] == "blocked"
    assert result["confirmed_metric_id"] == task.metric_id
    assert guarded.publish()["accepted_value"] is None
    assert guarded.task is task
    monkeypatch.undo()
    repaired = guarded.execute(ORDER_SQL, "order_revenue")
    assert repaired["verification_status"] == "verified"
    assert guarded.publish()["metric_id"] == "order_revenue"
    assert guarded.publish()["accepted_value"] == 280


def test_rejected_query_can_be_repaired_without_changing_the_task(database):
    before = database_sha256(database)
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("订单收入", "order_revenue"))
    rejected = guarded.execute(ORDER_JOIN, "order_revenue")
    assert rejected["value"] == 540
    assert rejected["error_code"] == "GRAIN_MULTIPLICATION"
    assert rejected["evidence"][0]["examples"]
    assert guarded.publish()["value"] is None
    guarded.execute(ORDER_SQL, "order_revenue")
    published = guarded.publish()
    assert published["status"] == "verified"
    assert published["value"] == 280
    assert published["verified_sql"] == ORDER_SQL
    assert published["intent_confirmation"] == "caller_confirmed_not_inferred"
    assert database_sha256(database) == before


def test_legitimate_line_grain_join_is_accepted(database):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("明细收入", "line_revenue"))
    assert guarded.execute(LINE_JOIN, "line_revenue")["verification_status"] == "verified"
    assert guarded.publish()["accepted_value"] == 280


def test_feedback_ablation_changes_evidence_not_rejection(database):
    task = ConfirmedMetricTask("订单收入", "order_revenue")
    code = GuardedTaskExecutor(database, task, feedback_mode="code")
    evidence = GuardedTaskExecutor(database, task, feedback_mode="evidence")
    a = code.execute(ORDER_JOIN, task.metric_id)
    b = evidence.execute(ORDER_JOIN, task.metric_id)
    for field in (
        "execution_status",
        "verification_status",
        "error_code",
        "accepted_value",
        "value",
    ):
        assert a[field] == b[field]
    assert a["feedback"] == "GRAIN_MULTIPLICATION"
    assert a["evidence"] == [] and a["checks"] == []
    assert b["evidence"] and b["checks"]
    assert code.last_result["evidence"] == evidence.last_result["evidence"]
    assert code.publish() == evidence.publish()


def test_external_result_mutation_and_claim_cannot_forge_publication(database):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("收入", "order_revenue"))
    assert guarded.publish()["error_code"] == "NO_VERIFIED_EVIDENCE"
    result = guarded.execute(ORDER_JOIN, "order_revenue")
    result.update(verification_status="verified", accepted_value=999999)
    audit = guarded.last_result
    audit.update(verification_status="verified", accepted_value=999999)
    with pytest.raises(TypeError):
        guarded.publish({"accepted_value": 999999, "verification_status": "verified"})
    assert guarded.publish()["value"] is None
    good_result = guarded.execute(ORDER_SQL, "order_revenue")
    good_result["accepted_value"] = 999999
    published = guarded.publish()
    published["value"] = 999999
    assert guarded.publish()["value"] == 280


@pytest.mark.parametrize("sql,metric", [(LINE_SQL, "line_revenue"), (ORDER_JOIN, "order_revenue")])
def test_new_failed_candidate_clears_previous_accepted_value(database, sql, metric):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("收入", "order_revenue"))
    guarded.execute(ORDER_SQL, "order_revenue")
    assert guarded.publish()["value"] == 280
    guarded.execute(sql, metric)
    assert guarded.publish()["value"] is None


def test_source_change_after_execution_prevents_publication_and_further_acceptance(database):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("收入", "order_revenue"))
    guarded.execute(ORDER_SQL, "order_revenue")
    with sqlite3.connect(database) as conn:
        conn.execute("UPDATE orders SET total=200 WHERE order_id=1")
    result = guarded.publish()
    assert result["status"] == "unverified" and result["value"] is None
    assert result["error_code"] == "SOURCE_CHANGED_DURING_TASK"
    assert guarded.execute(ORDER_SQL, "order_revenue")["execution_status"] == "blocked"


def test_source_change_during_checker_execution_rejects_returned_value(database, monkeypatch):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("收入", "order_revenue"))
    original = CheckedQueryExecutor.execute

    def query_then_external_commit(self, sql, metric_id):
        result = original(self, sql, metric_id)
        with sqlite3.connect(database) as conn:
            conn.execute("UPDATE orders SET total=200 WHERE order_id=1")
        return result

    monkeypatch.setattr(CheckedQueryExecutor, "execute", query_then_external_commit)
    result = guarded.execute(ORDER_SQL, "order_revenue")
    assert result["execution_status"] == "executed"
    assert result["verification_status"] == "rejected"
    assert result["error_code"] == "SOURCE_CHANGED_DURING_TASK"
    assert result["accepted_value"] is None
    assert guarded.publish()["value"] is None


def test_source_removal_fails_closed_without_leaking_filesystem_error(database):
    guarded = GuardedTaskExecutor(database, ConfirmedMetricTask("收入", "order_revenue"))
    guarded.execute(ORDER_SQL, "order_revenue")
    database.unlink()
    result = guarded.publish()
    assert result["error_code"] == "SOURCE_UNAVAILABLE"
    assert result["accepted_value"] is None
