"""Real SQLite tests: successful execution is not sufficient for acceptance."""

import sqlite3

import pytest

from dataverify.checker import CheckedQueryExecutor
from dataverify.contracts import describe_contracts


@pytest.fixture
def executor(tmp_path):
    db = tmp_path / "experiment.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY,total REAL NOT NULL);"
            "CREATE TABLE order_lines(line_id INTEGER PRIMARY KEY,order_id INTEGER,amount REAL NOT NULL);"
            "INSERT INTO orders VALUES(1,100),(2,100),(3,80);"
            "INSERT INTO order_lines VALUES(1,1,40),(2,1,60),(3,2,100),(4,3,20),(5,3,30),(6,3,30);"
        )
    return CheckedQueryExecutor(db)


def test_order_metric_direct_sum(executor):
    result = executor.execute("SELECT SUM(total) AS revenue FROM orders")
    assert result["execution_status"] == "executed"
    assert result["verification_status"] == "verified"
    assert result["accepted_value"] == 280
    assert all(item["passed"] for item in result["checks"])


def test_line_metric_valid_many_to_one_join(executor):
    result = executor.execute(
        "SELECT SUM(l.amount) FROM orders o INNER JOIN order_lines l ON o.order_id=l.order_id",
        "line_revenue",
    )
    assert result["verification_status"] == "verified"
    assert result["accepted_value"] == 280


def test_bad_join_executes_but_is_never_accepted(executor):
    result = executor.execute(
        "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"
    )
    assert result["execution_status"] == "executed"
    assert result["value"] == 540
    assert result["accepted_value"] is None
    assert result["verification_status"] == "rejected"
    assert result["error_code"] == "GRAIN_MULTIPLICATION"
    assert result["evidence"][0]["examples"] == [
        {"grain_value": 3, "multiplicity": 3},
        {"grain_value": 1, "multiplicity": 2},
    ]


def test_distinct_value_is_not_grain_dedup(executor):
    result = executor.execute(
        "SELECT SUM(DISTINCT o.total) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"
    )
    assert result["execution_status"] == "executed"
    assert result["value"] == 180
    assert result["accepted_value"] is None
    assert result["error_code"] == "VALUE_DISTINCT_NOT_GRAIN_DEDUP"
    assert result["evidence"][1]["examples"] == [
        {"measure_value": 100.0, "distinct_grain_count": 2}
    ]


def test_distinct_rejected_even_without_join(executor):
    result = executor.execute("SELECT SUM(DISTINCT total) FROM orders")
    assert result["value"] == 180
    assert result["verification_status"] == "rejected"


def test_metric_must_match_source_column(executor):
    result = executor.execute("SELECT SUM(amount) FROM order_lines")
    assert result["error_code"] == "METRIC_SOURCE_MISMATCH"
    assert result["accepted_value"] is None


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA writable_schema=ON",
        "SELECT SUM(total) FROM orders; DELETE FROM orders",
        "SELECT SUM(rootpage) FROM sqlite_master",
        "SELECT load_extension('anything') FROM orders",
        "SELECT SUM(abs(total)) FROM orders",
        "SELECT SUM(total) FROM orders WHERE order_id=1",
        "SELECT SUM(total) FROM orders GROUP BY order_id",
        "WITH x AS (SELECT * FROM orders) SELECT SUM(total) FROM x",
        "SELECT SUM(total) FROM (SELECT * FROM orders)",
        "SELECT SUM(total) OVER () FROM orders",
        "SELECT SUM(total) FROM orders UNION SELECT SUM(total) FROM orders",
        "SELECT SUM(total) FROM main.orders",
        "SELECT SUM(o.total) FROM orders o LEFT JOIN order_lines l ON o.order_id=l.order_id",
        "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.total=l.amount",
        "SELECT SUM(o.total) FROM orders o, order_lines l",
        "SELECT SUM(total), COUNT(*) FROM orders",
        "SELECT SUM(total) FROM orders LIMIT 1",
        "SELECT SUM(total) FROM orders ORDER BY total",
    ],
)
def test_unsupported_or_dangerous_sql_fails_closed(executor, sql):
    result = executor.execute(sql)
    assert result["execution_status"] == "blocked", result
    assert result["verification_status"] == "unsupported", result
    assert result["accepted_value"] is None
    assert result["value"] is None
    with sqlite3.connect(executor.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


def test_unknown_metric_fails_closed(executor):
    result = executor.execute("SELECT SUM(total) FROM orders", "invented_revenue")
    assert result["error_code"] == "UNKNOWN_METRIC"
    assert result["accepted_value"] is None


@pytest.mark.parametrize("metric_id", [None, [], {}, 123])
def test_malformed_metric_id_returns_structured_rejection(executor, metric_id):
    result = executor.execute("SELECT SUM(total) FROM orders", metric_id)
    assert result["execution_status"] == "blocked"
    assert result["error_code"] == "INVALID_METRIC_INPUT"
    assert result["accepted_value"] is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(total) FROM orders /*",
        "SELECT SUM(total) FROM orders WHERE total='unclosed",
        'SELECT SUM("total) FROM orders',
        "SELECT SUM(" + "(" * 600 + "total" + ")" * 600 + ") FROM orders",
    ],
)
def test_tokenization_and_parser_depth_failures_return_structured_rejection(executor, sql):
    result = executor.execute(sql)
    assert result["execution_status"] == "blocked"
    assert result["verification_status"] == "unsupported"
    assert result["error_code"] == "SQL_PARSE_ERROR"
    assert result["accepted_value"] is None


def test_inner_join_cannot_drop_source_grains(executor):
    with sqlite3.connect(executor.db_path) as connection:
        connection.execute("INSERT INTO orders VALUES(4,25)")
        connection.execute("DELETE FROM order_lines WHERE line_id IN (2,5,6)")
    result = executor.execute(
        "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"
    )
    assert result["value"] == 280
    assert result["error_code"] == "MISSING_GRAIN"
    assert result["accepted_value"] is None
    assert result["evidence"] == [
        {
            "kind": "missing_grain_coverage",
            "source_table": "orders",
            "grain_column": "order_id",
            "source_grains": 4,
            "query_grains": 3,
            "examples": [{"grain_value": 4}],
        }
    ]


def test_duplicate_and_missing_grains_both_produce_counterexamples(executor):
    with sqlite3.connect(executor.db_path) as connection:
        connection.execute("INSERT INTO orders VALUES(4,25)")
    result = executor.execute(
        "SELECT SUM(o.total) FROM orders o JOIN order_lines l ON o.order_id=l.order_id"
    )
    assert result["error_code"] == "GRAIN_MULTIPLICATION"
    assert result["accepted_value"] is None
    by_kind = {evidence["kind"]: evidence for evidence in result["evidence"]}
    assert by_kind["duplicate_grain"]["examples"][0] == {"grain_value": 3, "multiplicity": 3}
    assert by_kind["missing_grain_coverage"]["examples"] == [{"grain_value": 4}]


def test_line_metric_reports_specific_orphaned_source_grain(executor):
    with sqlite3.connect(executor.db_path) as connection:
        connection.execute("INSERT INTO order_lines VALUES(7,999,25)")
    result = executor.execute(
        "SELECT SUM(l.amount) FROM order_lines l JOIN orders o ON l.order_id=o.order_id",
        "line_revenue",
    )
    assert result["value"] == 280
    assert result["error_code"] == "MISSING_GRAIN"
    assert result["evidence"][0]["examples"] == [{"grain_value": 7}]


def test_sqlite_layers_independently_reject_writes_and_system_reads(executor):
    connection = executor._connect()
    try:
        for sql in [
            "DELETE FROM orders",
            "SELECT name FROM sqlite_master",
            "PRAGMA user_version",
            "ATTACH ':memory:' AS x",
        ]:
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(sql)
        assert connection.execute("SELECT SUM(total) FROM orders").fetchone()[0] == 280
    finally:
        connection.close()


def test_ro_uri_does_not_create_missing_database(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    result = CheckedQueryExecutor(missing).execute("SELECT SUM(total) FROM orders")
    assert result["execution_status"] == "error"
    assert result["accepted_value"] is None
    assert not missing.exists()


def test_contract_metadata_is_json_serializable():
    import json

    metadata = json.loads(json.dumps(describe_contracts()))
    assert metadata["metrics"]["order_revenue"]["grain_column"] == "order_id"
    assert len(metadata["relations"]) == 1
