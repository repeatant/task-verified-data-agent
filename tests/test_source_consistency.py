"""Regressions for WAL source changes, consistent reads and invalid measures.

All changes are local to pytest's workspace-local temporary directory. Separate
SQLite connections and synchronous hooks make the commit timing deterministic;
these tests neither use a language model nor race against background threads.
"""

import hashlib
import sqlite3

from dataverify.checker import CheckedQueryExecutor
from dataverify.database import seed_demo_database
from dataverify.engine import AgentRunner
from dataverify.providers import Action


def _main_file_hash(path):
    """The main file alone remains unchanged while committed writes live in WAL."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wal_commit_between_planner_steps_invalidates_run(tmp_path):
    database = seed_demo_database(tmp_path / "wal-between-steps.sqlite3")
    with sqlite3.connect(database) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        main_before = _main_file_hash(database)

        class CommitBeforeQueryPlanner:
            name = "test-concurrent-commit"
            real_model = False

            def choose(self, messages):
                if not any(message.get("role") == "tool" for message in messages):
                    writer.execute("UPDATE orders SET total=1000 WHERE order_id=1")
                    writer.commit()
                    return Action("tool", "inspect_schema", {}, "schema")
                return Action(
                    "tool",
                    "query_metric",
                    {"sql": "SELECT SUM(total) FROM orders", "metric_id": "order_revenue"},
                    "query",
                )

        result = AgentRunner(database, planner=CommitBeforeQueryPlanner()).run()
        # Reproduces the old hash blind spot: no checkpoint has copied this
        # committed update into the main SQLite file while writer stays open.
        assert _main_file_hash(database) == main_before
        assert writer.execute("SELECT SUM(total) FROM orders").fetchone()[0] == 1180
        assert result["status"] == "unverified"
        assert result["value"] is None
        assert result["stop_reason"] == "source_changed_during_run"


def test_query_and_probes_share_snapshot_and_changed_source_is_not_final_fact(
    tmp_path, monkeypatch
):
    database = seed_demo_database(tmp_path / "wal-between-read-and-probe.sqlite3")
    observed_snapshots = []
    original_verify = CheckedQueryExecutor._verify

    with sqlite3.connect(database) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        main_before = _main_file_hash(database)

        def commit_before_probe(connection, shape, result):
            # execute() has already read 280. A WAL writer can commit while the
            # query connection holds its read transaction; all following probes
            # must continue to use that transaction's original snapshot.
            assert result["value"] == 280
            writer.execute("UPDATE orders SET total=1000 WHERE order_id=1")
            writer.commit()
            observed_snapshots.append(
                connection.execute("SELECT SUM(total) FROM orders").fetchone()[0]
            )
            return original_verify(connection, shape, result)

        monkeypatch.setattr(CheckedQueryExecutor, "_verify", staticmethod(commit_before_probe))
        result = AgentRunner(database).run(case="clean_order_total")
        assert observed_snapshots == [280]
        assert _main_file_hash(database) == main_before
        assert writer.execute("SELECT SUM(total) FROM orders").fetchone()[0] == 1180
        assert result["status"] == "unverified"
        assert result["value"] is None
        assert result["stop_reason"] == "source_changed_during_run"


def test_non_numeric_measure_is_rejected_even_with_valid_source_grain(tmp_path):
    database = seed_demo_database(tmp_path / "non-numeric-measure.sqlite3")
    with sqlite3.connect(database) as writer:
        # SQLite's REAL affinity and NOT NULL constraint do not prevent text.
        # SUM would silently treat this value as zero without an explicit check.
        writer.execute("UPDATE orders SET total='damaged' WHERE order_id=1")
    result = CheckedQueryExecutor(database).execute("SELECT SUM(total) FROM orders")
    assert result["verification_status"] == "rejected"
    assert result["accepted_value"] is None
    assert result["error_code"] == "INVALID_MEASURE_VALUE"
