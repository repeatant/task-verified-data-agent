"""Public benchmark isolation, support selection, and read-only scoring."""

import json
import sqlite3

import pytest

from dataverify.evaluation.public_benchmark import (
    SupportPolicy,
    compare_results,
    execute_original,
    file_digest,
    inspect_database,
    partition_records,
    prepare_manifest,
    public_input,
    stratified_ids,
    support_reasons,
)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            'CREATE TABLE "orders" (id INTEGER PRIMARY KEY, value REAL, label TEXT);'
            "INSERT INTO orders VALUES (1, 2.5, 'b'), (2, 2.5, 'a');"
        )
    return path


def test_public_input_is_a_strict_allowlist(database):
    annotation = {
        "question_id": 3,
        "question": "Sum value.",
        "evidence": "Include all orders.",
        "SQL": "SELECT SUM(value) FROM orders",
        "difficulty": "simple",
        "gold_rows": [[5]],
        "selected_because": "some reason",
        "db_id": "shop",
    }
    result = public_input(annotation, inspect_database(database))
    assert result["task_id"] == "bird-3"
    assert result["definitions"] == "Include all orders."
    assert set(result) == {
        "task_id",
        "question",
        "definitions",
        "schema",
        "column_types",
        "constraints",
        "tables",
        "preview_notice",
    }
    assert "SQL" not in result and "gold_rows" not in result


def test_inspection_keeps_types_and_full_schema(database):
    inspected = inspect_database(database)
    assert inspected["schema"] == {"orders": ["id", "value", "label"]}
    assert inspected["column_types"] == {"orders": ["INTEGER", "REAL", "TEXT"]}
    assert inspected["tables"]["orders"] == [[1, 2.5, "b"], [2, 2.5, "a"]]
    assert inspected["constraints"]["orders"]["primary_key"] == ["id"]


def test_preview_does_not_replace_original_database(database):
    preview = inspect_database(database, SupportPolicy(preview_rows_per_table=1))
    assert len(preview["tables"]["orders"]) == 1
    assert execute_original("SELECT SUM(value) FROM orders", database) == [[5.0]]


def test_scorer_separates_official_set_and_strict_bag():
    metrics = compare_results([[1], [1]], [[1]])
    assert metrics == {"bird_execution_match": True, "bag_match": False, "ordered_match": False}


def test_scorer_separates_order():
    metrics = compare_results([[2], [1]], [[1], [2]])
    assert metrics["bird_execution_match"] and metrics["bag_match"]
    assert not metrics["ordered_match"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "PRAGMA user_version=1",
        "ATTACH DATABASE 'secret.db' AS secret",
        "SELECT load_extension('anything')",
    ],
)
def test_score_readonly_and_no_external_io(database, sql):
    original_digest = file_digest(database)
    with pytest.raises(ValueError):
        execute_original(sql, database)
    assert file_digest(database) == original_digest


def test_score_limits_result_size(database):
    with pytest.raises(ValueError, match="SCORER_RESULT_ROW_LIMIT"):
        execute_original("SELECT * FROM orders", database, max_rows=1)


def test_score_timeout(database):
    with pytest.raises(ValueError, match="SCORER_SQL_ERROR"):
        execute_original(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<10000000) "
            "SELECT sum(x) FROM n",
            database,
            timeout=0.001,
        )


@pytest.mark.parametrize(
    "sql,reason",
    [
        ("WITH s AS (SELECT * FROM orders) SELECT * FROM s", "unsupported_sql_construct"),
        ("SELECT ROW_NUMBER() OVER () FROM orders", "unsupported_sql_construct"),
        ("SELECT id FROM orders UNION SELECT id FROM orders", "unsupported_sql_construct"),
        ("SELECT random() FROM orders", "unsupported_sql_function"),
        ("SELECT * FROM secret", "unknown_table"),
    ],
)
def test_support_filter_before_model_results(database, sql, reason):
    assert reason in support_reasons({"SQL": sql}, inspect_database(database), SupportPolicy())


def test_scope_accepts_safe_scalar_and_cast(database):
    reasons = support_reasons(
        {"SQL": "SELECT CAST(SUM(value) AS REAL), substr(label,1,1) FROM orders GROUP BY label"},
        inspect_database(database),
        SupportPolicy(),
    )
    assert reasons == []


def test_split_is_stable_and_stratified():
    records = [{"db_id": str(index % 5), "question_id": index} for index in range(80)]
    chosen = stratified_ids(records, 20, seed=42)
    assert chosen == stratified_ids(list(reversed(records)), 20, seed=42)
    assert len(chosen) == len(set(chosen)) == 20
    assert {
        str(index % 5): sum(value % 5 == index % 5 for value in chosen) for index in range(5)
    } == dict.fromkeys(map(str, range(5)), 4)


def test_development_and_test_databases_are_disjoint_when_feasible():
    records = [{"db_id": str(index // 30), "question_id": index} for index in range(240)]
    split = partition_records(records, 100, 20, 20261107)
    assert len(split["test_ids"]) == 100
    assert len(split["development_ids"]) == 20
    assert len(split["test_databases"]) == 6
    assert not set(split["test_databases"]) & set(split["development_databases"])
    assert split == partition_records(list(reversed(records)), 100, 20, 20261107)


def test_prepare_manifest_preserves_gold_separation_and_disjoint_split(database):
    data_root = database.parent
    rows = [
        {
            "question_id": i,
            "db_id": "shop",
            "question": f"Task {i}: count orders.",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM orders",
            "difficulty": "simple",
        }
        for i in range(10)
    ]
    (data_root / "mini_dev_sqlite.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (data_root / "provenance.json").write_text(
        json.dumps({"source": "unit-test"}), encoding="utf-8"
    )
    manifest = prepare_manifest(data_root, test_count=6, dev_count=3)
    assert manifest["eligible"] == 10
    assert not set(manifest["test_ids"]) & set(manifest["development_ids"])
    assert len(manifest["model_inputs"]) == 9
    assert manifest["eligibility"][0]["gold_rows"] == [[2]]
    assert all("gold_sql" not in item for item in manifest["model_inputs"].values())


def test_weird_identifiers_are_quoted(tmp_path):
    path = tmp_path / "quoted.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "weird""table" ("odd name" TEXT)')
        connection.execute('INSERT INTO "weird""table" VALUES (?)', ("value",))
    assert inspect_database(path)["tables"] == {'weird"table': [["value"]]}


def test_blob_and_oversized_samples_are_disclosed_not_changed(database):
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO orders VALUES (3, 1, ?)", (b"private",))
        connection.execute("INSERT INTO orders VALUES (4, 1, ?)", ("z" * 1001,))
    result = inspect_database(database)
    assert len(result["tables"]["orders"]) == 2
    assert result["preview_omitted_rows"] == {"orders": 2}
