"""Evidence uses full trusted data while preserving query and output boundaries."""

import json
import sqlite3
from contextlib import closing

import pytest

from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.verification.discovery_runtime import execute_readonly
from dataverify.verification.relational_data import RelationalData, SQLiteSource
from dataverify.verification.sql_tools import SQLLimits, SQLTools, execute_observation


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source with spaces # ? &.sqlite"
    # '?' is forbidden in Windows filenames; URI escaping still covers '#' and '&'.
    path = path.with_name(path.name.replace("?", "question"))
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, amount REAL, kind TEXT)")
        connection.executemany(
            "INSERT INTO items VALUES(?,?,?)", [(1, 10, "sale"), (2, 20, "sale"), (3, -7, "refund")]
        )
        connection.commit()
    return SQLiteSource.from_path(path)


def test_query_reads_rows_missing_from_preview_without_mutation(database):
    tool = SQLTools(database)
    before = database.path.read_bytes()
    observation = tool.execute("SELECT SUM(amount) AS total FROM items", observation_id="q1")
    assert observation["status"] == "ok"
    assert observation["rows"] == [[23.0]]
    assert observation["columns"] == ["total"]
    assert observation["complete"] is True
    assert observation["scope"] == "original_database"
    assert execute_readonly("SELECT SUM(amount) FROM items", database.preview(2)) == [[30.0]]
    assert database.path.read_bytes() == before
    assert str(database.path) not in json.dumps(tool.metadata())
    assert str(database.path) not in json.dumps(observation)
    assert "gold" not in tool.metadata()


@pytest.mark.parametrize("case", get_discovery_cases(), ids=lambda case: case.name)
def test_controlled_sources_match_existing_executor(case):
    observation = execute_observation(case.good_sql, case.source, "candidate")
    assert observation["status"] == "ok"
    assert observation["rows"] == execute_readonly(case.good_sql, case.source)
    assert observation["scope"] == "provided_fixture"


def test_relational_fixture_and_original_use_same_result(database):
    query = "SELECT kind, COUNT(*) FROM items GROUP BY kind ORDER BY kind"
    original = SQLTools(database).execute(query, observation_id="original")
    fixture = SQLTools(database.preview(10)).execute(query, observation_id="fixture")
    assert fixture["rows"] == original["rows"]
    assert fixture["limits"] == original["limits"]


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM items",
        "DROP TABLE items",
        "UPDATE items SET amount=0",
        "SELECT * FROM items; DELETE FROM items",
        "ATTACH DATABASE 'elsewhere' AS other",
        "PRAGMA query_only=OFF",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM main.items",
        "SELECT load_extension('anything') FROM items",
        "SELECT readfile('anything') FROM items",
        "SELECT writefile('anything', amount) FROM items",
        "SELECT randomblob(999999999) FROM items",
        "SELECT * FROM items UNION SELECT * FROM items",
        "SELECT SUM(amount) OVER () FROM items",
        "WITH x AS (SELECT * FROM items) SELECT * FROM x",
        "SELECT 1",
    ],
)
def test_unsafe_or_unsupported_queries_never_mutate_source(database, query):
    before = database.path.read_bytes()
    result = SQLTools(database).execute(query, observation_id="rejected")
    # SQLite reports an unavailable dangerous function as a schema error before
    # calling the authorizer; neither classification admits execution.
    assert result["status"] in {"rejected", "error"}
    assert result["complete"] is False
    assert result["rows"] == []
    assert database.path.read_bytes() == before


def test_parameter_binding_treats_injection_as_a_value(database):
    result = SQLTools(database).execute(
        "SELECT amount FROM items WHERE kind = ?",
        ["sale'; DROP TABLE items; --"],
        observation_id="bound",
    )
    assert result["status"] == "ok"
    assert result["rows"] == []
    assert SQLTools(database).execute("SELECT COUNT(*) FROM items", observation_id="intact")[
        "rows"
    ] == [[3]]


@pytest.mark.parametrize(
    "parameters",
    [{"kind": "sale"}, [True], [float("nan")], [10**16], ["x" * 4001], ["\x00"], [None] * 101],
)
def test_parameter_types_and_size_are_bounded(database, parameters):
    result = SQLTools(database).execute(
        "SELECT * FROM items WHERE kind = ?", parameters, observation_id="p"
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"].startswith("SQL_PARAMETER")


def test_error_observations_are_specific_and_not_partial_results(database):
    result = SQLTools(database).execute(
        "SELECT nonexistent FROM items", observation_id="bad-column"
    )
    assert result["status"] == "error"
    assert result["error"]["code"] == "SQLITE_ERROR"
    assert "no such column" in result["error"]["message"]
    assert result["complete"] is False
    assert result["rows"] == []


def test_row_truncation_is_not_a_complete_result(database):
    result = SQLTools(database, limits=SQLLimits(max_rows=2)).execute(
        "SELECT id FROM items ORDER BY id", observation_id="limited"
    )
    assert result["status"] == "truncated"
    assert result["rows"] == [[1], [2]]
    assert result["complete"] is False
    assert result["truncation_reason"] == "row_limit"
    exact = SQLTools(database, limits=SQLLimits(max_rows=3)).execute(
        "SELECT id FROM items ORDER BY id", observation_id="exact"
    )
    assert exact["complete"] is True


@pytest.mark.parametrize(
    "limits,reason",
    [
        (SQLLimits(max_cell_chars=3), "cell_length_limit"),
        (SQLLimits(max_result_chars=10), "result_char_limit"),
    ],
)
def test_result_character_limits_never_silently_edit_a_cell(database, limits, reason):
    result = SQLTools(database, limits=limits).execute(
        "SELECT kind FROM items ORDER BY id", observation_id="bounded"
    )
    assert result["status"] == "truncated"
    assert result["rows"] == []
    assert result["truncation_reason"] == reason
    assert result["complete"] is False


def test_column_limit_and_numeric_limit(database):
    tool = SQLTools(database)
    width = tool.execute("SELECT id,id,id,id,id,id,id,id,id FROM items", observation_id="wide")
    assert width["status"] == "rejected"
    assert width["error"]["code"] == "RESULT_COLUMNS"
    number = tool.execute("SELECT 1e100 FROM items", observation_id="number")
    assert number["status"] == "truncated"
    assert number["truncation_reason"] == "numeric_limit"


def test_vm_limit_aborts_computation_with_no_partial_success():
    source = RelationalData({"numbers": ["value"]}, {"numbers": [[i] for i in range(200)]})
    result = SQLTools(source, limits=SQLLimits(max_vm_steps=1000)).execute(
        "SELECT SUM(a.value+b.value+c.value) FROM numbers a, numbers b, numbers c",
        observation_id="expensive",
    )
    assert result["status"] == "resource_limit"
    assert result["rows"] == []
    assert result["complete"] is False


def test_source_changes_invalidate_task_binding(database):
    tool = SQLTools(database)
    with closing(sqlite3.connect(database.path)) as connection:
        connection.execute("INSERT INTO items VALUES(4, 99, 'new')")
        connection.commit()
    result = tool.execute("SELECT COUNT(*) FROM items", observation_id="stale")
    assert result["error"]["code"] == "SOURCE_CHANGED"
    assert result["rows"] == []
    assert SQLTools(database).source_id != tool.source_id


def test_observation_provenance_is_stable_but_binds_values_and_query(database):
    tool = SQLTools(database)
    first = tool.execute("SELECT amount FROM items WHERE id = ?", [1], observation_id="q1")
    replay = tool.execute("SELECT amount FROM items WHERE id = ?", [1], observation_id="q1")
    different = tool.execute("SELECT amount FROM items WHERE id = ?", [2], observation_id="q2")
    assert first["query_digest"] == replay["query_digest"]
    assert first["observation_digest"] == replay["observation_digest"]
    assert first["query_digest"] != different["query_digest"]
    assert first["source_id"] == different["source_id"]


def test_model_cannot_supply_path_or_enlarge_resource_limits():
    with pytest.raises(TypeError, match="SOURCE_HANDLE"):
        SQLTools("arbitrary.sqlite")
    with pytest.raises(ValueError, match="RESOURCE_LIMIT"):
        SQLLimits(max_vm_steps=10_000_000)


def test_huge_integer_parameter_is_rejected_without_overflow(database):
    result = SQLTools(database).execute(
        "SELECT amount FROM items WHERE id = ?", [10**10_000], observation_id="overflow"
    )
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "SQL_PARAMETER_NUMBER"


def test_host_schema_metadata_change_invalidates_binding(database):
    tool = SQLTools(database)
    database.primary_keys["items"] = ()
    result = tool.execute("SELECT COUNT(*) FROM items", observation_id="changed-schema")
    assert result["error"]["code"] == "SOURCE_CHANGED"


def test_schema_metadata_comes_from_bound_source(database):
    metadata = SQLTools(database).metadata()
    assert metadata["schema"] == {"items": ["id", "amount", "kind"]}
    assert metadata["primary_keys"] == {"items": ["id"]}
    assert "tables" not in metadata
