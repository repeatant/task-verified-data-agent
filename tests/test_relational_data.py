"""Generic fixtures preserve schema, keys, read-only boundaries and benchmark scope."""

import sqlite3

import pytest

from dataverify.verification.discovery_runtime import (
    DiscoveryTask,
    check_candidate,
    execute_readonly,
    fixture_data,
    named_tables,
    validate_fixtures,
    validate_model_fixtures,
    validate_plan,
)
from dataverify.verification.relational_data import RelationalData, SQLiteSource, sqlite_affinity

SCHEMA = {"people": ["id", "name"], "payments": ["id", "person_id", "amount"]}
TYPES = {"people": ["INTEGER", "TEXT"], "payments": ["INTEGER", "INTEGER", "REAL"]}
KEYS = {"people": ["id"], "payments": ["id"]}
LINKS = [
    {
        "table": "payments",
        "columns": ["person_id"],
        "references": "people",
        "reference_columns": ["id"],
    }
]
TABLES = {"people": [[1, "Alice"], [2, None]], "payments": [[1, 1, 2.5], [2, 2, -1.0]]}


def data():
    return RelationalData(SCHEMA, TABLES, TYPES, KEYS, LINKS)


def task():
    return DiscoveryTask(
        "generic",
        "Sum all payment amounts.",
        "Include negative amounts.",
        schema=SCHEMA,
        column_types=TYPES,
        primary_keys=KEYS,
        foreign_keys=LINKS,
    )


def plan():
    return {
        "action": "plan",
        "basis": [{"source": "question", "quote": "Sum all payment amounts."}],
        "assumptions": [],
        "reference": "[[sum(p['amount'] for p in tables['payments'])]]",
        "fixtures": [{"name": "example", "data": TABLES}],
    }


def test_roundtrip_and_named_rows():
    assert fixture_data(data().to_dict()).to_dict() == data().to_dict()
    assert named_tables(data())["people"][0] == {"id": 1, "name": "Alice"}


def test_task_uses_only_its_real_schema():
    public = task().public_input()
    assert set(public["schema"]) == {"people", "payments"}
    assert public["foreign_keys"] == LINKS
    assert "orders" not in public["schema"]


def test_plan_normalizes_fixtures_without_mutating_model_payload():
    original = plan()
    normalized = validate_plan(original, task())
    assert "tables" in normalized["fixtures"][0]["data"]
    assert original["fixtures"][0]["data"] == TABLES
    assert check_candidate(
        "SELECT SUM(amount) FROM payments", normalized, normalized["fixtures"], source=data()
    )["all_match"]


def test_typed_sql_execution_join_and_nulls():
    assert execute_readonly("SELECT SUM(amount) FROM payments", data()) == [[1.5]]
    assert execute_readonly(
        "SELECT p.name FROM people p JOIN payments x ON p.id=x.person_id WHERE x.amount < 0", data()
    ) == [[None]]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM people",
        "SELECT * FROM people; DROP TABLE people",
        "SELECT load_extension('bad') FROM people",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM main.people",
        "WITH x AS (SELECT * FROM people) SELECT * FROM x",
        "SELECT * FROM people UNION SELECT * FROM people",
        "SELECT row_number() OVER () FROM people",
        "PRAGMA table_info(people)",
        "ATTACH DATABASE 'elsewhere' AS elsewhere",
        "SELECT readfile('/etc/passwd') FROM people",
    ],
)
def test_reject_unsafe_or_unsupported_sql(sql):
    with pytest.raises(ValueError):
        execute_readonly(sql, data())


def test_generic_safe_sqlite_functions():
    assert execute_readonly("SELECT lower(substr(name,1,2)) FROM people WHERE id=1", data()) == [
        ["al"]
    ]


def test_dynamic_sqlite_storage_types_are_shared_by_reference_and_sql():
    fixture = RelationalData(
        {"t": ["number", "date", "text", "raw"]},
        {"t": [["12", "2020-01-01", 17, "0012"]]},
        {"t": ["INTEGER", "NUMERIC", "TEXT", "BLOB"]},
    )
    assert fixture.tables["t"] == ((12, "2020-01-01", "17", "0012"),)
    assert execute_readonly("SELECT * FROM T", fixture) == [[12, "2020-01-01", "17", "0012"]]
    assert list(named_tables(fixture)["t"][0].values()) == [12, "2020-01-01", "17", "0012"]


def test_huge_integer_rejected_as_validation_error():
    with pytest.raises(ValueError, match="RELATIONAL_NUMBER"):
        RelationalData({"t": ["n"]}, {"t": [[10**1000]]})


def test_schema_envelope_cannot_override_task():
    value = data().to_dict()
    value["schema"] = {"fake": ["value"]}
    with pytest.raises(ValueError, match="SCHEMA_MISMATCH"):
        fixture_data(value, SCHEMA, TYPES, KEYS, LINKS)


def test_model_cannot_remove_key_constraints():
    value = data().to_dict()
    value["primary_keys"] = {}
    value["tables"]["people"].append([1, "Duplicate"])
    with pytest.raises(ValueError, match="PRIMARY_KEY"):
        fixture_data(value, SCHEMA, TYPES, KEYS, LINKS)


@pytest.mark.parametrize(
    "tables, error",
    [
        ({**TABLES, "payments": [[1, 77, 2.0]]}, "FOREIGN_KEY"),
        ({**TABLES, "people": [[None, "Alice"]]}, "PRIMARY_KEY"),
        ({**TABLES, "people": [[1, "Alice"], [1, "Again"]]}, "PRIMARY_KEY"),
        ({**TABLES, "people": [[True, "Alice"]]}, "SCALAR"),
        ({**TABLES, "payments": [[1, 1, float("nan")]]}, "NUMBER"),
        ({**TABLES, "people": [[1, "x" * 4001]]}, "STRING"),
        ({**TABLES, "people": [[1]]}, "ROW_COLUMNS"),
    ],
)
def test_invalid_rows(tables, error):
    with pytest.raises(ValueError, match=error):
        RelationalData(SCHEMA, tables, TYPES, KEYS, LINKS)


def test_fixture_and_source_have_different_limits():
    source = RelationalData({"t": ["n"]}, {"t": [[i] for i in range(51)]})
    with pytest.raises(ValueError, match="FIXTURE_ROW_LIMIT"):
        fixture_data(source.to_dict())
    assert len(source.tables["t"]) == 51


def test_empty_or_null_foreign_keys_legal():
    assert (
        RelationalData(
            SCHEMA, {"people": [], "payments": [[1, None, 0.0]]}, TYPES, KEYS, LINKS
        ).tables["payments"][0][1]
        is None
    )


def test_identifier_quoting_is_not_sql_injection():
    table = 'people"; DROP TABLE payments; --'
    unusual = RelationalData({table: ["n"]}, {table: [[1]]})
    quoted = '"' + table.replace('"', '""') + '"'
    assert execute_readonly(f"SELECT n FROM {quoted}", unusual) == [[1]]


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "original.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute(
            "CREATE TABLE payments(id INTEGER PRIMARY KEY, person_id INTEGER "
            "REFERENCES people(id), amount REAL)"
        )
        connection.executemany("INSERT INTO people VALUES(?,?)", [(1, "Alice"), (2, "Bob")])
        connection.executemany("INSERT INTO payments VALUES(?,?,?)", [(1, 2, 40), (2, 1, 60)])
    return path


def test_original_database_scoring_is_not_preview_scoring(database):
    source = SQLiteSource.from_path(database)
    preview = source.preview(1)
    assert execute_readonly("SELECT SUM(amount) FROM payments", source) == [[100.0]]
    # First payment refers to omitted person #2; closure drops the orphan.
    assert preview.tables["payments"] == ()
    assert execute_readonly("SELECT SUM(amount) FROM payments", preview) == [[None]]
    with pytest.raises(ValueError, match="FULL_DATABASE_REFERENCE_NOT_SUPPORTED"):
        named_tables(source)


def test_original_database_is_read_only(database):
    source = SQLiteSource.from_path(database)
    with source.connect() as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM people")
    assert execute_readonly("SELECT COUNT(*) FROM people", source) == [[2]]


def test_preview_char_budget_omits_whole_rows_and_preserves_keys(database):
    import json

    source = SQLiteSource.from_path(database)
    preview = source.preview(2, max_chars=20)
    assert (
        sum(
            len(json.dumps(row, ensure_ascii=False))
            for rows in preview.tables.values()
            for row in rows
        )
        <= 20
    )
    assert fixture_data(preview.to_dict()).to_dict() == preview.to_dict()
    assert source.preview(2, max_chars=0).tables == {"payments": (), "people": ()}
    with pytest.raises(ValueError, match="CHAR_LIMIT"):
        source.preview(2, max_chars=-1)


def test_schema_and_keys_introspected(database):
    source = SQLiteSource.from_path(database)
    assert source.primary_keys == {"payments": ("id",), "people": ("id",)}
    assert list(source.foreign_keys) == LINKS
    assert source.preview(2).to_dict()["column_types"]["payments"] == TYPES["payments"]


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("VARCHAR(30)", "TEXT"),
        ("DOUBLE", "REAL"),
        ("INT", "INTEGER"),
        ("BOOLEAN", "NUMERIC"),
        ("", "BLOB"),
        ("BLOB", "BLOB"),
    ],
)
def test_sqlite_affinity(declared, expected):
    assert sqlite_affinity(declared) == expected


def test_validation_returns_normalized_copy():
    fixtures = [{"name": "one", "data": TABLES}]
    normalized = validate_fixtures(fixtures, SCHEMA, TYPES, KEYS, LINKS)
    assert fixture_data(normalized[0]["data"]).schema == data().schema
    assert "schema" not in fixtures[0]["data"]


def test_legacy_model_boundary_rejects_generic_envelope():
    legacy_task = DiscoveryTask("legacy", "Sum settled amounts.")
    foreign = RelationalData({"unauthorized": ["x"]}, {"unauthorized": [[1]]})
    fixtures = [{"name": "override", "data": foreign.to_dict()}]
    with pytest.raises(ValueError, match="MODEL_FIXTURE_TASK_SCHEMA_MISMATCH"):
        validate_model_fixtures(fixtures, legacy_task)
    # Internal replay still accepts a normalized generic fixture, independent
    # of any model boundary, so generic evidence merging is not regressed.
    assert validate_fixtures(fixtures) == fixtures
    proposed = {
        "action": "plan",
        "basis": [{"source": "question", "quote": "Sum settled amounts."}],
        "assumptions": [],
        "reference": "[[sum(r['x'] for r in tables['unauthorized'])]]",
        "fixtures": fixtures,
    }
    with pytest.raises(ValueError, match="MODEL_FIXTURE_TASK_SCHEMA_MISMATCH"):
        validate_plan(proposed, legacy_task)


def test_generic_model_boundary_uses_task_keys_not_model_keys():
    value = data().to_dict()
    value["primary_keys"] = {}
    value["foreign_keys"] = []
    value["column_types"] = {"people": ["BLOB", "BLOB"], "payments": ["BLOB"] * 3}
    value["tables"]["people"].append([1, "Duplicate"])
    with pytest.raises(ValueError, match="PRIMARY_KEY"):
        validate_model_fixtures([{"name": "override", "data": value}], task())


def test_legacy_models_retain_original_validation():
    fixtures = [{"name": "legacy", "data": {"orders": [[1, 100, "web"]], "lines": []}}]
    assert validate_model_fixtures(fixtures, DiscoveryTask("legacy", "Count orders.")) == fixtures


def test_source_only_plan_is_opt_in_and_still_executed():
    from dataclasses import replace

    proposed = {**plan(), "fixtures": []}
    with pytest.raises(ValueError, match="ONE_TO_THREE"):
        validate_plan(proposed, task())
    opted = replace(task(), allow_source_only_plan=True)
    normalized = validate_plan(proposed, opted)
    assert normalized["fixtures"] == []
    assert opted.public_input()["allow_source_only_plan"] is True
    assert check_candidate("SELECT SUM(amount) FROM payments", normalized, [], source=data())[
        "all_match"
    ]
    assert not check_candidate(
        "SELECT SUM(amount) + 1 FROM payments", normalized, [], source=data()
    )["all_match"]
    # Empty probes remain invalid, even when the initial plan may be source-only.
    with pytest.raises(ValueError, match="ONE_TO_THREE"):
        validate_model_fixtures([], opted)


def test_source_only_plan_still_requires_safe_reference_program():
    from dataclasses import replace

    with pytest.raises(ValueError):
        validate_plan(
            {**plan(), "fixtures": [], "reference": "__import__('os')"},
            replace(task(), allow_source_only_plan=True),
        )


@pytest.mark.parametrize("value", [True, 1, "yes"])
def test_source_only_option_cannot_relax_legacy_tasks(value):
    with pytest.raises(ValueError, match="SOURCE_ONLY_PLAN_POLICY"):
        DiscoveryTask("legacy", "Count orders.", allow_source_only_plan=value)


def test_parent_retention_keeps_actual_child_outside_initial_parent_sample(database):
    source = SQLiteSource.from_path(database)
    old = source.preview(1)
    retained = source.preview(1, retain_parents=True)
    assert old.tables["payments"] == ()
    assert retained.tables["payments"] == ((1, 2, 40.0),)
    assert retained.tables["people"] == ((1, "Alice"), (2, "Bob"))
    assert retained.schema == old.schema
    assert execute_readonly("SELECT SUM(amount) FROM payments", retained) == [[40.0]]
    assert retained.to_dict() == source.preview(1, retain_parents=True).to_dict()


def test_parent_retention_never_writes_source(database):
    before = database.read_bytes()
    SQLiteSource.from_path(database).preview(1, retain_parents=True)
    assert database.read_bytes() == before


def test_parent_retention_respects_null_foreign_keys(database):
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE payments SET person_id=NULL WHERE id=1")
    retained = SQLiteSource.from_path(database).preview(1, retain_parents=True)
    assert retained.tables["payments"] == ((1, None, 40.0),)
    assert retained.tables["people"] == ((1, "Alice"),)


def test_parent_retention_handles_cyclic_reference(tmp_path):
    path = tmp_path / "cycles.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE nodes(id INTEGER PRIMARY KEY,parent INTEGER REFERENCES nodes(id))"
        )
        connection.executemany("INSERT INTO nodes VALUES(?,?)", [(1, 3), (2, None), (3, 1)])
    source = SQLiteSource.from_path(path)
    assert source.preview(1).tables["nodes"] == ()
    assert source.preview(1, retain_parents=True).tables["nodes"] == ((1, 3), (3, 1))


def test_parent_retention_char_cap_and_missing_parent_remain_safe(database):
    import json

    source = SQLiteSource.from_path(database)
    retained = source.preview(1, max_chars=15, retain_parents=True)
    assert (
        sum(
            len(json.dumps(row, ensure_ascii=False))
            for rows in retained.tables.values()
            for row in rows
        )
        <= 15
    )
    assert retained.tables["payments"] == ()
    assert fixture_data(retained.to_dict()).to_dict() == retained.to_dict()


def test_parent_retention_per_table_cap_terminates_long_chain(tmp_path):
    path = tmp_path / "chain.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE nodes(id INTEGER PRIMARY KEY,parent INTEGER REFERENCES nodes(id))"
        )
        connection.executemany(
            "INSERT INTO nodes VALUES(?,?)", [(i, i + 1 if i < 60 else None) for i in range(1, 61)]
        )
    source = SQLiteSource.from_path(path)
    retained = source.preview(1, retain_parents=True)
    # The capped tail cannot be completed, so orphan removal propagates back.
    assert retained.tables["nodes"] == ()


def test_parent_retention_policy_requires_boolean(database):
    with pytest.raises(ValueError, match="PARENT_POLICY"):
        SQLiteSource.from_path(database).preview(1, retain_parents="yes")


def source_patch(changes, **extra):
    return [
        {"name": "edited source", "data": {"base": "bound_source", "changes": changes, **extra}}
    ]


def test_source_patch_expands_full_source_without_mutation():
    original = data()
    snapshot = original.to_dict()
    patch = source_patch([{"table": "payments", "row": 1, "values": {"amount": 8.0}}])
    normalized = validate_model_fixtures(patch, task(), source=original)
    changed = fixture_data(normalized[0]["data"])
    assert changed.tables["payments"][1] == (2, 2, 8.0)
    assert changed.tables["people"] == original.tables["people"]
    assert original.to_dict() == snapshot
    assert patch[0]["data"]["base"] == "bound_source"
    assert execute_readonly("SELECT SUM(amount) FROM payments", changed) == [[10.5]]
    normalized_plan = validate_plan(plan(), task())
    assert check_candidate(
        "SELECT SUM(amount) FROM payments", normalized_plan, normalized, source=original
    )["all_match"]


@pytest.mark.parametrize(
    "change, error",
    [
        ({"table": "missing", "row": 0, "values": {"amount": 1}}, "PATCH_TABLE"),
        ({"table": "payments", "row": -1, "values": {"amount": 1}}, "PATCH_ROW"),
        ({"table": "payments", "row": 2, "values": {"amount": 1}}, "PATCH_ROW"),
        ({"table": "payments", "row": True, "values": {"amount": 1}}, "PATCH_ROW"),
        ({"table": "payments", "row": 0, "values": {"missing": 1}}, "PATCH_COLUMNS"),
        ({"table": "payments", "row": 0, "values": {}}, "PATCH_COLUMNS"),
        ({"table": "payments", "row": 0, "values": {"person_id": 99}}, "FOREIGN_KEY"),
        ({"table": "payments", "row": 0, "values": {"id": 2}}, "PRIMARY_KEY"),
        ({"table": "payments", "row": 0, "values": {"amount": [1, 2]}}, "SCALAR"),
        ({"table": "payments", "row": 0, "values": {"amount": 10**16}}, "NUMBER"),
        (
            {"table": "payments", "row": 0, "values": {"amount": 1}, "expected": [[1]]},
            "CHANGE_FIELDS",
        ),
    ],
)
def test_source_patch_rejects_illegal_edits(change, error):
    with pytest.raises(ValueError, match=error):
        validate_model_fixtures(source_patch([change]), task(), source=data())


def test_source_patch_bounds_and_metadata_cannot_be_overridden():
    change = {"table": "payments", "row": 0, "values": {"amount": 1}}
    for changes in ([], [change] * 13):
        with pytest.raises(ValueError, match="CHANGE_LIMIT"):
            validate_model_fixtures(source_patch(changes), task(), source=data())
    with pytest.raises(ValueError, match="PATCH_FIELDS"):
        validate_model_fixtures(source_patch([change], primary_keys={}), task(), source=data())
    with pytest.raises(ValueError, match="REQUIRES_GENERIC_BOUND_SOURCE"):
        validate_model_fixtures(source_patch([change]), task())
    with pytest.raises(ValueError, match="TASK_SCHEMA_MISMATCH"):
        validate_model_fixtures(
            source_patch([change]), DiscoveryTask("legacy", "Count orders."), source=data()
        )


def test_source_patch_validates_coordinated_key_edits_after_all_changes():
    edits = source_patch(
        [
            {"table": "people", "row": 0, "values": {"id": 5}},
            {"table": "payments", "row": 0, "values": {"person_id": 5}},
        ]
    )
    normalized = validate_model_fixtures(edits, task(), source=data())
    changed = fixture_data(normalized[0]["data"])
    assert changed.tables["people"][0][0] == changed.tables["payments"][0][1] == 5


def test_plain_table_named_base_is_not_mistaken_for_patch():
    generic = DiscoveryTask(
        "base-table", "Count base rows.", "All rows count.", schema={"base": ["x"]}
    )
    normalized = validate_model_fixtures([{"name": "plain", "data": {"base": [[1]]}}], generic)
    assert fixture_data(normalized[0]["data"]).tables["base"] == ((1,),)
