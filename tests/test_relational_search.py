"""Generic witness search proves disagreement, preserving declared key constraints."""

import pytest

from dataverify.verification.discovery_runtime import DiscoveryTask, execute_readonly, fixture_data
from dataverify.verification.reference_audit import (
    find_distinguishing_fixture,
    inspect_dispute,
    merge_fixtures,
)
from dataverify.verification.reference_expression import calculate
from dataverify.verification.relational_data import RelationalData


def source():
    return RelationalData(
        {"accounts": ["id", "label"], "entries": ["id", "account_id", "amount"]},
        {"accounts": [[1, "A"], [2, None]], "entries": [[1, 1, 10], [2, 2, -10]]},
        {"accounts": ["INTEGER", "TEXT"], "entries": ["INTEGER", "INTEGER", "INTEGER"]},
        {"accounts": ["id"], "entries": ["id"]},
        [
            {
                "table": "entries",
                "columns": ["account_id"],
                "references": "accounts",
                "reference_columns": ["id"],
            }
        ],
    )


SQL = "SELECT COALESCE(SUM(amount),0) FROM entries"
WRONG_SQL = "SELECT COALESCE(SUM(amount),0)*0 FROM entries"
REF = "[[sum(r['amount'] for r in tables['entries'])]]"


def test_search_finds_error_hidden_by_original_data_coincidence():
    original = source()
    assert execute_readonly(SQL, original) == execute_readonly(WRONG_SQL, original) == [[0]]
    found = find_distinguishing_fixture(WRONG_SQL, REF, REF, 40, source=original)
    assert found is not None
    probe = fixture_data(found["data"])
    assert execute_readonly(WRONG_SQL, probe) != calculate(REF, probe.named_tables())
    assert probe.primary_keys == original.primary_keys
    assert probe.foreign_keys == original.foreign_keys
    # Legal source-derived values are recombined, not guessed business constants.
    for table, rows in probe.tables.items():
        for row in rows:
            for index, value in enumerate(row):
                assert value in {item[index] for item in original.tables[table]}


def test_equal_programs_produce_no_witness():
    assert find_distinguishing_fixture(SQL, REF, REF, 15, source=source()) is None


def test_correlated_wrong_programs_are_not_detected_by_agreement():
    # Both references share the same constant-zero mistake. No witness is a
    # bounded negative search result, never a certificate of semantic correctness.
    assert find_distinguishing_fixture(WRONG_SQL, "[[0]]", "[[0]]", 15, source=source()) is None


def test_every_generated_database_preserves_keys(monkeypatch):
    from dataverify.verification import reference_audit

    inspected = []
    original_candidate = reference_audit._candidate_result

    def inspect(sql, data):
        serialized = data.to_dict()
        rebuilt = fixture_data(serialized)  # Rechecks all PK and FK constraints.
        assert rebuilt.to_dict() == serialized
        inspected.append(serialized)
        return original_candidate(sql, data)

    monkeypatch.setattr(reference_audit, "_candidate_result", inspect)
    assert find_distinguishing_fixture(SQL, REF, REF, 20, source=source()) is None
    assert len(inspected) > 1


def test_canonicalization_deduplicates_row_order_with_mixed_nulls():
    original = source()
    reordered = RelationalData(
        original.schema,
        {name: list(reversed(rows)) for name, rows in original.tables.items()},
        original.column_types,
        original.primary_keys,
        original.foreign_keys,
    )
    fixtures = merge_fixtures(
        [{"name": "first", "data": original.to_dict()}],
        [{"name": "reordered", "data": reordered.to_dict()}],
    )
    assert len(fixtures) == 1
    assert fixtures[0]["name"] == "first"


def test_three_way_evidence_executes_generic_data():
    original = source()
    task = DiscoveryTask(
        "generic",
        "Sum entries.",
        "Include all signs.",
        schema=original.schema,
        column_types=original.column_types,
        primary_keys=original.primary_keys,
        foreign_keys=original.foreign_keys,
    )
    probe = find_distinguishing_fixture(WRONG_SQL, REF, REF, 40, source=original)
    evidence = inspect_dispute(
        task, original, WRONG_SQL, {"reference": REF}, {"reference": REF}, [probe]
    )
    assert evidence["summary"]["candidate_difference_ids"]
    assert not evidence["summary"]["reference_difference_ids"]
    assert "not proof" in evidence["scope"]


@pytest.mark.parametrize("limit", [0, -1, 201, True])
def test_search_bound_required(limit):
    with pytest.raises(ValueError, match="EXAMPLE_LIMIT"):
        find_distinguishing_fixture(SQL, REF, REF, limit, source=source())
