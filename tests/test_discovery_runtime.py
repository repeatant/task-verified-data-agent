"""Safety boundaries and independent arithmetic checks for generated programs."""

import pytest

from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.verification.discovery_runtime import (
    check_candidate,
    execute_readonly,
    named_tables,
    rows_equal,
    validate_plan,
)
from dataverify.verification.reference_expression import ExpressionError, calculate


@pytest.mark.parametrize("case", get_discovery_cases(), ids=lambda c: c.name)
def test_new_calculations_execute_and_distinguish(case):
    plan = validate_plan(case.offline_plan(), case.task)
    fixtures = [case.offline_probe()["fixtures"][0]]
    assert check_candidate(case.bad_sql, plan, fixtures)["all_match"] is False
    assert check_candidate(case.good_sql, plan, fixtures)["all_match"] is True
    assert rows_equal(execute_readonly(case.good_sql, case.source), case.expected)
    assert rows_equal(calculate(plan["reference"], named_tables(case.source)), case.expected)


@pytest.mark.parametrize(
    "code",
    [
        "__import__('os').environ",
        "tables.__class__",
        "open('file')",
        "eval('1')",
        "(lambda:1)()",
        "10**100000",
        "[0]*999999999",
        "'x'*999999999",
        "sum(x for x in range(10000000))",
    ],
)
def test_reference_cannot_access_host_or_allocate_arbitrarily(code):
    with pytest.raises(ExpressionError):
        calculate(code, {})


def test_comprehensions_share_operation_budget():
    with pytest.raises(ExpressionError, match="BUDGET"):
        calculate(
            "sum(a*b for a in tables['x'] for b in tables['x'])",
            {"x": list(range(100))},
            budget=100,
        )


def test_reference_program_has_local_bindings_without_mutating_tables():
    tables = {"x": [1, 2, 3]}
    assert calculate("xs = tables['x']\ntotal = sum(xs)\n[[total / len(xs)]]", tables) == [[2]]
    assert tables == {"x": [1, 2, 3]}


@pytest.mark.parametrize(
    "code",
    [
        "tables = {}\n[[1]]",
        "tables['x'] = []\n[[1]]",
        "x = tables['x']\nx[0] = 5\n[[1]]",
        "sum = 2\n[[sum]]",
        "import os\n[[1]]",
    ],
)
def test_reference_program_rejects_mutation_and_host_statements(code):
    with pytest.raises(ExpressionError):
        calculate(code, {"x": [1]})


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "SELECT * FROM orders; DELETE FROM orders",
        "SELECT load_extension('anything') FROM orders",
        "SELECT * FROM sqlite_master",
        "ATTACH DATABASE 'file' AS other",
        "SELECT randomblob(1000000000) FROM orders",
        "WITH RECURSIVE t AS (SELECT 1 UNION ALL SELECT 1 FROM t) SELECT * FROM t",
    ],
)
def test_sql_cannot_write_or_escape_fixture_database(sql):
    with pytest.raises(ValueError):
        execute_readonly(sql, get_discovery_cases()[0].source)


def test_basis_must_be_in_original_source():
    case = get_discovery_cases()[0]
    plan = case.offline_plan()
    plan["basis"][0]["quote"] = "nonexistent business rule"
    with pytest.raises(ValueError, match="SOURCE"):
        validate_plan(plan, case.task)


def test_comparison_preserves_duplicates_and_tolerates_numeric_rounding():
    assert rows_equal([[1], [2]], [[2.0], [1.0]])
    assert not rows_equal([[1], [1]], [[1]])
    assert not rows_equal([[1]], [["1"]])


def test_large_totals_do_not_hide_whole_unit_errors():
    assert not rows_equal([[2_000_000_001]], [[2_000_000_000]])
    assert not rows_equal([[2_000_000_001.0]], [[2_000_000_000]])
    assert not rows_equal([[2_000_000_000.1]], [[2_000_000_000.0]])
    assert rows_equal([[1 / 3]], [[0.3333333333333333]])


@pytest.mark.parametrize("case", get_discovery_cases(), ids=lambda c: c.name)
def test_post_run_grading_checks_more_than_one_data_instance(case):
    from dataverify.evaluation.discovery_cases import grade_discovery_sql

    assert grade_discovery_sql(case, case.good_sql)["all_match"]
    assert not grade_discovery_sql(case, case.bad_sql)["all_match"]


@pytest.mark.parametrize(
    "value,digits,expected",
    [(0.125, 2, 0.13), (2.675, 2, 2.68), (-0.125, 2, -0.13), (2.5, 0, 3.0)],
)
def test_reference_half_up_rounding_is_explicit_and_handles_decimal_ties(value, digits, expected):
    actual = calculate(f"round_half_up({value}, {digits})", {})
    assert type(actual) is float
    assert actual == expected


def test_half_up_defaults_to_zero_digits_and_preserves_legacy_round_semantics():
    assert calculate("round_half_up(-2.5)", {}) == -3.0
    assert calculate("[[round(0.125,2), round_half_up(0.125,2)]]", {}) == [[0.12, 0.13]]


@pytest.mark.parametrize("digits", [-1, 10, 1_000_000_000, 1.5, True, "2"])
def test_reference_half_up_rejects_invalid_precision_before_decimal_work(digits):
    with pytest.raises(ExpressionError, match="REFERENCE_ROUND_PRECISION"):
        calculate("round_half_up(0.125, tables['digits'])", {"digits": digits})


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "0.125", 10**300]
)
def test_reference_half_up_rejects_nonfinite_non_numeric_and_unbounded_values(value):
    with pytest.raises(ExpressionError, match="REFERENCE_ROUND_VALUE"):
        calculate("round_half_up(tables['value'], 2)", {"value": value})


def test_half_up_does_not_allow_programs_to_access_decimal_methods():
    with pytest.raises(ExpressionError):
        calculate("round_half_up(0.125,2).quantize(1)", {})


def test_requested_half_up_matches_sqlite_round_at_eight_order_mean():
    from dataverify.evaluation.discovery_cases import data

    source = data([1, 0, 0, 0, 0, 0, 0, 0])
    sql = "SELECT ROUND(AVG(amount),2) FROM order_lines"
    reference = (
        "xs=[r['amount'] for r in tables['order_lines']]\n[[round_half_up(sum(xs)/len(xs),2)]]"
    )
    assert execute_readonly(sql, source) == [[0.13]]
    assert calculate(reference, named_tables(source)) == [[0.13]]


def test_dictionary_comprehension_groups_values_without_mutating_input():
    from copy import deepcopy

    from dataverify.evaluation.discovery_cases import data

    tables = named_tables(data([100, 70, -20], order_ids=[1, 2, 2]))
    original = deepcopy(tables)
    program = (
        "lines=tables['order_lines']\n"
        "ids=set(r['order_id'] for r in lines)\n"
        "totals={oid:sum(r['amount'] for r in lines if r['order_id']==oid) for oid in ids}\n"
        "[[oid,totals[oid]] for oid in sorted(totals)]"
    )
    assert calculate(program, tables) == [[1, 100], [2, 50]]
    assert tables == original


def test_dictionary_comprehension_shares_budget_across_nested_value_calculations():
    with pytest.raises(ExpressionError, match="REFERENCE_OPERATION_BUDGET"):
        calculate(
            "{a:sum(a*b for b in tables['x']) for a in tables['x']}",
            {"x": list(range(100))},
            budget=100,
        )


def test_dictionary_comprehension_bounds_new_mapping_size():
    with pytest.raises(ExpressionError, match="REFERENCE_COLLECTION_LIMIT"):
        calculate("{x:x for x in tables['x']}", {"x": list(range(1001))})


def test_dictionary_comprehension_invalid_keys_use_existing_error_boundary():
    with pytest.raises(ExpressionError, match="REFERENCE_EVALUATION_ERROR"):
        calculate("{[x]:x for x in tables['x']}", {"x": [1, 2]})


@pytest.mark.parametrize(
    "program",
    [
        "{x.__class__:x for x in tables['x']}",
        "{x:open('file') for x in tables['x']}",
        "{x:__import__('os').environ for x in tables['x']}",
        "{x:x for x in tables.items()}",
    ],
)
def test_dictionary_comprehension_does_not_expose_host_objects_or_methods(program):
    with pytest.raises(ExpressionError):
        calculate(program, {"x": [1, 2]})
