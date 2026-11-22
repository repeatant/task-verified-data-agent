"""Explicit scalar conversion extends syntax without opening Python protocols."""

import pytest

from dataverify.verification.reference_expression import ExpressionError, calculate


@pytest.mark.parametrize(
    "program,expected",
    [
        ("[[int('12')]]", [[12]]),
        ("[[int(-1.9)]]", [[-1]]),
        ("[[float('1.25')]]", [[1.25]]),
        ("[[str(12)]]", [["12"]]),
        ("[[str(1.25)]]", [["1.25"]]),
        ("list({'x':1, 'y':2})", ["x", "y"]),
        ("list('ab')", ["a", "b"]),
        ("list((1,2))", [1, 2]),
    ],
)
def test_allowed_explicit_conversions(program, expected):
    assert calculate(program, {}) == expected


def test_numeric_text_field_can_be_compared_explicitly():
    table = {"measurements": [{"value": "12.5"}, {"value": None}, {"value": "2"}]}
    expression = "[[sum(float(r['value']) for r in tables['measurements'] if r['value'] is not None and float(r['value']) > 10)]]"
    assert calculate(expression, table) == [[12.5]]


@pytest.mark.parametrize(
    "expression",
    [
        "int('1' * 999999999)",
        "float('nan')",
        "float('inf')",
        "float('1e309')",
        "int('1000000000000001')",
        "float('1000000000000001')",
        "int(True)",
        "str(None)",
        "str({'x':1})",
        "str([1,2])",
        "list(999999999)",
        "list(None)",
        "int('ff',16)",
        "float('2',3)",
        "str(1,2)",
        "list([1],2)",
        "int = 3\n[[int]]",
        "float = 3\n[[float]]",
        "str = 3\n[[str]]",
        "list = 3\n[[list]]",
        "list.__class__",
        "str.__call__(1)",
        "int.__subclasses__()",
    ],
)
def test_conversions_do_not_expand_host_or_allocation_boundary(expression):
    with pytest.raises(ExpressionError):
        calculate(expression, {})


@pytest.mark.parametrize("function", ["int", "float", "str"])
def test_scalar_conversion_rejects_long_strings_before_call(function):
    with pytest.raises(ExpressionError, match="CONVERSION_STRING_LIMIT"):
        calculate(f"{function}(tables['value'])", {"value": "1" * 101})


def test_list_conversion_charges_budget_before_copy():
    with pytest.raises(ExpressionError, match="BUDGET"):
        calculate("list(tables['values'])", {"values": list(range(100))}, budget=10)
    with pytest.raises(ExpressionError, match="COLLECTION_LIMIT"):
        calculate("list(tables['values'])", {"values": list(range(1001))})


def test_list_conversion_does_not_mutate_source():
    source = {"values": [1, 2, 3]}
    result = calculate("list(tables['values'])", source)
    result.append(4)
    assert source == {"values": [1, 2, 3]}


@pytest.mark.parametrize(
    "program,expected",
    [
        ("'2026-11-07'[:4]", "2026"),
        ("'abcdef'[1:5:2]", "bd"),
        ("'abcdef'[-3:]", "def"),
        ("'abc'[::-1]", "cba"),
        ("[1,2,3][:2]", [1, 2]),
        ("[1,2,3][1000:]", []),
        ("'abcd'[:3][1:]", "bc"),
    ],
)
def test_bounded_slices(program, expected):
    assert calculate(program, {}) == expected


@pytest.mark.parametrize(
    "program",
    [
        "'abc'[::0]",
        "'abc'[1.1:]",
        "'abc'[True:]",
        "'abc'[:'2']",
        "{'a':1}[:]",
        "{1,2}[:]",
        "'abc'[slice(1)]",
    ],
)
def test_invalid_slices_rejected(program):
    with pytest.raises(ExpressionError):
        calculate(program, {})


def test_slice_copy_limit_and_budget_checked_before_allocation():
    with pytest.raises(ExpressionError, match="COLLECTION_LIMIT"):
        calculate("tables['values'][:]", {"values": list(range(1001))})
    with pytest.raises(ExpressionError, match="BUDGET"):
        calculate("tables['values'][:100]", {"values": list(range(100))}, budget=10)
