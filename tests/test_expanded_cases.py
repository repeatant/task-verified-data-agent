"""Audit authored semantic faults against separate SQL, DSL and arithmetic paths."""

import pytest

from dataverify.evaluation.expanded_cases import (
    CONDITIONS,
    FAMILIES,
    expanded_cases,
    expected,
    grade_sql,
    scoring_data,
)
from dataverify.verification.discovery_runtime import (
    execute_readonly,
    named_tables,
    rows_equal,
    validate_plan,
)
from dataverify.verification.reference_expression import calculate

PARENTS = [case for case in expanded_cases() if case.condition == "correct"]


def test_population_counts_do_not_overstate_independent_tasks():
    cases = expanded_cases()
    assert len(cases) == 100
    assert len({case.parent_id for case in cases}) == 20
    # Upper/lower boundary variants share the same business question: count
    # them as different injected mechanisms, never independent task clusters.
    assert len({case.task.task_id for case in cases}) == 18
    assert {case.family for case in cases} == set(FAMILIES)
    assert len(FAMILIES) == 10
    for parent in PARENTS:
        siblings = [case for case in cases if case.parent_id == parent.parent_id]
        assert {case.condition for case in siblings} == set(CONDITIONS)
        assert all(case.task == parent.task and case.source == parent.source for case in siblings)


@pytest.mark.parametrize("case", PARENTS, ids=lambda c: c.parent_id)
def test_every_gold_sql_and_reference_matches_separate_oracle(case):
    datasets = scoring_data(case)
    assert len(datasets) == 25
    for name, data in datasets.items():
        truth = expected(case, data)
        assert rows_equal(execute_readonly(case.good_sql, data), truth), name
        assert rows_equal(calculate(case.good_reference, named_tables(data)), truth), name
    assert grade_sql(case, case.good_sql)["correct"]


@pytest.mark.parametrize("case", PARENTS, ids=lambda c: c.parent_id)
def test_each_injected_fault_is_hidden_on_source_but_revealed_on_holdout(case):
    assert rows_equal(
        execute_readonly(case.good_sql, case.source), execute_readonly(case.bad_sql, case.source)
    )
    sql_wrong, ref_wrong = [], []
    for name, data in scoring_data(case).items():
        truth = expected(case, data)
        sql_wrong.append(not rows_equal(execute_readonly(case.bad_sql, data), truth))
        ref_wrong.append(not rows_equal(calculate(case.bad_reference, named_tables(data)), truth))
    assert any(sql_wrong), "ineffective SQL mutation"
    assert any(ref_wrong), "ineffective reference mutation"
    assert sql_wrong == ref_wrong, "correlated-fault pair does not implement the same fault"
    assert not grade_sql(case, case.bad_sql)["correct"]


@pytest.mark.parametrize("case", expanded_cases(), ids=lambda c: c.case_id)
def test_all_initial_role_programs_are_valid_and_only_injected_where_declared(case):
    first = validate_plan(case.initial_plan(), case.task)
    audit = validate_plan(case.initial_plan(auditor=True), case.task)
    assert first["reference"] == (
        case.bad_reference
        if case.condition in {"reference_wrong", "correlated_wrong"}
        else case.good_reference
    )
    assert audit["reference"] == (
        case.bad_reference if case.condition == "auditor_wrong" else case.good_reference
    )
    assert case.initial_sql == (
        case.bad_sql if case.condition in {"sql_wrong", "correlated_wrong"} else case.good_sql
    )
    public = str(case.task.public_input())
    assert case.condition not in public
    assert case.family not in case.task.task_id
    assert case.good_sql not in public and case.bad_sql not in public


@pytest.mark.parametrize(
    "family,value",
    [
        ("upper_boundary", 160),
        ("lower_boundary", 160),
        ("refund_sign", 160),
        ("status_scope", 160),
        ("zero_activity", 4),
        ("null_channel", 71),
        ("duplicate_amounts", 5),
        ("aggregate_threshold", 0),
        ("population_denominator", 40),
        ("absolute_net", 160),
    ],
)
def test_hand_checked_fixture_has_independent_numeric_answers(family, value):
    case = next(case for case in PARENTS if case.family == family and case.variant == 1)
    assert expected(case, scoring_data(case)["boundaries_and_duplicates"]) == [[value]]


def test_scoring_generation_is_reproducible_and_conditions_share_data():
    cases = expanded_cases()
    for parent in PARENTS:
        expected_data = {name: data.to_dict() for name, data in scoring_data(parent).items()}
        for sibling in [case for case in cases if case.parent_id == parent.parent_id]:
            assert {
                name: data.to_dict() for name, data in scoring_data(sibling).items()
            } == expected_data
