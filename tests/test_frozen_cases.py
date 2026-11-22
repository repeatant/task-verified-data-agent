"""Validate frozen scoring assumptions and the accounting of unsuccessful runs."""

from copy import deepcopy
from dataclasses import replace

import pytest

from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.evaluation.frozen_cases import (
    CONDITIONS,
    FAMILIES,
    expected_rows,
    get_frozen_cases,
    grade_frozen_run,
    scoring_datasets,
)
from dataverify.verification.discovery_runtime import (
    execute_readonly,
    named_tables,
    rows_equal,
    validate_plan,
)
from dataverify.verification.reference_expression import calculate

BASE_CASES = {case.name: case for case in get_discovery_cases()}
CASES = get_frozen_cases()
CORRECT_CASES = [case for case in CASES if case.condition == "correct"]


def successful_record(case):
    base = BASE_CASES[case.family]
    return {
        "status": "reviewed",
        "candidate": {"sql": base.good_sql},
        "publication": {
            "status": "reviewed",
            "sql": base.good_sql,
            "rows": expected_rows(case.family, case.source),
        },
        "plan": base.offline_plan(),
        "audit_plan": base.offline_plan(),
    }


def test_condition_cross_product_has_no_prompt_labels_or_shared_plan_mutations():
    assert len(CASES) == 16
    assert len({case.case_id for case in CASES}) == 16
    assert {(case.family, case.condition) for case in CASES} == {
        (family, condition) for family in FAMILIES for condition in CONDITIONS
    }
    for case in CASES:
        assert case.task.task_id == case.family
        assert case.condition not in str(case.task.public_input())
        assert validate_plan(case.initial_plan, case.task) == case.initial_plan
    mutated = get_frozen_cases()
    mutated[0].initial_plan["fixtures"][0]["name"] = "mutated"
    assert all(c.initial_plan["fixtures"][0]["name"] == "initial" for c in get_frozen_cases())
    assert mutated[1].initial_plan["fixtures"][0]["name"] == "initial"


@pytest.mark.parametrize("case", CORRECT_CASES, ids=lambda case: case.family)
def test_seeded_scoring_is_deterministic_and_shared_across_conditions(case):
    first, second = scoring_datasets(case), scoring_datasets(case)
    assert first == second
    assert len(first) == 31
    assert len([name for name in first if name.startswith("seeded_")]) == 24
    assert first == scoring_datasets(replace(case, condition="sql_wrong"))


@pytest.mark.parametrize("case", CORRECT_CASES, ids=lambda case: case.family)
def test_python_oracle_crosschecks_sql_and_correct_dsl_on_every_dataset(case):
    base = BASE_CASES[case.family]
    for name, dataset in scoring_datasets(case).items():
        expected = expected_rows(case.family, dataset)
        assert rows_equal(execute_readonly(base.good_sql, dataset), expected), name
        assert rows_equal(calculate(base.offline_reference, named_tables(dataset)), expected), name
    assert expected_rows(case.family, case.source) == base.expected


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.condition == "correlated_wrong"], ids=lambda c: c.family
)
def test_each_injected_fault_is_executable_but_semantically_wrong(case):
    disagreements = []
    for name, dataset in scoring_datasets(case).items():
        sql_rows = execute_readonly(case.initial_sql, dataset)
        ref_rows = calculate(case.initial_plan["reference"], named_tables(dataset))
        assert rows_equal(sql_rows, ref_rows), name
        if not rows_equal(sql_rows, expected_rows(case.family, dataset)):
            disagreements.append(name)
    assert disagreements


def test_authored_arithmetic_is_independently_inspectable():
    datasets = scoring_datasets(CORRECT_CASES[0])
    boundary = datasets["time_and_status_boundaries"]
    refund = datasets["refunds_and_zero_net_order"]
    join = datasets["join_and_grain"]
    assert expected_rows("order_average", boundary) == [[21]]
    assert expected_rows("web_scope", boundary) == [[13]]
    assert expected_rows("distinct_orders", refund) == [[2]]
    assert expected_rows("order_average", refund) == [[-19.5]]
    assert expected_rows("web_scope", refund) == [[0]]
    assert expected_rows("money_units", refund) == [[-0.39]]
    assert expected_rows("web_scope", join) == [[103]]
    assert expected_rows("order_average", join) == [[54.25]]
    assert expected_rows("distinct_orders", join) == [[4]]
    assert expected_rows("money_units", join) == [[2.17]]
    with pytest.raises(ValueError, match="UNKNOWN_SCORING_FAMILY"):
        expected_rows("invented", boundary)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_correct_published_triplet_counts_as_joint_delivery(case):
    grade = grade_frozen_run(case, successful_record(case))
    assert grade["correct_delivery"]
    assert grade["joint_validated_delivery"]
    assert not grade["wrong_release"]
    assert not grade["correct_sql_damaged"]
    assert not grade["initial_correct_sql_not_delivered"]
    for component in ("candidate", "reference", "audit"):
        assert grade[component]["passes_all"]
        assert grade[component]["passed"] == grade[component]["total"] == 31
        assert grade[component]["failed_datasets"] == []
        assert "expected" not in grade[component]
        assert "actual" not in grade[component]


def test_semantic_sql_protection_allows_equivalent_reformatting():
    case = CORRECT_CASES[0]
    record = successful_record(case)
    record["publication"]["sql"] = "  " + record["publication"]["sql"] + "  "
    assert record["publication"]["sql"] != case.initial_sql
    grade = grade_frozen_run(case, record)
    assert grade["correct_delivery"]
    assert not grade["correct_sql_damaged"]


def test_wrong_reference_does_not_disappear_behind_correct_sql():
    case = next(c for c in CASES if c.condition == "reference_wrong")
    record = successful_record(case)
    record["plan"] = case.initial_plan
    grade = grade_frozen_run(case, record)
    assert grade["correct_delivery"]
    assert not grade["joint_validated_delivery"]
    assert not grade["reference"]["passes_all"]
    assert grade["audit"]["passes_all"]


def test_refusal_is_neither_delivery_nor_wrong_release():
    case = CORRECT_CASES[0]
    record = successful_record(case)
    record["status"] = "needs_clarification"
    record["publication"] = {"status": "needs_clarification", "sql": None, "rows": None}
    grade = grade_frozen_run(case, record)
    assert grade["candidate"]["passes_all"]
    assert not grade["correct_delivery"]
    assert not grade["wrong_release"]
    assert grade["initial_correct_sql_not_delivered"]


def test_wrong_sql_is_a_wrong_release_and_semantic_damage_even_if_source_rows_are_right():
    case = CORRECT_CASES[0]
    record = successful_record(case)
    record["publication"]["sql"] = BASE_CASES[case.family].bad_sql
    grade = grade_frozen_run(case, record)
    assert not grade["correct_delivery"]
    assert grade["wrong_release"]
    assert grade["correct_sql_damaged"]
    assert grade["initial_correct_sql_not_delivered"]


def test_published_rows_are_checked_as_well_as_sql():
    case = CORRECT_CASES[0]
    record = successful_record(case)
    record["publication"]["rows"] = [[123456]]
    grade = grade_frozen_run(case, record)
    assert grade["candidate"]["passes_all"]
    assert grade["wrong_release"]
    assert not grade["source_rows_match"]
    assert not grade["correct_delivery"]


@pytest.mark.parametrize(
    "bad_plan", [None, {}, {"reference": "[[1/0]]"}, {"reference": "import os"}, {"reference": [1]}]
)
def test_missing_or_invalid_auditor_is_scored_without_crashing(bad_plan):
    case = CORRECT_CASES[0]
    record = successful_record(case)
    record["audit_plan"] = bad_plan
    grade = grade_frozen_run(case, record)
    assert grade["correct_delivery"]
    assert not grade["joint_validated_delivery"]
    assert grade["audit"]["total"] == len(grade["audit"]["error_datasets"])


def test_missing_record_cannot_count_as_success():
    grade = grade_frozen_run(CORRECT_CASES[0], {})
    assert not grade["correct_delivery"]
    assert not grade["joint_validated_delivery"]
    assert not grade["wrong_release"]
    assert grade["candidate"]["total"] == len(grade["candidate"]["error_datasets"])


def test_natural_generation_has_no_injected_correctness_denominator():
    case = replace(
        CORRECT_CASES[0],
        case_id="order_average__natural",
        condition="natural",
        initial_sql=None,
        initial_plan=None,
    )
    record = deepcopy(successful_record(case))
    grade = grade_frozen_run(case, record)
    assert grade["correct_delivery"]
    assert grade["initial_candidate"] is None
    assert not grade["correct_sql_damaged"]
    assert not grade["initial_correct_sql_not_delivered"]
