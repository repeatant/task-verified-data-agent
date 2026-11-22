"""Independent, literal checks for the forward termination-policy challenge."""

import pytest

from dataverify.evaluation import terminal_scenarios as scenarios
from dataverify.evaluation.dispute_recovery import study_cases as prior_cases
from dataverify.verification.discovery_runtime import execute_readonly, named_tables
from dataverify.verification.reference_expression import calculate


def test_scenarios_are_distinct_from_the_known_development_set():
    questions = [case.base.task.question for case in scenarios.study_cases()]
    assert len(set(questions)) == 3
    assert not set(questions).intersection(case.base.task.question for case in prior_cases())
    manifest = scenarios.frozen_cases()
    assert len({item["requirement"] for item in manifest}) == 3
    assert len({item["full_source"] for item in manifest}) == 1
    assert len({item["preview"] for item in manifest}) == 1


@pytest.mark.parametrize(
    "dataset,counts",
    [
        ("preview", [1, 0, 0]),
        ("empty", [0, 0, 0]),
        ("bare_orders", [0, 0, 0]),
        ("duplicates_only", [1, 2, 0]),
        ("mixed_and_inactive", [3, 1, 2]),
        ("sign_and_channel", [3, 2, 1]),
    ],
)
def test_boundary_counts_are_independently_known(dataset, counts):
    data = scenarios.scoring_sources()[dataset]
    for case, count in zip(scenarios.study_cases(), counts, strict=True):
        assert scenarios.expected(case, data) == [[count]]
        assert execute_readonly(case.base.good_sql, data) == [[count]]
        assert calculate(case.base.good_reference, named_tables(data)) == [[count]]


def test_real_publication_rows_and_all_programs_must_match():
    case = scenarios.study_cases()[1]
    data = scenarios.scoring_sources()["full_source"]
    rows = scenarios.expected(case, data)
    good = case.base.good_sql
    assert rows != scenarios.expected(case, scenarios.PREVIEW)
    record = {
        "status": "reviewed",
        "candidate": {"sql": good},
        "publication": {"sql": good, "rows": rows},
        "plan": {"reference": case.base.good_reference},
        "audit_plan": {"reference": case.base.good_reference},
    }
    assert scenarios.grade(case, record)["joint_correct_delivery"]
    record["plan"] = {"reference": case.base.bad_reference}
    score = scenarios.grade(case, record)
    assert score["correct_delivery"] and not score["joint_correct_delivery"]
    record["publication"]["rows"] = [[0]]
    assert scenarios.grade(case, record)["wrong_release"]


def test_three_matching_mistakes_are_visible_to_the_external_scorer():
    case = scenarios.study_cases()[2]
    sql = case.initial_sql
    record = {
        "status": "reviewed",
        "candidate": {"sql": sql},
        "plan": case.reference_plan,
        "audit_plan": case.auditor_plan,
        "publication": {
            "sql": sql,
            "rows": execute_readonly(sql, scenarios.scoring_sources()["full_source"]),
        },
    }
    on_preview = execute_readonly(sql, scenarios.PREVIEW)
    for plan in [case.reference_plan, case.auditor_plan]:
        assert on_preview == calculate(plan["reference"], named_tables(scenarios.PREVIEW))
    score = scenarios.grade(case, record)
    assert (
        score["wrong_release"]
        and not score["reference"]["correct"]
        and not score["auditor"]["correct"]
    )
