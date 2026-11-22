"""Guard the experiment against publication-source swaps and unequal arm inputs."""

from dataclasses import asdict

import pytest

from dataverify.agent.dispute_no_query import QueryDisabledDisputeModel, prompt_without_queries
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation import query_ablation as experiment
from dataverify.evaluation.dispute_recovery import study_cases
from dataverify.evaluation.expanded_cases import expected
from dataverify.verification.discovery_runtime import execute_readonly


def test_paired_arms_differ_only_by_query_permission():
    on, off = (asdict(experiment.policy_for(method)) for method in ("query_on", "query_off"))
    assert on.pop("tools_enabled") is True
    assert off.pop("tools_enabled") is False
    assert on == off


def test_grade_rejects_accurate_sql_with_rows_from_wrong_source():
    case = study_cases()[3]
    sql = case.base.good_sql
    full = experiment.full_source(case.base.source)
    good_rows = expected(case.base, full)
    assert execute_readonly(sql, full) == good_rows
    small_rows = execute_readonly(sql, case.base.source)
    assert small_rows != good_rows
    record = {
        "status": "reviewed",
        "candidate": {"sql": sql},
        "publication": {"sql": sql, "rows": small_rows},
    }
    assert experiment.grade(case, record)["wrong_release"]
    record["publication"]["rows"] = good_rows
    assert experiment.grade(case, record)["correct_delivery"]


def test_disabled_model_advertises_only_actions_the_host_accepts():
    model = object.__new__(QueryDisabledDisputeModel)
    assert '"action":"query"' not in prompt_without_queries()
    choices = {v["properties"]["action"]["enum"][0] for v in model.response_schema()["anyOf"]}
    assert choices == {"finish", "clarify", "revise"}
    with pytest.raises(GraphModelError, match="DISPUTE_QUERY_UNAVAILABLE"):
        model.parse_action('{"action":"query","sql":"SELECT 1","hypothesis":"test"}')
