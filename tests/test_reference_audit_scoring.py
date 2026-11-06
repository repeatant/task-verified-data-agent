"""Post-run oracle scoring must expose errors hidden by source-only SQL checks."""

from copy import deepcopy
from dataclasses import replace

from dataverify.evaluation.reference_audit_cases import audit_scenarios
from dataverify.evaluation.reference_audit_scoring import grade_audit_run


def scenario_named(name):
    return next(s for s in audit_scenarios() if s.name == name)


def record_for(scenario):
    return {
        "status": "reviewed",
        "publication": {"status": "reviewed", "sql": scenario.final_sql},
        "plan": deepcopy(scenario.independent_plan),
        "audit_plan": deepcopy(scenario.independent_plan),
    }


def test_python_round_reference_error_is_visible_even_when_sql_passes():
    scenario = scenario_named("requested_rounding")
    record = record_for(scenario)
    for key in ("plan", "audit_plan"):
        record[key]["reference"] = record[key]["reference"].replace("round_half_up(", "round(")
    result = grade_audit_run(scenario, record)
    assert result["candidate_passes_all"]
    assert not result["reference_passes_all"]
    assert not result["independent_passes_all"]
    tie = next(x for x in result["checks"] if x["name"] == "half_tie")
    assert tie["gold"]["rows"] == tie["candidate"]["rows"] == [[0.13]]
    assert tie["reference"]["rows"] == tie["independent"]["rows"] == [[0.12]]
    assert not tie["reference"]["match"] and not tie["independent"]["match"]


def test_half_up_reference_matches_explicit_rounding():
    scenario = scenario_named("requested_rounding")
    record = record_for(scenario)
    for key in ("plan", "audit_plan"):
        record[key]["reference"] = record[key]["reference"].replace("round(", "round_half_up(")
    result = grade_audit_run(scenario, record)
    assert result["candidate_passes_all"]
    assert result["reference_passes_all"]
    assert result["independent_passes_all"]


def test_unrequested_rounding_is_detected_on_recurring_decimals():
    scenario = scenario_named("wrong_reference_rounding")
    record = record_for(scenario)
    record["plan"] = deepcopy(scenario.reference_plan)
    result = grade_audit_run(scenario, record)
    recurring = next(x for x in result["checks"] if x["name"] == "recurring_decimal")
    assert result["candidate_passes_all"] and result["independent_passes_all"]
    assert not result["reference_passes_all"]
    assert recurring["reference"]["rows"] == [[10.33]]
    assert recurring["candidate"]["rows"] == [[31 / 3]]


def test_scoring_covers_seven_groups_without_mutating_the_record():
    scenario = scenario_named("correct_pair")
    record = record_for(scenario)
    before = deepcopy(record)
    result = grade_audit_run(scenario, record)
    assert {x["name"] for x in result["checks"]} == {
        "source",
        "order_grain",
        "empty",
        "recurring_decimal",
        "half_tie",
        "refunds",
        "mixed_boundaries",
    }
    assert result["candidate_passes_all"] and result["reference_passes_all"]
    assert record == before


def test_missing_and_invalid_references_are_reported_not_hidden():
    scenario = scenario_named("correct_pair")
    record = record_for(scenario)
    record["plan"]["reference"] = "[[1/0]]"
    record.pop("audit_plan")
    record["publication"]["sql"] = "DELETE FROM orders"
    result = grade_audit_run(scenario, record)
    assert not result["candidate_passes_all"]
    assert not result["reference_passes_all"]
    assert not result["independent_passes_all"]
    assert all(x["candidate"]["status"] == "error" for x in result["checks"])
    assert all(x["reference"]["status"] == "error" for x in result["checks"])
    assert all(x["independent"]["error"] == "SCORING_REFERENCE_MISSING" for x in result["checks"])


def test_unpublished_candidate_is_diagnostic_only():
    scenario = scenario_named("correct_pair")
    record = record_for(scenario)
    record.update(
        status="needs_clarification",
        publication={"sql": None},
        candidate={"sql": scenario.final_sql},
    )
    result = grade_audit_run(scenario, record)
    assert result["candidate_passes_all"]
    assert result["candidate_origin"] == "last_candidate_unpublished"
    assert result["run_status"] == "needs_clarification"


def test_invalid_gold_setup_cannot_yield_an_overall_pass():
    scenario = scenario_named("correct_pair")
    scenario = replace(scenario, expected_rows=[[999999]])
    result = grade_audit_run(scenario, record_for(scenario))
    assert not result["gold_source_consistent"]
    assert not result["gold_valid_on_scoring_data"]
    assert not result["candidate_passes_all"]
    assert not result["reference_passes_all"]
    assert not result["independent_passes_all"]


def test_correlated_error_without_anchor_is_scored_as_wrong():
    scenario = scenario_named("correlated_error_without_anchor")
    record = record_for(scenario)
    record["publication"]["sql"] = scenario.initial_sql
    record["plan"] = deepcopy(scenario.reference_plan)
    result = grade_audit_run(scenario, record)
    assert result["gold_valid_on_scoring_data"]
    assert not result["candidate_passes_all"]
    assert not result["reference_passes_all"]
    assert not result["independent_passes_all"]
    grain = next(x for x in result["checks"] if x["name"] == "order_grain")
    assert grain["gold"]["rows"] == [[100]]
    assert grain["candidate"]["rows"] == [[200 / 3]]
