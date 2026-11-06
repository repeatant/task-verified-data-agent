"""Format feedback remains answer-blind and keeps the provenance guard strict."""

import pytest

from dataverify.agent.discovery_model import plan_validation_feedback
from dataverify.evaluation.discovery_cases import get_discovery_cases
from dataverify.verification.discovery_runtime import validate_plan
from dataverify.verification.reference_expression import calculate, parse_expression


def test_quote_feedback_offers_only_actual_source_fragments():
    case = get_discovery_cases()[0]
    plan = case.offline_plan()
    plan["basis"] = [{"source": "question", "quote": "some...paraphrase"}]
    with pytest.raises(ValueError, match="BASIS_NOT_IN_SOURCE"):
        validate_plan(plan, case.task)
    feedback = plan_validation_feedback(plan, "BASIS_NOT_IN_SOURCE", case.task)
    assert feedback["source_fragments"]
    assert all(piece in case.task.question for piece in feedback["source_fragments"])
    assert case.good_sql not in str(feedback)
    assert "reference" not in feedback and "expected" not in feedback


def test_syntax_feedback_locates_error_without_replacing_program():
    case = get_discovery_cases()[0]
    feedback = plan_validation_feedback(
        {"reference": "x = [1,\n[[x]]"}, "REFERENCE_SYNTAX", case.task
    )
    assert feedback["syntax"]["line"] == 1
    assert feedback["syntax"]["column"] > 0
    assert "reference" not in feedback


@pytest.mark.parametrize(
    "program,code,node,line",
    [
        ("result = [[1]]", "REFERENCE_FINAL_EXPRESSION_REQUIRED", "Assign", 1),
        (
            "values = []\nfor item in [1]:\n    values = [item]\n[[1]]",
            "REFERENCE_ONLY_LOCAL_ASSIGNMENTS_ALLOWED",
            "For",
            2,
        ),
        (
            "value = 1\nif value:\n    value = 2\n[[value]]",
            "REFERENCE_ONLY_LOCAL_ASSIGNMENTS_ALLOWED",
            "If",
            2,
        ),
        (
            "mapping = {'a': 1}\n[[mapping.get('a', 0)]]",
            "REFERENCE_CALL_NOT_ALLOWED",
            "Attribute",
            2,
        ),
        (
            "mapping = {'a': 1}\nvalues = mapping.values()\n[[sum(values)]]",
            "REFERENCE_CALL_NOT_ALLOWED",
            "Attribute",
            2,
        ),
        (
            "mapping = {'a': 1}\nmapping['a'] = 2\n[[1]]",
            "REFERENCE_ONLY_LOCAL_ASSIGNMENTS_ALLOWED",
            "Assign",
            2,
        ),
    ],
)
def test_dsl_feedback_locates_rejected_construct_without_relaxing_interpreter(
    program, code, node, line
):
    case = get_discovery_cases()[0]
    with pytest.raises(ValueError, match=code):
        parse_expression(program)
    feedback = plan_validation_feedback({"reference": program}, code, case.task)
    assert any(d["node"] == node and d["line"] == line for d in feedback["dsl_diagnostics"])
    # The format example is executable, generic and independent of business data.
    assert calculate(feedback["syntax_example"], {}) == [[1]]
    assert case.good_sql not in str(feedback)
    assert case.offline_reference not in str(feedback)
    assert "reference" not in feedback and "expected" not in feedback
    with pytest.raises(ValueError, match=code):
        parse_expression(program)


def test_diagnostics_show_multiple_mistakes_from_one_rejected_program():
    program = (
        "mapping = {'a': 1}\nfor key in mapping:\n    value = mapping.get(key)\nresult = [[value]]"
    )
    feedback = plan_validation_feedback(
        {"reference": program}, "REFERENCE_FINAL_EXPRESSION_REQUIRED", get_discovery_cases()[0].task
    )
    assert {d["node"] for d in feedback["dsl_diagnostics"]} == {"Assign", "For", "Attribute"}
    assert len(feedback["dsl_diagnostics"]) <= 6


def test_runtime_calculation_error_does_not_receive_irrelevant_syntax_example():
    feedback = plan_validation_feedback(
        {"reference": "[[1 / 0]]"}, "REFERENCE_EVALUATION", get_discovery_cases()[0].task
    )
    assert "syntax_example" not in feedback
