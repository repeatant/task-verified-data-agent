"""Execution attribution cannot treat a mistaken oracle as an answer authority."""

from copy import deepcopy

import pytest

from dataverify.evaluation.discovery_cases import data, get_discovery_cases
from dataverify.verification.discovery_runtime import DiscoveryTask, fixture_data, named_tables
from dataverify.verification.reference_audit import (
    authorize_decision,
    find_distinguishing_fixture,
    inspect_dispute,
    merge_fixtures,
    validate_confirmed_examples,
)


def average_dispute():
    case = get_discovery_cases()[0]
    correct = case.offline_plan()
    wrong = deepcopy(correct)
    wrong["reference"] = (
        "amounts = [r['amount'] for r in tables['order_lines']]\n"
        "[[round(sum(amounts)/len(amounts),2) if amounts else 0]]"
    )
    precision = data([10, 10, 11])
    evidence = inspect_dispute(case.task, precision, case.good_sql, wrong, correct, [])
    return case, correct, wrong, precision, evidence


def attribution(task, evidence, target):
    return {
        "action": "attribute",
        "target": target,
        "reason": "The original requirement and the executed example distinguish the programs.",
        "basis": [{"source": "question", "quote": task.question}],
        "checks": evidence["summary"].get(f"{target}_difference_ids", []),
    }


def test_rounding_error_is_attributed_to_reference_not_valid_sql():
    case, _, _, _, evidence = average_dispute()
    check = evidence["checks"][0]
    assert check["candidate"]["rows"] == [[31 / 3]]
    assert check["reference"]["rows"] == [[10.33]]
    assert check["independent"]["rows"] == [[31 / 3]]
    decision = authorize_decision(
        attribution(case.task, evidence, "reference"), case.task, evidence
    )
    assert decision["target"] == "reference"
    assert decision["evidence_level"] == "model_supported_not_proven"


@pytest.mark.parametrize("case", get_discovery_cases(), ids=lambda case: case.name)
def test_candidate_repair_needs_executed_supporting_calculations(case):
    reference = case.offline_plan()
    evidence = inspect_dispute(
        case.task, case.source, case.bad_sql, reference, reference, reference["fixtures"]
    )
    decision = authorize_decision(
        attribution(case.task, evidence, "candidate"), case.task, evidence
    )
    assert decision["target"] == "candidate"
    assert decision["evidence_level"] == "model_supported_not_proven"


def test_explicit_rounding_is_not_globally_banned():
    _, unrounded, rounded, source, _ = average_dispute()
    task = DiscoveryTask(
        "rounded", "Return the mean of line amounts rounded to two decimal places."
    )
    sql = "SELECT ROUND(AVG(amount),2) FROM order_lines"
    evidence = inspect_dispute(task, source, sql, unrounded, rounded, [])
    assert (
        authorize_decision(attribution(task, evidence, "reference"), task, evidence)["target"]
        == "reference"
    )


def test_unrelated_matching_check_cannot_authorize_repair():
    case, _, _, _, evidence = average_dispute()
    action = attribution(case.task, evidence, "candidate")
    action["checks"] = [evidence["checks"][0]["check_id"]]
    result = authorize_decision(action, case.task, evidence)
    assert result["target"] == "ambiguous"
    assert result["reason"] == "ATTRIBUTION_CITES_NO_RELEVANT_DISAGREEMENT"


def test_caller_confirmed_example_provides_explicit_bounded_authority():
    case, correct, wrong, source, _ = average_dispute()
    confirmed = [
        {
            "name": "manual-example",
            "data": source.to_dict(),
            "expected": [[31 / 3]],
            "source": "Caller-confirmed arithmetic: 31/3; no rounding.",
        }
    ]
    evidence = inspect_dispute(case.task, source, case.good_sql, wrong, correct, [], confirmed)
    decision = authorize_decision(
        attribution(case.task, evidence, "reference"), case.task, evidence
    )
    assert decision["evidence_level"] == "caller_confirmed_example"
    assert evidence["summary"]["confirmed_count"] == 1
    assert len(evidence["checks"]) == 1


def test_confirmed_data_can_block_two_agreeing_wrong_calculations():
    case, correct, wrong, source, _ = average_dispute()
    confirmed = [
        {
            "name": "exact-mean",
            "data": source.to_dict(),
            "expected": [[31 / 3]],
            "source": "Caller confirmed exact output.",
        }
    ]
    evidence = inspect_dispute(case.task, source, case.good_sql, wrong, wrong, [], confirmed)
    assert evidence["summary"]["reference_matches_independent"]
    result = authorize_decision(attribution(case.task, evidence, "candidate"), case.task, evidence)
    assert result["target"] == "ambiguous"
    assert result["reason"] == "SUPPORTING_CALCULATIONS_VIOLATE_CONFIRMED_EXAMPLE"


def test_agreement_of_all_three_does_not_override_confirmed_data():
    case, _, wrong, source, _ = average_dispute()
    confirmed = [
        {
            "name": "exact-mean",
            "data": source.to_dict(),
            "expected": [[31 / 3]],
            "source": "Caller confirmed exact output.",
        }
    ]
    evidence = inspect_dispute(
        case.task,
        source,
        "SELECT ROUND(AVG(amount),2) FROM order_lines",
        wrong,
        wrong,
        [],
        confirmed,
    )
    assert evidence["summary"]["candidate_matches_reference"]
    assert not evidence["summary"]["candidate_matches_confirmed"]
    assert not evidence["summary"]["independent_matches_confirmed"]


def test_supporting_calculation_must_be_defined_on_every_retained_fixture():
    case, correct, wrong, source, _ = average_dispute()
    undefined = deepcopy(correct)
    undefined["reference"] = "[[1/0]]"
    evidence = inspect_dispute(case.task, source, case.good_sql, wrong, undefined, [])
    assert evidence["checks"][0]["independent"]["status"] == "error"
    action = attribution(case.task, evidence, "reference")
    action["checks"] = [evidence["checks"][0]["check_id"]]
    assert authorize_decision(action, case.task, evidence)["target"] == "ambiguous"


def test_candidate_execution_failure_can_be_attributed_when_references_execute():
    case = get_discovery_cases()[0]
    reference = case.offline_plan()
    evidence = inspect_dispute(
        case.task, case.source, "DELETE FROM orders", reference, reference, []
    )
    assert evidence["checks"][0]["candidate"]["status"] == "error"
    assert (
        authorize_decision(attribution(case.task, evidence, "candidate"), case.task, evidence)[
            "target"
        ]
        == "candidate"
    )


def test_reference_execution_failure_can_be_attributed_when_other_programs_execute():
    case, correct, wrong, source, _ = average_dispute()
    wrong["reference"] = "[[1/0]]"
    evidence = inspect_dispute(case.task, source, case.good_sql, wrong, correct, [])
    assert evidence["checks"][0]["reference"]["status"] == "error"
    assert (
        authorize_decision(attribution(case.task, evidence, "reference"), case.task, evidence)[
            "target"
        ]
        == "reference"
    )


def test_cherry_picking_a_supporting_check_cannot_hide_another_disagreement():
    case = get_discovery_cases()[0]
    reference = case.offline_plan()
    independent = deepcopy(reference)
    independent["reference"] = reference["reference"].replace("else 0", "else 1")
    evidence = inspect_dispute(
        case.task,
        case.source,
        case.bad_sql,
        reference,
        independent,
        [{"name": "retained-empty", "data": {"orders": [], "lines": []}}],
    )
    action = attribution(case.task, evidence, "candidate")
    assert action["checks"]  # The source supports repair, but the empty fixture does not.
    decision = authorize_decision(action, case.task, evidence)
    assert decision["target"] == "ambiguous"
    assert decision["reason"] == "SUPPORTING_CALCULATIONS_DISAGREE"


def test_summary_cannot_forge_execution_relationships():
    case, _, _, _, evidence = average_dispute()
    action = attribution(case.task, evidence, "candidate")
    action["checks"] = [evidence["checks"][0]["check_id"]]
    evidence["summary"]["reference_matches_independent"] = True
    evidence["summary"]["candidate_difference_ids"] = action["checks"]
    assert authorize_decision(action, case.task, evidence)["target"] == "ambiguous"


@pytest.mark.parametrize(
    "field,value",
    [
        ("checks", ["made-up"]),
        ("basis", [{"source": "question", "quote": "Invented rounding requirement"}]),
    ],
)
def test_fabricated_execution_or_requirement_citations_are_rejected(field, value):
    case, _, _, _, evidence = average_dispute()
    action = attribution(case.task, evidence, "reference")
    action[field] = value
    with pytest.raises(ValueError):
        authorize_decision(action, case.task, evidence)


def test_attribution_is_bound_to_original_task():
    case, _, _, _, evidence = average_dispute()
    other = DiscoveryTask(case.name, "A different question.")
    with pytest.raises(ValueError, match="TASK_MISMATCH"):
        authorize_decision(attribution(other, evidence, "reference"), other, evidence)


def test_ambiguous_decision_requires_no_fabricated_winner():
    case, _, _, _, evidence = average_dispute()
    action = {
        "action": "attribute",
        "target": "ambiguous",
        "reason": "Business policy is missing.",
        "basis": [],
        "checks": [],
    }
    assert authorize_decision(action, case.task, evidence)["target"] == "ambiguous"


def test_merge_deduplicates_row_encodings_without_changing_original_records():
    source = data([10, -2])
    array_rows = {"name": "original", "data": source.to_dict()}
    named_rows = {"name": "duplicate", "data": named_tables(source)}
    named_rows["data"]["order_lines"].reverse()
    distinct = {"name": "new", "data": data([0]).to_dict()}
    before = deepcopy(array_rows)
    merged = merge_fixtures([array_rows], [named_rows, distinct])
    assert merged == [array_rows, distinct]
    assert array_rows == before
    merged[0]["name"] = "changed"
    assert array_rows["name"] == "original"


def test_merge_refuses_to_discard_old_data_when_limit_is_reached():
    old = [{"name": str(i), "data": data([i]).to_dict()} for i in range(12)]
    with pytest.raises(ValueError, match="FIXTURE_LIMIT"):
        merge_fixtures(old, [{"name": "new", "data": data([13]).to_dict()}])
    assert len(old) == 12


def test_conflicting_confirmed_examples_are_not_resolved_by_voting():
    item = {
        "name": "confirmed",
        "data": data([10]).to_dict(),
        "expected": [[10]],
        "source": "Caller.",
    }
    with pytest.raises(ValueError, match="CONFLICTING_CONFIRMED"):
        validate_confirmed_examples([item, {**item, "expected": [[11]]}])


def test_confirmed_examples_require_explicit_caller_provenance():
    with pytest.raises(ValueError, match="SOURCE"):
        validate_confirmed_examples(
            [{"name": "x", "data": data([10]).to_dict(), "expected": [[10]], "source": ""}]
        )


def test_hypothesis_search_finds_and_replays_arithmetic_disagreement():
    sql = "SELECT COALESCE(AVG(amount),0) FROM order_lines"
    exact = "xs=[r['amount'] for r in tables['order_lines']]\n[[sum(xs)/len(xs) if xs else 0]]"
    rounded = (
        "xs=[r['amount'] for r in tables['order_lines']]\n[[round(sum(xs)/len(xs),2) if xs else 0]]"
    )
    witness = find_distinguishing_fixture(sql, rounded, exact, max_examples=80)
    assert witness is not None
    source = fixture_data(witness["data"])
    task = DiscoveryTask("mean", "Return the exact mean of all line amounts.")
    evidence = inspect_dispute(task, source, sql, {"reference": rounded}, {"reference": exact}, [])
    assert not evidence["summary"]["reference_matches_independent"]
    assert evidence["summary"]["candidate_matches_independent"]


def test_no_search_witness_is_not_a_claim_of_equivalence():
    sql = "SELECT COUNT(*) FROM order_lines"
    reference = "[[len(tables['order_lines'])]]"
    assert find_distinguishing_fixture(sql, reference, reference, max_examples=8) is None
