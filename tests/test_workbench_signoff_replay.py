"""Verify the downloadable replay against executable data, even without paid records."""

import json

from dataverify.evaluation.dispute_recovery import study_cases
from dataverify.evaluation.expanded_cases import expected, scoring_data
from dataverify.evaluation.query_ablation import full_source
from dataverify.verification.discovery_runtime import execute_readonly
from tools.export_split_signoff_replay import PUBLIC, RECORDS, export


def test_public_signoff_replay_preserves_decisions_and_reproduces_correct_rows():
    fixture = json.loads(PUBLIC.read_text(encoding="utf-8"))
    case = next(c for c in study_cases() if c.case_id == fixture["provenance"]["case"])
    result = fixture["result"]
    rejection, check, support, delivery = result["evidence"]
    assert "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK" == rejection["scope"]
    assert check["id"] in rejection["detail"] and support["id"] in rejection["detail"]
    assert result["finish_citations"] == {
        "signoff_ids": [check["id"]],
        "supporting_ids": [support["id"]],
    }
    for name in ("candidate", "reference", "independent"):
        assert check["sample"][name]["rows"] == expected(case.base, case.base.source) == [[11]]
    source = full_source(case.base.source)
    for observation in (support, delivery):
        assert execute_readonly(observation["sql"], source) == observation["rows"] == [[669]]
    assert result["rows"] == expected(case.base, source)
    scoring = scoring_data(case.base)
    assert len(scoring) == 25
    for data in scoring.values():
        assert execute_readonly(result["sql"], data) == expected(case.base, data)
    if RECORDS.exists():
        assert export() == fixture
