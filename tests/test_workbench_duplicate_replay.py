"""Published evidence must reproduce the historical fixture results locally."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from dataverify.evaluation.dispute_recovery import visible_diagnostic_fixture
from dataverify.verification.discovery_runtime import execute_readonly, named_tables
from dataverify.verification.reference_expression import calculate

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "dataverify/workbench/data/duplicate-amount.json"
ORIGINAL = ROOT / "reports/dispute-recovery-v1/records/dispute-04--dispute_resolver.json"


def test_public_history_executes_on_its_declared_fixture():
    item = json.loads(PUBLIC.read_text(encoding="utf-8"))
    result = item["result"]
    source = visible_diagnostic_fixture()
    candidate, reference = result["revisions"]
    assert item["mode"] == "replay"
    assert candidate["target"] == "candidate" and reference["target"] == "reference"
    assert execute_readonly(candidate["before"], source) == [[5]]
    assert execute_readonly(candidate["after"], source) == result["rows"] == [[6]]
    assert calculate(reference["before"], named_tables(source)) == [[5]]
    assert calculate(reference["after"], named_tables(source)) == [[6]]
    assert result["sql"] == candidate["after"]
    assert execute_readonly(result["evidence"][1]["sql"], source) == [[6]]
    assert result["evidence"][1]["id"] in candidate["evidence_ids"]


@pytest.mark.skipif(
    not ORIGINAL.exists(), reason="Private archived trace is not part of the public repository"
)
def test_export_is_an_exact_projection_of_the_pinned_trace():
    spec = importlib.util.spec_from_file_location(
        "replay_export", ROOT / "tools/export_duplicate_replay.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == exporter.RECORD_SHA256
    assert exporter.export(ORIGINAL) == json.loads(PUBLIC.read_text(encoding="utf-8"))
