"""Input integrity and no-gold loading for the independent evaluator."""

import importlib.util
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

spec = importlib.util.spec_from_file_location(
    "holdout_evaluator_test_module",
    Path(__file__).resolve().parents[1] / "tools/evaluate_independent_holdout.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def mini_document():
    task = {
        "task_id": "loader-test",
        "question": "统计三月已结算净额，无匹配时为零。",
        "properties": ["exclude_end", "exclude_status", "include_start", "include_negative"],
        "output_kind": "scalar",
        "start": "2026-03-01T00:00:00",
        "end": "2026-04-01T00:00:00",
        "status": "settled",
        "other_label": "其他",
    }
    data = {
        "orders": [{"order_id": 1, "total": 999, "channel": None}],
        "lines": [
            {
                "line_id": 1,
                "order_id": 1,
                "amount": 5,
                "status": "settled",
                "settled_at": "2026-03-15T00:00:00",
            }
        ],
    }
    sql = "SELECT COALESCE(SUM(l.amount),0) FROM order_lines l WHERE l.status='settled' AND l.settled_at>='2026-03-01T00:00:00' AND l.settled_at<'2026-04-01T00:00:00'"
    return {
        "challenges": [
            {
                "id": f"h{i:02d}",
                "category": "legal",
                "task": deepcopy(task),
                "data": deepcopy(data),
                "candidate_sql": sql,
                "expected_rows": [[5]],
                "reason": "Loader fixture, not a held-out task.",
            }
            for i in range(1, 5)
        ],
        "natural_probe_ids": ["h01", "h02", "h03", "h04"],
    }


def test_object_source_rows_are_loaded_without_a_gold_query():
    loaded = module.load_challenges(mini_document())
    assert len(loaded) == 4
    assert loaded[0].data.lines == ((1, 1, 5, "settled", "2026-03-15T00:00:00"),)
    assert not hasattr(loaded[0], "reference_sql")
    result = module.score(loaded[0])
    assert result["passed"] and result["publication"]["rows"] == [[5]]


@pytest.mark.parametrize(
    "corruption", ["target", "extra_column", "unsafe_id", "duplicate_id", "unknown_natural"]
)
def test_corrupt_or_changed_challenge_inputs_fail_before_model_use(corruption):
    document = mini_document()
    if corruption == "target":
        document["challenges"][0]["expected_rows"] = [[999]]
    elif corruption == "extra_column":
        document["challenges"][0]["data"]["lines"][0]["secret"] = "not-a-field"
    elif corruption == "unsafe_id":
        document["challenges"][0]["id"] = "../escape"
    elif corruption == "duplicate_id":
        document["challenges"][1]["id"] = "h01"
    else:
        document["natural_probe_ids"][-1] = "h99"
    with pytest.raises(ValueError):
        module.load_challenges(document)


def test_semantic_wrong_release_and_scope_breach_are_separate():
    challenge = module.load_challenges(mini_document())[0]

    class FakeGuard:
        def __init__(self, *args, **kwargs):
            self.last_result = {}

        def execute(self, *args):
            pass

        def publish(self):
            return {"status": "verified", "rows": [[999]], "verified_sql": "fake"}

    verifier = replace(module.current_verifier(), guard=FakeGuard)
    result = module.score(challenge, verifier=verifier)
    assert result["semantic_wrong_release"]
    assert not result["scope_breach"]
    assert not result["passed"]


def test_current_manifest_detects_a_changed_source_without_rewriting_evidence():
    manifest = module.capture_current_verifier()
    assert all(module.verify_source_manifest(manifest).values())
    source = next(iter(manifest["files"]))
    manifest["files"][source] = "0" * 64
    assert module.verify_source_manifest(manifest)[source] is False


def test_archived_manifest_is_rejected_instead_of_silently_rebased():
    historical = (
        Path(__file__).resolve().parent
        / "fixtures/independent_holdout/current_verifier_manifest.json"
    )
    original = historical.read_bytes()
    with pytest.raises(ValueError, match="unavailable source files"):
        module.verify_source_manifest(json.loads(original))
    assert historical.read_bytes() == original


@pytest.mark.parametrize(
    "outcome,exit_code,status,error_code",
    [
        ("source_change", 1, "invalid", "VERIFIER_SOURCE_CHANGED"),
        ("operational_error", 1, "error", "COMPONENT_OPERATIONAL_ERROR"),
        ("failed_challenge", 0, "completed", None),
    ],
)
def test_cli_does_not_confuse_run_integrity_with_challenge_results(
    tmp_path, monkeypatch, outcome, exit_code, status, error_code
):
    """Preserve actual manifest/file checks; isolate only the scoring outcome."""
    original_root = Path(__file__).resolve().parents[1]
    for name in module.VERIFIER_SOURCES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, destination)
    input_dir = tmp_path / "reports" / "input"
    input_dir.mkdir(parents=True)
    fixture_dir = original_root / "tests/fixtures/independent_holdout"
    for name in ("frozen_challenges.json", "author_manifest.json", "label_audit.json"):
        shutil.copyfile(fixture_dir / name, input_dir / name)
    output = tmp_path / "reports" / "output"
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def controlled_score(challenge, *, verifier):
        if outcome == "operational_error":
            raise RuntimeError("Injected component failure")
        if outcome == "source_change":
            source = tmp_path / module.VERIFIER_SOURCES[0]
            source.write_text(
                source.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8"
            )
        return {
            "id": challenge.identifier,
            "category": challenge.category,
            "passed": outcome != "failed_challenge",
            "semantic_wrong_release": False,
            "scope_breach": False,
            "refused_legal": False,
            "dynamic_new_detection": False,
        }

    monkeypatch.setattr(module, "score", controlled_score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_independent_holdout.py",
            "--dataset",
            str(input_dir / "frozen_challenges.json"),
            "--output",
            str(output),
        ],
    )
    assert module.main() == exit_code
    workflow = json.loads((output / "workflow.json").read_text(encoding="utf-8"))
    assert workflow["status"] == status
    assert workflow.get("error_code") == error_code
    summary = workflow["initial_summary"]
    assert summary["scheduled"] == 24
    assert summary["operational_errors"] == (24 if outcome == "operational_error" else 0)
    if outcome == "failed_challenge":
        assert summary["passed"] == 0
    elif outcome == "source_change":
        assert not all(workflow["verifier_unchanged_after"].values())
