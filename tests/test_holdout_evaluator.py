"""Input integrity and no-gold loading for the independent evaluator."""

import importlib.util
import sys
from copy import deepcopy
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


def test_semantic_wrong_release_and_scope_breach_are_separate(monkeypatch):
    challenge = module.load_challenges(mini_document())[0]

    class FakeGuard:
        def __init__(self, *args, **kwargs):
            self.last_result = {}

        def execute(self, *args):
            pass

        def publish(self):
            return {"status": "verified", "rows": [[999]], "verified_sql": "fake"}

    monkeypatch.setattr(module, "ProbeGuard", FakeGuard)
    result = module.score(challenge)
    assert result["semantic_wrong_release"]
    assert not result["scope_breach"]
    assert not result["passed"]
