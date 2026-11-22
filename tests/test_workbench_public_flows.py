"""Exercise public-data tasks through the same graph as live questions."""

import asyncio

import pytest

from dataverify.agent.graph_model import ScriptedGraphModel
from dataverify.workbench import penguin_source as penguins
from dataverify.workbench import runner
from dataverify.workbench.sample_source import NET_AMOUNT, bind_source


def run(directory, question, definitions=""):
    return asyncio.run(
        runner.run_task(
            run_id="public-example",
            question=question,
            definitions=definitions,
            source_id=penguins.DATASET,
            mode="offline",
            directory=directory,
        )
    )


@pytest.mark.parametrize(
    ("question", "definition", "rows"),
    [
        (penguins.SPECIES_COUNTS, "", [["Adelie", 152], ["Chinstrap", 68], ["Gentoo", 124]]),
        (penguins.MASS_COVERAGE, "", [[342]]),
        (penguins.AMBIGUOUS_COUNT, "全部记录", [[344]]),
        (penguins.AMBIGUOUS_COUNT, "有体重记录", [[342]]),
    ],
)
def test_result_comes_from_full_public_source(tmp_path, question, definition, rows):
    result = run(tmp_path, question, definition)
    assert result["status"] == "reviewed"
    assert result["rows"] == rows
    assert result["evidence"][-1]["scope"] == "original_database"
    assert "344" in result["evidence"][-2]["detail"]
    assert result["usage"]["requests"] == 0


def test_ambiguous_or_unregistered_question_never_publishes(tmp_path):
    result = run(tmp_path / "ask", penguins.AMBIGUOUS_COUNT)
    assert result["status"] == "needs_clarification"
    assert "有体重记录" in result["clarification"]
    assert result["rows"] is None
    unknown = run(tmp_path / "unknown", NET_AMOUNT)
    assert unknown["status"] == "inconclusive" and unknown["rows"] is None
    unsupported = run(tmp_path / "other", penguins.AMBIGUOUS_COUNT, "按岛屿计数")
    assert unsupported["status"] == "inconclusive" and unsupported["sql"] is None


def test_query_that_passes_preview_but_drops_unseen_species_cannot_ship(tmp_path, monkeypatch):
    db = bind_source(penguins.DATASET, tmp_path)
    preview = db.preview(rows_per_table=50, retain_parents=True)
    assert {row["species"] for row in preview.named_tables()["penguins"]} == {"Adelie"}
    registered_models = runner.offline_models

    def faulty_writer(task, definitions, **kwargs):
        _, reviewer, auditor, investigator = registered_models(task, definitions, **kwargs)
        sql = "SELECT COUNT(body_mass_g) FROM penguins WHERE species != 'Gentoo'"
        writer = ScriptedGraphModel([{"task_id": task.task_id, "sql": sql}])
        return writer, reviewer, auditor, investigator

    monkeypatch.setattr(runner, "offline_models", faulty_writer)
    result = run(tmp_path, penguins.MASS_COVERAGE)
    assert result["status"] == "inconclusive"
    assert result["reason"] == "FULL_REFERENCE_DISAGREEMENT"
    assert result["sql"] is None and result["rows"] is None
    assert result["candidate_sql"] is not None


def test_unverified_rows_above_full_replay_limit_are_never_delivered(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_FULL_REPLAY_MAX_ROWS", 343)
    result = run(tmp_path, penguins.MASS_COVERAGE)
    assert result["reason"] == "FULL_REFERENCE_ROW_LIMIT"
    assert result["status"] == "inconclusive"
    assert result["rows"] is None and result["sql"] is None


def test_both_references_must_agree_on_unseen_species(tmp_path, monkeypatch):
    registered_models = runner.offline_models

    def faulty_audit(task, definitions, **kwargs):
        writer, reviewer, auditor, investigator = registered_models(task, definitions, **kwargs)
        auditor.plan["reference"] = (
            "[[len([r for r in tables['penguins'] "
            "if r['body_mass_g'] is not None and r['species'] != 'Gentoo'])]]"
        )
        return writer, reviewer, auditor, investigator

    monkeypatch.setattr(runner, "offline_models", faulty_audit)
    result = run(tmp_path, penguins.MASS_COVERAGE)
    assert result["reason"] == "FULL_REFERENCE_DISAGREEMENT"
    assert result["rows"] is None and result["sql"] is None
