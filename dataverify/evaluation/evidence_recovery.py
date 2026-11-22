"""Opt-in paired recovery experiment; scoring truth never enters model context.

The experiment distinguishes changed-SQL repair from releasing an unchanged SQL.
All methods see the same initial SQL and original-database execution. Only the
evidence method can choose extra read-only queries; no_tools is its ablation.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from dataverify.agent.evidence_graph import EvidencePolicy, run_evidence_graph
from dataverify.agent.evidence_model import QwenEvidenceModel
from dataverify.agent.graph_model import GraphModelError, QwenGraphModel
from dataverify.evaluation.expanded_experiment import SQL_PROMPT, _events, message, public_bundles
from dataverify.evaluation.expanded_scoring import score_public
from dataverify.evaluation.public_benchmark import file_digest, public_input
from dataverify.evaluation.structured_confirmation import begin, cached
from dataverify.runtime.campaign_budget import CellBudget
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.relational_data import SQLiteSource
from dataverify.verification.sql_tools import SQLTools

METHODS = ("self_review", "no_tools", "evidence")
SEED = 20261107
INPUT_CHAR_CAP = 11_000
OUTPUT_TOKEN_CAP = 1800
ACTION_CAP = 3
DEVELOPMENT_COUNT = 6
HOLDOUT_COUNT = 20


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_binding(bundle):
    """Freeze original database content as well as public task/source metadata."""
    source = bundle.source
    if isinstance(source, SQLiteSource):
        path = Path(source.path)
        wal = Path(str(path) + "-wal")
        identity = {
            "metadata": SQLTools(source).metadata(),
            "database_sha256": file_digest(path),
            "wal_sha256": file_digest(wal) if wal.exists() else None,
        }
    else:
        identity = source.to_dict()
    return {
        "task_digest": digest_json(bundle.task.public_input()),
        "source_digest": digest_json(identity),
    }


def select_question_ids(manifest, previous_confirmation):
    """Use only prior membership, static eligibility and database names.

    The pre-existing support filter was based on declared SQL/runtime limits.
    This sampler does not inspect gold SQL, answers, questions or model outcomes.
    """
    used = set(manifest["test_ids"] + manifest["development_ids"])
    used.update(previous_confirmation["test_ids"])
    pools = defaultdict(list)
    for entry in manifest["eligibility"]:
        if not entry["reasons"] and entry["question_id"] not in used:
            pools[entry["db_id"]].append(entry["question_id"])
    rng = random.Random(SEED)
    for database in sorted(pools):
        pools[database].sort()
        rng.shuffle(pools[database])
    selected = []
    while len(selected) < HOLDOUT_COUNT and any(pools.values()):
        for database in sorted(pools):
            if pools[database] and len(selected) < HOLDOUT_COUNT:
                selected.append(pools[database].pop())
    if len(selected) != HOLDOUT_COUNT:
        raise ValueError("RECOVERY_INSUFFICIENT_UNSEEN_QUESTIONS")
    return {
        "seed": SEED,
        "development_ids": manifest["development_ids"][:DEVELOPMENT_COUNT],
        "holdout_ids": selected,
        "selection": "seeded database round-robin over unused supported questions",
        "scope": "new questions in existing databases, not unseen-database generalization",
    }


def make_manifest(original, annotations_path, question_ids):
    """Create scoring records separately from the explicit public-input allowlist."""
    if file_digest(annotations_path) != original["annotations_sha256"]:
        raise ValueError("RECOVERY_PUBLIC_ANNOTATIONS_CHANGED")
    selected = set(question_ids)
    inputs = {}
    for line in Path(annotations_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["question_id"] in selected:
            inputs[str(row["question_id"])] = public_input(row, original["databases"][row["db_id"]])
    if set(inputs) != {str(i) for i in selected}:
        raise ValueError("RECOVERY_PUBLIC_INPUT_MISSING")
    manifest = deepcopy(original)
    manifest.update(test_ids=list(question_ids), development_ids=[], model_inputs=inputs)
    manifest["eligibility"] = [
        entry for entry in manifest["eligibility"] if entry["question_id"] in selected
    ]
    return manifest


def recovery_bundles(manifest):
    """Bind all arms to the same original database, rather than a fixture preview."""
    return [
        replace(
            bundle,
            source=SQLiteSource.from_path(Path(bundle.truth["db_path"])),
            task=replace(
                bundle.task,
                data_constraints=(
                    "Tools query the original read-only SQLite database under the same "
                    "limits for every method. Query results may be bounded; do not infer "
                    "business correctness from successful execution. Output order is not "
                    "used by the primary execution metric."
                ),
            ),
        )
        for bundle in public_bundles(manifest, "test")
    ]


class BoundedCellBudget(CellBudget):
    """Refuse oversized prompts before payment; never refund historical reserves."""

    def reserve(self, chars):
        if chars > INPUT_CHAR_CAP:
            raise GraphModelError("RECOVERY_INPUT_CHAR_CAP")
        return super().reserve(chars)


def max_reservation(calls):
    return calls * ((4 * INPUT_CHAR_CAP + 4096) * 0.8 + OUTPUT_TOKEN_CAP * 2.7) / 1_000_000


def forecast():
    dev_calls = DEVELOPMENT_COUNT * (1 + 2 * ACTION_CAP)
    holdout_calls = HOLDOUT_COUNT * (2 + 2 * ACTION_CAP)
    return {
        "development_max_calls": dev_calls,
        "holdout_max_calls": holdout_calls,
        "total_max_calls": dev_calls + holdout_calls,
        "development_max_reserved_cny": max_reservation(dev_calls),
        "holdout_max_reserved_cny": max_reservation(holdout_calls),
        "total_max_reserved_cny": max_reservation(dev_calls + holdout_calls),
        "basis": "cached development SQL; unchanged 1800-token conservative reservation",
        "note": "No promise to execute holdout; lane distribution and quality gate still apply.",
    }


def imported_initial(bundle, old_path, output_path):
    """Reuse a previously paid development SQL without carrying its old reference."""
    old = read_json(old_path)
    binding = {
        **task_binding(bundle),
        "stage": "imported_development_sql",
        "source_path": str(Path(old_path).resolve()),
        "source_sha256": file_digest(old_path),
    }
    previous = cached(output_path, binding)
    if previous is not None:
        return previous
    if not isinstance(old.get("sql"), str) or not old["sql"].strip():
        raise ValueError("RECOVERY_DEVELOPMENT_SQL_MISSING")
    result = {
        "binding": binding,
        "completed": True,
        "sql": old["sql"],
        "http_events": [],
        "historical_sql_events": [
            e for e in old.get("http_events", []) if e.get("role") == "initial_writer"
        ],
        "note": "Historical SQL reused identically; no old reference, review or gold enters context.",
    }
    write_json(output_path, result)
    return result


async def shared_initial(bundle, config, lane, path):
    binding = {**task_binding(bundle), "stage": "shared_sql", "model": config.model}
    previous = cached(path, binding)
    if previous is not None:
        return previous
    begin(path, binding)
    start = lane.snapshot()
    model = QwenGraphModel(config, BoundedCellBudget(lane, 1), max_tokens=OUTPUT_TOKEN_CAP)
    result = {"binding": binding, "sql": None}
    try:
        action = await model.generate(
            [message("system", SQL_PROMPT), message("user", bundle.task.public_input())]
        )
        if action["task_id"] != bundle.task.task_id:
            raise ValueError("TASK_ID_MISMATCH")
        result["sql"] = action["sql"]
    except (GraphModelError, ValueError) as error:
        result["error"] = str(error)[:160]
    finally:
        await model.aclose()
    result.update(
        completed=True,
        http_events=_events([("shared_writer", model)], lane),
        contexts=model.requests,
        budget_start=start,
        budget_end=lane.snapshot(),
    )
    write_json(path, result)
    return result


async def self_review(bundle, sql, model):
    """One review receives full-DB execution evidence under the shared executor."""
    tools = SQLTools(bundle.source)
    before = tools.execute(sql, observation_id="e0001", purpose="initial_candidate")
    # This is an explicitly bounded context view, not a second full-data result.
    from dataverify.agent.evidence_graph import _prompt_observation

    action = await model.generate(
        [
            message("system", SQL_PROMPT),
            message(
                "user",
                {
                    "task": bundle.task.public_input(),
                    "candidate": {"sql": sql},
                    "execution": _prompt_observation(before),
                    "instruction": "Review the original task and this execution. Fix SQL if needed; otherwise preserve it. Successful execution alone does not establish correctness.",
                },
            ),
        ]
    )
    if action["task_id"] != bundle.task.task_id:
        raise ValueError("TASK_ID_MISMATCH")
    final = tools.execute(action["sql"], observation_id="e0002", purpose="reviewed_candidate")
    deliver = final["status"] == "ok" and final["complete"]
    return {
        "status": "reviewed" if deliver else "inconclusive",
        "candidate": {"sql": action["sql"]},
        "publication": {"sql": action["sql"], "rows": final["rows"]} if deliver else {},
        "observations": [before, final],
        "model_calls": 1,
        "query_calls": 0,
    }


async def paired_cell(bundle, method, common, config, lane, output, protocol_digest):
    if method not in METHODS:
        raise ValueError("RECOVERY_METHOD")
    path = output / "records" / f"{bundle.case_id}--{method}.json"
    binding = {
        **task_binding(bundle),
        "method": method,
        "model": config.model,
        "shared_sql_digest": digest_json(common),
        "protocol_digest": protocol_digest,
    }
    previous = cached(path, binding)
    if previous is not None:
        return previous
    begin(path, binding)
    start, started = lane.snapshot(), time.perf_counter()
    limit = 1 if method == "self_review" else ACTION_CAP
    cls = QwenGraphModel if method == "self_review" else QwenEvidenceModel
    model = cls(config, BoundedCellBudget(lane, limit), max_tokens=OUTPUT_TOKEN_CAP)
    record = {"status": "initial_sql_invalid", "candidate": {}, "publication": {}}
    try:
        if common["sql"]:
            if method == "self_review":
                record = await self_review(bundle, common["sql"], model)
            else:
                record = await run_evidence_graph(
                    bundle.task,
                    bundle.source,
                    common["sql"],
                    model,
                    policy=EvidencePolicy(
                        tools_enabled=method == "evidence",
                        max_actions=ACTION_CAP,
                        max_queries=2,
                        max_revisions=1,
                    ),
                )
    except (GraphModelError, ValueError) as error:
        record = {
            "status": "inconclusive",
            "candidate": {"sql": common["sql"]},
            "publication": {},
            "error": str(error)[:160],
        }
    finally:
        await model.aclose()
    events = _events([(method, model)], lane)
    score = score_public({**bundle.truth, "initial_sql": common["sql"]}, record)
    changed = bool(record.get("candidate", {}).get("sql")) and (
        record["candidate"]["sql"].strip() != (common["sql"] or "").strip()
    )
    score.update(
        sql_text_changed=changed,
        repaired_initial_error=(
            score["initial"]["correct"] is False and score["final_sql_correct"] and changed
        ),
        preserved_correct_sql=score["initial_sql_correct"] and score["final_sql_correct"],
        unchanged_delivery=score["delivered"] and not changed,
    )
    row = {
        "case_id": bundle.case_id,
        "family": bundle.family,
        "method": method,
        "status": record["status"],
        "score": score,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "query_calls": record.get("query_calls", 0),
        "usage": summarize_usage(events),
        "http_events": events,
    }
    result = {
        "binding": binding,
        "completed": True,
        "row": row,
        "record": record,
        "contexts": model.requests,
        "http_events": events,
        "budget_start": start,
        "budget_end": lane.snapshot(),
    }
    write_json(path, result)
    return result


def summarize(output, protocol):
    records = [read_json(path) for path in (output / "records").glob("*.json")]
    rows = [r["row"] for r in records if r.get("completed") and "row" in r]
    keys = {
        "initial_sql_correct",
        "final_sql_correct",
        "correct_sql_damaged",
        "delivered",
        "correct_delivery",
        "wrong_release",
        "correct_task_rejected",
        "sql_text_changed",
        "repaired_initial_error",
        "preserved_correct_sql",
        "unchanged_delivery",
    }
    methods = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        methods[method] = {
            "tasks": len(selected),
            **{key: sum(row["score"].get(key) is True for row in selected) for key in keys},
            "query_calls": sum(row["query_calls"] for row in selected),
            "usage": summarize_usage([e for row in selected for e in row["http_events"]]),
        }
    initial = [read_json(path) for path in (output / "initial").glob("*.json")]
    all_events = [e for row in rows for e in row["http_events"]]
    all_events += [e for record in initial for e in record.get("http_events", [])]
    observed = {(row["case_id"], row["method"]) for row in rows}
    missing = [
        [case["case_id"], method]
        for case in protocol["cases"]
        for method in METHODS
        if (case["case_id"], method) not in observed
    ]
    result = {
        "protocol_digest": digest_json(protocol),
        "phase": protocol["phase"],
        "complete": not missing,
        "completed_cells": len(rows),
        "expected_cells": len(protocol["cases"]) * len(METHODS),
        "missing": missing,
        "methods": methods,
        "usage_all_new_requests": summarize_usage(all_events),
        "rows": rows,
        "limitations": protocol["limitations"],
    }
    write_json(output / "summary.json", result)
    return result


def development_gate(summary):
    """A stop/go development rule, not a statistical generalization claim."""
    methods = summary["methods"]
    evidence = methods["evidence"]
    reasons = []
    if not summary["complete"] or summary["phase"] != "development":
        reasons.append("development_incomplete")
    if evidence["correct_sql_damaged"]:
        reasons.append("initially_correct_sql_damaged")
    if not evidence["repaired_initial_error"] or not evidence["query_calls"]:
        reasons.append("no_observed_tool_supported_sql_repair")
    if evidence["final_sql_correct"] <= max(
        methods[method]["final_sql_correct"] for method in ("self_review", "no_tools")
    ):
        reasons.append("no_candidate_accuracy_gain_over_both_comparators")
    if evidence["wrong_release"] > methods["self_review"]["wrong_release"]:
        reasons.append("wrong_release_increased")
    return {"passed": not reasons, "reasons": reasons, "rule": "development stop/go only"}
