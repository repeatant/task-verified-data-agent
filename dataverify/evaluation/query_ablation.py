"""Paired, controlled development experiment with a single tool toggle.

The six inherited fault trajectories are development inputs (including a shared
error control). A seeded synthetic source is identical across arms. No hand-made
diagnostic examples or hidden grading datasets are passed to the model.
"""

from __future__ import annotations

import random
import time
from dataclasses import replace
from datetime import date, timedelta

from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
from dataverify.agent.dispute_model import QwenDisputeModel
from dataverify.agent.graph_model import GraphModelError, QwenGraphModel
from dataverify.evaluation import dispute_recovery as previous
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation.expanded_experiment import _events
from dataverify.evaluation.expanded_scoring import score_controlled
from dataverify.evaluation.structured_confirmation import begin, cached
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.probe_runtime import ProbeData

METHODS = ("self_review", "query_off", "query_on")
SOURCE_SEED = 20261121


def full_source(preview):
    """Extend the original one-row preview uniformly without access to fault labels."""
    rng = random.Random(SOURCE_SEED)
    orders = preview.orders + tuple(
        (i, rng.randint(10, 800), rng.choice(("web", "store", None))) for i in range(2, 18)
    )
    origin = date(2026, 8, 10)
    lines = preview.lines + tuple(
        (
            i,
            rng.randint(1, len(orders)),
            rng.randint(-15, 55),
            rng.choice(("settled", "settled", "pending", "cancelled")),
            (origin + timedelta(days=rng.randrange(95))).isoformat() + "T00:00:00",
        )
        for i in range(2, 121)
    )
    return ProbeData(orders, lines)


def frozen_cases():
    cases = previous.study_cases()
    return [
        {
            "id": case.case_id,
            "parent": case.base.parent_id,
            "requirement": case.base.task.task_id,
            "initial_sql": digest_json(case.initial_sql),
            "reference": digest_json(case.reference_plan),
            "auditor": digest_json(case.auditor_plan),
            "question": digest_json(case.base.task.public_input()),
            "preview": digest_json(case.base.source.to_dict()),
            "full_source": digest_json(full_source(case.base.source).to_dict()),
            "grading": previous.case_binding(case)["scoring_suite_digest"],
        }
        for case in cases
    ]


def policy_for(method):
    if method not in ("query_on", "query_off"):
        raise ValueError("Unknown dispute method")
    return DisputePolicy(
        tools_enabled=method == "query_on",
        max_actions=previous.MAX_ACTIONS,
        max_queries=previous.MAX_QUERIES,
        max_revisions=previous.MAX_REVISIONS,
        max_context_chars=bounded.INPUT_CHAR_CAP,
    )


async def run_arm(case, method, model, *, reserve_final_decision=False):
    """Run a single arm without exposing grading inputs to the model."""
    source = full_source(case.base.source)
    if method == "self_review":
        return await bounded.self_review(replace(case.base, source=source), case.initial_sql, model)
    policy = policy_for(method)
    if reserve_final_decision:
        policy = replace(policy, reserve_final_decision=True)
    return await run_dispute_graph(
        case.base.task,
        case.base.source,
        case.initial_sql,
        case.reference_plan,
        case.auditor_plan,
        model,
        query_source=source,
        policy=policy,
    )


def grade(case, record):
    from dataverify.evaluation.expanded_scoring import _outcomes
    from dataverify.verification.discovery_runtime import execute_readonly, rows_equal

    # Retain all original 25 grading fixtures and also verify the rows served to the user.
    score = score_controlled(case.base, record)
    pub = record.get("publication", {})
    correct_rows = False
    if pub.get("sql"):
        try:
            correct_rows = rows_equal(
                execute_readonly(pub["sql"], full_source(case.base.source)), pub.get("rows")
            )
        except (ValueError, TypeError):
            pass
    score.update(
        _outcomes(
            record,
            score["initial"],
            score["candidate"],
            score["publication"],
            publication_rows_consistent=correct_rows,
        )
    )
    if score["joint_evaluable"]:
        score["joint_correct_delivery"] = (
            score["correct_delivery"]
            and score["reference"]["correct"]
            and score["auditor"]["correct"]
        )
    return score


async def run_cell(
    case,
    method,
    config,
    lane,
    output,
    protocol_id,
    *,
    model_type=None,
    reserve_final_decision=False,
    grader=grade,
):
    path = output / "records" / f"{case.case_id}--{method}.json"
    binding = {
        "protocol": protocol_id,
        "case": case.case_id,
        "method": method,
        "model": config.model,
    }
    if reserve_final_decision:
        binding["reserve_final_decision"] = True
    old = cached(path, binding)
    if old is not None:
        return old
    begin(path, binding)
    started = time.perf_counter()
    calls = 1 if method == "self_review" else previous.MAX_ACTIONS
    budget = bounded.BoundedCellBudget(lane, calls)
    cls = model_type or (QwenGraphModel if method == "self_review" else QwenDisputeModel)
    model = cls(config, budget, max_tokens=bounded.OUTPUT_TOKEN_CAP)
    try:
        record = await run_arm(case, method, model, reserve_final_decision=reserve_final_decision)
    except (ValueError, GraphModelError) as exc:
        record = {
            "status": "inconclusive",
            "reason": str(exc)[:160],
            "candidate": {"sql": case.initial_sql},
            "publication": {},
        }
        if method != "self_review":
            record.update(plan=case.reference_plan, audit_plan=case.auditor_plan)
    finally:
        await model.aclose()
    events = _events([(method, model)], lane)
    result = {
        "completed": True,
        "binding": binding,
        "record": record,
        "score": grader(case, record),
        "http_events": events,
        "usage": summarize_usage(events),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "requests": model.requests,
    }
    write_json(path, result)
    return result


def summarize(output, protocol):
    protocol_id = digest_json(protocol)
    totals = {}
    rows = []
    for method in METHODS:
        selected = []
        for item in protocol["cases"]:
            key = {
                "protocol": protocol_id,
                "case": item["id"],
                "method": method,
                "model": protocol["model"],
            }
            result = cached(output / "records" / f"{item['id']}--{method}.json", key)
            if result is None:
                continue
            score = result["score"]
            row = {
                "case": item["id"],
                "method": method,
                "status": result["record"]["status"],
                "correct": score["correct_delivery"],
                "wrong": score["wrong_release"],
                "undelivered": not score["delivered"],
                "repaired": score["initial"]["correct"] is False and score["final_sql_correct"],
                "damaged": score["correct_sql_damaged"],
                "correct_rejected": score["correct_task_rejected"],
                "queries": result["record"].get("query_calls", 0),
                "elapsed_seconds": result["elapsed_seconds"],
            }
            rows.append(row)
            selected.append((row, result))
        totals[method] = {
            "completed": len(selected),
            **{
                k: sum(r[k] for r, _ in selected)
                for k in (
                    "correct",
                    "wrong",
                    "undelivered",
                    "repaired",
                    "damaged",
                    "correct_rejected",
                    "queries",
                    "elapsed_seconds",
                )
            },
            "usage": summarize_usage([e for _, result in selected for e in result["http_events"]]),
        }
    comparisons = []
    for item in protocol["cases"]:
        arms = {r["method"]: r["correct"] for r in rows if r["case"] == item["id"]}
        if set(arms) == set(METHODS):
            comparisons.append(
                {"case": item["id"], "on_minus_off": int(arms["query_on"]) - int(arms["query_off"])}
            )
    report = {
        "protocol": protocol_id,
        "completed_cells": len(rows),
        "paired_cases": comparisons,
        "totals": totals,
        "rows": rows,
    }
    write_json(output / "summary.json", report)
    return report
