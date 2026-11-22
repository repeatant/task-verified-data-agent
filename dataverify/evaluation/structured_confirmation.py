"""Paired format-only confirmation: shared SQL, independently generated references.

No gold enters model messages. All stages retain failures, bounded calls and the
same full verification policy. The driver owns existing budget leases and freeze.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import QwenReviewModel, plan_prompt, plan_validation_feedback
from dataverify.agent.graph_model import GraphModelError, InjectedFirstGraphModel, QwenGraphModel
from dataverify.evaluation.discovery_roles import InjectedFirstReviewModel
from dataverify.evaluation.expanded_experiment import SQL_PROMPT, _events, message
from dataverify.evaluation.expanded_scoring import score_public
from dataverify.runtime.campaign_budget import CellBudget
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.discovery_runtime import validate_plan

METHODS = ("json_object", "strict_schema")


def cached(path: Path, binding: dict):
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("binding") != binding or value.get("completed") is not True:
            raise ValueError("CONFIRMATION_INCOMPLETE_OR_CHANGED_RECORD_REQUIRES_REVIEW")
        return value
    return None


def begin(path: Path, binding: dict):
    # A crash after payment must not silently repeat the same stage.
    write_json(path, {"binding": binding, "completed": False})


def task_binding(bundle):
    return {
        "task_digest": digest_json(bundle.task.public_input()),
        "source_digest": digest_json(bundle.source.to_dict()),
    }


async def shared_sql(bundle, config, lane, path):
    binding = {**task_binding(bundle), "stage": "shared_sql", "model": config.model}
    previous = cached(path, binding)
    if previous is not None:
        return previous
    begin(path, binding)
    start = lane.snapshot()
    model = QwenGraphModel(config, CellBudget(lane, 1), max_tokens=1800)
    record = {"binding": binding, "sql": None}
    try:
        action = await model.generate(
            [message("system", SQL_PROMPT), message("user", bundle.task.public_input())]
        )
        if action["task_id"] != bundle.task.task_id:
            raise ValueError("TASK_ID_MISMATCH")
        record["sql"] = action["sql"]
    except (GraphModelError, ValueError) as error:
        record["error"] = str(error)[:160]
    finally:
        await model.aclose()
    record.update(
        completed=True,
        http_events=_events([("shared_writer", model)], lane),
        contexts=model.requests,
        budget_start=start,
        budget_end=lane.snapshot(),
    )
    record["usage"] = summarize_usage(record["http_events"])
    write_json(path, record)
    return record


async def initial_reference(bundle, method, config, lane, path):
    binding = {
        **task_binding(bundle),
        "stage": "reference",
        "method": method,
        "model": config.model,
    }
    previous = cached(path, binding)
    if previous is not None:
        return previous
    begin(path, binding)
    start = lane.snapshot()
    model = QwenReviewModel(
        config, CellBudget(lane, 3), max_tokens=1800, structured_output=method == "strict_schema"
    )
    record = {"binding": binding, "plan": None, "validation_failures": []}
    messages = [
        message("system", plan_prompt(bundle.task)),
        message("user", bundle.task.public_input()),
    ]
    try:
        for _ in range(3):
            action = None
            try:
                action = await model.generate(messages)
                record["plan"] = validate_plan(action, bundle.task)
                break
            except (GraphModelError, ValueError) as error:
                record["error"] = str(error)[:160]
                record["validation_failures"].append(str(error)[:160])
                if lane.snapshot()["stopped_reason"]:
                    break
                messages += [
                    message("assistant", action or {"error": str(error)}),
                    message("user", plan_validation_feedback(action, str(error), bundle.task)),
                ]
    finally:
        await model.aclose()
    record.update(
        completed=True,
        http_events=_events([("initial_reference", model)], lane),
        contexts=model.requests,
        budget_start=start,
        budget_end=lane.snapshot(),
    )
    record["usage"] = summarize_usage(record["http_events"])
    write_json(path, record)
    return record


async def paired_cell(bundle, method, common, config, lane, output):
    if method not in METHODS:
        raise ValueError("CONFIRMATION_METHOD")
    path = output / "records" / f"{bundle.case_id}--{method}.json"
    binding = {
        **task_binding(bundle),
        "method": method,
        "model": config.model,
        "shared_sql_digest": digest_json(common),
    }
    previous = cached(path, binding)
    if previous is not None:
        return previous
    checkpoint = output / "checkpoints" / f"{bundle.case_id}--{method}.sqlite3"
    if checkpoint.exists():
        raise ValueError("CONFIRMATION_UNRECORDED_CHECKPOINT_REQUIRES_REVIEW")
    begin(path, binding)
    start, started = lane.snapshot(), time.perf_counter()
    initial = {"plan": None, "http_events": []}
    models, events, contexts = [], [], {}
    if not common["sql"]:
        record = {"status": "initial_sql_invalid", "publication": {}}
    else:
        initial = await initial_reference(
            bundle, method, config, lane, output / "references" / f"{bundle.case_id}--{method}.json"
        )
        if not initial["plan"]:
            record = {
                "status": "initial_reference_invalid",
                "candidate": {"sql": common["sql"]},
                "publication": {},
            }
        else:
            budget = CellBudget(lane, 12)
            strict = method == "strict_schema"
            writer = InjectedFirstGraphModel(
                QwenGraphModel(config, budget, max_tokens=1800, structured_output=strict),
                {"sql": common["sql"], "task_id": bundle.task.task_id},
            )
            reviewer = InjectedFirstReviewModel(
                QwenReviewModel(config, budget, max_tokens=1800, structured_output=strict),
                initial["plan"],
            )
            auditor = QwenReviewModel(config, budget, max_tokens=1800, structured_output=strict)
            models = [("writer", writer), ("reviewer", reviewer), ("auditor", auditor)]
            try:
                record = await run_discovery(
                    bundle.task,
                    bundle.source,
                    writer,
                    reviewer,
                    auditor=auditor,
                    checkpoint_path=checkpoint,
                    thread_id="confirmation",
                    max_attempts=3,
                    max_reviews=3,
                    max_challenge_examples=40,
                )
            except Exception as error:
                record = {
                    "status": "runner_error",
                    "error_type": type(error).__name__,
                    "publication": {},
                }
            finally:
                for _, model in models:
                    await model.aclose()
            events = _events(models, lane)
            contexts = {role: model.requests for role, model in models}
    events = initial["http_events"] + events
    score = score_public({**bundle.truth, "initial_sql": common["sql"]}, record)
    row = {
        "case_id": bundle.case_id,
        "parent_id": bundle.parent_id,
        "cluster_id": bundle.cluster_id,
        "family": bundle.family,
        "condition": "natural",
        "kind": "public",
        "method": method,
        "status": record["status"],
        "score": score,
        "initial_reference_accepted": initial.get("plan") is not None,
        "reference_validation_failures": len(initial.get("validation_failures", [])),
        "reference_generation_calls": len(initial["http_events"]),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": summarize_usage(events),
        "http_events": events,
        "shared_initial_usage": summarize_usage(common["http_events"]),
        "shared_initial_http_events": common["http_events"],
    }
    result = {
        "binding": binding,
        "completed": True,
        "row": row,
        "record": record,
        "reference_digest": digest_json(initial),
        "contexts": contexts,
        "http_events": events,
        "budget_start": start,
        "budget_end": lane.snapshot(),
    }
    write_json(path, result)
    return result
