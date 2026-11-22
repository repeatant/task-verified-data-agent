"""Offline counterfactual replay of saved model actions after a finish-ID fix.

No model client, credential loading, API request or budget mutation is performed.
Saved parsed responses are returned verbatim and in order, independent of new
prompts. Results explain a host-interface change; they are NOT new live-model
successes and must never be merged into the frozen real experiment's scores.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.dispute_graph import DisputePolicy, run_dispute_graph
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation import dispute_recovery as experiment
from dataverify.evaluation.evidence_recovery import read_json
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.runtime.reporting import digest_json, write_json

ALLOWED_CHANGED_SOURCE = {
    "dataverify/agent/dispute_graph.py",
    "dataverify/agent/dispute_model.py",
}


class SavedResponseModel:
    """A response tape, not a model: prompts cannot rewrite or add responses."""

    client_kind = "offline_saved_live_actions_counterfactual"

    def __init__(self, actions):
        self.actions = deepcopy(actions)
        self.requests = []
        self.returned_action_digests = []
        self.exhausted = False

    async def generate(self, messages):
        self.requests.append(deepcopy(messages))
        index = len(self.returned_action_digests)
        if index >= len(self.actions):
            self.exhausted = True
            raise GraphModelError("OFFLINE_SAVED_RESPONSES_EXHAUSTED")
        action = deepcopy(self.actions[index])
        self.returned_action_digests.append(digest_json(action))
        return action

    async def aclose(self):
        return None


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("OFFLINE_REPLAY_EXISTING_ARTIFACT_CHANGED")
    else:
        write_json(path, value)


def check_paths(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("OFFLINE_REPLAY_MUST_USE_SEPARATE_OUTPUT_DIRECTORY")


def saved_actions(original):
    """Only replay a complete one-to-one HTTP/context/parsed-action history."""
    actions = original["record"]["actions"]
    events = original["http_events"]
    contexts = original["contexts"]["dispute_resolver"]
    if (
        original.get("completed") is not True
        or len(actions) != len(events)
        or len(actions) != len(contexts)
        or len(actions) != original["record"]["model_calls"]
        or any(
            e.get("status") != "completed" or e.get("usage_complete") is not True for e in events
        )
        or any(e.get("action_error_code") for e in events)
    ):
        raise ValueError("OFFLINE_REPLAY_INCOMPLETE_OR_UNPARSED_RESPONSE_HISTORY")
    return deepcopy(actions)


def _programs(record):
    return {
        "candidate": record.get("candidate", {}).get("sql"),
        "reference": record.get("plan", {}).get("reference"),
        "auditor": record.get("audit_plan", {}).get("reference"),
    }


def _context_binding(messages):
    context = json.loads(messages[-1]["content"])
    return {
        "task": context["task"],
        "programs": context["programs"],
        "versions": context["versions"],
        "latest_replay_id": context["latest_replay"]["id"],
        "query_source": context["query_source"],
    }


def compare_query_observations(original, replayed):
    """Reexecuted queries must match identity, source, outputs and limits.

    Wall-clock elapsed_ms may change; it is explicitly the only omitted field.
    Version bindings before each response are checked separately against saved
    contexts, so a stable query identifier alone is not accepted as evidence.
    """
    previous = {item["id"]: item for item in original}
    checks = []
    for actual in replayed:
        if actual["id"] not in previous:
            raise ValueError("OFFLINE_REPLAY_NEW_QUERY_OBSERVATION")
        before = previous[actual["id"]]
        left = {key: value for key, value in before.items() if key != "elapsed_ms"}
        right = {key: value for key, value in actual.items() if key != "elapsed_ms"}
        if left != right:
            raise ValueError("OFFLINE_REPLAY_QUERY_EVIDENCE_CHANGED")
        checks.append({"id": actual["id"], "matched": True, "content_digest": digest_json(right)})
    return {
        "checks": checks,
        "original_query_count": len(original),
        "reexecuted_query_count": len(replayed),
        "original_query_ids_not_reexecuted": sorted(
            set(previous) - {item["id"] for item in replayed}
        ),
        "excluded_comparison_fields": ["elapsed_ms"],
    }


async def replay_one(case, original, source_path, output, protocol_digest):
    path = output / "records" / (case.case_id + ".json")
    binding = {
        "protocol_digest": protocol_digest,
        "original_record_path": str(source_path.resolve()),
        "original_record_sha256": file_digest(source_path),
        **experiment.case_binding(case),
    }
    if path.exists():
        previous = read_json(path)
        if previous.get("binding") != binding or previous.get("completed") is not True:
            raise ValueError("OFFLINE_REPLAY_CHANGED_OR_INCOMPLETE_RECORD")
        return previous
    actions = saved_actions(original)
    old_grade = experiment.score_result(case, "dispute_resolver", original["record"])
    if old_grade != original["row"]["score"]:
        raise ValueError("OFFLINE_REPLAY_ORIGINAL_GRADE_CHANGED")
    write_json(path, {"binding": binding, "completed": False})
    model = SavedResponseModel(actions)
    record = await run_dispute_graph(
        case.base.task,
        case.base.source,
        case.initial_sql,
        case.reference_plan,
        case.auditor_plan,
        model,
        query_source=experiment.visible_diagnostic_fixture(),
        fixtures=(
            {
                "name": "shared_diagnostic",
                "data": experiment.visible_diagnostic_fixture().to_dict(),
            },
        ),
        confirmed_examples=(),
        policy=DisputePolicy(**original["record"]["policy"]),
    )
    await model.aclose()
    consumed = len(model.returned_action_digests)
    expected_digests = [digest_json(action) for action in actions[:consumed]]
    if model.returned_action_digests != expected_digests or record["actions"] != actions[:consumed]:
        raise ValueError("OFFLINE_REPLAY_SAVED_ACTIONS_WERE_CHANGED")
    # The revised system prompt may differ; actual task, program versions and
    # source bindings before every returned action must stay exactly comparable.
    old_contexts = original["contexts"]["dispute_resolver"]
    context_checks = []
    for index in range(consumed):
        previous = _context_binding(old_contexts[index])
        actual = _context_binding(model.requests[index])
        if previous != actual:
            raise ValueError("OFFLINE_REPLAY_PRE_ACTION_PROGRAM_OR_SOURCE_CHANGED")
        context_checks.append(
            {"response_index": index, "matched": True, "binding_digest": digest_json(actual)}
        )
    query_checks = compare_query_observations(
        original["record"]["query_observations"], record["query_observations"]
    )
    # This six-case tape has no post-termination query response. Do not silently
    # call an unexecuted original query "reverified" in a future replay dataset.
    old_programs, new_programs = _programs(original["record"]), _programs(record)
    new_grade = experiment.score_result(case, "dispute_resolver", record)
    result = {
        "binding": binding,
        "completed": True,
        "kind": "offline_counterfactual_saved_response_replay",
        "count_as_new_live_model_success": False,
        "new_api_calls": 0,
        "new_model_tokens": 0,
        "new_budget_reservation_cny": 0,
        "case_id": case.case_id,
        "condition": case.condition,
        "task_id": case.base.task.task_id,
        "saved_original_response_count": len(actions),
        "consumed_original_response_count": consumed,
        "unused_saved_response_count": len(actions) - consumed,
        "requested_offline_turns": len(model.requests),
        "response_tape_exhausted": model.exhausted,
        "original": {
            "status": original["record"]["status"],
            "score": old_grade,
            "programs": old_programs,
        },
        "counterfactual": {
            "status": record["status"],
            "score": new_grade,
            "programs": new_programs,
        },
        "program_changes_from_original_final": {
            role: {
                "changed": old_programs[role] != new_programs[role],
                "before_digest": digest_json(old_programs[role]),
                "after_digest": digest_json(new_programs[role]),
            }
            for role in old_programs
        },
        "pre_action_binding_checks": context_checks,
        "query_evidence_reexecution": query_checks,
        "saved_actions_digest": digest_json(actions),
        "returned_action_digests": model.returned_action_digests,
        "record": record,
        "offline_contexts": model.requests,
    }
    write_json(path, result)
    return result


def build_protocol(source):
    original = read_json(source / "protocol.json")
    current_hashes = {
        relative: file_digest(ROOT / relative) for relative in original["source_hashes"]
    }
    changed = {
        relative
        for relative, digest in current_hashes.items()
        if digest != original["source_hashes"][relative]
    }
    if not changed <= ALLOWED_CHANGED_SOURCE:
        raise ValueError("OFFLINE_REPLAY_UNEXPECTED_RUNTIME_OR_SCORER_CHANGE")
    cases = experiment.study_cases()
    old_cases = {case["case_id"]: case for case in original["cases"]}
    files = {"protocol.json": file_digest(source / "protocol.json")}
    for case in cases:
        for key, value in experiment.case_binding(case).items():
            if old_cases[case.case_id][key] != value:
                raise ValueError("OFFLINE_REPLAY_FROZEN_INPUT_CHANGED")
        relative = "records/" + case.case_id + "--dispute_resolver.json"
        record = read_json(source / relative)
        if record["binding"]["protocol_digest"] != digest_json(original):
            raise ValueError("OFFLINE_REPLAY_ORIGINAL_PROTOCOL_BINDING")
        saved_actions(record)
        files[relative] = file_digest(source / relative)
    return {
        "format": "dataverify-finish-id-offline-counterfactual-v1",
        "kind": "offline_counterfactual_not_new_live_experiment",
        "source_directory": str(source.resolve()),
        "original_source_files": files,
        "original_protocol_digest": digest_json(original),
        "current_runtime_hashes": current_hashes,
        "changed_runtime_paths": sorted(changed),
        "replay_tool_sha256": file_digest(Path(__file__)),
        "controlled_trajectories": len(cases),
        "distinct_requirements": len({case.base.task.task_id for case in cases}),
        "response_policy": "Return the exact saved parsed actions in original order. Never regenerate, normalize citations, append actions, or use the changed prompt to author a response.",
        "query_policy": "Execute saved query actions against the identical source; compare complete observations except elapsed_ms and verify program-version/source context before every action.",
        "live_model_successes_credited": 0,
        "new_api_calls": 0,
        "limitations": [
            "A host-policy counterfactual reuses past responses, including responses conditioned on old prompts and feedback.",
            "Earlier termination may leave saved responses unused. This is not measured token/cost savings from a new real-model run.",
            "An increase in deliverable outcomes cannot be claimed as a new autonomous-model success rate.",
            "The original frozen experiment and its paid-call accounting remain unchanged.",
        ],
    }


async def run(source, output, protocol):
    rows = []
    for case in experiment.study_cases():
        source_path = source / "records" / f"{case.case_id}--dispute_resolver.json"
        rows.append(
            await replay_one(
                case, read_json(source_path), source_path, output, digest_json(protocol)
            )
        )
    for relative, expected in protocol["original_source_files"].items():
        if file_digest(source / relative) != expected:
            raise ValueError("OFFLINE_REPLAY_ORIGINAL_ARTIFACT_CHANGED_DURING_RUN")
    outcome_keys = (
        "final_sql_correct",
        "correct_delivery",
        "wrong_release",
        "correct_task_rejected",
        "joint_correct_delivery",
        "wrong_reference_released",
        "wrong_auditor_released",
    )
    summary = {
        "kind": protocol["kind"],
        "protocol_digest": digest_json(protocol),
        "new_api_calls": 0,
        "new_tokens": 0,
        "new_budget_reservation_cny": 0,
        "live_model_successes_credited": 0,
        "controlled_trajectories": len(rows),
        "distinct_requirements": protocol["distinct_requirements"],
        "original_model_responses_available": sum(
            row["saved_original_response_count"] for row in rows
        ),
        "saved_responses_consumed_offline": sum(
            row["consumed_original_response_count"] for row in rows
        ),
        "original_and_counterfactual_counts": {
            key: {
                side: sum(row[side]["score"].get(key) is True for row in rows)
                for side in ("original", "counterfactual")
            }
            for key in outcome_keys
        },
        "programs_changed_from_saved_final": {
            role: sum(row["program_changes_from_original_final"][role]["changed"] for row in rows)
            for role in ("candidate", "reference", "auditor")
        },
        "queries_reexecuted_and_matched": sum(
            len(row["query_evidence_reexecution"]["checks"]) for row in rows
        ),
        "source_artifacts_unchanged": True,
        "cases": [
            {
                key: row[key]
                for key in (
                    "case_id",
                    "condition",
                    "saved_original_response_count",
                    "consumed_original_response_count",
                    "unused_saved_response_count",
                    "response_tape_exhausted",
                    "original",
                    "counterfactual",
                    "program_changes_from_original_final",
                    "query_evidence_reexecution",
                )
            }
            for row in rows
        ],
        "limitations": protocol["limitations"],
    }
    immutable(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/dispute-recovery-v1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/dispute-recovery-finish-replay"
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    check_paths(source, output)
    protocol = build_protocol(source)
    (output / "records").mkdir(parents=True, exist_ok=True)
    immutable(output / "protocol.json", protocol)
    summary = asyncio.run(run(source, output, protocol))
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "new_api_calls": 0,
                "original_responses": summary["original_model_responses_available"],
                "consumed_offline": summary["saved_responses_consumed_offline"],
                "counts": summary["original_and_counterfactual_counts"],
                "program_changes": summary["programs_changed_from_saved_final"],
                "queries_reexecuted": summary["queries_reexecuted_and_matched"],
                "source_artifacts_unchanged": summary["source_artifacts_unchanged"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
