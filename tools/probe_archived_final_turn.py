"""Re-run one frozen fourth decision after an exact archived three-action prefix.

Default mode verifies the control and freezes the fork without model credentials.
--execute allows a single metered call. --audit replays its saved response locally.
The source snapshot and receipts are historical local artifacts, not shipped inputs.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = ROOT / "reports/terminal-dev-v1"
PARENT = ROOT / "reports/query-ablation-dev-20261121"
OUTPUT = ROOT / "reports/matched-final-dispute01"
CONTROL_SHA = "4d4f96564bbfc50c2682f2808b56d5316006b24abf48c438d3a2bf4d207c4c3c"
SNAPSHOT_PROTOCOL_SHA = "aded34bc70d36659726649d05f0b44b1aa0c2cfd011965ca919403713379c8f3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(ok, error):
    if not ok:
        raise ValueError(error)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify_archive():
    snapshot = TERMINAL / "source"
    protocol_file = TERMINAL / "protocol.json"
    control = PARENT / "records/dispute-01--query_on.json"
    ensure(sha(protocol_file) == SNAPSHOT_PROTOCOL_SHA, "Terminal protocol changed")
    protocol = read(protocol_file)
    ensure(sha(control) == protocol["controls"][control.name] == CONTROL_SHA, "Control changed")
    for name, value in protocol["source_hashes"].items():
        ensure(sha(snapshot / name) == value, f"Snapshot changed: {name}")
    return snapshot, protocol, read(control)


class ArchivedPrefix:
    def __init__(self, archive, final, *, reserved):
        self.archive, self.final, self.reserved = archive, final, reserved
        self.requests = []

    async def generate(self, messages):
        from dataverify.agent.dispute_model import is_final_decision
        from dataverify.agent.graph_model import GraphModelError

        i = len(self.requests)
        ensure(i < 4, "Unexpected extra call")
        self.requests.append(deepcopy(messages))
        old = self.archive["requests"][i]
        if i != 3 or not self.reserved:
            ensure(old == messages, f"Original full prompt mismatch before call {i + 1}")
            if i == 0:
                reason = self.archive["http_events"][0]["action_error_code"]
                ensure(reason == "DISPUTE_INVALID_ACTION", "First parser outcome changed")
                raise GraphModelError(reason, retryable=True)
            return deepcopy(self.archive["record"]["actions"][i - 1])

        ensure(is_final_decision(messages), "Final decision prompt missing")
        original_ctx, new_ctx = json.loads(old[1]["content"]), json.loads(messages[1]["content"])
        ensure(
            original_ctx["remaining"]["actions_including_this_one"] == 1, "Not the same final slot"
        )
        expected_ctx = deepcopy(original_ctx)
        expected_ctx["remaining"].update(queries=0, revisions=0, tools_enabled=False)
        ensure(expected_ctx == new_ctx, "Program, source, evidence or other fork context changed")
        return await self.final(messages)


def stable_observation(observation):
    return {k: v for k, v in observation.items() if k != "elapsed_ms"}


async def inspect_prefix(archive):
    from dataverify.agent.graph_model import GraphModelError
    from dataverify.evaluation import query_ablation as paired
    from dataverify.evaluation.dispute_recovery import study_cases
    from dataverify.runtime.reporting import digest_json

    case = next(x for x in study_cases() if x.case_id == "dispute-01")
    old = archive["record"]
    ensure(
        [x["action"] for x in old["actions"]] == ["revise", "query", "query"],
        "Invalid control tape",
    )
    baseline = ArchivedPrefix(archive, None, reserved=False)
    replay = await paired.run_arm(case, "query_on", baseline)
    for key in (
        "trace",
        "actions",
        "versions",
        "replays",
        "history",
        "attempts",
        "status",
        "reason",
        "publication",
    ):
        ensure(old[key] == replay[key], f"Control replay mismatch: {key}")
    ensure(
        [stable_observation(a) for a in old["query_observations"]]
        == [stable_observation(a) for a in replay["query_observations"]],
        "Original execution changed",
    )

    async def stop_at_fork(_messages):
        raise GraphModelError("PROBE_BEFORE_NETWORK")

    witness = ArchivedPrefix(archive, stop_at_fork, reserved=True)
    check = await paired.run_arm(case, "query_on", witness, reserve_final_decision=True)
    ensure(
        check["reason"] == "PROBE_BEFORE_NETWORK" and len(witness.requests) == 4,
        "Fork is unreachable",
    )
    ensure(
        check["actions"] == old["actions"][:2]
        and check["replays"] == old["replays"]
        and check["trace"][:-1] == old["trace"][:5],
        "Fork state differs",
    )
    return case, {
        "format": "single-fixed-historical-fourth-decision-v1",
        "case": case.case_id,
        "archived_record_sha256": CONTROL_SHA,
        "software_protocol_sha256": SNAPSHOT_PROTOCOL_SHA,
        "probe_sha256": sha(Path(__file__)),
        "prefix_prompt_digests": [digest_json(p) for p in witness.requests[:3]],
        "old_final_prompt": digest_json(baseline.requests[-1]),
        "new_final_prompt": digest_json(witness.requests[-1]),
        "archived_events": [event["event_id"] for event in archive["http_events"]],
        "prefix_state": digest_json(
            [
                check["replays"],
                check["versions"],
                check["trace"][:-1],
                [stable_observation(o) for o in check["query_observations"]],
            ]
        ),
        "max_new_requests": 1,
        "grading": "Compare recorded fourth action to one new constrained draw at identical state; use existing independent controlled scorer. No estimate of average effect.",
    }


async def run(mode):
    snapshot, protocol, archive = verify_archive()
    sys.path[:0] = [str(snapshot), str(snapshot / "tools")]
    from evaluate_evidence_recovery import accounted, choose_lane
    from evaluate_structured_confirmation import immutable, load_driver

    from dataverify.agent.dispute_model import QwenDisputeModel
    from dataverify.agent.graph_checkpoint import exclusive_run
    from dataverify.evaluation import evidence_recovery as bounded
    from dataverify.evaluation import query_ablation as paired
    from dataverify.evaluation.expanded_experiment import _events
    from dataverify.evaluation.structured_confirmation import begin, cached
    from dataverify.integrations.qwen_config import QwenConfig
    from dataverify.runtime.probe_cost_budget import validate_pricing
    from dataverify.runtime.reporting import summarize_usage, write_json

    case, binding = await inspect_prefix(archive)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    immutable(OUTPUT / "protocol.json", binding)
    record_path = OUTPUT / "result.json"
    if mode == "prepare":
        print("All three archived prompts match; fourth-slot fork frozen.")
        return
    if mode == "execute" and not record_path.exists():
        config = QwenConfig.from_local_environment()
        validate_pricing(config)
        ensure(config.model == protocol["model"], "Live model changed")
        main = ROOT.parent / "github-publish-v1-20261107"
        campaign = main / "reports/expanded-v1-20261107"
        ensure(
            (campaign / "budget120/authorization.json").is_file(), "Original authorization absent"
        )
        driver = load_driver(main / "tools/continue_expanded.py", protocol["driver"])
        with ExitStack() as locks:
            locks.enter_context(exclusive_run(campaign / "allocation"))
            for i in range(4):
                locks.enter_context(exclusive_run(campaign / f"lane-{i}-execution"))
            ensure(not record_path.exists(), "Attempt already started")
            directory, receipt = driver.allocate_extension(
                campaign, campaign / "runs/main-v1", approved_limit=120
            )
            _, lane = choose_lane(driver, directory, receipt, 1)
            begin(record_path, binding)
            client = QwenDisputeModel(
                config, bounded.BoundedCellBudget(lane, 1), max_tokens=bounded.OUTPUT_TOKEN_CAP
            )
            try:
                tape = ArchivedPrefix(archive, client.generate, reserved=True)
                end = await paired.run_arm(case, "query_on", tape, reserve_final_decision=True)
            finally:
                await client.aclose()
            events = _events([("reserved_fourth", client)], lane)
            ensure(
                len(events) == 1 and accounted({"http_events": events}, lane),
                "Unresolved accounting",
            )
            ensure(events[0]["event_id"] not in binding["archived_events"], "Event ID reused")
            ensure(
                len(client.requests) == 1 and client.requests == tape.requests[3:],
                "Prefix sent to provider",
            )
            write_json(
                record_path,
                {
                    "completed": True,
                    "binding": binding,
                    "record": end,
                    "score": paired.grade(case, end),
                    "event": events[0],
                    "usage": summarize_usage(events),
                    "messages": client.requests[0],
                },
            )
    result = cached(record_path, binding)
    ensure(result is not None, "Missing live result")
    ensure(summarize_usage([result["event"]]) == result["usage"], "Usage drift")

    async def recorded(messages):
        from dataverify.agent.graph_model import GraphModelError

        ensure(messages == result["messages"], "Paid final prompt differs")
        if result["event"].get("action_error_code"):
            raise GraphModelError(result["event"]["action_error_code"])
        ensure(len(result["record"]["actions"]) == 3, "Missing fourth response")
        return deepcopy(result["record"]["actions"][-1])

    reproduction = await paired.run_arm(
        case,
        "query_on",
        ArchivedPrefix(archive, recorded, reserved=True),
        reserve_final_decision=True,
    )
    for key in ("actions", "status", "reason", "versions", "replays", "attempts", "publication"):
        ensure(reproduction[key] == result["record"][key], f"Final replay differs: {key}")
    score = paired.grade(case, reproduction)
    ensure(score == result["score"], "Offline scoring mismatch")
    print(
        json.dumps(
            {
                "case": case.case_id,
                "before": archive["record"]["status"],
                "after": reproduction["status"],
                "final_action": reproduction["actions"][-1]["action"]
                if len(reproduction["actions"]) == 3
                else "invalid",
                "correct_delivery": score["correct_delivery"],
                "joint_correct": score["joint_correct_delivery"],
                "usage": result["usage"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "execute", "audit"], nargs="?", default="prepare"
    )
    asyncio.run(run(parser.parse_args().mode))
