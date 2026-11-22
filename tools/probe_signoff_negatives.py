"""Four prespecified, schema-constrained real decisions on two anonymized faults."""

import argparse
import asyncio
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/split-signoff-v1"
SOURCE = PARENT / "source"
OUTPUT = ROOT / "reports/split-negative-v2"
sys.path[:0] = [str(SOURCE), str(SOURCE / "tools")]

from evaluate_split_signoff import (
    FAULTS,
    QwenConfig,
    QwenDisputeModel,
    _events,
    accounted,
    begin,
    bounded,
    cached,
    choose_lane,
    digest_json,
    exclusive_run,
    file_digest,
    immutable,
    load_driver,
    negative,
    paired,
    read,
    require,
    run_dispute_graph,
    summarize_usage,
    validate_pricing,
    write_json,
)

PROMPT = (
    "Assess eligibility for delivering this result. Return a schema-valid finish if "
    "the evidence supports delivery; otherwise return a schema-valid clarify or abstain "
    "and provide the business question or technical reason. Use only the defined fields."
)
ORDER = [("wrong-store", False), ("wrong-store", True), ("wrong-null", True), ("wrong-null", False)]
PARENT_HASH = "a4aec2738361eee626809ca4a4ed04034afe707218a852ed5cd4c5167f4047a3"


class Prefix:
    def __init__(self, final):
        self.final, self.prefix_digest, self.final_prompt = final, None, None

    async def generate(self, messages):
        require(all(name not in str(messages) for name in FAULTS), "Fault label leaked")
        if self.prefix_digest is None:
            self.prefix_digest = digest_json(json.loads(messages[1]["content"]))
            return {
                "action": "query",
                "sql": "SELECT COUNT(*) FROM orders",
                "hypothesis": "Establish the number of orders on the bound source.",
            }
        require(self.final_prompt is None, "More than one paid decision")
        ctx = json.loads(messages[1]["content"])
        ctx["decision_request"] = PROMPT
        self.final_prompt = [
            messages[0],
            {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)},
        ]
        return await self.final(self.final_prompt)


async def run_fork(name, split, final):
    case = FAULTS[name]
    task = replace(case.task, task_id="t_" + digest_json(case.question)[:12])
    model = Prefix(final)
    rec = await run_dispute_graph(
        task,
        case.preview,
        case.sql,
        case.plan(case.reference),
        case.plan(case.auditor),
        model,
        query_source=paired.full_source(case.preview),
        policy=replace(
            paired.policy_for("query_on"),
            max_actions=2,
            reserve_final_decision=True,
            split_finish_evidence=split,
        ),
    )
    return rec, model


async def protocol():
    parent = read(PARENT / "protocol.json")
    require(
        digest_json(parent) == PARENT_HASH
        and all(file_digest(SOURCE / k) == v for k, v in parent["sources"].items()),
        "Frozen parent mismatch",
    )
    forks = {}

    async def stop(_):
        from dataverify.agent.graph_model import GraphModelError

        raise GraphModelError("BEFORE_PAYMENT")

    for name, split in ORDER:
        rec, tape = await run_fork(name, split, stop)
        require(rec["reason"] == "BEFORE_PAYMENT", "Probe did not reach decision")
        state = {
            "context": digest_json(json.loads(tape.final_prompt[1]["content"])),
            "first_user": tape.prefix_digest,
            "replays": digest_json(rec["replays"]),
        }
        entry = forks.setdefault(name, {"state": state})
        require(entry["state"] == state, "Arms see different business evidence")
        entry["split" if split else "old"] = digest_json(tape.final_prompt)
    p = {
        "parent": PARENT_HASH,
        "driver": parent["driver"],
        "model": parent["model"],
        "probe": file_digest(Path(__file__)),
        "forks": forks,
        "order": [list(cell) for cell in ORDER],
        "max_new_calls": 4,
        "method": "Two anonymized wrong candidates, one authentic Qwen decision per arm with strict schema. Identical neutral decision request and fixed SQL investigation; score to independent data. No retries for invalid output; no gold or condition IDs are supplied.",
        "decision": "Any incorrect delivery precludes promotion. Abstention, valid finish rejected by guard, and invalid output classified separately. Forced well-formed finish is an offline control only.",
    }
    OUTPUT.mkdir(exist_ok=True)
    immutable(OUTPUT / "protocol.json", p)
    return p


async def campaign(mode):
    p = await protocol()
    if mode == "prepare":
        print(digest_json(p))
        return
    main = ROOT.parent / "github-publish-v1-20261107"
    acct = main / "reports/expanded-v1-20261107"
    rows, events = [], []
    with ExitStack() as locks:
        if mode == "execute":
            cfg = QwenConfig.from_local_environment()
            validate_pricing(cfg)
            require(
                cfg.model == p["model"] and (acct / "budget120/authorization.json").is_file(),
                "Billing or model binding",
            )
            driver = load_driver(main / "tools/continue_expanded.py", p["driver"])
            locks.enter_context(exclusive_run(acct / "allocation"))
            for n in range(4):
                locks.enter_context(exclusive_run(acct / f"lane-{n}-execution"))
            directory, receipt = driver.allocate_extension(
                acct, acct / "runs/main-v1", approved_limit=120
            )
        for index, (name, split) in enumerate(ORDER):
            path = OUTPUT / f"decision-{index}.json"
            binding = {"protocol": digest_json(p), "case": name, "arm": "split" if split else "old"}
            item = cached(path, binding)
            if mode == "execute" and item is None:
                _, lane = choose_lane(driver, directory, receipt, 1)
                begin(path, binding)
                client = QwenDisputeModel(
                    cfg,
                    bounded.BoundedCellBudget(lane, 1),
                    max_tokens=bounded.OUTPUT_TOKEN_CAP,
                    structured_output=True,
                )
                try:
                    rec, tape = await run_fork(name, split, client.generate)
                finally:
                    await client.aclose()
                ev = _events([("signoff", client)], lane)
                require(
                    len(ev) == 1 and accounted({"http_events": ev}, lane), "Missing payment receipt"
                )
                item = {
                    "completed": True,
                    "binding": binding,
                    "event": ev[0],
                    "record": rec,
                    "prompt": tape.final_prompt,
                }
                write_json(path, item)
            require(item is not None, "Missing decision")
            event = item["event"]
            require(event["http_status"] == 200 and event["status"] == "completed", "HTTP error")

            async def replay(messages):
                from dataverify.agent.graph_model import GraphModelError

                require(messages == item["prompt"], "Different prompt in replay")
                if event.get("action_error_code"):
                    raise GraphModelError(event["action_error_code"])
                return item["record"]["actions"][-1]

            rec, tape = await run_fork(name, split, replay)
            require(
                digest_json(tape.final_prompt) == p["forks"][name][binding["arm"]],
                "Paid prompt not frozen",
            )
            for k in ("publication", "status", "reason", "replays", "actions", "trace"):
                require(rec[k] == item["record"][k], "Reproduction failure: " + k)
            valid = not event.get("action_error_code")
            score = negative.grade(FAULTS[name], rec)
            rows.append(
                {
                    **binding,
                    **score,
                    "status": rec["status"],
                    "valid_action": valid,
                    "action": rec["actions"][-1]["action"] if valid else None,
                    "reason": rec["reason"],
                    "rejection": rec["trace"][-1].get("code"),
                }
            )
            events.append(event)
    require(len(set(e["event_id"] for e in events)) == 4, "Duplicate receipt")
    result = {"protocol": digest_json(p), "rows": rows, "usage": summarize_usage(events)}
    write_json(OUTPUT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "audit"))
    asyncio.run(campaign(parser.parse_args().mode))
