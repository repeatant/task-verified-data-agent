"""Freeze, execute or reproduce an equal-state, one-call signoff comparison."""

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from evaluate_evidence_recovery import accounted, choose_lane
from evaluate_query_ablation import hash_sources
from evaluate_structured_confirmation import immutable, load_driver, read

from dataverify.agent.dispute_graph import _DisputeNodes, run_dispute_graph
from dataverify.agent.dispute_model import QwenDisputeModel
from dataverify.agent.graph_checkpoint import exclusive_run
from dataverify.agent.graph_model import GraphModelError
from dataverify.evaluation import dispute_recovery as development
from dataverify.evaluation import evidence_recovery as bounded
from dataverify.evaluation import query_ablation as paired
from dataverify.evaluation import signoff_controls as negative
from dataverify.evaluation import terminal_scenarios as forward
from dataverify.evaluation.expanded_experiment import _events
from dataverify.evaluation.public_benchmark import file_digest
from dataverify.evaluation.structured_confirmation import begin, cached
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.probe_cost_budget import PRICE_MODEL, validate_pricing
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json

PARENT = "18be459a5987939311d06aa86d8e15c12682bf9fd394968ba234f843b145a7c4"
CODE = "DISPUTE_FINISH_REQUIRES_CURRENT_REPLAY_OR_CHECK"
FOCAL = ("dispute-01", "dispute-03", "dispute-04", "terminal-02")
CASES = {c.case_id: c for c in development.study_cases() + forward.study_cases()}
FAULTS = {c.name: c for c in negative.cases()}


def require(check, message):
    if not check:
        raise ValueError(message)


def schedule():
    keys = (
        [("terminal", case) for case in FOCAL]
        + [("negative", name) for name in FAULTS]
        + [("ordinary", case) for case in FOCAL[:2]]
    )
    for i, (group, case) in enumerate(keys):
        for split in (False, True) if i % 2 == 0 else (True, False):
            yield group, case, split


def old_file(parent, name):
    return parent / "last_decision" / "records" / (name + "--query_on.json")


def verify_cohort(parent):
    require(digest_json(read(parent / "protocol.json")) == PARENT, "Parent changed")
    selected, files = [], {}
    for name in CASES:
        file = old_file(parent, name)
        item = read(file)
        require(item["completed"] and item["binding"]["protocol"] == PARENT, "Archive mismatch")
        files[name] = file_digest(file)
        rec = item["record"]
        if ("code", CODE) in rec["trace"][-1].items() and len(item["requests"]) == 4:
            selected.append(name)
    require(set(selected) == set(FOCAL), "Unexpected set of archived final citation refusals")
    return files


class Tape:
    """Replay recorded parser outcomes; the final arm gets one real model response."""

    def __init__(self, archived, final):
        self.archived, self.final = archived, final
        self.calls, self.actions = 0, 0
        self.last_prompt = None
        self.prefix = []

    async def generate(self, messages):
        index = self.calls
        self.calls += 1
        at_fork = 3 if self.archived else 1
        require(index <= at_fork, "Extra model call")
        if index == at_fork:
            self.last_prompt = messages
            return await self.final(messages)
        self.prefix.append(digest_json(messages))
        if self.archived:
            require(messages == self.archived["requests"][index], "Historic prompt drift")
            error = self.archived["http_events"][index].get("action_error_code")
            if error:
                raise GraphModelError(error, retryable=True)
            value = self.archived["record"]["actions"][self.actions]
            self.actions += 1
            return value
        return {
            "action": "query",
            "sql": "SELECT COUNT(*) FROM orders",
            "hypothesis": "Establish the number of orders on the bound source.",
        }


async def at_fork(parent, group, name, split, final):
    # Execute the historical prefix with its original parser and gates. The patch
    # is confined to this sequential research process. Both forks run the actual
    # production graph, SQL executions, citation checks and publication boundary.
    target = 1 if group == "negative" else 3
    old = None if group == "negative" else read(old_file(parent, name))
    model = Tape(old, final)
    base_policy = replace(
        paired.policy_for("query_on"), max_actions=target + 1, reserve_final_decision=True
    )
    original = _DisputeNodes.act

    async def branch(nodes, state):
        if state["model_calls"] == target:
            nodes.policy = replace(
                base_policy, split_finish_evidence=split, reserve_final_decision=group != "ordinary"
            )
        return await original(nodes, state)

    if old:
        case = CASES[name]
        task, preview = case.base.task, case.base.source
        sql, ref, aud = case.initial_sql, case.reference_plan, case.auditor_plan
    else:
        case = FAULTS[name]
        task, preview = case.task, case.preview
        sql, ref, aud = case.sql, case.plan(case.reference), case.plan(case.auditor)
    with patch.object(_DisputeNodes, "act", branch):
        result = await run_dispute_graph(
            task,
            preview,
            sql,
            ref,
            aud,
            model,
            policy=base_policy,
            query_source=paired.full_source(preview),
        )
    require(model.calls == target + 1, "Could not reach designated decision")
    return result, model


def grade(group, name, rec):
    if group == "negative":
        return negative.grade(FAULTS[name], rec)
    grader = forward.grade if name.startswith("terminal-") else paired.grade
    got = grader(CASES[name], rec)
    return {
        "sql_good": got["candidate"]["correct"],
        "reference_good": got["reference"]["correct"],
        "auditor_good": got["auditor"]["correct"],
        "correct_delivery": got["correct_delivery"],
        "wrong_release": got["wrong_release"],
    }


async def offline_forks(parent):
    checks = {}

    async def probe(_):
        raise GraphModelError("PRE_PAYMENT_STOP")

    for group, name, split in schedule():
        rec, tape = await at_fork(parent, group, name, split, probe)
        require(rec["reason"] == "PRE_PAYMENT_STOP", "Unexpected simulated outcome")
        key = group + "/" + name
        ctx = json.loads(tape.last_prompt[1]["content"])
        if group == "terminal":
            historic = read(old_file(parent, name))
            require(
                json.loads(historic["requests"][3][1]["content"]) == ctx, "Changed evidence at fork"
            )
            if not split:
                require(historic["requests"][3] == tape.last_prompt, "Historical baseline drift")
        identity = {
            "prefix": tape.prefix,
            "context": digest_json(ctx),
            "programs": digest_json(rec["history"]),
            "replays": digest_json(rec["replays"]),
        }
        entry = checks.setdefault(key, {"identity": identity})
        require(identity == entry["identity"], "Unequal contexts or executable state")
        entry["split_prompt" if split else "old_prompt"] = digest_json(tape.last_prompt)
        g = grade(group, name, rec)
        if group == "negative":
            require(not g["sql_good"], "Fault cannot be detected by independent grade")
            require(
                rec["replay"]["agreement"] == (name == "wrong-null"),
                "Negative control scope changed",
            )
        else:
            require(
                all(g[k] for k in ("sql_good", "reference_good", "auditor_good")),
                "Historical state is not all correct",
            )
    return checks


def freeze(parent, out, main):
    sources = hash_sources(ROOT)
    relative = "tools/evaluate_split_signoff.py"
    sources[relative] = file_digest(ROOT / relative)
    snapshot = out / "source"
    archives = verify_cohort(parent)
    driver = file_digest(main / "tools/continue_expanded.py")
    manifest = out / "protocol.json"
    if manifest.exists():
        p = read(manifest)
        require(
            archives == p["archives"]
            and driver == p["driver"]
            and all(file_digest(snapshot / k) == v for k, v in p["sources"].items()),
            "Frozen input drift",
        )
        return p
    forks = asyncio.run(offline_forks(parent))
    p = {
        "model": PRICE_MODEL,
        "parent": PARENT,
        "sources": sources,
        "driver": driver,
        "archives": archives,
        "forks": forks,
        "schedule": list(schedule()),
        "max_new_http": 16,
        "new_control_scoring": {
            k: digest_json(
                {n: [d.to_dict(), negative.expected(c, d)] for n, d in negative.sources(c).items()}
            )
            for k, c in FAULTS.items()
        },
        "selection": "All four final citation rejections among nine archived affected-study cases. The two ordinary-policy probes reuse the first two cases in ID order. Negative definitions and fixed SQL chosen before new responses.",
        "primary": "At identical decision states, count independently correct programs withheld, with citation-specific rejection separate from choosing to abstain/query/revise. Grade every publication SQL, both references and source-bound returned rows.",
        "decision": "Require net additional correct signoffs and no new wrong delivery or wrong reference. Any released negative control (including same-sample common-mode error) defeats safety promotion. Ordinary workflow requires its own net gain without regression. One draw per cell; no further sampling based on results.",
    }
    for name, hash_value in sources.items():
        dest = snapshot / name
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(ROOT / name, dest)
        require(file_digest(dest) == hash_value, "Source changed during freeze")
    immutable(manifest, p)
    return p


async def cells(mode, parent, out, p, account=None):
    protocol = digest_json(p)
    require(await offline_forks(parent) == p["forks"], "Trial inputs changed")
    ids, events, rows = set(), [], []
    if mode == "execute":
        cfg = QwenConfig.from_local_environment()
        validate_pricing(cfg)
        require(cfg.model == p["model"], "Model changed")
    for group, name, split in schedule():
        arm = "split" if split else "old"
        path = out / "records" / f"{group}-{name}-{arm}.json"
        binding = {"protocol": protocol, "group": group, "case": name, "arm": arm}
        item = cached(path, binding)
        if mode == "execute" and item is None:
            _, lane = choose_lane(*account, 1)
            begin(path, binding)
            client = QwenDisputeModel(
                cfg, bounded.BoundedCellBudget(lane, 1), max_tokens=bounded.OUTPUT_TOKEN_CAP
            )
            try:
                rec, tape = await at_fork(parent, group, name, split, client.generate)
            finally:
                await client.aclose()
            ev = _events([(arm, client)], lane)
            require(
                len(ev) == 1 and accounted({"http_events": ev}, lane),
                "Missing or uncertain receipt",
            )
            require(client.requests == [tape.last_prompt], "Prefix sent to provider")
            item = {
                "completed": True,
                "binding": binding,
                "event": ev[0],
                "record": rec,
                "final_prompt": tape.last_prompt,
            }
            write_json(path, item)
            print(group, name, arm, rec["status"], flush=True)
        require(item is not None, "Incomplete trial")
        e = item["event"]
        require(
            e["event_id"] not in ids and e["http_status"] == 200 and e["status"] == "completed",
            "Duplicate or failed HTTP request",
        )
        ids.add(e["event_id"])
        events.append(e)

        async def recorded(messages):
            require(messages == item["final_prompt"], "Paid final prompt drift")
            if e.get("action_error_code"):
                raise GraphModelError(
                    e["action_error_code"],
                    retryable=e["action_error_code"] != "DISPUTE_INVALID_FINAL_DECISION",
                )
            return item["record"]["actions"][-1]

        replay, tape = await at_fork(parent, group, name, split, recorded)
        rec = item["record"]
        require(
            digest_json(tape.last_prompt) == p["forks"][group + "/" + name][arm + "_prompt"],
            "Paid prompt differs from freeze",
        )
        for key in ("status", "reason", "trace", "actions", "versions", "publication", "replays"):
            require(rec[key] == replay[key], "Record does not reproduce: " + key)
        g = grade(group, name, rec)
        final_event = rec["trace"][-1]
        rejected = final_event.get("code") if final_event["node"] == "action_rejected" else None
        rows.append(
            {
                **binding,
                **g,
                "status": rec["status"],
                "final_action": None
                if e.get("action_error_code")
                else rec["actions"][-1]["action"],
                "rejection": rejected,
                "edit_count": rec["revision_attempts"],
                "all_current_correct": all(
                    g[k] for k in ("sql_good", "reference_good", "auditor_good")
                ),
            }
        )
    report = {"rows": rows, "usage": summarize_usage(events), "protocol": protocol}
    write_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "execute", "audit"])
    for flag in ("main", "parent", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    main_repo, parent, out = (d.resolve() for d in (args.main, args.parent, args.output))
    require(out != parent and not out.is_relative_to(parent), "Use a fresh trial directory")
    (out / "records").mkdir(parents=True, exist_ok=True)
    protocol = freeze(parent, out, main_repo)
    if args.mode == "prepare":
        print(digest_json(protocol), protocol["max_new_http"])
        return
    snapshot = out / "source"
    if ROOT != snapshot:
        subprocess.run(
            [
                sys.executable,
                str(snapshot / "tools/evaluate_split_signoff.py"),
                args.mode,
                "--main",
                str(main_repo),
                "--parent",
                str(parent),
                "--output",
                str(out),
            ],
            cwd=snapshot,
            check=True,
        )
        return
    if args.mode == "execute":
        campaign = main_repo / "reports/expanded-v1-20261107"
        require((campaign / "budget120/authorization.json").is_file(), "No original authorization")
        driver = load_driver(main_repo / "tools/continue_expanded.py", protocol["driver"])
        with ExitStack() as locks:
            locks.enter_context(exclusive_run(campaign / "allocation"))
            for lane in range(4):
                locks.enter_context(exclusive_run(campaign / f"lane-{lane}-execution"))
            directory, receipt = driver.allocate_extension(
                campaign, campaign / "runs/main-v1", approved_limit=120
            )
            result = asyncio.run(
                cells(args.mode, parent, out, protocol, (driver, directory, receipt))
            )
    else:
        result = asyncio.run(cells(args.mode, parent, out, protocol))
    print(json.dumps(result["usage"]))


if __name__ == "__main__":
    main()
