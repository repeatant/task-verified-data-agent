"""Run disclosed discovery experiments; gold answers stay outside model context.

Offline actions are authored integration fixtures. --live normally generates
references and probes with Qwen; --audit-scenario explicitly injects the first
SQL and reference plan, while the independent audit and later actions are real.
--track otherwise identifies injected versus naturally generated first SQL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataverify.agent.discovery_graph import run_discovery
from dataverify.agent.discovery_model import QwenReviewModel, ScriptedReviewModel
from dataverify.agent.graph_checkpoint import (
    JournaledBudget,
    exclusive_run,
    read_budget,
    resume_budget,
)
from dataverify.agent.graph_model import InjectedFirstGraphModel, QwenGraphModel, ScriptedGraphModel
from dataverify.evaluation.discovery_cases import get_discovery_cases, grade_discovery_sql
from dataverify.evaluation.discovery_roles import (
    AuthoredReviewModel,
    InjectedFirstReviewModel,
    authored_attribution,
)
from dataverify.evaluation.reference_audit_cases import audit_scenarios
from dataverify.integrations.qwen_config import QwenConfig
from dataverify.runtime.discovery_budget import reviewed_truncation
from dataverify.runtime.experiment_budget import restore_budget
from dataverify.runtime.probe_cost_budget import validate_pricing
from dataverify.runtime.reporting import digest_json, summarize_usage, write_json
from dataverify.verification.discovery_runtime import DiscoveryTask, fixture_data, rows_equal
from dataverify.verification.probe_runtime import ProbeTask, static_check


def offline_roles(case, track):
    initial = case.good_sql if track == "correct" else case.bad_sql
    writer = ScriptedGraphModel(
        [{"task_id": case.name, "sql": initial}, {"task_id": case.name, "sql": case.good_sql}]
    )
    actions = [case.offline_plan()]
    if track != "correct":
        actions.append(authored_attribution(case.task, "candidate"))
    actions.append(case.offline_probe())
    actions.append({"action": "finish", "reason": "Authored bounded review."})
    return writer, AuthoredReviewModel(actions), ScriptedReviewModel([case.offline_plan()])


async def evaluate(args, budget=None):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = budget.snapshot() if budget else None
    config = QwenConfig.from_local_environment() if args.live else None
    if config:
        validate_pricing(config)
    cases = [c for c in get_discovery_cases() if not args.cases or c.name in args.cases]
    if args.audit_scenario:
        cases = [c for c in audit_scenarios() if c.name in args.audit_scenario]
    confirmed_examples = []
    if args.input:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        if not {"task", "source"} <= set(document) or set(document) - {
            "task",
            "source",
            "confirmed_examples",
        }:
            raise ValueError("INPUT_REQUIRES_TASK_AND_SOURCE")
        confirmed_examples = document.get("confirmed_examples", [])
        task_doc = document["task"]
        legacy = task_doc.get("legacy_task")
        task = DiscoveryTask(**{**task_doc, "legacy_task": ProbeTask(**legacy) if legacy else None})
        cases = [(task, fixture_data(document["source"]))]
    results, all_events = [], []
    for index, case in enumerate(cases):
        task, source = case if args.input else (case.task, case.source)
        if args.audit_scenario:
            confirmed_examples = case.confirmed_examples
        if args.live:
            writer = QwenGraphModel(config, budget)
            reviewer = QwenReviewModel(config, budget)
            auditor = QwenReviewModel(config, budget)
            if args.audit_scenario:
                writer = InjectedFirstGraphModel(
                    writer, {"task_id": task.task_id, "sql": case.initial_sql}
                )
                reviewer = InjectedFirstReviewModel(reviewer, case.reference_plan)
            elif args.track != "natural":
                writer = InjectedFirstGraphModel(
                    writer,
                    {
                        "task_id": task.task_id,
                        "sql": case.good_sql if args.track == "correct" else case.bad_sql,
                    },
                )
        else:
            writer, reviewer, auditor = offline_roles(case, args.track)
        record = {}
        try:
            record = await run_discovery(
                task,
                source,
                writer,
                reviewer,
                auditor=auditor,
                confirmed_examples=confirmed_examples,
                checkpoint_path=output / f"case-{index}.sqlite3",
                thread_id="run",
            )
        finally:
            await writer.aclose()
            await reviewer.aclose()
            await auditor.aclose()
            events = sorted(
                [
                    {**event, "role": role}
                    for role, model in (
                        ("writer", writer),
                        ("reviewer", reviewer),
                        ("auditor", auditor),
                    )
                    for event in getattr(model, "http_events", [])
                ],
                key=lambda event: event["attempt"],
            )
            all_events.extend(events)
            record["http_events"] = events
            record["bound_source"] = source.to_dict()
            record["budget_end"] = budget.snapshot() if budget else None
            record["model_contexts"] = {
                "writer": writer.requests,
                "reviewer": reviewer.requests,
                "auditor": auditor.requests,
            }
            record["controlled_injections"] = {
                "scenario": case.name if args.audit_scenario else None,
                "initial_sql": bool(args.audit_scenario or args.track != "natural"),
                "initial_reference": bool(args.audit_scenario),
                "writer_turn_origins": getattr(writer, "turn_origins", []),
                "reviewer_turn_origins": getattr(reviewer, "turn_origins", []),
                "independent_audit": "real_qwen" if args.live else "authored_offline_script",
            }
            write_json(output / f"case-{index}.json", record)
        score = {
            "task_id": task.task_id,
            "track": "controlled_reference_audit" if args.audit_scenario else args.track,
            "status": record["status"],
            "writer_attempts": record["attempts"],
            "generated_fixtures": len(record["fixtures"]),
            "observed_disagreement": any(
                any(c["status"] == "disagreement" for c in event["observation"]["checks"])
                for event in record["trace"]
                if event["node"] == "execute"
            ),
            "http_attempts": len(events),
            "reason": record["publication"]["reason"],
        }
        if args.audit_scenario:
            score.update(
                audit_scenario=case.name,
                published_source_rows_match_gold=record["publication"]["rows"] is not None
                and rows_equal(record["publication"]["rows"], case.expected_rows),
                sql_changed=record["publication"]["sql"] is not None
                and record["publication"]["sql"] != case.initial_sql,
                reference_revisions=record["reference_revisions"],
                confirmed_examples_supplied=len(case.confirmed_examples),
                scoring_scope="Source rows only; expected rows and final SQL are scoring-only.",
            )
        elif not args.input:
            score["post_run_grade"] = grade_discovery_sql(case, record["publication"]["sql"])
            score["gold_result_match"] = record["publication"]["rows"] is not None and rows_equal(
                record["publication"]["rows"], case.expected
            )
            legacy_task = ProbeTask(
                task.task_id,
                task.question,
                ("include_negative", "exclude_end", "exclude_status", "include_start"),
            )
            score["legacy_wrong_candidate"] = static_check(case.bad_sql, legacy_task)
            score["legacy_correct_candidate"] = static_check(case.good_sql, legacy_task)
        results.append(score)
        print(json.dumps(score, ensure_ascii=False), flush=True)
        summary = {
            "scope": "Disclosed development fixtures, not an independent benchmark.",
            "oracle_generation": (
                "controlled_initial_reference_then_real_qwen"
                if args.audit_scenario
                else "real_qwen"
                if args.live
                else "authored_offline_script"
            ),
            "real_model_used": bool(all_events),
            "cases": results,
            "usage": summarize_usage(all_events),
            "budget_start": start,
            "budget_end": budget.snapshot() if budget else None,
            "source_integrity_ok": digest_json(source.to_dict())
            == record["binding"]["source_digest"],
            "audit_ledger_complete": (
                budget.snapshot()["reported_cost_complete"] if budget else True
            ),
            "http_events": all_events,
        }
        write_json(output / "summary.json", summary)
        if budget and (
            budget.snapshot().get("stopped_reason")
            or record["status"] == "inconclusive"
            and any(e.get("status") != "completed" for e in events)
        ):
            break
    return results


def apply_technical_limits(budget, *, token_stop=None, input_char_stop=None):
    """Change only explicitly selected technical stops, retaining all paid usage."""
    if token_stop is not None and (
        type(token_stop) is not int or not budget.reported_tokens <= token_stop <= 500_000
    ):
        raise ValueError("Token stop must cover consumed tokens and be at most 500000")
    if input_char_stop is not None and (
        type(input_char_stop) is not int or not budget.input_chars <= input_char_stop <= 2_000_000
    ):
        raise ValueError("Input character stop must cover consumed input and be at most 2000000")
    if token_stop is not None:
        budget.max_reported_tokens = token_stop
    if input_char_stop is not None:
        budget.max_input_chars = input_char_stop


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--track", choices=("controlled", "correct", "natural"), default="controlled"
    )
    parser.add_argument("--cases", nargs="*", choices=[c.name for c in get_discovery_cases()])
    parser.add_argument(
        "--audit-scenario",
        nargs="+",
        choices=[c.name for c in audit_scenarios()],
        help="Live controlled injection of initial SQL/reference; independent audit stays real",
    )
    parser.add_argument(
        "--input", type=Path, help="Arbitrary task/source JSON; requires --live --track natural"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / ("discovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")),
    )
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument(
        "--prior-journal",
        type=Path,
        help="Initialize once from the last cumulative journal; do not fork allowances",
    )
    parser.add_argument("--max-extra-requests", type=int, choices=range(1, 37), default=18)
    parser.add_argument(
        "--token-stop",
        type=int,
        help="Technical cumulative token stop for a new continuation (money allowance is unchanged)",
    )
    parser.add_argument(
        "--input-char-stop",
        type=int,
        help="Technical cumulative input character stop for a NEW live continuation; money unchanged",
    )
    parser.add_argument(
        "--recover-truncation",
        type=Path,
        help="Explicitly review a fully accounted truncation report when continuing into a NEW journal",
    )
    args = parser.parse_args(argv)
    if args.audit_scenario and (not args.live or args.cases is not None or args.input):
        parser.error("--audit-scenario requires --live and excludes --cases/--input")
    if args.token_stop is not None and (
        not args.live or not args.prior_journal or not 1 <= args.token_stop <= 500000
    ):
        parser.error("--token-stop requires live continuation from a prior journal; maximum 500000")
    if args.input_char_stop is not None and (
        not args.live
        or not args.prior_journal
        or not args.budget_file
        or args.budget_file.exists()
        or not 1 <= args.input_char_stop <= 2_000_000
    ):
        parser.error("--input-char-stop requires a NEW live continuation journal; maximum 2000000")
    if args.input and (not args.live or args.track != "natural" or args.cases):
        parser.error("--input requires --live --track natural and no --cases")
    if not args.live:
        if args.track == "natural" or args.budget_file or args.prior_journal:
            parser.error("Natural generation and budgets require --live")
        asyncio.run(evaluate(args))
        return
    if args.recover_truncation and (
        not args.prior_journal or args.budget_file and args.budget_file.exists()
    ):
        parser.error("Recovery requires --prior-journal and a new --budget-file")
    if not args.budget_file:
        parser.error("--live requires --budget-file")
    with exclusive_run(args.budget_file):
        if args.budget_file.exists():
            if args.prior_journal:
                parser.error("Cannot reseed an existing journal")
            budget = JournaledBudget(resume_budget(read_budget(args.budget_file)), args.budget_file)
        else:
            if not args.prior_journal or args.prior_journal.resolve() == args.budget_file.resolve():
                parser.error("New journal requires a distinct existing --prior-journal")
            with exclusive_run(args.prior_journal):
                prior = read_budget(args.prior_journal)
                parent_digest = digest_json(prior)
                if args.recover_truncation:
                    prior = reviewed_truncation(
                        prior, json.loads(args.recover_truncation.read_text(encoding="utf-8"))
                    )
                restored = restore_budget(
                    prior,
                    max_extra_requests=args.max_extra_requests,
                    money_limit_cny=prior["money_limit_cny"],
                )
                try:
                    apply_technical_limits(
                        restored, token_stop=args.token_stop, input_char_stop=args.input_char_stop
                    )
                except ValueError as error:
                    parser.error(str(error))
                budget = JournaledBudget(restored, args.budget_file)
                write_json(
                    args.budget_file.with_suffix(".lineage.json"),
                    {
                        "parent": str(args.prior_journal.resolve()),
                        "parent_digest": parent_digest,
                        "recovery_report": str(args.recover_truncation)
                        if args.recover_truncation
                        else None,
                        "token_stop_before": prior["reported_token_stop_threshold"],
                        "token_stop_after": restored.max_reported_tokens,
                        "input_char_stop_before": prior["input_char_limit"],
                        "input_char_stop_after": restored.max_input_chars,
                        "money_allowance_unchanged": True,
                    },
                )
        asyncio.run(evaluate(args, budget))


if __name__ == "__main__":
    main()
