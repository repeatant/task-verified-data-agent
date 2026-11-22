"""Gold-isolated program grading and paired, clustered experiment summaries.

Correct execution on a finite suite is evidence, not a proof. A correct program
that was withheld is not a successful delivery. Reference and auditor programs
are graded separately so a correct SQL cannot conceal an incorrect reference.
"""

from __future__ import annotations

import random
from collections import defaultdict

from sqlglot.errors import SqlglotError

from dataverify.evaluation.expanded_cases import expected, scoring_data
from dataverify.evaluation.public_benchmark import compare_results, execute_original
from dataverify.runtime.reporting import summarize_usage
from dataverify.verification.discovery_runtime import execute_readonly, named_tables, rows_equal
from dataverify.verification.reference_expression import calculate


def _program_grade(case, program, *, sql: bool) -> dict:
    datasets = scoring_data(case)
    if not isinstance(program, str) or not program.strip():
        return {
            "present": False,
            "correct": None,
            "passed": 0,
            "total": len(datasets),
            "failures": [],
        }
    passed, failures = 0, []
    for name, data in datasets.items():
        try:
            actual = (
                execute_readonly(program, data) if sql else calculate(program, named_tables(data))
            )
            matched = rows_equal(actual, expected(case, data))
            if not matched:
                failures.append({"dataset": name, "reason": "output_mismatch"})
            passed += matched
        except (ValueError, TypeError, KeyError, ArithmeticError, SqlglotError) as error:
            failures.append(
                {"dataset": name, "reason": type(error).__name__, "detail": str(error)[:200]}
            )
    return {
        "present": True,
        "correct": passed == len(datasets),
        "passed": passed,
        "total": len(datasets),
        "failures": failures,
    }


def _field(record: dict, field: str, key: str):
    value = record.get(field)
    return value.get(key) if isinstance(value, dict) else None


def _outcomes(record, initial, candidate, publication, *, publication_rows_consistent=True) -> dict:
    delivered = record.get("status") == "reviewed" and publication["present"]
    correct_delivery = delivered and publication["correct"] is True and publication_rows_consistent
    initially_correct = initial["correct"] is True
    return {
        "delivered": delivered,
        "correct_delivery": correct_delivery,
        "wrong_release": delivered and not correct_delivery,
        "initial_sql_correct": initially_correct,
        "final_sql_correct": candidate["correct"] is True,
        "correct_sql_damaged": initially_correct and candidate["correct"] is False,
        "correct_task_rejected": initially_correct and not delivered,
        "publication_rows_consistent": publication_rows_consistent if delivered else None,
        "initial": initial,
        "candidate": candidate,
        "publication": publication,
    }


def score_controlled(case, record: dict) -> dict:
    """Grade the 25 hidden datasets without passing their results to an Agent."""
    initial = _program_grade(case, case.initial_sql, sql=True)
    candidate = _program_grade(case, _field(record, "candidate", "sql"), sql=True)
    publication = _program_grade(case, _field(record, "publication", "sql"), sql=True)
    reference = _program_grade(case, _field(record, "plan", "reference"), sql=False)
    auditor = _program_grade(case, _field(record, "audit_plan", "reference"), sql=False)
    publication_rows_consistent = False
    if publication["present"]:
        try:
            rows = execute_readonly(_field(record, "publication", "sql"), case.source)
            publication_rows_consistent = rows_equal(rows, _field(record, "publication", "rows"))
        except (ValueError, TypeError):
            pass
    score = _outcomes(
        record,
        initial,
        candidate,
        publication,
        publication_rows_consistent=publication_rows_consistent,
    )
    score.update(
        reference=reference, auditor=auditor, metric_scope="25-dataset controlled semantic suite"
    )
    joint_evaluable = reference["present"] and auditor["present"]
    score["joint_evaluable"] = joint_evaluable
    score["joint_correct_delivery"] = (
        (score["correct_delivery"] and reference["correct"] and auditor["correct"])
        if joint_evaluable
        else None
    )
    return score


def score_public(case: dict, record: dict) -> dict:
    """Score final SQL on the original full database using official BIRD EX.

    Independent programs are not graded on a preview as though it were the full
    database. Their public-benchmark correctness is explicitly not measured.
    """
    cache = {}

    def grade(sql):
        if not isinstance(sql, str) or not sql.strip():
            return {"present": False, "correct": None, "passed": 0, "total": 1}
        if sql in cache:
            return cache[sql]
        try:
            rows = execute_original(sql, case["db_path"])
            matches = compare_results(rows, case["gold_rows"])
            result = {
                "present": True,
                "correct": matches["bird_execution_match"],
                "passed": int(matches["bird_execution_match"]),
                "total": 1,
                **matches,
            }
        except (ValueError, TypeError, KeyError, ArithmeticError, SqlglotError) as error:
            result = {
                "present": True,
                "correct": False,
                "passed": 0,
                "total": 1,
                "error": str(error)[:300],
            }
        cache[sql] = result
        return result

    initial = grade(case.get("initial_sql"))
    candidate = grade(_field(record, "candidate", "sql"))
    publication = grade(_field(record, "publication", "sql"))
    # Runtime publication rows are a fixture preview, not full-DB output. The
    # public endpoint is the delivered SQL executed here on the original DB.
    score = _outcomes(record, initial, candidate, publication)
    score.update(
        metric_scope="BIRD EX on original full SQLite database",
        reference={
            "present": bool(_field(record, "plan", "reference")),
            "correct": None,
            "reason": "not_scored_on_full_database",
        },
        auditor={
            "present": bool(_field(record, "audit_plan", "reference")),
            "correct": None,
            "reason": "not_scored_on_full_database",
        },
        joint_evaluable=False,
        joint_correct_delivery=None,
        publication_rows_consistent=None,
    )
    return score


_METRICS = {
    "correct_delivery": "all",
    "wrong_release": "all",
    "correct_sql_damaged": "initial_sql_correct",
    "correct_task_rejected": "initial_sql_correct",
    "joint_correct_delivery": "joint_evaluable",
}


def _rate(rows, key):
    criterion = _METRICS[key]
    eligible = [row for row in rows if criterion == "all" or row["score"].get(criterion) is True]
    successes = sum(row["score"].get(key) is True for row in eligible)
    return {
        "numerator": successes,
        "denominator": len(eligible),
        "rate": successes / len(eligible) if eligible else None,
    }


def _quantile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _cluster_key(row):
    return str(row.get("cluster_id") or row.get("parent_id") or row["case_id"])


def _sum_usage(summaries):
    keys = (
        "prompt_tokens_known",
        "completion_tokens_known",
        "total_tokens_known",
        "event_count",
        "unknown_usage_events",
    )
    result = {key: sum(item.get(key, 0) for item in summaries) for key in keys}
    result["usage_complete"] = all(item.get("usage_complete", False) for item in summaries)
    return result


def _actual_usage(rows, initial_records):
    """Deduplicate shared caches by stable lane/request event IDs, never tokens."""
    events = []
    for row in rows:
        events.extend(row.get("http_events", []))
        events.extend(row.get("shared_initial_http_events", []))
    for record in initial_records:
        events.extend(record.get("http_events", []))
    if any(not event.get("event_id") for event in events):
        return {
            "available": False,
            "reason": "Stable lane-qualified event IDs required; use durable budget ledgers for actual charges.",
        }
    unique = {}
    for event in events:
        event_id = event["event_id"]
        if event_id in unique and unique[event_id].get("usage") != event.get("usage"):
            raise ValueError("EVENT_ID_USAGE_CONFLICT")
        unique[event_id] = event
    return {
        "available": True,
        "scope": "Observed row and supplied initial-cache events, deduplicated; durable ledgers remain authoritative.",
        **summarize_usage(list(unique.values())),
    }


def _summary(rows, repetitions, seed):
    metrics = {key: _rate(rows, key) for key in _METRICS}
    groups = defaultdict(list)
    for row in rows:
        groups[_cluster_key(row)].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    distributions = defaultdict(list)
    for _ in range(repetitions):
        sampled = (
            [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]] if keys else []
        )
        for metric in _METRICS:
            rate = _rate(sampled, metric)["rate"]
            if rate is not None:
                distributions[metric].append(rate)
    for metric, value in metrics.items():
        values = distributions[metric]
        value["cluster_bootstrap_ci95"] = (
            [_quantile(values, 0.025), _quantile(values, 0.975)] if values else None
        )
        value["bootstrap_valid_replicates"] = len(values)
    return {
        "trajectories": len(rows),
        "clusters": len(groups),
        "metrics": metrics,
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in rows),
        "elapsed_scope": "Post-initial trajectory wall time; excludes shared cached first generations.",
        "post_initial_usage": _sum_usage([row.get("usage", {}) for row in rows]),
        "cold_start_effective_usage": _sum_usage(
            [
                usage
                for row in rows
                for usage in (row.get("usage", {}), row.get("shared_initial_usage", {}))
            ]
        ),
        "cold_start_cost_note": "Initial generations counted once per applicable method: virtual uncached comparison, not actual total billing.",
    }


def _paired(rows, reference_method, repetitions, seed):
    by_method = defaultdict(dict)
    for row in rows:
        if row["case_id"] in by_method[row["method"]]:
            raise ValueError("DUPLICATE_CASE_METHOD; separate repeats with a trial ID")
        by_method[row["method"]][row["case_id"]] = row
    reference = by_method.get(reference_method, {})
    result = {}
    for method, candidates in sorted(by_method.items()):
        if method == reference_method:
            continue
        shared = sorted(set(reference) & set(candidates))
        groups = defaultdict(list)
        for case_id in shared:
            if _cluster_key(reference[case_id]) != _cluster_key(candidates[case_id]):
                raise ValueError("PAIRED_CLUSTER_MISMATCH")
            groups[_cluster_key(reference[case_id])].append(case_id)
        keys = sorted(groups)
        rng = random.Random(seed)
        distributions = defaultdict(list)
        for _ in range(repetitions):
            ids = (
                [case_id for key in rng.choices(keys, k=len(keys)) for case_id in groups[key]]
                if keys
                else []
            )
            for metric in _METRICS:
                ref_rate = _rate([reference[i] for i in ids], metric)["rate"]
                candidate_rate = _rate([candidates[i] for i in ids], metric)["rate"]
                if ref_rate is not None and candidate_rate is not None:
                    distributions[metric].append(ref_rate - candidate_rate)
        metrics = {}
        for metric in _METRICS:
            ref_metric = _rate([reference[i] for i in shared], metric)
            candidate_metric = _rate([candidates[i] for i in shared], metric)
            delta = (
                ref_metric["rate"] - candidate_metric["rate"]
                if ref_metric["rate"] is not None and candidate_metric["rate"] is not None
                else None
            )
            values = distributions[metric]
            metrics[metric] = {
                "reference_minus_method": delta,
                "reference": ref_metric,
                "method": candidate_metric,
                "cluster_bootstrap_ci95": [_quantile(values, 0.025), _quantile(values, 0.975)]
                if values
                else None,
                "bootstrap_valid_replicates": len(values),
            }
        result[method] = {
            "paired_tasks": len(shared),
            "clusters": len(groups),
            "reference_only_tasks": len(set(reference) - set(candidates)),
            "method_only_tasks": len(set(candidates) - set(reference)),
            "metrics": metrics,
        }
    return result


def aggregate_runs(
    rows: list[dict],
    *,
    reference_method="full",
    bootstrap_repetitions=2000,
    seed=20261107,
    initial_records=(),
) -> dict:
    """Summarize each population separately; paired comparisons use same tasks.

    Cluster IDs should group fault variants of a parent task (or a whole error
    family for a more conservative design). Do not pool public EX and controlled
    semantic-suite success into a single headline accuracy.
    """
    if bootstrap_repetitions < 0:
        raise ValueError("BOOTSTRAP_REPETITIONS_NEGATIVE")
    populations = defaultdict(list)
    for row in rows:
        populations[row["kind"]].append(row)
    result = {
        "seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "reference_method": reference_method,
        "populations": {},
        "actual_logged_usage": _actual_usage(rows, initial_records),
    }
    for kind, population in sorted(populations.items()):
        methods = defaultdict(list)
        strata = defaultdict(lambda: defaultdict(list))
        conditions = defaultdict(lambda: defaultdict(list))
        for row in population:
            methods[row["method"]].append(row)
            strata[str(row.get("db_id") or row.get("family") or "unspecified")][
                row["method"]
            ].append(row)
            conditions[str(row.get("condition", "unspecified"))][row["method"]].append(row)
        result["populations"][kind] = {
            "methods": {
                method: _summary(items, bootstrap_repetitions, seed)
                for method, items in sorted(methods.items())
            },
            "strata": {
                group: {
                    method: _summary(items, 0, seed)
                    for method, items in sorted(group_methods.items())
                }
                for group, group_methods in sorted(strata.items())
            },
            "paired": _paired(population, reference_method, bootstrap_repetitions, seed),
            "conditions": {
                condition: {
                    method: _summary(items, 0, seed)
                    for method, items in sorted(group_methods.items())
                }
                for condition, group_methods in sorted(conditions.items())
            },
        }
    return result
