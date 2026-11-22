"""Export a completed audit to public aggregates and a reproducible result plot.

This tool reads saved evidence only. It does not import the Agent, read budget
journals, load credentials or make model requests. Use --summary to redraw the
figure using only the public JSON; matplotlib is the only plotting dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

METHODS = ("direct", "self_repair", "two_way", "full")
LABELS = ("Direct", "Self repair", "Two-way", "Full")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def terminal_category(document):
    """One terminal category per trajectory, not an inferred root-cause claim."""
    row, record = document["row"], document["record"]
    reason = record.get("reason") or record.get("publication", {}).get("reason") or ""
    if row["score"]["correct_delivery"]:
        return "correct_delivery"
    if row["score"]["wrong_release"]:
        return "wrong_sql_released"
    if row["status"] == "initial_reference_invalid":
        return "initial_reference_invalid"
    if reason in {"DISCOVERY_INVALID_REVIEW_ACTION", "GRAPH_INVALID_ACTION_FIELDS"}:
        return "invalid_action_shape"
    if reason in {"BASIS_NOT_IN_SOURCE", "ATTRIBUTION_BASIS_NOT_IN_SOURCE"}:
        return "evidence_quote_not_in_source"
    if reason == "ATTRIBUTION_CITES_NO_RELEVANT_DISAGREEMENT":
        return "evidence_gate_no_relevant_disagreement"
    if reason.startswith("REFERENCE_"):
        return "reference_dsl_failure"
    if reason == "REVIEW_BUDGET_EXHAUSTED":
        return "review_budget_exhausted"
    if reason.startswith("GRAPH_QWEN_TRANSPORT_"):
        return "transport_failure"
    if reason == "GRAPH_QWEN_RESPONSE_TRUNCATED":
        return "output_truncated"
    if reason == "GRAPH_SINGLE_REQUEST_INPUT_TOO_LARGE":
        return "input_limit"
    if reason.startswith("settled_at") or reason.startswith("RELATIONAL_"):
        return "fixture_constraints"
    if row["status"] == "needs_clarification":
        return "model_clarification"
    return "other_non_delivery"


def public_projection(audit_path):
    audit = read(audit_path)
    if (
        audit["status"] != "complete"
        or audit["errors"]
        or not audit["rescore"]["requested"]
        or audit["rescore"]["mismatches"]
        or not audit["stability"]["rescore"]["requested"]
        or audit["stability"]["rescore"]["mismatches"]
    ):
        raise ValueError("COMPLETE_RECOMPUTED_AUDIT_REQUIRED")
    taxonomy, conditions = {}, {}
    for population in ("public", "controlled"):
        taxonomy[population] = {"all_full": Counter(), "initially_correct_rejected": Counter()}
        conditions[population] = {}
    for provenance in audit["provenance"]:
        document = read(provenance["selected"])
        row = document["row"]
        if row["method"] != "full":
            continue
        category = terminal_category(document)
        taxonomy[row["kind"]]["all_full"][category] += 1
        if row["score"]["correct_task_rejected"]:
            taxonomy[row["kind"]]["initially_correct_rejected"][category] += 1
        counts = conditions[row["kind"]].setdefault(row["condition"], Counter())
        counts["trajectories"] += 1
        for key in (
            "correct_delivery",
            "wrong_release",
            "initial_sql_correct",
            "correct_sql_damaged",
            "correct_task_rejected",
            "final_sql_correct",
        ):
            counts[key] += row["score"][key]
    budget = audit["budget_audit"]
    result = {
        "format": "dataverify-expanded-public-summary-v1",
        "date": "2026-11-07",
        "source_commit": "3b1fab4",
        "audit_sha256": hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
        "protocol_digest": audit["source_audit"]["protocol_digest"],
        "database_digests": audit["source_audit"]["database_digests"],
        "audit_status": "complete",
        "coverage": {
            "main": audit["coverage"]["available"],
            "stability": audit["stability"]["available"],
        },
        "population": audit["population"],
        "analysis": audit["analysis"],
        "outcomes": audit["observed_outcomes"],
        "full_terminal_categories": taxonomy,
        "full_conditions": conditions,
        "stability": {
            key: audit["stability"][key]
            for key in (
                "expected",
                "available",
                "status",
                "outcomes",
                "per_original_task",
                "actual_repeat_usage",
                "rescore",
            )
        },
        "actual_main_usage": audit["actual_main_usage"],
        "budget_snapshot": {
            **{
                key: budget[key]
                for key in (
                    "scope",
                    "cumulative_authorized_cny",
                    "cumulative_reserved_cny",
                    "cumulative_known_estimated_cny",
                    "known_cost_is_incomplete",
                )
            },
            "unknown_reservations": len(budget["unknown_reservations"]),
            "unknown_reserved_cny": sum(
                item["reserved_upper_cny"] for item in budget["unknown_reservations"]
            ),
            "snapshot_cutoff": "After main + stability, before structured-output confirmation.",
        },
        "metric_notes": [
            "Public: original-database BIRD execution-set match for delivered SQL; reference program correctness on the full DB is not measured.",
            "Controlled: all 25 scoring datasets; 100 fault trajectories cover 20 task/fault parent combinations and 18 original requirements.",
            "Non-delivery is not successful correction. Wrong releases must also be divided by delivered answers.",
            "Zero observed errors can produce a degenerate empirical bootstrap [0,0]; this is not a useful population-risk upper bound.",
            "Ablation comparisons use their matched subset, not full-method totals. Intervals are exploratory, unadjusted cluster bootstrap intervals.",
            "Actual calls are deduplicated; per-method cold-start effective usage includes shared initial calls and is not additional billing.",
            "This public projection contains aggregate results only, not API credentials, local paths, database contents or private contexts.",
        ],
    }
    return result


def baseline_rows(summary):
    rows = []
    for population in ("public", "controlled"):
        for method in METHODS:
            outcome = summary["outcomes"][population][method]
            interval = summary["analysis"]["populations"][population]["methods"][method]["metrics"][
                "correct_delivery"
            ]["cluster_bootstrap_ci95"]
            row = {
                "population": population,
                "method": method,
                "scheduled": outcome["available"],
                "correct_delivery": outcome["correct_delivery"],
                "wrong_release": outcome["wrong_release"],
                "non_delivery": outcome["available"] - outcome["delivered"],
                "correct_delivery_ci95_low": interval[0],
                "correct_delivery_ci95_high": interval[1],
            }
            if (
                row["scheduled"] != 100
                or sum(row[key] for key in ("correct_delivery", "wrong_release", "non_delivery"))
                != 100
            ):
                raise ValueError("EXPECTED_SEPARATE_100_TASK_DENOMINATORS")
            rows.append(row)
    return rows


def render(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = baseline_rows(summary)
    colors = ("#2474A5", "#CB574D", "#DCE1E8")
    keys = ("correct_delivery", "wrong_release", "non_delivery")
    labels = ("Correct SQL delivery", "Wrong SQL release", "Non-delivery")
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), sharey=True)
    titles = (
        "Public: 100 questions / 6 databases",
        "Controlled: 100 trajectories / 18 requirements",
    )
    for index, (axis, population, title) in enumerate(zip(axes, ("public", "controlled"), titles)):
        selected = [row for row in rows if row["population"] == population]
        left = [0] * 4
        for key, color, label in zip(keys, colors, labels):
            values = [row[key] for row in selected]
            axis.barh(range(4), values, left=left, height=0.58, color=color, label=label)
            for position, (start, value) in enumerate(zip(left, values)):
                if value >= 3:
                    axis.text(
                        start + value / 2,
                        position,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#17202A" if key == "non_delivery" else "white",
                    )
            left = [a + b for a, b in zip(left, values)]
        for position, row in enumerate(selected):
            small = [
                f"{row[key]} {label}"
                for key, label in (("wrong_release", "wrong"), ("non_delivery", "not delivered"))
                if 0 < row[key] < 3
            ]
            if small:
                axis.text(
                    100,
                    position + 0.34,
                    "; ".join(small),
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#465362",
                )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18)
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.set_yticks(range(4), LABELS)
        axis.tick_params(axis="both", length=0, labelsize=10)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if index == 0:
            axis.invert_yaxis()
    figure.suptitle(
        "Correct SQL delivery, wrong release and non-delivery",
        x=0.05,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.05,
        0.055,
        "Counts equal percentages within each separate population. Non-delivery includes clarification, invalid references and failures.",
        fontsize=9,
        color="#465362",
    )
    figure.text(
        0.05,
        0.018,
        "Frozen Qwen3.8-flash run, 2026-11-07. Public: original-DB execution match. Controlled: 25-dataset oracle; authored fault mix.",
        fontsize=9,
        color="#465362",
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.25, wspace=0.12)
    figure.savefig(output / "expanded_delivery.png", dpi=200, facecolor="white")
    figure.savefig(output / "expanded_delivery.svg", facecolor="white", metadata={"Date": None})
    plt.close(figure)
    with (output / "expanded_baselines.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audit", type=Path)
    source.add_argument("--summary", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/experiments/expanded-results")
    )
    args = parser.parse_args()
    summary = public_projection(args.audit) if args.audit else read(args.summary)
    if summary.get("audit_status") != "complete":
        raise ValueError("COMPLETED_PUBLIC_SUMMARY_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render(summary, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "baseline_rows": 8,
                "main_cells": summary["coverage"]["main"],
                "stability_cells": summary["coverage"]["stability"],
            }
        )
    )


if __name__ == "__main__":
    main()
