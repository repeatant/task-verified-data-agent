"""Post-change boundary regressions; evaluation-only, NOT unseen templates.

Eight legal queries, eight faulty candidates and four unsupported candidates
use existing task families with new dates/statuses/data combinations. Reference
SQL and expected acceptance never enter the runtime or model context.
"""

from dataclasses import dataclass, replace

from .probe_cases import Case
from .probe_runtime import ProbeData, ProbeTask


@dataclass(frozen=True)
class CapabilityCase:
    case: Case
    expected_acceptance: bool
    category: str


def queries(task: ProbeTask) -> tuple[str, str, str]:
    period = (
        f"l.status = '{task.status}' AND l.settled_at >= '{task.start}' "
        f"AND l.settled_at < '{task.end}'"
    )
    scalar = "SELECT COALESCE(SUM(l.amount),0) FROM order_lines l WHERE " + period
    orders = (
        "SELECT o.order_id, COALESCE(SUM(l.amount),0) FROM orders o LEFT JOIN order_lines l "
        "ON o.order_id = l.order_id AND " + period + " GROUP BY o.order_id"
    )
    channels = (
        f"SELECT COALESCE(o.channel,'{task.other_label}'), SUM(l.amount) FROM orders o "
        "INNER JOIN order_lines l ON o.order_id = l.order_id WHERE "
        + period
        + f" GROUP BY COALESCE(o.channel,'{task.other_label}')"
    )
    return scalar, orders, channels


def get_capability_cases() -> tuple[CapabilityCase, ...]:
    scalar = ProbeTask(
        "v-scalar",
        "统计七月已取消明细净额，包含退款，无匹配明细时返回零。",
        ("exclude_end", "exclude_status", "include_start", "include_negative"),
        start="2026-07-01T00:00:00",
        end="2026-08-01T00:00:00",
        status="cancelled",
    )
    orders = replace(
        scalar,
        task_id="v-orders",
        output_kind="orders",
        properties=("preserve_zero_orders",),
        question="逐订单统计七月已取消明细净额，包含退款；没有有效明细的订单保留零值。",
    )
    channels = replace(
        scalar,
        task_id="v-channels",
        output_kind="channels",
        properties=("merge_other_channels",),
        other_label="未标注",
        question="按渠道统计七月已取消明细净额，包含退款；NULL渠道归入未标注并合并同名渠道。",
    )
    middle = "2026-07-16T12:00:00"
    mixed = ProbeData(
        ((101, 999, None), (102, 888, "未标注"), (103, 777, "web"), (104, 666, "offline")),
        (
            (1, 101, 100, "cancelled", scalar.start),
            (2, 101, -25, "cancelled", middle),
            (3, 102, 30, "cancelled", middle),
            (4, 102, 800, "cancelled", scalar.end),
            (5, 103, 700, "pending", middle),
        ),
    )
    empty = ProbeData((), ())
    no_match = ProbeData(((201, 100, "web"),), ((201, 201, 100, "pending", middle),))
    cancellation = ProbeData(
        ((301, 555, "web"),),
        (
            (301, 301, 41, "cancelled", middle),
            (302, 301, -41, "cancelled", middle),
            (303, 301, 0, "cancelled", middle),
        ),
    )
    a, b, c = queries(scalar)[0], queries(orders)[1], queries(channels)[2]
    entries = [
        (
            "v01",
            scalar,
            mixed,
            a,
            a,
            True,
            "legal",
            "Mixed boundary/refund/source-column mismatch.",
        ),
        (
            "v02",
            orders,
            mixed,
            b,
            b,
            True,
            "legal",
            "LEFT JOIN retains no-match and no-child orders.",
        ),
        (
            "v03",
            channels,
            mixed,
            c,
            c,
            True,
            "legal",
            "NULL and literal fallback channel collide and must merge.",
        ),
        ("v04", scalar, empty, a, a, True, "legal", "Completely empty scalar source."),
        ("v05", scalar, no_match, a, a, True, "legal", "No eligible child rows."),
        ("v06", orders, empty, b, b, True, "legal", "An empty grouped relation is valid."),
        (
            "v07",
            channels,
            cancellation,
            c,
            c,
            True,
            "legal",
            "A channel has a real zero net amount.",
        ),
        (
            "v08",
            scalar,
            cancellation,
            a,
            a,
            True,
            "legal",
            "Positive/refund/zero lines cancel to zero.",
        ),
        (
            "v09",
            scalar,
            mixed,
            a.replace("l.settled_at <", "l.settled_at <="),
            a,
            False,
            "task_fault",
            "Includes excluded end.",
        ),
        (
            "v10",
            scalar,
            mixed,
            a.replace("l.settled_at >=", "l.settled_at >"),
            a,
            False,
            "task_fault",
            "Excludes included start.",
        ),
        (
            "v11",
            scalar,
            mixed,
            a.replace("l.status = 'cancelled' AND ", ""),
            a,
            False,
            "task_fault",
            "Drops status requirement.",
        ),
        (
            "v12",
            scalar,
            mixed,
            a + " AND l.amount > 0",
            a,
            False,
            "task_fault",
            "Drops valid refunds.",
        ),
        (
            "v13",
            orders,
            mixed,
            b.replace(" AND l.status", " WHERE l.status", 1),
            b,
            False,
            "task_fault",
            "WHERE loses zero orders.",
        ),
        (
            "v14",
            channels,
            mixed,
            c.replace("GROUP BY COALESCE(o.channel,'未标注')", "GROUP BY o.channel"),
            c,
            False,
            "task_fault",
            "Duplicate displayed groups.",
        ),
        (
            "v15",
            scalar,
            empty,
            a.replace("COALESCE(SUM(l.amount),0)", "SUM(l.amount)"),
            a,
            False,
            "result_contract",
            "NULL scalar output.",
        ),
        (
            "v16",
            scalar,
            no_match,
            a.replace("COALESCE(SUM(l.amount),0)", "SUM(l.amount)"),
            a,
            False,
            "result_contract",
            "NULL despite nonempty source.",
        ),
        (
            "v17",
            scalar,
            mixed,
            a.replace("SUM(l.amount)", "SUM(CASE WHEN l.amount > 0 THEN l.amount ELSE 0 END)"),
            a,
            False,
            "unsupported_candidate",
            "CASE aggregate is outside grammar.",
        ),
        (
            "v18",
            scalar,
            mixed,
            "WITH eligible AS (" + a + ") SELECT * FROM eligible",
            a,
            False,
            "unsupported_candidate",
            "CTE outside grammar.",
        ),
        (
            "v19",
            orders,
            mixed,
            b + " ORDER BY o.order_id LIMIT 2",
            b,
            False,
            "unsupported_candidate",
            "Ordering/limiting outside grammar.",
        ),
        (
            "v20",
            scalar,
            mixed,
            "DELETE FROM order_lines",
            a,
            False,
            "unsupported_candidate",
            "Write operation is forbidden.",
        ),
    ]
    return tuple(
        CapabilityCase(
            Case(case_id, task, source, candidate, reference, None if accepted else category, note),
            accepted,
            category,
        )
        for case_id, task, source, candidate, reference, accepted, category, note in entries
    )
