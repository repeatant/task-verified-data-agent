"""Explicit business semantics for the deliberately small offline experiment.

These declarations come from the experiment's metric definitions. The Agent must
not infer them from a column name, and the checker must not let model output
change them. Adding a new metric requires an explicit, reviewed contract.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricContract:
    metric_id: str
    source_table: str
    value_column: str
    grain_column: str
    description: str
    aggregation: str = "sum"
    scope: str = "all_source_rows"


@dataclass(frozen=True)
class RelationContract:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    cardinality: str


TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "orders": ("order_id", "total"),
    "order_lines": ("line_id", "order_id", "amount"),
}

METRIC_CATALOG: dict[str, MetricContract] = {
    "order_revenue": MetricContract(
        metric_id="order_revenue",
        source_table="orders",
        value_column="total",
        grain_column="order_id",
        description="订单收入：每个订单累计一次订单总额；包含所有订单。",
    ),
    "line_revenue": MetricContract(
        metric_id="line_revenue",
        source_table="order_lines",
        value_column="amount",
        grain_column="line_id",
        description="明细收入：每条订单明细累计一次明细金额；包含所有明细。",
    ),
}

RELATIONS: tuple[RelationContract, ...] = (
    RelationContract("orders", "order_id", "order_lines", "order_id", "one_to_many"),
)


def describe_contracts() -> dict:
    """Return JSON-safe metadata for inspect_schema and experiment reports."""
    return {
        "tables": {name: {"columns": list(columns)} for name, columns in TABLE_COLUMNS.items()},
        "metrics": {name: asdict(metric) for name, metric in METRIC_CATALOG.items()},
        "relations": [asdict(relation) for relation in RELATIONS],
        "supported_sql": {
            "projection": "exactly one SUM(column), optionally aliased",
            "source": "one registered table or one registered INNER JOIN",
            "scope": "all source rows; no WHERE, GROUP BY, CTE, subquery, UNION or window",
            "distinct": "executed for diagnosis but rejected by row-grain SUM contracts",
        },
    }
