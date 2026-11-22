"""Pinned public SQL benchmark preparation and gold-isolated execution scoring.

Model prompts are constructed with ``public_input`` only. Annotations, gold SQL,
selection reasons and expected rows belong to the evaluator, never an Agent.
The original SQLite file is used for scoring; previews are explicitly incomplete.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp


@dataclass(frozen=True)
class SupportPolicy:
    """Technical inclusion rules fixed before observing any model output."""

    max_tables: int = 8
    max_columns: int = 120
    max_sql_nodes: int = 500
    max_schema_chars: int = 16000
    preview_rows_per_table: int = 10
    preview_chars: int = 12000
    max_result_rows: int = 200
    max_result_columns: int = 8


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _connect(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def inspect_database(path: Path, policy: SupportPolicy = SupportPolicy()) -> dict:
    """Read every user table's schema, without using a question or gold SQL.

    Rows are small deterministic previews, not replacement evaluation databases.
    Oversized and BLOB-containing rows are omitted rather than misrepresented.
    """
    schema, types, preview, constraints, omitted = {}, {}, {}, {}, {}
    remaining = policy.preview_chars
    with closing(_connect(path)) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (table,) in tables:
            columns = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
            schema[table] = [column[1] for column in columns]
            types[table] = [column[2] or "BLOB" for column in columns]
            primary = [column[1] for column in sorted(columns, key=lambda c: c[5]) if column[5]]
            constraints[table] = {
                "primary_key": primary,
                "not_null": [column[1] for column in columns if column[3]],
                "foreign_keys": [
                    {"column": row[3], "table": row[2], "target_column": row[4]}
                    for row in connection.execute(f"PRAGMA foreign_key_list({_quote(table)})")
                ],
            }
            order = primary or schema[table]
            sql = f"SELECT * FROM {_quote(table)} ORDER BY "
            sql += ",".join(_quote(column) for column in order) + " LIMIT ?"
            preview[table], omitted[table] = [], 0
            for row in connection.execute(sql, (policy.preview_rows_per_table,)):
                if any(isinstance(value, bytes) for value in row):
                    omitted[table] += 1
                    continue
                encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                if len(encoded) > remaining or any(
                    isinstance(value, str) and len(value) > 1000 for value in row
                ):
                    omitted[table] += 1
                    continue
                preview[table].append(list(row))
                remaining -= len(encoded)
    return {
        "schema": schema,
        "column_types": types,
        "constraints": constraints,
        "tables": preview,
        "preview_notice": "Incomplete deterministic schema-independent preview; "
        "final SQL is scored on the original full SQLite database.",
        "preview_omitted_rows": omitted,
    }


def support_reasons(annotation: dict, database: dict, policy: SupportPolicy) -> list[str]:
    """Gold syntax classifies support only; never selects inputs given to models."""
    reasons = []
    schema = database["schema"]
    if len(schema) > policy.max_tables:
        reasons.append("table_count")
    if sum(map(len, schema.values())) > policy.max_columns:
        reasons.append("column_count")
    if len(json.dumps(schema, ensure_ascii=False)) > policy.max_schema_chars:
        reasons.append("schema_chars")
    try:
        trees = sqlglot.parse(annotation["SQL"], read="sqlite")
        tree = trees[0]
        if len(trees) != 1 or not isinstance(tree, exp.Select):
            reasons.append("select_only")
        if tree.find(exp.With, exp.Window, exp.Union, exp.Intersect, exp.Except, exp.Into):
            reasons.append("unsupported_sql_construct")
        if len(list(tree.walk())) > policy.max_sql_nodes:
            reasons.append("sql_node_count")
        allowed = {
            "SUM",
            "AVG",
            "COUNT",
            "MIN",
            "MAX",
            "COALESCE",
            "NULLIF",
            "ABS",
            "ROUND",
            "AND",
            "OR",
            "CAST",
            "CASE",
            "IF",
            "EXISTS",
            "SUBSTRING",
            "STR_POSITION",
            "TIME_TO_STR",
            "TS_OR_DS_TO_TIMESTAMP",
            "REPLACE",
            "TRIM",
            "LENGTH",
            "LOWER",
            "UPPER",
            "DATE",
        }
        if any(function.sql_name() not in allowed for function in tree.find_all(exp.Func)):
            reasons.append("unsupported_sql_function")
        if any(
            table.name not in schema or table.db or table.catalog
            for table in tree.find_all(exp.Table)
        ):
            reasons.append("unknown_table")
    except (sqlglot.errors.ParseError, IndexError, AttributeError):
        reasons.append("sql_parse")
    return sorted(set(reasons))


def execute_original(
    sql: str, database_path: Path, *, timeout: float = 10, max_rows: int = 10000
) -> list:
    """Run one SELECT against the immutable original file with bounded resources.

    This scorer permits SQL constructs beyond the Agent's declared support so
    that eligibility is not silently conflated with benchmark correctness.
    SQLite's authorizer and read-only connection still reject writes and I/O.
    """
    trees = sqlglot.parse(sql, read="sqlite")
    if len(trees) != 1 or not isinstance(
        trees[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)
    ):
        raise ValueError("SCORER_READONLY_SELECT_REQUIRED")
    with closing(_connect(database_path)) as connection:
        connection.enable_load_extension(False)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        start = time.monotonic()
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
        unsafe_functions = {"load_extension", "readfile", "writefile", "eval", "fts3_tokenizer"}

        def authorize(action, arg1, arg2, database, trigger):
            if action in allowed:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in unsafe_functions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 1000)
        try:
            rows = connection.execute(sql).fetchmany(max_rows + 1)
        except sqlite3.Error as error:
            raise ValueError(f"SCORER_SQL_ERROR:{error}") from None
        if len(rows) > max_rows:
            raise ValueError("SCORER_RESULT_ROW_LIMIT")
        return [list(row) for row in rows]


def compare_results(predicted: list, gold: list) -> dict:
    """BIRD's official EX uses sets; strict bag/order diagnostics stay separate."""
    predicted_rows, gold_rows = list(map(tuple, predicted)), list(map(tuple, gold))
    return {
        "bird_execution_match": set(predicted_rows) == set(gold_rows),
        "bag_match": Counter(predicted_rows) == Counter(gold_rows),
        "ordered_match": predicted_rows == gold_rows,
    }


def public_input(annotation: dict, database: dict) -> dict:
    """Explicit allowlist prevents gold, difficulty and selection labels leaking."""
    return {
        "task_id": f"bird-{annotation['question_id']}",
        "question": annotation["question"],
        "definitions": annotation.get("evidence", ""),
        **{
            key: database[key]
            for key in ("schema", "column_types", "constraints", "tables", "preview_notice")
        },
    }


def stratified_ids(records: list[dict], count: int, seed: int) -> list[int]:
    """Round-robin databases then seeded task order; never uses model outcomes."""
    groups = defaultdict(list)
    for record in records:
        groups[record["db_id"]].append(record["question_id"])
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort()
        rng.shuffle(groups[key])
    selected = []
    while len(selected) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def partition_records(records: list[dict], test_count: int, dev_count: int, seed: int) -> dict:
    """Prefer two development databases and at least five held-out test databases.

    Choose a seeded pair using only eligible counts, before any model output.
    If the population cannot support this design, disclose question-only split.
    """
    names = sorted({record["db_id"] for record in records})
    pairs = list(combinations(names, 2))
    random.Random(seed).shuffle(pairs)
    for pair in pairs:
        dev = [row for row in records if row["db_id"] in pair]
        test = [row for row in records if row["db_id"] not in pair]
        if len(dev) >= dev_count and len(test) >= test_count and len(names) - 2 >= 5:
            return {
                "test_ids": stratified_ids(test, test_count, seed),
                "development_ids": stratified_ids(dev, dev_count, seed + 1),
                "split_claim": "Database-disjoint development/test within the declared supported subset.",
                "development_databases": list(pair),
                "test_databases": [name for name in names if name not in pair],
            }
    test_ids = stratified_ids(records, test_count, seed)
    remaining = [row for row in records if row["question_id"] not in test_ids]
    dev_ids = stratified_ids(remaining, dev_count, seed + 1)
    return {
        "test_ids": test_ids,
        "development_ids": dev_ids,
        "split_claim": "Question-disjoint, schema-shared; no unseen-database claim.",
        "development_databases": sorted(
            {row["db_id"] for row in records if row["question_id"] in dev_ids}
        ),
        "test_databases": sorted(
            {row["db_id"] for row in records if row["question_id"] in test_ids}
        ),
    }


def prepare_manifest(
    data_root: Path, *, test_count: int = 100, dev_count: int = 20, seed: int = 20261107
) -> dict:
    """Prepare explicit eligibility and split before any model predictions.

    Prefer database-disjoint development/test; the manifest discloses a fallback
    to disjoint questions when the supported population is insufficient.
    """
    data_root = Path(data_root).resolve()
    annotations = data_root / "mini_dev_sqlite.jsonl"
    records = [json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines()]
    if len({record["question_id"] for record in records}) != len(records):
        raise ValueError("DUPLICATE_QUESTION_ID")
    policy = SupportPolicy()
    databases, available = {}, {}
    for name in sorted({record["db_id"] for record in records}):
        matches = sorted(data_root.rglob(name + ".sqlite"))
        if len(matches) != 1:
            raise ValueError(f"DATABASE_PATH_NOT_UNIQUE:{name}:{len(matches)}")
        path = matches[0]
        available[name] = path
        databases[name] = {
            "path": str(path),
            "sha256": file_digest(path),
            **inspect_database(path, policy),
        }
    audited, eligible = [], []
    for record in records:
        reasons = support_reasons(record, databases[record["db_id"]], policy)
        scoring = {}
        if not reasons:
            try:
                rows = execute_original(record["SQL"], available[record["db_id"]])
                if len(rows) > policy.max_result_rows:
                    reasons.append("result_rows")
                if any(len(row) > policy.max_result_columns for row in rows):
                    reasons.append("result_columns")
                scoring = {"gold_rows": rows, "gold_sql": record["SQL"]}
            except (ValueError, sqlglot.errors.ParseError) as error:
                reasons.append("gold_execution_error")
                scoring = {"gold_execution_error": str(error)}
        audited.append(
            {
                "question_id": record["question_id"],
                "db_id": record["db_id"],
                "reasons": reasons,
                **scoring,
            }
        )
        if not reasons:
            eligible.append(record)
    split = partition_records(eligible, test_count, dev_count, seed)
    test_ids, dev_ids = split["test_ids"], split["development_ids"]
    selected = set(test_ids + dev_ids)
    return {
        "format": "dataverify-bird-public-v1",
        "seed": seed,
        "policy": asdict(policy),
        "annotations_sha256": file_digest(annotations),
        "upstream": json.loads((data_root / "provenance.json").read_text(encoding="utf-8")),
        "total": len(records),
        "eligible": len(eligible),
        "eligible_by_database": dict(Counter(row["db_id"] for row in eligible)),
        "excluded_reasons": dict(Counter(reason for row in audited for reason in row["reasons"])),
        **split,
        "databases": databases,
        "eligibility": audited,
        "model_inputs": {
            str(row["question_id"]): public_input(row, databases[row["db_id"]])
            for row in records
            if row["question_id"] in selected
        },
    }
