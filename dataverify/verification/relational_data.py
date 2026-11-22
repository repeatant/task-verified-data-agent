"""Typed, bounded relational fixtures and trusted read-only database handles.

Model output may supply rows, never database paths or executable DDL. SQLiteSource
is created by the evaluation harness and keeps full-database scoring separate
from the bounded preview used during investigation.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


def quote_identifier(value: str) -> str:
    """Quote a validated identifier; identifier quoting is not SQL-value escaping."""
    if type(value) is not str or not value or len(value) > 200 or "\x00" in value:
        raise ValueError("RELATIONAL_IDENTIFIER")
    return '"' + value.replace('"', '""') + '"'


def normalize_schema(schema: dict) -> dict[str, tuple[str, ...]]:
    if not isinstance(schema, dict) or not 1 <= len(schema) <= 64:
        raise ValueError("RELATIONAL_SCHEMA")
    result = {}
    for table, columns in schema.items():
        quote_identifier(table)
        if table.lower().startswith("sqlite_"):
            raise ValueError("RELATIONAL_RESERVED_TABLE")
        if not isinstance(columns, (tuple, list)) or not 1 <= len(columns) <= 128:
            raise ValueError("RELATIONAL_COLUMNS")
        for column in columns:
            quote_identifier(column)
        if len({column.casefold() for column in columns}) != len(columns):
            raise ValueError("RELATIONAL_DUPLICATE_COLUMNS")
        result[table] = tuple(columns)
    if len({name.casefold() for name in result}) != len(result):
        raise ValueError("RELATIONAL_DUPLICATE_TABLES")
    return result


def sqlite_affinity(declared_type: str) -> str:
    """Apply SQLite's documented five-affinity rules to existing column metadata."""
    name = declared_type.upper()
    if "INT" in name:
        return "INTEGER"
    if any(part in name for part in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if not name or "BLOB" in name:
        return "BLOB"
    if any(part in name for part in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def normalize_types(schema: dict, column_types: dict | None) -> dict:
    if column_types is None:
        # BLOB affinity means no coercion, not permission to accept byte payloads.
        return {table: ("BLOB",) * len(columns) for table, columns in schema.items()}
    if not isinstance(column_types, dict) or set(column_types) != set(schema):
        raise ValueError("RELATIONAL_COLUMN_TYPES")
    result = {}
    for table, columns in schema.items():
        values = column_types[table]
        if not isinstance(values, (list, tuple)) or len(values) != len(columns):
            raise ValueError("RELATIONAL_COLUMN_TYPES")
        if any(
            type(v) is not str or v not in {"INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC"}
            for v in values
        ):
            raise ValueError("RELATIONAL_COLUMN_TYPES")
        result[table] = tuple(values)
    return result


def _scalar(value):
    if type(value) not in (int, float, str, type(None)):
        raise ValueError("RELATIONAL_SCALAR")
    if type(value) in (int, float) and (abs(value) > 10**15 or not math.isfinite(value)):
        raise ValueError("RELATIONAL_NUMBER")
    if isinstance(value, str) and (len(value) > 4000 or "\x00" in value):
        raise ValueError("RELATIONAL_STRING")
    return value


@dataclass(frozen=True)
class RelationalData:
    """Full schema with an explicit bounded set of rows; not a full-DB assertion."""

    schema: dict
    tables: dict
    column_types: dict | None = None
    primary_keys: dict | None = None
    foreign_keys: tuple | list = ()

    def __post_init__(self):
        schema = normalize_schema(self.schema)
        types = normalize_types(schema, self.column_types)
        if not isinstance(self.tables, dict) or set(self.tables) != set(schema):
            raise ValueError("RELATIONAL_TABLES")
        if any(not isinstance(rows, (list, tuple)) for rows in self.tables.values()):
            raise ValueError("RELATIONAL_ROWS")
        if sum(len(rows) for rows in self.tables.values()) > 5000:
            raise ValueError("RELATIONAL_SOURCE_ROW_LIMIT")
        tables = {}
        for table, columns in schema.items():
            converted = []
            for row in self.tables[table]:
                if isinstance(row, dict):
                    if set(row) != set(columns):
                        raise ValueError("RELATIONAL_ROW_COLUMNS")
                    row = [row[column] for column in columns]
                if not isinstance(row, (list, tuple)) or len(row) != len(columns):
                    raise ValueError("RELATIONAL_ROW_COLUMNS")
                values = tuple(_scalar(value) for value in row)
                converted.append(values)
            tables[table] = tuple(converted)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "column_types", types)
        object.__setattr__(self, "tables", tables)
        # SQLite is dynamically typed: e.g. a DATETIME/NUMERIC column may store
        # ISO text. Apply the engine's actual affinity rules once, then give the
        # reference interpreter exactly those storage values. No Python cast or
        # guessed parsing silently changes dates or mixed-type business values.
        with closing(self.connect()) as connection:
            tables = {
                table: tuple(
                    connection.execute(f"SELECT * FROM {quote_identifier(table)}").fetchall()
                )
                for table in schema
            }
        object.__setattr__(self, "tables", tables)
        primary_keys, foreign_keys = normalize_keys(schema, self.primary_keys, self.foreign_keys)
        named = self.named_tables()
        for table, columns in primary_keys.items():
            keys = [tuple(row[column] for column in columns) for row in named[table]]
            if any(None in key for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("RELATIONAL_PRIMARY_KEY")
        for link in foreign_keys:
            target = {
                tuple(row[column] for column in link["reference_columns"])
                for row in named[link["references"]]
            }
            for row in named[link["table"]]:
                key = tuple(row[column] for column in link["columns"])
                if None not in key and key not in target:
                    raise ValueError("RELATIONAL_FOREIGN_KEY")
        object.__setattr__(self, "primary_keys", primary_keys)
        object.__setattr__(self, "foreign_keys", foreign_keys)

    def to_dict(self):
        return {
            "schema": {table: list(columns) for table, columns in self.schema.items()},
            "column_types": {table: list(types) for table, types in self.column_types.items()},
            "tables": {table: [list(row) for row in rows] for table, rows in self.tables.items()},
            "primary_keys": {table: list(columns) for table, columns in self.primary_keys.items()},
            "foreign_keys": [dict(link) for link in self.foreign_keys],
        }

    def named_tables(self):
        return {
            table: [dict(zip(self.schema[table], row)) for row in rows]
            for table, rows in self.tables.items()
        }

    def connect(self):
        connection = sqlite3.connect(":memory:")
        try:
            for table, columns in self.schema.items():
                declarations = ",".join(
                    f"{quote_identifier(column)} {affinity}"
                    for column, affinity in zip(columns, self.column_types[table])
                )
                connection.execute(f"CREATE TABLE {quote_identifier(table)} ({declarations})")
                placeholders = ",".join("?" for _ in columns)
                connection.executemany(
                    f"INSERT INTO {quote_identifier(table)} VALUES ({placeholders})",
                    self.tables[table],
                )
            connection.commit()
            return connection
        except BaseException:
            connection.close()
            raise


@dataclass(frozen=True)
class SQLiteSource:
    """Harness-created original database; never constructed from model JSON."""

    path: Path
    schema: dict
    column_types: dict
    primary_keys: dict | None = None
    foreign_keys: tuple | list = ()

    def __post_init__(self):
        path = Path(self.path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("SQLITE_SOURCE_FILE")
        schema = normalize_schema(self.schema)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "column_types", normalize_types(schema, self.column_types))
        primary_keys, foreign_keys = normalize_keys(schema, self.primary_keys, self.foreign_keys)
        object.__setattr__(self, "primary_keys", primary_keys)
        object.__setattr__(self, "foreign_keys", foreign_keys)

    def connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.enable_load_extension(False)
        return connection

    @classmethod
    def from_path(cls, path: Path):
        path = Path(path).resolve(strict=True)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA trusted_schema=OFF")
            tables = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            if any("CREATE VIRTUAL TABLE" in (ddl or "").upper() for _, ddl in tables):
                raise ValueError("SQLITE_VIRTUAL_TABLE_UNSUPPORTED")
            schema, types, primary_keys, raw_links = {}, {}, {}, []
            for table, _ in tables:
                columns = connection.execute(
                    f"PRAGMA table_info({quote_identifier(table)})"
                ).fetchall()
                schema[table] = [column[1] for column in columns]
                types[table] = [sqlite_affinity(column[2]) for column in columns]
                keys = [column[1] for column in sorted(columns, key=lambda c: c[5]) if column[5]]
                if keys:
                    primary_keys[table] = keys
                links = connection.execute(
                    f"PRAGMA foreign_key_list({quote_identifier(table)})"
                ).fetchall()
                grouped = {}
                for link in links:
                    grouped.setdefault(link[0], []).append(link)
                for group in grouped.values():
                    group.sort(key=lambda item: item[1])
                    raw_links.append(
                        {
                            "table": table,
                            "columns": [item[3] for item in group],
                            "references": group[0][2],
                            "reference_columns": [item[4] for item in group],
                        }
                    )
            for link in raw_links:
                table_names = {table.casefold(): table for table in schema}
                link["references"] = table_names.get(
                    link["references"].casefold(), link["references"]
                )
                if link["references"] in schema:
                    column_names = {
                        column.casefold(): column for column in schema[link["references"]]
                    }
                    link["reference_columns"] = [
                        column_names.get(column.casefold(), column) if column is not None else None
                        for column in link["reference_columns"]
                    ]
                if all(column is None for column in link["reference_columns"]):
                    link["reference_columns"] = primary_keys.get(link["references"], [])
        return cls(path, schema, types, primary_keys, raw_links)

    def preview(
        self, rows_per_table: int = 10, *, max_chars: int = 12_000, retain_parents: bool = False
    ) -> RelationalData:
        """Bounded real rows; optional FK completion uses no task or answer data.

        Parent retention adds only actual referenced rows, up to 50 per table,
        500 in total and the shared character budget. Remaining orphans are
        dropped. The result remains a preview, never a full-database assertion.
        """
        if type(rows_per_table) is not int or not 0 <= rows_per_table <= 50:
            raise ValueError("SQLITE_PREVIEW_LIMIT")
        if type(max_chars) is not int or not 0 <= max_chars <= 100_000:
            raise ValueError("SQLITE_PREVIEW_CHAR_LIMIT")
        if type(retain_parents) is not bool:
            raise ValueError("SQLITE_PREVIEW_PARENT_POLICY")
        with closing(self.connect()) as connection:
            tables, remaining, count = {table: [] for table in self.schema}, max_chars, 0

            def append(table, row):
                nonlocal remaining, count
                if len(tables[table]) >= 50 or count >= (500 if retain_parents else 5000):
                    return False
                # Do not cut strings or partially represent a source row.
                try:
                    for value in row:
                        _scalar(value)
                    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                except (TypeError, ValueError):
                    return False
                if len(encoded) > remaining:
                    return False
                tables[table].append(row)
                remaining -= len(encoded)
                count += 1
                return True

            for table in self.schema:
                order = self.primary_keys.get(table) or self.schema[table]
                order_sql = ",".join(quote_identifier(column) for column in order)
                rows = connection.execute(
                    f"SELECT * FROM {quote_identifier(table)} ORDER BY {order_sql} LIMIT ?",
                    (rows_per_table,),
                ).fetchall()
                for row in rows:
                    append(table, row)
            if retain_parents:
                self._retain_preview_parents(connection, tables, append)
        # Independent LIMITs need not preserve FK closure. Drop orphan child rows
        # to a fixed point; no synthetic values or missing parent rows are invented.
        changed = True
        while changed:
            changed = False
            for link in self.foreign_keys:
                target_indexes = [
                    self.schema[link["references"]].index(c) for c in link["reference_columns"]
                ]
                indexes = [self.schema[link["table"]].index(c) for c in link["columns"]]
                targets = {
                    tuple(row[i] for i in target_indexes) for row in tables[link["references"]]
                }
                old = tables[link["table"]]
                tables[link["table"]] = [
                    row
                    for row in old
                    if None in tuple(row[i] for i in indexes)
                    or tuple(row[i] for i in indexes) in targets
                ]
                changed |= len(old) != len(tables[link["table"]])
        return RelationalData(
            self.schema, tables, self.column_types, self.primary_keys, self.foreign_keys
        )

    def _retain_preview_parents(self, connection, tables, append):
        """Follow real foreign keys to a bounded fixed point, including cycles."""
        queried, changed = set(), True
        while changed and len(queried) < 500:
            changed = False
            for link in self.foreign_keys:
                parent = link["references"]
                target_columns = tuple(link["reference_columns"])
                target_indexes = [self.schema[parent].index(c) for c in target_columns]
                child_indexes = [self.schema[link["table"]].index(c) for c in link["columns"]]
                known = {tuple(row[i] for i in target_indexes) for row in tables[parent]}
                # Snapshot: self-referencing additions are inspected next pass.
                for row in list(tables[link["table"]]):
                    key = tuple(row[i] for i in child_indexes)
                    query_key = (parent, target_columns, key)
                    if None in key or key in known or query_key in queried:
                        continue
                    if len(queried) >= 500 or len(tables[parent]) >= 50:
                        continue
                    queried.add(query_key)
                    predicates = " AND ".join(f"{quote_identifier(c)}=?" for c in target_columns)
                    order = self.primary_keys.get(parent) or self.schema[parent]
                    order_sql = ",".join(quote_identifier(c) for c in order)
                    # Identifiers originate only from introspected schema; all
                    # source key values remain bound parameters, never SQL text.
                    found = connection.execute(
                        f"SELECT * FROM {quote_identifier(parent)} WHERE {predicates} "
                        f"ORDER BY {order_sql} LIMIT 1",
                        key,
                    ).fetchone()
                    if found is not None and append(parent, found):
                        known.add(tuple(found[i] for i in target_indexes))
                        changed = True


def normalize_keys(schema, primary_keys, foreign_keys):
    primary_keys = {} if primary_keys is None else primary_keys
    if not isinstance(primary_keys, dict):
        raise ValueError("RELATIONAL_KEY_METADATA")
    for table, columns in primary_keys.items():
        if (
            table not in schema
            or not isinstance(columns, (list, tuple))
            or not columns
            or len(set(columns)) != len(columns)
            or any(column not in schema[table] for column in columns)
        ):
            raise ValueError("RELATIONAL_KEY_METADATA")
    if not isinstance(foreign_keys, (list, tuple)) or len(foreign_keys) > 128:
        raise ValueError("RELATIONAL_KEY_METADATA")
    normalized = []
    for link in foreign_keys:
        if not isinstance(link, dict) or set(link) != {
            "table",
            "columns",
            "references",
            "reference_columns",
        }:
            raise ValueError("RELATIONAL_KEY_METADATA")
        columns, targets = link["columns"], link["reference_columns"]
        if (
            link["table"] not in schema
            or link["references"] not in schema
            or not isinstance(columns, (list, tuple))
            or not isinstance(targets, (list, tuple))
            or not columns
            or len(columns) != len(targets)
            or any(column not in schema[link["table"]] for column in columns)
            or any(column not in schema[link["references"]] for column in targets)
        ):
            raise ValueError("RELATIONAL_KEY_METADATA")
        normalized.append({**link, "columns": list(columns), "reference_columns": list(targets)})
    return {table: tuple(columns) for table, columns in primary_keys.items()}, tuple(normalized)
