"""Build a read-only workbench source from the pinned Palmer Penguins CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from dataverify.verification.relational_data import SQLiteSource

DATASET = "palmer-penguins"
SPECIES_COUNTS = "分别统计每个物种的观测记录数，按物种名称升序排列。"
MASS_COVERAGE = "统计有体重测量值的观测记录数。"
AMBIGUOUS_COUNT = "统计可用于分析的观测记录数。"
DEFINITIONS = (
    "penguins 每行是一条实地观测。species 是物种，island 是岛屿，year 是采样年份。"
    "bill_length_mm、bill_depth_mm、flipper_length_mm 的单位为毫米，body_mass_g 的单位为克。"
    "数据包中的 NA 已转为 SQL NULL。观测总数包含有缺失测量的行；"
    "针对特定测量的均值和有效样本数只计入该列非空的记录。"
)
CSV = Path(__file__).with_name("data") / "palmer-penguins.csv"
CSV_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
COLUMNS = {
    "species": "TEXT",
    "island": "TEXT",
    "bill_length_mm": "REAL",
    "bill_depth_mm": "REAL",
    "flipper_length_mm": "INTEGER",
    "body_mass_g": "INTEGER",
    "sex": "TEXT",
    "year": "INTEGER",
}


def catalog_entry() -> dict:
    return {
        "id": DATASET,
        "name": "Palmer 企鹅观测",
        "description": "2007–2009 年在南极帕尔默群岛采集的 344 条实地观测，涵盖三个岛屿和三个企鹅物种。",
        "attribution": {
            "title": "Palmer Station LTER · palmerpenguins",
            "url": "https://allisonhorst.github.io/palmerpenguins/",
            "license": "CC0",
        },
        "schema": {"penguins": list(COLUMNS)},
        "definitions": DEFINITIONS,
        "examples": [SPECIES_COUNTS, MASS_COVERAGE, AMBIGUOUS_COUNT],
    }


def _source_rows() -> list[tuple]:
    raw = CSV.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("PENGUINS_DATA_DIGEST_MISMATCH")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != list(COLUMNS):
        raise ValueError("PENGUINS_CSV_SCHEMA_MISMATCH")
    convert = {"TEXT": str, "INTEGER": int, "REAL": float}
    return [
        tuple(
            None if row[column] == "NA" else convert[affinity](row[column])
            for column, affinity in COLUMNS.items()
        )
        for row in reader
    ]


def bind(directory: Path) -> SQLiteSource:
    """Keep each run linked to the full published records, including after resume."""
    rows = _source_rows()
    path = directory / "palmer-penguins.sqlite3"
    if not path.exists():
        with TemporaryDirectory(dir=directory) as staging:
            tmp = Path(staging) / path.name
            with closing(sqlite3.connect(tmp)) as db, db:
                columns = ", ".join(f"{name} {affinity}" for name, affinity in COLUMNS.items())
                db.execute(f"CREATE TABLE penguins ({columns})")
                db.executemany("INSERT INTO penguins VALUES (?,?,?,?,?,?,?,?)", rows)
            tmp.replace(path)
    source = SQLiteSource.from_path(path)
    with closing(source.connect()) as db:
        if (
            tuple(source.schema.get("penguins", ())) != tuple(COLUMNS)
            or set(source.schema) != {"penguins"}
            or tuple(source.column_types["penguins"]) != tuple(COLUMNS.values())
            or db.execute("SELECT * FROM penguins ORDER BY rowid").fetchall() != rows
        ):
            raise ValueError("PENGUINS_STORED_SOURCE_CHANGED")
    return source
