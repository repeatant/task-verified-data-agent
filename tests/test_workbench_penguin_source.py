import hashlib
import sqlite3
from contextlib import closing

import pytest

from dataverify.workbench import penguin_source
from dataverify.workbench.sample_source import bind_source, get_source


def test_public_dataset_carries_its_source_and_exact_snapshot():
    catalog = get_source("palmer-penguins")
    assert catalog["attribution"]["license"] == "CC0"
    assert catalog["attribution"]["url"] == "https://allisonhorst.github.io/palmerpenguins/"
    assert catalog["schema"] == {"penguins": list(penguin_source.COLUMNS)}
    assert hashlib.sha256(penguin_source.CSV.read_bytes()).hexdigest() == (
        "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
    )


def test_full_source_preserves_species_and_missing_measurements(tmp_path):
    source = bind_source("palmer-penguins", tmp_path)
    with closing(source.connect()) as connection:
        assert connection.execute(
            "SELECT COUNT(*), COUNT(body_mass_g), COUNT(sex), COUNT(DISTINCT island), "
            "MIN(year), MAX(year) FROM penguins"
        ).fetchone() == (344, 342, 333, 3, 2007, 2009)
        assert connection.execute(
            "SELECT species, COUNT(*), COUNT(body_mass_g), ROUND(AVG(body_mass_g), 2) "
            "FROM penguins GROUP BY species ORDER BY species"
        ).fetchall() == [
            ("Adelie", 152, 151, 3700.66),
            ("Chinstrap", 68, 68, 3733.09),
            ("Gentoo", 124, 123, 5076.02),
        ]
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("UPDATE penguins SET body_mass_g = 0")
    assert bind_source("palmer-penguins", tmp_path).path == source.path


def test_resume_detects_changes_even_beyond_the_preview(tmp_path):
    source = bind_source("palmer-penguins", tmp_path)
    with closing(sqlite3.connect(source.path)) as connection, connection:
        connection.execute("UPDATE penguins SET body_mass_g = 10 WHERE rowid = 340")

    with pytest.raises(ValueError, match="PENGUINS_STORED_SOURCE_CHANGED"):
        bind_source("palmer-penguins", tmp_path)
