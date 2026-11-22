import csv
import io
from collections import Counter
from hashlib import sha256

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from dataverify.workbench.app import PREFIX, create_app, csv_result


def test_download_preserves_numeric_types_and_escapes_spreadsheet_text():
    row = {"=header": -3, "name": " =SUM(1,1)", "nullable": None, "notes": '测量,"一"\n次'}
    run = {"id": "test", "status": "reviewed", "result": {"columns": list(row), "rows": [row]}}
    response = csv_result(run)
    assert list(csv.reader(io.StringIO(response.body.decode("utf-8-sig")))) == [
        ["'=header", "name", "nullable", "notes"],
        ["-3", "' =SUM(1,1)", "", row["notes"]],
    ]
    with pytest.raises(HTTPException) as exc:
        csv_result({**run, "status": "inconclusive"})
    assert exc.value.status_code == 409


def test_preview_is_bounded_and_exports_are_read_only(tmp_path):
    with TestClient(create_app(tmp_path, get_qwen_status=lambda: (False, "offline"))) as client:
        url = f"{PREFIX}/sources"
        assert client.get(f"{url}/unknown/preview").status_code == 404
        sample = client.get(f"{url}/demo-orders/preview").json()["tables"]
        assert len(sample["order_lines"]) == 5
        assert len(sample["orders"]) == 3
        export = client.get(f"{PREFIX}/replays/duplicate-amount/result.csv")
        assert export.status_code == 200 and "attachment" in export.headers["Content-Disposition"]
        assert client.get(f"{PREFIX}/runs").json() == {"runs": []}


def test_recorded_qwen_result_matches_upstream_csv_and_can_be_downloaded(tmp_path):
    from dataverify.verification.sql_tools import SQLTools
    from dataverify.workbench.penguin_source import CSV, bind

    with TestClient(create_app(tmp_path, get_qwen_status=lambda: (False, "offline"))) as client:
        path = f"{PREFIX}/replays/penguins-2009"
        replay = client.get(path).json()
        with CSV.open(encoding="utf-8", newline="") as stream:
            by_year = Counter(row["year"] for row in csv.DictReader(stream))
        result = replay["result"]
        source = SQLTools(bind(tmp_path))
        assert sha256(CSV.read_bytes()).hexdigest() == replay["provenance"]["csv_sha256"]
        observed = source.execute(result["sql"], observation_id="public_replay")
        assert observed["complete"] and observed["rows"] == result["rows"] == [[by_year["2009"]]]
        assert result["usage"] == {"requests": 5, "total_tokens": 14105, "complete": True}
        assert (
            client.get(path + "/result.csv").content.decode("utf-8-sig").splitlines()[-1] == "120"
        )
