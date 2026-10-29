"""Interface contracts and local-only defaults; no external model requests."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dataverify import __main__ as cli
from dataverify.api import create_app


@pytest.fixture
def local_app(tmp_path):
    return create_app(tmp_path / "demo.sqlite3", tmp_path / "traces")


def test_health_makes_offline_scope_explicit(local_app):
    with TestClient(local_app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    health = response.json()
    assert health["default_provider"] == "offline"
    assert health["execution_mode"] == "offline-scripted"
    assert health["real_model_validation"] is False


def test_api_repairs_and_persists_trace(local_app, tmp_path):
    with TestClient(local_app) as client:
        response = client.post("/runs", json={"case": "join_fanout"})
    assert response.status_code == 200
    result = response.json()
    assert result["provider"] == "offline-scripted"
    assert result["real_model_used"] is False
    assert result["status"] == "verified"
    assert result["value"] == 280
    assert result["trace"]
    trace_path = Path(result["trace_path"])
    assert trace_path.is_relative_to(tmp_path / "traces")
    assert trace_path.is_file()


def test_api_without_repair_does_not_verify_bad_query(local_app):
    with TestClient(local_app) as client:
        response = client.post("/runs", json={"case": "join_fanout", "max_repairs": 0})
    assert response.status_code == 200
    assert response.json()["status"] == "unverified"


@pytest.mark.parametrize(
    "payload",
    [
        {"case": "arbitrary_case"},
        {"max_repairs": -1},
        {"max_repairs": 3},
        {"max_repairs": True},
        {"question": "   "},
        {"provider": "unrecognized"},
        {"db_path": "C:/arbitrary/data.sqlite3"},
    ],
)
def test_api_rejects_invalid_input_and_arbitrary_database_paths(local_app, payload):
    with TestClient(local_app) as client:
        response = client.post("/runs", json=payload)
    assert response.status_code == 422


def test_api_reuses_existing_database_without_reseeding(tmp_path):
    database = tmp_path / "demo.sqlite3"
    traces = tmp_path / "traces"
    with TestClient(create_app(database, traces)) as client:
        assert client.get("/health").status_code == 200
    original = database.read_bytes()
    with TestClient(create_app(database, traces)) as client:
        assert client.get("/health").status_code == 200
    assert database.read_bytes() == original


def test_api_invalid_input_does_not_echo_private_values(local_app):
    private_value = "validation-canary-not-a-real-credential"
    with TestClient(local_app) as client:
        response = client.post("/runs", json={"api_key": private_value})
    assert response.status_code == 422
    assert private_value not in response.text
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_api_storage_failure_returns_safe_operational_error(local_app):
    class UnavailableStorageRunner:
        def run(self, **kwargs):
            raise OSError("private-directory-canary: storage unavailable")

    with TestClient(local_app) as client:
        local_app.state.runner = UnavailableStorageRunner()
        response = client.post("/runs", json={})
    assert response.status_code == 503
    assert "private-directory-canary" not in response.text


def test_api_qwen_without_configuration_does_not_call_a_model(local_app, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with TestClient(local_app) as client:
        response = client.post("/runs", json={"provider": "qwen"})
    assert response.status_code in (400, 503)
    assert response.json()["detail"]


def test_cli_demo_emits_json_and_stores_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.main(["demo", "--case", "distinct_fix"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["provider"] == "offline-scripted"
    assert result["value"] == 280
    assert Path(result["trace_path"]).is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["demo", "--case", "join_fanout", "--max-repairs", "0"],
        ["demo", "--case", "unsupported"],
    ],
)
def test_cli_unverified_run_has_nonzero_exit_status(tmp_path, monkeypatch, capsys, arguments):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.main(arguments) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unverified"
    assert result["value"] is None
    assert Path(result["trace_path"]).is_file()


def test_cli_storage_failure_has_safe_json_error(monkeypatch, capsys):
    def unavailable_runner():
        raise OSError("private-directory-canary: storage unavailable")

    monkeypatch.setattr(cli, "_runner", unavailable_runner)
    assert cli.main(["demo"]) == 2
    output = capsys.readouterr().out
    assert "private-directory-canary" not in output
    assert json.loads(output)["status"] == "error"


def test_cli_evaluation_is_explicitly_not_an_llm_benchmark(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.main(["evaluate"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["llm_evaluation"] is False
    assert report["evaluation_type"] == "offline-scripted-pipeline-validation"
    assert report["case_count"] == 5
    assert report["pipeline_checks_passed"] == 5
    assert {item["case"] for item in report["results"]} == set(cli.CASES)
    unsupported = next(item for item in report["results"] if item["case"] == "unsupported")
    assert unsupported["status"] == "unverified"
    stored = json.loads(
        (tmp_path / "reports" / "offline_evaluation.json").read_text(encoding="utf-8")
    )
    assert stored == report


def test_cli_missing_model_configuration_returns_actionable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert cli.main(["demo", "--provider", "qwen"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "--provider offline" in result["message"]


def test_cli_server_cannot_be_bound_to_an_external_interface():
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--host", "0.0.0.0"])
    assert error.value.code == 2


@pytest.mark.parametrize("port", ["-1", "0", "65536"])
def test_cli_server_rejects_invalid_ports_before_starting(port):
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--port", port])
    assert error.value.code == 2
