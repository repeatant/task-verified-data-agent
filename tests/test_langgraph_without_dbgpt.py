"""Run public offline paths in a fresh process that cannot import DB-GPT.

The import guard raises a runtime failure, rather than an ImportError that an
optional-dependency fallback could silently swallow. Historical source fixtures
are excluded from package discovery and remain immutable replay evidence.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ISOLATED_RUN = r"""
import importlib
import importlib.abc
import json
import pkgutil
import runpy
import sys

class ForbiddenRuntime(RuntimeError):
    pass

class NoDBGPTRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dbgpt" or fullname.startswith("dbgpt."):
            raise ForbiddenRuntime("The offline runner attempted to import DB-GPT")
        return None

# Make the check independent of whether a developer happens to have DB-GPT
# installed. Validate the guard itself before checking the package behavior.
for name in list(sys.modules):
    if name == "dbgpt" or name.startswith("dbgpt."):
        del sys.modules[name]
sys.meta_path.insert(0, NoDBGPTRuntime())
try:
    importlib.import_module("dbgpt")
except ForbiddenRuntime:
    pass
else:
    raise AssertionError("Import isolation was not installed")

def no_external_call(*args, **kwargs):
    raise AssertionError("Offline validation must not read credentials or call a model")

import httpx
httpx.Client.request = no_external_call
httpx.AsyncClient.request = no_external_call

import dataverify
from dataverify.integrations.qwen_config import QwenConfig
QwenConfig.from_local_environment = staticmethod(no_external_call)
for module in pkgutil.walk_packages(dataverify.__path__, dataverify.__name__ + "."):
    importlib.import_module(module.name)

entrypoint, argv = json.loads(sys.argv[1])
sys.argv = [entrypoint, *argv]
runpy.run_path(entrypoint, run_name="__main__")
"""


@pytest.mark.parametrize("entrypoint", ["run_graph", "evaluate_discovery"])
def test_offline_cli_executes_with_all_dbgpt_imports_blocked(tmp_path, entrypoint):
    """Exercise repair, evaluation and publication, not just module imports."""
    if entrypoint == "run_graph":
        report_path = tmp_path / "graph.json"
        arguments = [
            "--checkpoint",
            str(tmp_path / "graph.sqlite3"),
            "--output",
            str(report_path),
            "--thread",
            "no-native-framework",
        ]
    else:
        output = tmp_path / "discovery"
        report_path = output / "case-0.json"
        arguments = ["--cases", "order_average", "--output", str(output)]

    payload = [str(ROOT / "tools" / f"{entrypoint}.py"), arguments]
    result = subprocess.run(
        [sys.executable, "-B", "-c", ISOLATED_RUN, json.dumps(payload)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["http_events"] == []
    if entrypoint == "run_graph":
        assert report["real_model_used"] is False
        assert report["runtime"] == "langgraph"
        assert report["publication"]["status"] == "verified"
        assert report["model_attempts"] == 2
    else:
        summary = json.loads((report_path.parent / "summary.json").read_text(encoding="utf-8"))
        assert summary["real_model_used"] is False
        assert summary["http_events"] == []
        assert report["status"] == "reviewed"
        assert report["publication"]["rows"] == [[60]]
        assert report["attempts"] == 2
