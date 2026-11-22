"""Offline cross-lane failure injection; no model or credential access occurs."""

import asyncio
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


def load_runner():
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_expanded.py"
    spec = importlib.util.spec_from_file_location("expanded_runner_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unclassified_worker_failure_stops_remaining_paid_cells(monkeypatch, tmp_path):
    runner = load_runner()
    monkeypatch.setattr(runner.QwenConfig, "from_local_environment", lambda: object())
    monkeypatch.setattr(runner, "validate_pricing", lambda config: None)
    monkeypatch.setattr(runner, "exclusive_run", lambda path: nullcontext())
    monkeypatch.setattr(runner, "source_hashes", lambda root: {})
    monkeypatch.setattr(runner, "save_summary", lambda output, protocol: {})
    monkeypatch.setattr(
        runner,
        "lane_budget",
        lambda *args: SimpleNamespace(snapshot=lambda: {"stopped_reason": None}),
    )
    monkeypatch.setattr(
        runner,
        "public_bundles",
        lambda *args: [SimpleNamespace(case_id="one"), SimpleNamespace(case_id="two")],
    )
    calls = []
    initial_count = 0

    async def fail_one_lane(*args):
        nonlocal initial_count
        initial_count += 1
        number = initial_count
        await asyncio.sleep(0)
        if number == 1:
            raise OSError("simulated journal persistence failure")
        return {}

    async def paid_cell(*args):
        calls.append(args[1])
        await asyncio.sleep(0)

    monkeypatch.setattr(runner, "prepare_first", fail_one_lane)
    monkeypatch.setattr(runner, "run_cell", paid_cell)
    args = SimpleNamespace(phase="development", limit=None, campaign=tmp_path)
    protocol = {"source_hashes": {}, "ablation_case_ids": []}
    asyncio.run(runner.run(args, {}, protocol, tmp_path, {"workers": 2}))
    assert len(calls) <= 1, f"other lane continued {len(calls)} paid cells after failure"
