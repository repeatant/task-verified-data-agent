"""Retry only atomic persistence, never model spending or network actions."""

import json
import os
import threading
from pathlib import Path

import pytest

from dataverify.agent.graph_checkpoint import JournaledBudget, read_budget
from dataverify.agent.probe_langgraph import runtime_fingerprint
from dataverify.runtime import atomic_files
from dataverify.runtime.probe_cost_budget import MonetaryRequestBudget
from dataverify.runtime.reporting import write_json


def windows_error(code):
    error = PermissionError("test replacement access denied")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32])
def test_transient_windows_replacement_retries_only_replace(tmp_path, monkeypatch, code):
    source, target = tmp_path / "pending.json", tmp_path / "saved.json"
    source.write_text("new")
    target.write_text("old")
    original = Path.replace
    calls, waits = [], []

    def replace(path, destination):
        calls.append((path, destination))
        if len(calls) < 3:
            assert target.read_text() == "old" and source.read_text() == "new"
            raise windows_error(code)
        return original(path, destination)

    monkeypatch.setattr(atomic_files, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_files.time, "sleep", waits.append)
    monkeypatch.setattr(Path, "replace", replace)
    atomic_files.replace_atomically(source, target)
    assert len(calls) == 3 and waits == [0.01, 0.02]
    assert target.read_text() == "new" and not source.exists()


@pytest.mark.parametrize(
    "is_windows,error",
    [
        (False, windows_error(5)),
        (True, windows_error(13)),
        (True, PermissionError("no Windows error code")),
        (True, FileNotFoundError("missing source")),
        (True, OSError("disk failure")),
    ],
)
def test_other_failures_are_not_retried(tmp_path, monkeypatch, is_windows, error):
    calls, waits = [], []

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(atomic_files, "_IS_WINDOWS", is_windows)
    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(atomic_files.time, "sleep", waits.append)
    with pytest.raises(type(error)) as captured:
        atomic_files.replace_atomically(tmp_path / "a", tmp_path / "b")
    assert captured.value is error and len(calls) == 1 and waits == []


class CountingBudget(MonetaryRequestBudget):
    reserve_calls = 0
    record_calls = 0

    def reserve(self, prompt_chars):
        self.reserve_calls += 1
        return super().reserve(prompt_chars)

    def record_tokens(self, usage):
        self.record_calls += 1
        return super().record_tokens(usage)


USAGE = {"prompt_tokens": 2532, "completion_tokens": 573, "total_tokens": 3105}


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows read-handle semantics")
@pytest.mark.parametrize("operation", ["reserve", "record_tokens", "report"])
def test_real_reader_release_allows_one_accounting_operation(tmp_path, monkeypatch, operation):
    path = tmp_path / "document.json"
    delegate = CountingBudget(money_limit_cny=1)
    budget = JournaledBudget(delegate, path)
    if operation == "record_tokens":
        budget.reserve(3000)
    old_bytes = path.read_bytes()
    ready, release, closed = (threading.Event() for _ in range(3))

    def reader():
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.read(1)
                ready.set()
                release.wait(3)
        finally:
            closed.set()

    thread = threading.Thread(target=reader)
    thread.start()
    assert ready.wait(3)
    waits = []

    def release_after_real_lock_failure(delay):
        waits.append(delay)
        assert path.read_bytes() == old_bytes
        release.set()
        assert closed.wait(3)

    monkeypatch.setattr(atomic_files.time, "sleep", release_after_real_lock_failure)
    try:
        if operation == "reserve":
            assert budget.reserve(3000) == 1
        elif operation == "record_tokens":
            budget.record_tokens(USAGE)
        else:
            write_json(path, {"state": "complete"})
    finally:
        release.set()
        thread.join(timeout=3)
    assert waits == [0.01]
    if operation == "report":
        assert json.loads(path.read_text()) == {"state": "complete"}
        assert delegate.reserve_calls == delegate.record_calls == 0
    else:
        assert read_budget(path) == budget.snapshot()
        assert delegate.reserve_calls == 1
        assert delegate.record_calls == (1 if operation == "record_tokens" else 0)
        assert len(budget.snapshot()["reservation_ledger"]) == 1
        assert budget.snapshot()["http_attempts"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows read-handle semantics")
@pytest.mark.parametrize("operation", ["reserve", "record_tokens"])
def test_permanent_reader_lock_stops_without_losing_old_journal(tmp_path, operation):
    path = tmp_path / "budget.json"
    delegate = CountingBudget(money_limit_cny=1)
    budget = JournaledBudget(delegate, path)
    if operation == "record_tokens":
        budget.reserve(3000)
    original = path.read_bytes()
    with path.open("r", encoding="utf-8") as reader:
        reader.read(1)
        with pytest.raises(PermissionError) as captured:
            if operation == "reserve":
                budget.reserve(3000)
            else:
                budget.record_tokens(USAGE)
        assert captured.value.winerror in {5, 32}
        assert path.read_bytes() == original
    assert delegate.reserve_calls == 1
    assert delegate.record_calls == (1 if operation == "record_tokens" else 0)
    assert budget.snapshot()["http_attempts"] == 1
    pending = json.loads(path.with_suffix(".part.json").read_text())
    assert pending["budget"] == budget.snapshot()
    # Failure is surfaced; no network action/re-reservation is attempted. Once
    # the caller stops, the same one-entry ledger can be durably recorded.
    budget.stop("CLIENT_ERROR")
    assert read_budget(path) == budget.snapshot()
    assert budget.snapshot()["http_attempts"] == 1


def test_exhaustion_has_finite_attempts_and_retains_original_exception(tmp_path, monkeypatch):
    error = windows_error(32)
    calls, waits = [], []

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(atomic_files, "_IS_WINDOWS", True)
    monkeypatch.setattr(Path, "replace", fail)
    monkeypatch.setattr(atomic_files.time, "sleep", waits.append)
    with pytest.raises(PermissionError) as captured:
        atomic_files.replace_atomically(tmp_path / "a", tmp_path / "b")
    assert captured.value is error and len(calls) == 6
    assert waits == list(atomic_files._RETRY_DELAYS_SECONDS)
    assert sum(waits) == pytest.approx(0.31)


def test_runtime_fingerprint_covers_the_atomic_helper(monkeypatch):
    original = Path.read_bytes
    baseline = runtime_fingerprint()

    def changed(path):
        value = original(path)
        return value + b"\n# diagnostic mutation" if path.name == "atomic_files.py" else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert runtime_fingerprint() != baseline
