"""No-network tests for preserving the money allowance across continuations."""

import importlib.util
from pathlib import Path
from copy import deepcopy

import pytest

if importlib.util.find_spec('dbgpt') is None:
    pytest.skip('Optional pinned DB-GPT runtime is not installed', allow_module_level=True)
pytest.importorskip('hypothesis')

from dataverify.probe_cost_budget import MonetaryRequestBudget

spec = importlib.util.spec_from_file_location('probe_workflow_tool', Path(__file__).resolve().parents[1] / 'tools/run_probe_workflow.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def completed_snapshot():
    budget = MonetaryRequestBudget()
    budget.reserve(100)
    budget.record_tokens({'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120})
    return budget.snapshot()


def test_continuation_reproduces_reservations_and_usage_exactly():
    snapshot = completed_snapshot()
    resumed = module.restore_completed_budget(snapshot)
    assert resumed.snapshot() == snapshot
    assert resumed.reserve(200) == 2
    assert resumed.snapshot()['reserved_upper_cny'] > snapshot['reserved_upper_cny']


@pytest.mark.parametrize('field,value', [('reported_cost_complete', False), ('stopped_reason', 'USAGE_UNAVAILABLE'),
                                      ('unknown_count', 1), ('in_flight_count', 1),
                                      ('unmatched_usage_events', 1), ('http_attempts', 0),
                                      ('estimated_reported_cny', 0), ('reserved_upper_cny', 0)])
def test_incomplete_or_inconsistent_ledger_cannot_restart_allowance(field, value):
    snapshot = completed_snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError):
        module.restore_completed_budget(snapshot)


def test_changed_price_or_missing_reservation_is_rejected():
    snapshot = completed_snapshot()
    snapshot['pricing']['input_cny_per_million_tokens'] = 0
    with pytest.raises(ValueError):
        module.restore_completed_budget(snapshot)
    snapshot = completed_snapshot()
    snapshot['reservation_ledger'] = []
    with pytest.raises(ValueError):
        module.restore_completed_budget(snapshot)
