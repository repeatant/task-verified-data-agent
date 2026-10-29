"""Manually calculated targets for the frozen boundary evaluation inputs."""

from collections import Counter
import pytest

pytest.importorskip('hypothesis')

from dataverify.capability_cases import get_capability_cases
from dataverify.probe_cases import reference_result


@pytest.mark.parametrize('case_id, expected', [
    ('v01', [[105]]),
    ('v02', [[101,75], [102,30], [103,0], [104,0]]),
    ('v03', [['未标注',105]]),
    ('v04', [[0]]), ('v05', [[0]]), ('v06', []),
    ('v07', [['web',0]]), ('v08', [[0]]),
])
def test_frozen_positive_targets_have_independent_manual_totals(case_id, expected):
    entry = next(entry for entry in get_capability_cases() if entry.case.case_id == case_id)
    assert entry.expected_acceptance
    assert reference_result(entry.case) == expected


def test_suite_is_explicitly_boundary_regression_with_unique_identifiers():
    entries = get_capability_cases()
    assert len(entries) == len({entry.case.case_id for entry in entries}) == 20
    assert Counter(entry.category for entry in entries) == {
        'legal': 8, 'task_fault': 6, 'result_contract': 2, 'unsupported_candidate': 4}
    assert all(entry.case.split == 'controlled_development' for entry in entries)
