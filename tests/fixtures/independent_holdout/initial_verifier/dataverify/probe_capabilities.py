"""Public execution capabilities and task-preserving repair advice.

This describes the existing closed SQL grammar. It does not expand that grammar,
generate a correct SQL answer, or import any evaluation task or target. Both
feedback arms use the same contract; legacy_v1 exists for frozen comparisons.
"""

from __future__ import annotations

from copy import deepcopy

from .probe_runtime import ProbeTask


POLICIES = ('legacy_v1', 'capability_v2')
_CAPABILITIES = {
    'version': 'probe-sql-v1',
    'aggregate': {'function': 'SUM', 'argument': 'order_lines.amount',
                  'zero_wrapper': 'COALESCE(...,0)'},
    'joins': {'maximum': 1, 'types': ['INNER', 'LEFT'],
              'key': 'orders.order_id = order_lines.order_id'},
    'filter_locations': ['WHERE', 'JOIN_ON'],
    'filters': 'AND of direct registered column/literal comparisons or IS NULL; no expression filters.',
    'grouping': 'One registered order or channel key; output and GROUP BY keys must agree.',
    'result_contract': 'A numeric measure is required; NULL measures are not publishable. Empty scalar totals require zero; empty grouped relations are permitted.',
    'unsupported': ['CASE', 'CTE', 'subquery', 'window', 'UNION', 'DISTINCT',
                    'arithmetic', 'ORDER BY', 'LIMIT', 'arbitrary function'],
    'repair_rule': 'Stay inside this grammar and preserve the confirmed task; do not remove requirements to obtain acceptance.',
}


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError('Unknown probe feedback policy')
    return policy


def sql_capabilities() -> dict:
    """Return an isolated descriptor of existing runtime capabilities."""
    return deepcopy(_CAPABILITIES)


def repair_guidance(error_code: str, task: ProbeTask) -> dict | None:
    """Explain a supported alternative, never supply a complete query.

    An unsupported expression can be valid general SQL: it is rejected because
    our verifier cannot check it. A missing zero order is a task violation.
    Keeping these diagnoses distinct prevents treating a grammar limitation as
    proof that the model's business logic was wrong.
    """
    if not isinstance(task, ProbeTask):
        raise ValueError('Repair advice requires a confirmed task')
    if error_code == 'ZERO_ORDER_WHERE_FILTER':
        classification = 'task_requirement'
        alternatives = [
            'For LEFT JOIN tasks that retain unmatched orders, use JOIN_ON for right-table eligibility filters; WHERE on right-table columns can remove those orders.',
            'Aggregate the direct registered amount column and use the supported zero wrapper for empty matches. CASE aggregation is outside the execution grammar.',
        ]
    elif error_code == 'UNSUPPORTED_EXPRESSION':
        classification = 'unsupported_syntax'
        alternatives = [
            'Use the direct registered column as the SUM argument. CASE, arithmetic and nested expressions in the SUM argument are not supported, even if valid in general SQL.',
            'Express eligible-row conditions using supported column/literal comparisons in WHERE or JOIN_ON. If the task retains unmatched orders, right-table eligibility filters belong in JOIN_ON.',
        ]
    elif error_code == 'INVALID_RESULT_CONTRACT':
        classification = 'result_contract'
        alternatives = [
            'The task result must contain numeric measures. For SUM with no matching rows, the supported zero wrapper supplies the required numeric zero.',
            'Keep the confirmed filters and all required groups. Do not add artificial rows or remove task requirements to avoid an empty aggregate.',
        ]
    else:
        return None
    return {
        'classification': classification, 'capability_version': _CAPABILITIES['version'],
        'allowed_alternatives': alternatives,
        'preserve_requirements': [
            'Confirmed task_id=' + task.task_id,
            'Measure remains order_lines.amount; include legal negative amounts.',
            'Confirmed status=' + task.status + '; time range [' + task.start + ', ' + task.end + ').',
            'Keep output_kind=' + task.output_kind + ' and every caller-confirmed property: ' + ', '.join(task.properties),
        ],
        'retryable': True,
    }
