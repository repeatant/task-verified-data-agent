"""Score independently authored challenges against a frozen verifier.

The initial score is recorded without adapting the verifier to challenge cases.
Live mode can run only the author's four preselected natural-language tasks,
under a continued budget. No gold SQL is required or sent to the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from dataverify.probe_guard import ProbeGuard
from dataverify.probe_cases import Case, grade_rows, reference_result
from dataverify.probe_runtime import ProbeData, ProbeTask


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    partial = path.with_suffix('.part.json')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Challenge:
    identifier: str
    category: str
    task: ProbeTask
    data: ProbeData
    sql: str
    expected: list
    reason: str


def load_challenges(document):
    result = []
    for item in document['challenges']:
        identifier = item['id']
        if not re.fullmatch(r'h[0-9]{2}', identifier):
            raise ValueError('Invalid independent challenge identifier')
        if item['category'] not in ('legal', 'semantic_fault', 'unsupported'):
            raise ValueError('Unknown independent challenge category')
        fields = {'orders': ('order_id','total','channel'),
                  'lines': ('line_id','order_id','amount','status','settled_at')}
        converted = {}
        for name, columns in fields.items():
            rows = []
            for row in item['data'][name]:
                if isinstance(row, dict):
                    if set(row) != set(columns):
                        raise ValueError('Unexpected source record fields')
                    rows.append(tuple(row[column] for column in columns))
                else:
                    rows.append(tuple(row))
            converted[name] = tuple(rows)
        task = ProbeTask(**item['task'])
        data = ProbeData(**converted)
        expected = item['expected_rows']
        # Existing record oracle is a cross-check, not the label author. No
        # correct SQL is invented: this placeholder is never executed or sent.
        case = Case(identifier, task, data, item['candidate_sql'], 'EVALUATOR_REFERENCE_UNUSED',
                    None if item['category'] == 'legal' else item['category'], item['reason'])
        if not grade_rows(reference_result(case), expected)['correct']:
            raise ValueError('Independent target disagrees with task-record oracle')
        result.append(Challenge(identifier, item['category'], task, data,
                                item['candidate_sql'], expected, item['reason']))
    if len(result) != len({item.identifier for item in result}):
        raise ValueError('Duplicate independent challenge IDs')
    selected = document['natural_probe_ids']
    by_id = {item.identifier: item for item in result}
    if (len(selected) != 4 or len(set(selected)) != 4
            or any(name not in by_id or by_id[name].category != 'legal' for name in selected)):
        raise ValueError('Exactly four preselected legal natural tasks are required')
    return tuple(result)


def score(challenge):
    guard = ProbeGuard(challenge.task, challenge.data, feedback_policy='capability_v2')
    guard.execute(challenge.sql, challenge.task.task_id)
    publication = guard.publish()
    accepted = publication['status'] == 'verified'
    grade = grade_rows(publication['rows'], challenge.expected) if accepted else None
    audit = guard.last_result
    probe = audit.get('probe_report') or {}
    semantic_wrong_release = accepted and not grade['correct']
    scope_breach = accepted and challenge.category == 'unsupported'
    refused_legal = not accepted and challenge.category == 'legal'
    passed = (accepted and grade['correct']) if challenge.category == 'legal' else not accepted
    return {'id': challenge.identifier, 'category': challenge.category, 'passed': bool(passed),
            'accepted': accepted, 'semantic_wrong_release': bool(semantic_wrong_release),
            'scope_breach': bool(scope_breach), 'refused_legal': refused_legal,
            'expected_rows': challenge.expected, 'grade': grade, 'publication': publication,
            'static': probe.get('static'), 'dynamic_status': probe.get('status'),
            'dynamic_new_detection': bool(probe.get('new_detection_beyond_static')),
            'source_output_check': audit.get('source_output_check'),
            'audit': audit}


async def evaluate(output: Path, dataset: Path, verifier_manifest: Path, *, live=False, previous_workflow=None,
                   regression_from: Path | None = None):
    output, dataset, verifier_manifest = output.resolve(), dataset.resolve(), verifier_manifest.resolve()
    readable_roots=((ROOT/'reports').resolve(),(ROOT/'examples').resolve())
    if (not output.is_relative_to((ROOT/'reports').resolve())
            or any(not any(path.is_relative_to(base) for base in readable_roots) for path in (dataset,verifier_manifest))):
        raise ValueError('Inputs must remain inside reports/examples; output must remain inside reports')
    freeze = read(verifier_manifest)
    unchanged = {name: sha(ROOT/name) == digest for name,digest in freeze['files'].items()}
    if not all(unchanged.values()):
        raise ValueError('Frozen verifier changed before initial evaluation')
    author = read(dataset.parent / 'author_manifest.json')
    label_audit = read(dataset.parent / 'label_audit.json')
    dataset_hash = sha(dataset)
    if (author['challenge_sha256'] != dataset_hash or label_audit['challenge_sha256'] != dataset_hash
            or label_audit['all_labels_valid'] is not True):
        raise ValueError('Challenge hash mismatch or unresolved independent label dispute')
    document = read(dataset)
    challenges = load_challenges(document)
    if len(challenges) != 24 or Counter(item.category for item in challenges) != {
            'legal':8, 'semantic_fault':10, 'unsupported':6}:
        raise ValueError('Independent suite must match its preregistered 24-case composition')
    if regression_from is not None:
        regression_from = regression_from.resolve()
        if not any(regression_from.is_relative_to(base) for base in readable_roots):
            raise ValueError('Original evaluation must be inside reports/examples')
        prior_manifest = read(regression_from/'evaluation_manifest.json')
        if prior_manifest['challenge_sha256'] != dataset_hash or live:
            raise ValueError('Regression must reuse the exact initial challenges; controlled live repairs use a separate run')
    budget = None
    config = None
    if live:
        if previous_workflow is None:
            raise ValueError('Live validation must continue an existing budget')
        from run_probe_workflow import restore_completed_budget
        from dataverify.dbgpt_qwen import QwenConfig
        from dataverify.probe_cost_budget import validate_pricing
        previous_workflow = Path(previous_workflow).resolve()
        if not previous_workflow.is_relative_to((ROOT / 'reports').resolve()):
            raise ValueError('Invalid budget continuation path')
        previous = read(previous_workflow)
        if previous.get('real_model_used') is not True:
            raise ValueError('Previous budget is not a live ledger')
        budget = restore_completed_budget(previous['budget'])
        if budget.snapshot()['http_attempt_limit'] - budget.snapshot()['http_attempts'] < 8:
            raise ValueError('Insufficient remaining allowance for four bounded tasks')
        config = QwenConfig.from_local_environment()
        validate_pricing(config)
    output.mkdir(parents=True, exist_ok=False)
    evaluation_manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                           'dataset': str(dataset.relative_to(ROOT)), 'challenge_sha256': dataset_hash,
                           'verifier_manifest': str(verifier_manifest.relative_to(ROOT)),
                           'verifier_manifest_sha256': sha(verifier_manifest),
                           'evaluator_sha256': sha(Path(__file__)),
                           'author_manifest_sha256': sha(dataset.parent/'author_manifest.json'),
                           'label_audit_sha256': sha(dataset.parent/'label_audit.json'),
                           'phase': 'converted_regression' if regression_from else 'initial_independent_challenge',
                           'scope': ('Post-fix regression on challenges already examined for fixes; NOT an untouched holdout score.'
                                     if regression_from else 'Prospective independently authored query challenges on fixed schema, no tuning during initial evaluation; not cross-schema or production generalization.'),
                           'natural_probe_ids': document['natural_probe_ids'],
                           'feedback_policy': 'capability_v2', 'real_model_used': live,
                           'budget_start': budget.snapshot() if budget else None,
                           'verifier_unchanged_before': unchanged}
    if regression_from:
        evaluation_manifest['original_evaluation'] = str(regression_from.relative_to(ROOT))
    write(output/'evaluation_manifest.json', evaluation_manifest)
    workflow = {'status': 'running', 'real_model_used': live, 'report_dirs': ['natural'] if live else [],
                'budget': budget.snapshot() if budget else None, 'scope': evaluation_manifest['scope']}
    if live:
        workflow['continued_from'] = str(previous_workflow.relative_to(ROOT))
    write(output/'workflow.json', workflow)
    try:
        rows = []
        for item in challenges:
            try:
                rows.append(score(item))
            except Exception as exc:
                rows.append({'id':item.identifier, 'category':item.category, 'passed':False,
                             'accepted':False, 'semantic_wrong_release':False, 'scope_breach':False,
                             'refused_legal':False, 'dynamic_new_detection':False,
                             'operational_error':type(exc).__name__})
        summary = {'scheduled': len(rows), 'completed': len(rows),
                   'passed': sum(row['passed'] for row in rows),
                   'category_counts': dict(Counter(row['category'] for row in rows)),
                   'semantic_wrong_releases': sum(row['semantic_wrong_release'] for row in rows),
                   'scope_breaches': sum(row['scope_breach'] for row in rows),
                   'refused_legal': sum(row['refused_legal'] for row in rows),
                   'operational_errors': sum('operational_error' in row for row in rows),
                   'dynamic_new_detections': sum(row['dynamic_new_detection'] for row in rows)}
        write(output/'initial_component_results.json', {'summary': summary, 'rows': rows})
        workflow['initial_summary'] = summary
        write(output/'workflow.json', workflow)
        if live:
            from dataverify.probe_live_benchmark import run_probe_benchmark
            by_id = {item.identifier: item for item in challenges}
            cases = tuple(Case(name, by_id[name].task, by_id[name].data, by_id[name].sql,
                               'EVALUATOR_REFERENCE_UNUSED', None, 'Independent preregistered natural task')
                          for name in document['natural_probe_ids'])
            natural = await run_probe_benchmark(output/'natural', live=True, config=config,
                                               evaluation_cases=cases, natural_case_ids=document['natural_probe_ids'],
                                               include_controlled=False, seeds=(43,), modes=('compact',),
                                               budget=budget, feedback_policy='capability_v2')
            workflow.update(natural_summary=natural['summary'], audit_ledger_complete=natural['audit_ledger_complete'])
        workflow.update(status='completed' if not live or natural['all_cells_completed'] else 'incomplete',
                        verifier_unchanged_after={name: sha(ROOT/name) == digest for name,digest in freeze['files'].items()})
    except BaseException as exc:
        if budget:
            budget.stop('INDEPENDENT_EVALUATION_ERROR')
        workflow.update(status='error', error_type=type(exc).__name__)
        raise
    finally:
        workflow['budget'] = budget.snapshot() if budget else None
        write(output/'workflow.json', workflow)
    return workflow


def main():
    import asyncio
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    public=ROOT/'examples/independent_holdout'
    parser.add_argument('--dataset',type=Path,default=(public/'frozen_challenges.json' if public.is_dir() else ROOT/'reports/independent_holdout_20261029/frozen_challenges.json'))
    parser.add_argument('--verifier-manifest',type=Path,default=(public/'current_verifier_manifest.json' if public.is_dir() else ROOT/'reports/holdout_verifier_freeze_20261029/manifest.json'))
    parser.add_argument('--output',type=Path,default=ROOT/'reports'/('independent_eval_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--previous-workflow',type=Path)
    parser.add_argument('--regression-from',type=Path,default=public/'initial_evaluation' if public.is_dir() else None)
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.output,args.dataset,args.verifier_manifest,
                                  live=args.live,previous_workflow=args.previous_workflow,
                                  regression_from=args.regression_from))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
