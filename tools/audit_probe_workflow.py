"""Audit saved probe evidence and summarize paired costs; never calls a model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def usage(events):
    prompt = sum(event['usage']['prompt_tokens'] for event in events)
    completion = sum(event['usage']['completion_tokens'] for event in events)
    return {'prompt_tokens': prompt, 'completion_tokens': completion,
            'total_tokens': prompt + completion,
            'estimated_cny': (prompt * 800 + completion * 2700) / 1_000_000_000}


def audit(path: Path):
    path = path.resolve()
    chain, seen = [], set()
    while True:
        if not path.is_relative_to((ROOT / 'reports').resolve()) or path in seen:
            raise ValueError('Invalid or cyclic workflow chain')
        seen.add(path)
        workflow = read(path)
        chain.append((path, workflow))
        if not workflow.get('continued_from'):
            break
        path = ROOT / workflow['continued_from']
    chain.reverse()
    events, traces, run_checks, failures = [], [], [], []
    for workflow_path, workflow in chain:
        for name in workflow.get('report_dirs', ('smoke', 'full')):
            folder = workflow_path.parent / name
            if not folder.resolve().is_relative_to(workflow_path.parent):
                raise ValueError('Report directories must remain inside their workflow')
            if not (folder / 'summary.json').is_file():
                continue
            report = read(folder / 'summary.json')
            manifest = read(folder / 'frozen_manifest.json')
            audit_events = read(folder / 'request_audit.json')
            events.extend(audit_events)
            run_checks.append({'path': str(folder.relative_to(ROOT)), 'status': report['status'],
                               'manifest_hash_matches': digest(manifest) == report['manifest_sha256'],
                               'audit_ledger_complete': report['audit_ledger_complete'],
                               'all_cells_completed': report['all_cells_completed']})
            for row in report['rows']:
                trace_path = folder / 'traces' / f"{row['case_id']}_{row['track']}_{row['seed']}_{row['mode']}.json"
                trace = read(trace_path) if trace_path.is_file() else {}
                record = trace.get('record', {})
                traces.append(record)
                if row['status'] != 'completed' or not row.get('grade', {}).get('correct_task'):
                    failures.append({'run': str(folder.relative_to(ROOT)), 'case_id': row['case_id'],
                                     'track': row['track'], 'mode': row['mode'], 'seed': row['seed'],
                                     'status': row['status'], 'grade': row.get('grade'),
                                     'final_error': (row.get('publication') or {}).get('error_code'),
                                     'trace': str(trace_path.relative_to(ROOT))})
    final = chain[-1][1]
    budget = final['budget']
    measured = usage(events)
    attempts = [event['attempt'] for event in events]
    ids = [event.get('request_id') or event.get('response_id') for event in events]
    checks = {
        'all_manifest_hashes_match': all(item['manifest_hash_matches'] for item in run_checks),
        'all_run_ledgers_complete': all(item['audit_ledger_complete'] for item in run_checks),
        'all_scheduled_cells_completed': all(item['all_cells_completed'] for item in run_checks),
        'http_count_matches_cumulative_budget': len(events) == budget['http_attempts'],
        'attempts_contiguous_without_reset': attempts == list(range(1, len(events) + 1)),
        'provider_ids_present_and_unique': all(ids) and len(set(ids)) == len(ids),
        'all_http_completed_with_known_usage': all(event.get('status') == 'completed' and event.get('usage_complete') for event in events),
        'all_response_models_match': all(event.get('response_model') == 'qwen3.8-flash' for event in events),
        'reported_tokens_match_ledger': measured['total_tokens'] == budget['reported_total_tokens'],
        'estimated_cost_matches_ledger': measured['estimated_cny'] == budget['estimated_reported_cny'],
        'within_authorized_cny5': budget['reserved_upper_cny'] <= 5 and budget['estimated_reported_cny'] <= 5,
        'no_incomplete_cost': budget['reported_cost_complete'] and budget['unknown_count'] == 0,
        'sources_unchanged': all(record.get('source_unchanged') for record in traces),
        'raw_db_query_bypasses_zero': all(record.get('raw_resource_queries') == 0 for record in traces),
        'no_evaluator_fields_in_model_requests': all(
            all(token not in json.dumps(record.get('llm_requests', []), ensure_ascii=False)
                for token in ('reference_sql', 'fault_label', 'boundary_fixtures', 'source_grade')) for record in traces),
    }
    full_path = chain[-1][0].parent / 'full' / 'summary.json'
    result = {'checks': checks, 'all_audit_checks_passed': all(checks.values()),
              'chain': [str(item[0].relative_to(ROOT)) for item in chain],
              'usage_all_paid_calls': measured, 'budget': budget, 'runs': run_checks,
              'failures_including_prior_smoke': failures,
              'limitation': 'Audits saved application/provider evidence, not cloud billing or a statistical generalization study.'}
    if full_path.is_file():
        full = read(full_path)
        pairs = [pair for pair in full['pairs'] if pair['eligible_initial_counterexample']]
        comparable = [pair for pair in pairs if pair['both_completed']
                      and pair.get('full_initial_evidence_matches_frozen') is True
                      and pair.get('compact_initial_evidence_matches_frozen') is True
                      and pair['full_usage']['usage_complete'] and pair['compact_usage']['usage_complete']]
        aggregate = {'scheduled_eligible_pairs': len(pairs), 'comparable_pairs': len(comparable),
                     'excluded_pairs': [pair for pair in pairs if pair not in comparable]}
        context_mismatches = []
        for pair in comparable:
            normalized = []
            for mode in ('full', 'compact'):
                record = read(full_path.parent / 'traces' / f"{pair['case_id']}_controlled_{pair['seed']}_{mode}.json")['record']
                diagnostic = record['actions'][0]['feedback']['diagnostic_evidence']
                requests = []
                for messages in record['llm_requests']:
                    normalized_messages = []
                    for message in messages:
                        message = dict(message)
                        text = message['content'].replace(json.dumps(diagnostic, ensure_ascii=False), '"<DIAGNOSTIC>"')
                        message['content'] = re.sub(r'The current time is:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                                                    'The current time is:<NATIVE_CLOCK>', text)
                        normalized_messages.append(message)
                    requests.append(normalized_messages)
                normalized.append(requests)
            if normalized[0] != normalized[1]:
                context_mismatches.append({'case_id': pair['case_id'], 'seed': pair['seed']})
        aggregate.update(context_matches_except_diagnostic_and_native_clock=not context_mismatches,
                         context_mismatches=context_mismatches,
                         native_clock_limitation='Inherited native prompts contain wall-clock time; both arms keep it. All other model context is compared after masking only the clock and diagnostic.')
        checks['paired_context_controlled'] = not context_mismatches
        result['all_audit_checks_passed'] = all(checks.values())
        for mode in ('full', 'compact'):
            selected = [row for row in full['rows'] if row['track'] == 'controlled' and row['mode'] == mode
                        and (row['case_id'], row['seed']) in {(pair['case_id'], pair['seed']) for pair in comparable}]
            aggregate[mode] = {**usage([event for row in selected for event in row['http_events']]),
                               'correct': sum(row['grade']['correct_task'] for row in selected),
                               'http_attempts': sum(row['http_attempts'] for row in selected),
                               'initial_diagnostic_characters': sum(len(read(full_path.parent / 'traces' / f"{row['case_id']}_controlled_{row['seed']}_{mode}.json")['record']['actions'][0]['feedback'].get('diagnostic_evidence', '')) for row in selected)}
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'estimated_cny', 'initial_diagnostic_characters'):
            base = aggregate['full'][field]
            aggregate[field + '_reduction_percent'] = 100 * (base - aggregate['compact'][field]) / base if base else None
        result.update(full_experiment_summary=full['summary'], eligible_paired_comparison=aggregate,
                      full_failure_codes=dict(Counter((row.get('publication') or {}).get('error_code') for row in full['rows'] if not row.get('grade', {}).get('correct_task'))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path)
    args = parser.parse_args()
    result = audit(args.workflow)
    output = args.workflow.resolve().parent / 'independent_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'all_audit_checks_passed': result['all_audit_checks_passed'],
                      'checks': result['checks'], 'usage': result['usage_all_paid_calls'],
                      'paired': result.get('eligible_paired_comparison'), 'output': str(output)}, ensure_ascii=False, indent=2))
    return 0 if result['all_audit_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
