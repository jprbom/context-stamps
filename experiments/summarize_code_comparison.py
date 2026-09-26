"""Descriptive paired quality/resource analysis; never activates a candidate."""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from code_adapter_generate import extract_program
from grade_code_generation import generation_rows
from humaneval_local import parse_result
from mbpp_training_data import ROOT, fresh, write_new
from source_evidence import verify_sources


def percentile(values, fraction):
    values = sorted(values)
    point = (len(values) - 1) * fraction
    lo, hi = math.floor(point), math.ceil(point)
    return values[lo] + (values[hi] - values[lo]) * (point - lo)


def paired_statistics(pairs):
    diffs = [int(new) - int(old) for old, new in pairs]
    gained, lost = diffs.count(1), diffs.count(-1)
    n = gained + lost
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n) if n else 1.
    rng = random.Random(71)
    boot = [sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000)]
    return {'gained': gained, 'regressed': lost, 'delta': statistics.mean(diffs),
            'paired_bootstrap_95_interval': [percentile(boot, .025), percentile(boot, .975)],
            'exact_two_sided_mcnemar_p': p,
            'interpretation': 'Descriptive task-level resampling. Assumes independent tasks; related tasks and four development-smoke tasks limit inference. Not an activation gate.'}


def validate_records(rows, plan, records, completed):
    """Bind every outcome to its original program; execute no generated code."""
    ids = plan['task_ids']
    if (ids != ['HumanEval/' + str(i) for i in range(164)] or plan['smoke']
            or completed['tasks'] != 164 or completed['evaluations'] != 328):
        raise ValueError('Complete paired full benchmark required')
    expected = {(k, arm) for k in ids for arm in ('base', 'adapter')}
    for values in (rows, records):
        if len(values) != 328 or {(r['task_id'], r['arm']) for r in values} != expected:
            raise ValueError('Incomplete or duplicate paired records')
    outputs = {(r['task_id'], r['arm']): r for r in rows}
    for row in rows:
        if any(row[k] != value for k, value in extract_program(row['text']).items()):
            raise ValueError('Output postprocessing changed')
    for task in ids:
        a, b = (outputs[(task, arm)] for arm in ('base', 'adapter'))
        if (a['prompt_sha256'] != b['prompt_sha256'] or a['prompt_tokens'] != b['prompt_tokens']
                or {a['order'], b['order']} != {0, 1}):
            raise ValueError('Unmatched paired generation')
    for r in records:
        row = outputs[(r['task_id'], r['arm'])]
        if type(r['passed']) is not bool or r['containers_remaining']:
            raise ValueError('Invalid outcome or incomplete cleanup')
        if not row['valid_format'] or row['reached_token_limit']:
            if r['passed'] or 'native' in r or r.get('reason') != 'invalid_format_or_generation_truncated':
                raise ValueError('Invalid or truncated output was accepted')
        elif 'native' in r:
            request = {'task_id': r['task_id'], 'mode': 'candidate', 'code': row['code']}
            if r['native'] != parse_result(r['execution'], request) or r['passed'] != r['native']['passed']:
                raise ValueError('Native result mismatch')
        elif r['passed'] or 'error' not in r:
            raise ValueError('Missing native grade or explicit execution error')
    errors = sorted([r['task_id'], r['arm']] for r in records if 'error' in r)
    if completed['activation'] or completed['containers_remaining'] or sorted(completed['errors']) != errors:
        raise ValueError('Inconsistent grading summary or unqualified activation')
    return outputs


def outcome_stage(row, record):
    if row['reached_token_limit']:
        return 'generation_truncated'
    if not row['valid_format']:
        return 'output_format'
    if 'error' in record:
        return 'driver_error'
    native = record['native']
    if 'checks' not in native:
        return 'native_execution_or_protocol'
    if record['passed']:
        return 'passed'
    base = native['checks']['base']
    if base['status'] != 'pass' or base['completed'] != base['inputs'] or base['passing'] != base['inputs']:
        return 'base_tests'
    return 'additional_tests'


def summarize_records(rows, plan, records, completed):
    outputs = validate_records(rows, plan, records, completed)
    ids = plan['task_ids']
    scores, report = {}, {'tasks': 164, 'activation': False, 'provider_charges_usd': 0, 'models': {}}
    for arm in ('base', 'adapter'):
        group = [r for r in rows if r['arm'] == arm]
        selected = [r for r in records if r['arm'] == arm]
        for r in selected:
            scores[(r['task_id'], arm)] = r['passed']
        batches = {}
        for r in group:
            key = int(r['task_id'].split('/')[1]) // plan['config']['batch_size']
            value = (r['batch_seconds'], r['batch_size'])
            if key in batches and batches[key] != value:
                raise ValueError('Inconsistent batch timing')
            batches[key] = value
        times = [v[0] for v in batches.values()]
        tokens = sum(r['generated_tokens'] for r in group)
        report['models'][arm] = {'correct_combined': sum(r['passed'] for r in selected),
                                'outcome_stages': dict(Counter(outcome_stage(outputs[(r['task_id'], arm)], r) for r in selected)),
                                'format_failures': dict(Counter(r['reason'] for r in group if not r['valid_format'])),
                                'truncated_generations': sum(r['reached_token_limit'] for r in group),
                                'execution_errors': [r['task_id'] for r in selected if 'error' in r],
                                'execution_boundary_failures': [r['task_id'] for r in selected
                                                               if r.get('execution', {}).get('boundary_failure')],
                                'input_tokens': sum(r['prompt_tokens'] for r in group),
                                'generated_tokens': tokens, 'generation_batches': len(times),
                                'total_batch_generation_seconds': sum(times), 'batch_p50_seconds': statistics.median(times),
                                'batch_p95_seconds': percentile(times, .95), 'generated_tokens_per_generation_second': tokens / sum(times),
                                'peak_torch_cuda_bytes': max(r['peak_allocated_cuda_bytes'] for r in group)}
        if report['models'][arm]['correct_combined'] != completed[arm + '_passed']:
            raise ValueError('Recorded summary differs from native task outcomes')
    report['paired'] = paired_statistics([(scores[(k, 'base')], scores[(k, 'adapter')]) for k in ids])
    report['gained_tasks'] = [k for k in ids if not scores[(k, 'base')] and scores[(k, 'adapter')]]
    report['regressed_tasks'] = [k for k in ids if scores[(k, 'base')] and not scores[(k, 'adapter')]]
    if report['gained_tasks'] != completed['gained'] or report['regressed_tasks'] != completed['regressed']:
        raise ValueError('Paired gains or regressions changed')
    report['limitations'] = ['One generic SFT control, one base model, one training seed and greedy decoding',
                             'No stamp-memory treatment, domain-retention result, production workload, frontier comparison or edge-device measurement',
                             'Benchmark task 32 has a retained reference/oracle numerical limitation; it remains in the denominator',
                             'Batch generation time is not per-request latency, time to first token or end-to-end workflow cost',
                             'No energy measurement; no automatic activation or domain-wide competence claim']
    return report


def summarize(generation, grading):
    rows, plan = generation_rows(generation)
    verify_sources(plan['source_hashes'])
    gp = json.loads((grading / 'plan.json').read_text())
    verify_sources(gp['sources'])
    if gp['generation_sha256'] != hashlib.sha256((generation / 'generations.jsonl').read_bytes()).hexdigest():
        raise ValueError('Grading does not bind these generations')
    if gp['generation_plan'] != plan:
        raise ValueError('Different generation plan')
    completed = json.loads((grading / 'summary.json').read_text())
    records = [json.loads((grading / (row['task_id'].split('/')[1] + '-' + row['arm'] + '.json')).read_text()) for row in rows]
    return summarize_records(rows, plan, records, completed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('generation', 'grading', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    report = summarize(args.generation, args.grading)
    out = fresh(args.out)
    report['analysis_source_sha256'] = hashlib.sha256((ROOT / 'experiments/summarize_code_comparison.py').read_bytes()).hexdigest()
    write_new(out / 'summary.json', report)
    print(json.dumps(report, indent=2))
