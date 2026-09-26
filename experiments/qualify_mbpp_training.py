"""Execute published training references and empty controls only in containers.

No generated model programs, fitting, hosted calls or benchmark test evaluation.
"""

import argparse
import asyncio
import base64
import json
import os
import platform
import shlex
import time
from pathlib import Path

from harbor_sandbox import PYTHON_IMAGE, ResearchSandbox
from mbpp_training_data import ROOT, fresh, sha, training_rows, write_new

# User code and all assertion evaluation stay in an isolated guest interpreter.
# This preserves assertions but is not a proof against deliberate grader tampering.
WORKER = r'''
import base64,json,sys
p=json.loads(base64.b64decode(sys.argv[1]));ns={}
result={'loaded':False,'tests':[],'passed':False}
try:
    exec(compile(p['setup'],'<setup>','exec'),ns)
    exec(compile(p['code'],'<reference>','exec'),ns)
    result['loaded']=True
except BaseException as e:
    result['load_error']={'type':type(e).__name__,'message':str(e)[:512]}
if result['loaded']:
    for i,test in enumerate(p['tests']):
        try:
            exec(compile(test,'<assertion>','exec'),ns)
            result['tests'].append({'index':i,'passed':True})
        except BaseException as e:
            result['tests'].append({'index':i,'passed':False,'error':type(e).__name__,'message':str(e)[:512]})
result['passed']=result['loaded'] and len(result['tests'])==3 and all(r['passed'] for r in result['tests'])
print('SCQR_MBPP_RESULT:'+json.dumps(result))
'''


def command(row, empty=False):
    if type(row['task_id']) is not int or not 601 <= row['task_id'] <= 974:
        raise ValueError('Only the published training split is executable here')
    payload = {'setup': row.get('test_setup_code', ''), 'code': '' if empty else row['code'], 'tests': row['test_list']}
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
    return 'python -I -B -c ' + shlex.quote(WORKER) + ' ' + shlex.quote(encoded)


def parse_result(value):
    if value['boundary_failure'] or value['return_code'] != 0:
        return {'passed': False, 'reason': 'execution_failure'}
    lines = [line[len('SCQR_MBPP_RESULT:'):] for line in value['stdout'].splitlines()
             if line.startswith('SCQR_MBPP_RESULT:')]
    if len(lines) != 1:
        return {'passed': False, 'reason': 'missing_or_ambiguous_worker_result'}
    record = json.loads(lines[0])
    if type(record.get('loaded')) is not bool or type(record.get('passed')) is not bool:
        raise ValueError('Malformed worker result')
    tests = record['tests']
    if (type(tests) is not list or len(tests) > 3
            or [r['index'] for r in tests] != list(range(len(tests)))
            or any(type(r['passed']) is not bool for r in tests)):
        raise ValueError('Malformed native assertion record')
    passed = record['loaded'] and len(tests) == 3 and all(r['passed'] for r in tests)
    if passed != record['passed']:
        raise ValueError('Inconsistent native result')
    return record


async def run(args):
    os.environ['HARBOR_TELEMETRY'] = 'off'
    rows, audit = training_rows(args.source)
    out = fresh(args.out)
    sources = ('experiments/qualify_mbpp_training.py', 'experiments/mbpp_training_data.py', 'experiments/harbor_sandbox.py')
    plan = {'schema': 1, 'task_ids': [r['task_id'] for r in rows], 'source_audit': audit,
            'source_hashes': {n: sha((ROOT / n).read_bytes()) for n in sources},
            'image': PYTHON_IMAGE, 'concurrency': 2, 'deadline_seconds': 10, 'host_python': platform.python_version(),
            'started_utc': time.time(), 'model_calls': 0, 'purpose': 'Native assertion filter for published training data',
            'test_partition_used_for_training': False,
            'overlap_audit': 'Exact normalized text or full AST equality against reserved IDs 1..600; no semantic decontamination claim'}
    write_new(out / 'plan.json', plan)
    semaphore = asyncio.Semaphore(2)
    records = []

    async def qualify(row):
        async with semaphore:
            record = {'task_id': row['task_id'], **audit[str(row['task_id'])]}
            sandbox = ResearchSandbox(out / ('task-' + str(row['task_id'])), memory_mb=1024)
            started = time.perf_counter()
            try:
                record['sandbox'] = await sandbox.start()
                for name, empty in (('empty', True), ('reference', False)):
                    raw = await sandbox.execute(command(row, empty), timeout=10)
                    record[name] = {'execution': raw, 'native': parse_result(raw)}
                    if raw['boundary_failure']:
                        break
                record['eligible'] = (not record['exact_reserved_overlap']
                                      and not record['empty']['native']['passed']
                                      and record.get('reference', {}).get('native', {}).get('passed', False))
            except Exception as error:
                record['error'] = {'type': type(error).__name__, 'message': str(error)[-1000:]}
                record['eligible'] = False
            finally:
                await sandbox.stop()
                record['containers_remaining'] = sandbox.ids()
                record['elapsed_seconds'] = time.perf_counter() - started
                write_new(out / (str(row['task_id']) + '.json'), record)
                records.append(record)
                if len(records) % 20 == 0:
                    print(json.dumps({'completed': len(records), 'eligible': sum(r['eligible'] for r in records)}), flush=True)
    await asyncio.gather(*(qualify(row) for row in rows))
    records.sort(key=lambda r: r['task_id'])
    summary = {'total': len(records), 'eligible': [r['task_id'] for r in records if r['eligible']],
               'reference_failures': [r['task_id'] for r in records if not r.get('reference', {}).get('native', {}).get('passed', False)],
               'empty_passes': [r['task_id'] for r in records if r.get('empty', {}).get('native', {}).get('passed', False)],
               'exact_reserved_overlap': [r['task_id'] for r in records if r['exact_reserved_overlap']],
               'errors': [r['task_id'] for r in records if 'error' in r],
               'containers_remaining': [c for r in records for c in r['containers_remaining']],
               'model_calls': 0, 'training_runs': 0}
    write_new(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    asyncio.run(run(a))
