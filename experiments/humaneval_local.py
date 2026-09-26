"""Local native HumanEval+ controls; no provider or generated-code host execution."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
import time
from pathlib import Path

from harbor_sandbox import ResearchSandbox
from mbpp_training_data import ROOT, fresh, write_new

PREFIX = 'SCQR_HUMANEVAL_RESULT:'


def parse_result(raw, request):
    if raw['boundary_failure'] or raw['return_code'] != 0:
        return {'passed': False, 'reason': 'execution_failure'}
    lines = [s[len(PREFIX):] for s in raw['stdout'].splitlines() if s.startswith(PREFIX)]
    if len(lines) != 1:
        return {'passed': False, 'reason': 'missing_or_ambiguous_worker_result'}
    result = json.loads(lines[0])
    if result['task_id'] != request['task_id'] or result['mode'] != request['mode']:
        raise ValueError('Wrong native task result')
    if type(result['passed']) is not bool or set(result['checks']) != {'base', 'plus'}:
        raise ValueError('Incomplete native result')
    for check in result['checks'].values():
        if (check['status'] not in ('pass', 'fail', 'timeout')
                or any(type(check[k]) is not int for k in ('inputs', 'completed', 'passing'))
                or not 0 <= check['passing'] <= check['completed'] <= check['inputs']
                or check['inputs'] < 1):
            raise ValueError('Invalid native input counts')
    passed = all(v['status'] == 'pass' and v['completed'] == v['inputs'] == v['passing']
                 for v in result['checks'].values())
    if passed != result['passed']:
        raise ValueError('False native pass claim')
    if request['mode'] != 'reference':
        code = '' if request['mode'] == 'empty' else request['code']
        if result['code_sha256'] != hashlib.sha256(code.encode()).hexdigest():
            raise ValueError('Wrong program result')
    return result


async def run(args):
    os.environ['HARBOR_TELEMETRY'] = 'off'
    if not args.image.startswith('sha256:') or len(args.image) != 71:
        raise ValueError('Pin a local image ID, not a mutable image tag')
    # This driver runs canonical/empty qualification only. Candidate execution
    # will need a separately registered generation and evaluation protocol.
    ids = [0, 32, 129] if args.smoke else list(range(164))
    out = fresh(args.out)
    names = ('experiments/humaneval_guest.py', 'experiments/humaneval_local.py', 'experiments/harbor_sandbox.py')
    write_new(out / 'plan.json', {'schema': 1, 'task_ids': ids, 'image_id': args.image,
              'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names},
              'smoke': args.smoke, 'model_calls': 0, 'concurrency': 2,
              'grading': 'Pinned native EvalPlus untrusted_check, base and full plus inputs, fail-fast',
              'purpose': 'Grader qualification, not a model benchmark', 'started_utc': time.time()})
    semaphore = asyncio.Semaphore(2)
    records = []

    async def task(task_id):
        async with semaphore:
            sandbox = ResearchSandbox(out / ('task-' + str(task_id)), image=args.image, memory_mb=1024)
            record = {'task_id': 'HumanEval/' + str(task_id), 'passed': False}
            try:
                record['sandbox'] = await sandbox.start()
                for mode in ('empty', 'reference'):
                    request = {'task_id': record['task_id'], 'mode': mode}
                    path = out / (str(task_id) + '-' + mode + '-request.json')
                    write_new(path, request)
                    encoded = base64.b64encode(path.read_bytes()).decode()
                    guest_path = '/app/request-' + mode + '.json'
                    writer = ('import os,base64; fd=os.open(' + repr(guest_path)
                              + ',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); '
                              + 'f=os.fdopen(fd,"wb"); f.write(base64.b64decode(' + repr(encoded) + ')); f.close()')
                    transfer = await sandbox.execute('python -I -B -c ' + shlex.quote(writer), timeout=10)
                    if transfer['return_code'] != 0 or transfer['boundary_failure']:
                        raise RuntimeError('Guest request transfer failed')
                    raw = await sandbox.execute('python -I -B /opt/humaneval_guest.py ' + guest_path, timeout=180)
                    record[mode] = {'execution': raw, 'native': parse_result(raw, request)}
                    if raw['boundary_failure']:
                        break
                record['passed'] = (not record['empty']['native']['passed']
                                    and record.get('reference', {}).get('native', {}).get('passed', False))
            except Exception as e:
                record['error'] = {'type': type(e).__name__, 'message': str(e)[-1000:]}
            finally:
                await sandbox.stop()
                record['containers_remaining'] = sandbox.ids()
                write_new(out / (str(task_id) + '.json'), record)
                records.append(record)
                if len(records) % 10 == 0:
                    print(json.dumps({'completed': len(records), 'qualified': sum(r['passed'] for r in records)}), flush=True)
    await asyncio.gather(*(task(k) for k in ids))
    summary = {'total': len(records), 'qualified': sum(r['passed'] for r in records),
               'failed': [r['task_id'] for r in records if not r['passed']], 'model_calls': 0,
               'containers_remaining': [c for r in records for c in r['containers_remaining']]}
    write_new(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    asyncio.run(run(parser.parse_args()))
