"""Grade retained model programs in separate offline containers, never on host."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
from pathlib import Path

from code_adapter_generate import extract_program
from harbor_sandbox import ResearchSandbox
from humaneval_local import parse_result
from mbpp_training_data import ROOT, fresh, write_new


def generation_rows(run):
    plan = json.loads((run / 'plan.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    rows = [json.loads(s) for s in (run / 'generations.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(k, arm) for k in plan['task_ids'] for arm in ('base', 'adapter')}
    if (len(rows) != len(expected) or completed['generations'] != len(rows)
            or {(r['task_id'], r['arm']) for r in rows} != expected):
        raise ValueError('Incomplete paired generation inventory')
    if any(r['task_id'] not in {'HumanEval/' + str(i) for i in range(164)} for r in rows):
        raise ValueError('Unknown benchmark task')
    for row in rows:
        if any(row[k] != v for k, v in extract_program(row['text']).items()):
            raise ValueError('Postprocessing differs from retained generation')
    return rows, plan


async def transfer_request(sandbox, request):
    data = json.dumps(request).encode()
    if len(data) > 131072:
        raise ValueError('Request size bound')
    for offset in range(0, len(data), 2048):
        payload = base64.b64encode(data[offset:offset + 2048]).decode()
        flags = 'os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW' if offset == 0 else 'os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW'
        writer = ('import os,base64;fd=os.open("/app/request.json",' + flags + ',0o600);'
                  'f=os.fdopen(fd,"wb");f.write(base64.b64decode(' + repr(payload) + '));f.close()')
        raw = await sandbox.execute('python -I -B -c ' + shlex.quote(writer), timeout=10)
        if raw['return_code'] != 0 or raw['boundary_failure']:
            raise ValueError('Guest request transfer failed')


async def run(args):
    os.environ['HARBOR_TELEMETRY'] = 'off'
    if not args.image.startswith('sha256:') or len(args.image) != 71:
        raise ValueError('Pinned image required')
    rows, generation_plan = generation_rows(args.run)
    out = fresh(args.out)
    source_names = ('experiments/grade_code_generation.py', 'experiments/code_adapter_generate.py',
                    'experiments/humaneval_local.py', 'experiments/humaneval_guest.py', 'experiments/harbor_sandbox.py')
    write_new(out / 'plan.json', {'schema': 1, 'image': args.image, 'concurrency': 2,
              'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in source_names},
              'generation_sha256': hashlib.sha256((args.run / 'generations.jsonl').read_bytes()).hexdigest(),
              'generation_plan': generation_plan, 'model_calls': 0,
              'grading': 'Native pinned EvalPlus base and plus tests; no changed tolerances or dropped tasks',
              'limitations': 'Outer isolation protects the host; native grading is not adversarially tamper-proof'})
    semaphore = asyncio.Semaphore(2)
    records = []

    async def grade(row):
        async with semaphore:
            number = row['task_id'].split('/')[1]
            name = number + '-' + row['arm']
            record = {'task_id': row['task_id'], 'arm': row['arm'], 'passed': False}
            if not row['valid_format'] or row['reached_token_limit']:
                record['reason'] = 'invalid_format_or_generation_truncated'
                record['containers_remaining'] = []
                write_new(out / (name + '.json'), record)
                records.append(record)
                return
            sandbox = ResearchSandbox(out / ('guest-' + name), image=args.image, memory_mb=1024)
            try:
                record['sandbox'] = await sandbox.start()
                request = {'task_id': row['task_id'], 'mode': 'candidate', 'code': row['code']}
                await transfer_request(sandbox, request)
                raw = await sandbox.execute('python -I -B /opt/humaneval_guest.py /app/request.json', timeout=180)
                record['execution'] = raw
                record['native'] = parse_result(raw, request)
                record['passed'] = record['native']['passed']
            except Exception as e:
                record['error'] = {'type': type(e).__name__, 'message': str(e)[-1000:]}
            finally:
                await sandbox.stop()
                record['containers_remaining'] = sandbox.ids()
                write_new(out / (name + '.json'), record)
                records.append(record)
                print(json.dumps({'completed': len(records), 'total': len(rows)}), flush=True)
    await asyncio.gather(*(grade(row) for row in rows))
    by_key = {(r['task_id'], r['arm']): r for r in records}
    ids = generation_plan['task_ids']
    summary = {'tasks': len(ids), 'evaluations': len(records),
               'base_passed': sum(by_key[(k, 'base')]['passed'] for k in ids),
               'adapter_passed': sum(by_key[(k, 'adapter')]['passed'] for k in ids),
               'gained': [k for k in ids if not by_key[(k, 'base')]['passed'] and by_key[(k, 'adapter')]['passed']],
               'regressed': [k for k in ids if by_key[(k, 'base')]['passed'] and not by_key[(k, 'adapter')]['passed']],
               'errors': [(r['task_id'], r['arm']) for r in records if 'error' in r],
               'containers_remaining': [c for r in records for c in r['containers_remaining']],
               'activation': False, 'provider_charges_usd': 0}
    write_new(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    asyncio.run(run(p.parse_args()))
