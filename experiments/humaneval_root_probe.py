"""Diagnostic numerical control, not a generated model result or training data."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
from pathlib import Path

from harbor_sandbox import ResearchSandbox
from humaneval_local import parse_result
from mbpp_training_data import ROOT, fresh, write_new

# The upstream oracle evaluates the residual, rather than requiring one root.
# Bracket/bisect is an independent diagnostic for the shipped Newton reference.
CONTROL = '''import math
def find_zero(xs):
    def poly(x):
        return sum(c * math.pow(x, i) for i, c in enumerate(xs))
    lo, hi = -1.0, 1.0
    for _ in range(100):
        flo, fhi = poly(lo), poly(hi)
        if flo == 0: return lo
        if fhi == 0: return hi
        if (flo < 0) != (fhi < 0): break
        lo, hi = lo * 2, hi * 2
    else: raise ValueError('No bracket found')
    best = lo if abs(flo) < abs(fhi) else hi
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = poly(mid)
        if abs(fm) < abs(poly(best)): best = mid
        if fm == 0 or mid == lo or mid == hi: break
        if (fm < 0) == (flo < 0): lo, flo = mid, fm
        else: hi = mid
    return best
'''


async def run(args):
    os.environ['HARBOR_TELEMETRY'] = 'off'
    if not args.image.startswith('sha256:') or len(args.image) != 71:
        raise ValueError('Pinned local image required')
    out = fresh(args.out)
    request = {'task_id': 'HumanEval/32', 'mode': 'candidate', 'code': CONTROL}
    write_new(out / 'request.json', request)
    report = {'purpose': 'Authored numerical diagnostic; never a model benchmark', 'image': args.image,
              'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                          ('experiments/humaneval_root_probe.py', 'experiments/humaneval_local.py',
                           'experiments/humaneval_guest.py', 'experiments/harbor_sandbox.py')},
              'model_calls': 0, 'training_runs': 0}
    sandbox = ResearchSandbox(out / 'guest', image=args.image, memory_mb=1024)
    try:
        report['sandbox'] = await sandbox.start()
        payload = base64.b64encode(json.dumps(request).encode()).decode()
        writer = ('import os,base64;fd=os.open("/app/request.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);'
                  'f=os.fdopen(fd,"wb");f.write(base64.b64decode(' + repr(payload) + '));f.close()')
        raw = await sandbox.execute('python -I -B -c ' + shlex.quote(writer), timeout=10)
        if raw['return_code'] != 0 or raw['boundary_failure']:
            raise ValueError('Transfer failure')
        raw = await sandbox.execute('python -I -B /opt/humaneval_guest.py /app/request.json', timeout=180)
        report.update(execution=raw, native=parse_result(raw, request))
    finally:
        await sandbox.stop()
        report['containers_remaining'] = sandbox.ids()
        write_new(out / 'result.json', report)
    print(json.dumps(report['native']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out', type=Path, required=True)
    asyncio.run(run(p.parse_args()))
