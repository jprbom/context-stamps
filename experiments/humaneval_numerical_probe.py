"""Inspect representable root residuals in the isolated benchmark environment."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
from pathlib import Path

from harbor_sandbox import ResearchSandbox
from humaneval_root_probe import CONTROL
from mbpp_training_data import ROOT, fresh, write_new

WORKER = '''import gzip,json,math,base64
from pathlib import Path
p=next(r for r in map(json.loads,gzip.decompress(Path('/opt/HumanEvalPlus.jsonl.gz').read_bytes()).splitlines()) if r['task_id']=='HumanEval/32')
xs=p['plus_input'][119][0]
records=[]
for name,code in [('canonical',p['prompt']+p['canonical_solution']),('bisection',base64.b64decode(CONTROL_B64).decode())]:
    ns={};exec(code,ns);root=ns['find_zero'](xs)
    nearby=[root];a,b=root,root
    for i in range(32):
        a=math.nextafter(a,-math.inf);b=math.nextafter(b,math.inf);nearby.extend([a,b])
    residuals=[{'root':x,'residual':sum(c*math.pow(x,i) for i,c in enumerate(xs))} for x in nearby]
    records.append({'method':name,'root':root,'residuals':residuals,'minimum_absolute_residual':min(abs(r['residual']) for r in residuals)})
print(json.dumps({'task_id':p['task_id'],'input_index':119,'coefficients':xs,'atol':p['atol'],'records':records}))
'''


async def run(args):
    os.environ['HARBOR_TELEMETRY'] = 'off'
    out = fresh(args.out)
    sandbox = ResearchSandbox(out / 'guest', image=args.image, memory_mb=1024)
    report = {'purpose': 'Numerical diagnostic, not a model benchmark', 'image': args.image,
              'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                          ('experiments/humaneval_numerical_probe.py', 'experiments/humaneval_root_probe.py',
                           'experiments/harbor_sandbox.py')}, 'model_calls': 0}
    try:
        report['sandbox'] = await sandbox.start()
        code = 'CONTROL_B64=' + repr(base64.b64encode(CONTROL.encode()).decode()) + '\n' + WORKER
        report['execution'] = await sandbox.execute('python -I -B -c ' + shlex.quote(code), timeout=30)
        if report['execution']['return_code'] == 0 and not report['execution']['boundary_failure']:
            report['result'] = json.loads(report['execution']['stdout'])
    finally:
        await sandbox.stop()
        report['containers_remaining'] = sandbox.ids()
        write_new(out / 'result.json', report)
    print(json.dumps({k: v for k, v in report.get('result', {}).items() if k != 'records'}))
    for r in report.get('result', {}).get('records', []):
        print(json.dumps({k: v for k, v in r.items() if k != 'residuals'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--out', type=Path, required=True)
    asyncio.run(run(p.parse_args()))
