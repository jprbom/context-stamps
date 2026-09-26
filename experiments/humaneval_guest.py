"""Container-only adapter to pinned, unmodified EvalPlus grading functions.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Never import or execute this module on the host with benchmark programs.
"""

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    # The outer Docker boundary, not EvalPlus's reliability_guard, isolates code.
    if not Path('/.dockerenv').is_file() or os.getuid() != 1000:
        raise RuntimeError('Registered non-root container required')
    os.environ['EVALPLUS_MAX_MEMORY_BYTES'] = str(768 * 1024**2)
    os.environ.pop('EVALPLUS_TIMEOUT_PER_TASK', None)
    from evalplus.eval import untrusted_check
    from evalplus.gen.util import trusted_exec

    request = json.loads(Path(sys.argv[1]).read_text())
    task_id = request['task_id']
    rows = [json.loads(s) for s in gzip.decompress(Path('/opt/HumanEvalPlus.jsonl.gz').read_bytes()).splitlines()]
    problem = next(r for r in rows if r['task_id'] == task_id)
    reference = problem['prompt'] + problem['canonical_solution']
    mode = request['mode']
    if mode not in ('reference', 'empty', 'candidate'):
        raise ValueError('Unknown grading mode')
    code = reference if mode == 'reference' else '' if mode == 'empty' else request['code']
    if type(code) is not str or len(code.encode()) > 32768:
        raise ValueError('Bounded program required')
    result = {'task_id': task_id, 'mode': mode, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'checks': {}, 'passed': False}
    for kind in ('base', 'plus'):
        inputs = problem[kind + '_input']
        started = time.perf_counter()
        # Canonical programs execute only inside this disposable guest as well.
        expected, ref_time = trusted_exec(reference, inputs, problem['entry_point'], record_time=True)
        oracle_seconds = time.perf_counter() - started
        started = time.perf_counter()
        status, details = untrusted_check('humaneval', code, inputs, problem['entry_point'],
                                         expected, problem['atol'], ref_time, fast_check=True)
        details = [bool(v) for v in details]
        result['checks'][kind] = {'status': status, 'inputs': len(inputs), 'completed': len(details),
                                  'passing': sum(details), 'details_sha256': hashlib.sha256(bytes(details)).hexdigest(),
                                  'failed_indices': [i for i, ok in enumerate(details) if not ok],
                                  'oracle_seconds': oracle_seconds, 'check_seconds': time.perf_counter() - started}
    result['passed'] = all(v['status'] == 'pass' and v['completed'] == v['inputs'] == v['passing']
                           for v in result['checks'].values())
    print('SCQR_HUMANEVAL_RESULT:' + json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
