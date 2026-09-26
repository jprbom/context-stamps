"""Export a completed local comparison as compact, data-only public evidence."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from mbpp_training_data import ROOT, fresh, write_new
from summarize_code_comparison import summarize


def export(generation, grading, destination):
    report = summarize(generation, grading)
    out = fresh(destination)
    def read(folder, name):
        return json.loads((folder / name).read_text(encoding='utf-8'))
    data = {'generation_plan': read(generation, 'plan.json'),
            'generation_completed': read(generation, 'completed.json'),
            'grading_plan': read(grading, 'plan.json'), 'grading_summary': read(grading, 'summary.json'),
            'grading_records': [read(grading, str(i) + '-' + arm + '.json')
                                for i in range(164) for arm in ('base', 'adapter')]}
    (out / 'records.json.gz').write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    (out / 'generations.jsonl.gz').write_bytes(gzip.compress((generation / 'generations.jsonl').read_bytes(), mtime=0))
    write_new(out / 'summary.json', report)
    names = ('experiments/summarize_code_comparison.py', 'experiments/export_code_comparison.py',
             'experiments/verify_full_code_eval.py', 'experiments/test_code_statistics.py')
    write_new(out / 'manifest.json', {'schema': 1,
              'scope': 'One local generic code-adaptation control; not Context Stamps treatment or automatic activation',
              'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
              'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names},
              'licenses': {'benchmark_notices': 'evidence/humaneval-grader-v1',
                           'training_attribution': 'evidence/mbpp-training-v1',
                           'adapter_card': 'evidence/mbpp-code-adapter-v1/MODEL_CARD.md'},
              'limitations': report['limitations']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('generation', 'grading', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.generation, args.grading, args.out), indent=2))
