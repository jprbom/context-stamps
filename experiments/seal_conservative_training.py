"""Seal the completed, inactive conservative adapter's public training record."""

import argparse
import hashlib
import json
from pathlib import Path

from mbpp_training_data import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(directory):
    directory = directory.resolve()
    if directory != (ROOT / 'evidence/mbpp-conservative-adapter-v2').resolve():
        raise ValueError('Expected the reviewed public evidence directory')
    target = directory / 'manifest.json'
    if target.exists():
        raise FileExistsError(target)
    files = {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob('*') if p.is_file()}
    sources = ('experiments/verify_conservative_code_adapter.py',
               'experiments/mbpp_conservative_adapter.py',
               'experiments/seal_conservative_training.py')
    manifest = {'schema': 1, 'files': files, 'sources': {n: digest(ROOT / n) for n in sources},
                'activation': False,
                'notes': 'Training record only. HumanEval+ was inspected during candidate design; repeated results are development evidence.'}
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'evidence/mbpp-conservative-adapter-v2')
    seal(parser.parse_args().directory)
