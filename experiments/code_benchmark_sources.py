"""Recreate the pinned minimal HumanEval+ grading image without hosted models."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from mbpp_training_data import ROOT, fresh, write_new

MANIFEST = ROOT / 'evidence/humaneval-grader-v1/source.json'
GRADER_FILES = ('evalplus/__init__.py', 'evalplus/config.py', 'evalplus/eval/__init__.py',
                'evalplus/eval/utils.py', 'evalplus/eval/_special_oracle.py',
                'evalplus/gen/__init__.py', 'evalplus/gen/util/__init__.py')
ALLOWED_HOSTS = {'raw.githubusercontent.com', 'github.com', 'release-assets.githubusercontent.com',
                 'objects.githubusercontent.com', 'files.pythonhosted.org'}


def checked_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.username
            or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Unreviewed source host')
    return url


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return super().redirect_request(request, fp, code, message, headers, checked_url(newurl))


def inventory():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def verify(source):
    for name, record in inventory()['files'].items():
        path = source / name
        if path.is_symlink() or path.stat().st_size != record['bytes']:
            raise ValueError('Changed source size: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Changed source checksum: ' + name)


def fetch(out):
    out = fresh(out)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), CheckedRedirect())
    for name, record in inventory()['files'].items():
        if '\\' in name or '..' in Path(name).parts or Path(name).is_absolute():
            raise ValueError('Invalid relative source path')
        with opener.open(checked_url(record['url']), timeout=60) as response:
            raw = response.read(record['bytes'] + 1)
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Download checksum mismatch: ' + name)
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    verify(out)


def build(source, out):
    verify(source)
    out = fresh(out)
    manifest = inventory()
    names = (*GRADER_FILES, 'LICENSE', 'HumanEvalPlus.jsonl.gz', 'HumanEval-LICENSE', 'HumanEvalPlus-LICENSE')
    for name in names:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (out / 'wheels').mkdir()
    for record in manifest['wheels'].values():
        shutil.copyfile(source / 'wheels' / record['filename'], out / 'wheels' / record['filename'])
    shutil.copyfile(ROOT / 'experiments/humaneval_guest.py', out / 'humaneval_guest.py')
    (out / 'requirements.txt').write_text(''.join(f"{n}=={r['version']} --hash=sha256:{r['sha256']}\n"
                                                 for n, r in manifest['wheels'].items()), encoding='utf-8', newline='\n')
    dockerfile = '''FROM BASE_IMAGE
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY wheels /opt/wheels
COPY requirements.txt /opt/requirements.txt
RUN python -I -m pip install --no-index --no-deps --require-hashes --no-cache-dir --find-links=/opt/wheels -r /opt/requirements.txt
COPY evalplus /usr/local/lib/python3.13/site-packages/evalplus
COPY HumanEvalPlus.jsonl.gz humaneval_guest.py LICENSE HumanEval-LICENSE HumanEvalPlus-LICENSE /opt/
USER 1000:1000
WORKDIR /app
'''.replace('BASE_IMAGE', manifest['base_image'])
    (out / 'Dockerfile').write_text(dockerfile, encoding='utf-8', newline='\n')
    write_new(out / 'build-manifest.json', {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in out.rglob('*') if p.is_file()})
    with (out / 'build.log').open('x', encoding='utf-8') as log:
        subprocess.run(['docker', 'build', '--network=none', '--iidfile', str(out / 'image-id.txt'), str(out)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    print((out / 'image-id.txt').read_text().strip())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('fetch', 'verify', 'build'))
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.action == 'fetch':
        if args.out is None:
            p.error('--out required')
        fetch(args.out)
    else:
        if args.source is None or (args.action == 'build' and args.out is None):
            p.error('--source required; build also requires --out')
        if args.action == 'verify':
            verify(args.source)
            print('Pinned dataset, grader, licenses and wheels verified')
        else:
            build(args.source, args.out)
