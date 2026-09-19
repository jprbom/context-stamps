"""Exercise the narrow allowlist against public IDs and generated fake canaries.

No real credentials are used or printed. Requires Gitleaks 8.30.1 or compatible.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path('evidence/replication-v1/per-query.jsonl')


def main(executable):
    executable = str(Path(executable).resolve())
    original = (ROOT / RELATIVE).read_text(encoding='utf-8').splitlines()[1731]
    # Generated, nonfunctional canary; never a user credential.
    fake = 'sk_' + hashlib.sha256(b'context-stamps-scanner-positive-control').hexdigest()
    with tempfile.TemporaryDirectory(prefix='context-stamps-scan-') as directory:
        folder = Path(directory).resolve()
        assert folder.is_relative_to(Path(tempfile.gettempdir()).resolve())
        scan = folder / 'scan'
        target = scan / RELATIVE
        target.parent.mkdir(parents=True)
        baseline = folder / 'baseline.toml'
        baseline.write_text('[extend]\nuseDefault = true\n', encoding='utf-8')

        def run(config):
            report = folder / 'report.json'
            result = subprocess.run([executable, 'dir', str(scan), '--config', str(config),
                                     '--report-format', 'json', '--report-path', str(report), '--redact'],
                                    capture_output=True, text=True, check=False)
            if result.returncode not in (0, 1):
                raise RuntimeError('scanner failed: ' + result.stderr)
            return json.loads(report.read_text(encoding='utf-8')) if report.exists() else []

        target.write_text(original + '\n', encoding='utf-8')
        assert len(run(baseline)) == 1, 'positive control must reproduce the original finding'
        assert not run(ROOT / '.gitleaks.toml'), 'reviewed public pair should be excluded only at its known path'

        # Same file and same line as the benign pair: the real rule remains active.
        target.write_text(original + ' api_key="' + fake + '"\n', encoding='utf-8')
        assert any(r['RuleID'] == 'generic-api-key' for r in run(ROOT / '.gitleaks.toml'))

        target.write_text('', encoding='utf-8')
        other = scan / 'application.txt'
        other.write_text(original + '\n', encoding='utf-8')
        assert run(ROOT / '.gitleaks.toml'), 'same pair outside the exact evidence path must remain detectable'

        other.write_text('', encoding='utf-8')
        target.write_text(original.replace('test-law-tahglcphsld-pro02a', fake) + '\n', encoding='utf-8')
        assert run(ROOT / '.gitleaks.toml'), 'changed neighboring token must remain detectable'

    print('Five scanner controls passed: default finding, scoped exclusion, same-line canary, wrong path, changed token.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gitleaks', required=True, type=Path)
    main(parser.parse_args().gitleaks)
