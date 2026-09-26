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

        target.write_text('', encoding='utf-8')
        checksum_cases = []
        for version in (1, 2):
            relative = Path(f'evidence/longbench-v2-pilot/run-v{version}/plan.json')
            lines = (ROOT / relative).read_text(encoding='utf-8').splitlines()
            for key in ('tokenizer.json', 'tokenizer_config.json'):
                line = next(line for line in lines if line.strip().startswith('"' + key + '":'))
                checksum_cases.append((relative, line))
        relative = Path('experiments/longbench_eval.py')
        line = next(line for line in (ROOT / relative).read_text().splitlines() if line.startswith('TOKEN_REV = '))
        checksum_cases.append((relative, line))
        for version in (1, 2):
            relative = Path(f'evidence/terminal-pilot-v1/attempt-{version}/plan.json')
            lines = (ROOT / relative).read_text(encoding='utf-8').splitlines()
            for key in ('experiments/qwen_token_count.py', 'tokenizer_sha256'):
                line = next(line for line in lines if line.strip().startswith('"' + key + '":'))
                checksum_cases.append((relative, line))
        relative = Path('evidence/terminal-pilot-v1/manifest.json')
        line = next(line for line in (ROOT / relative).read_text().splitlines()
                    if line.strip().startswith('"experiments/qwen_token_count.py":'))
        checksum_cases.append((relative, line))
        for relative, line in checksum_cases:
            control = scan / relative
            control.parent.mkdir(parents=True, exist_ok=True)
            control.write_text(line + '\n', encoding='utf-8')
            assert len(run(baseline)) == 1, 'public checksum should reproduce default finding'
            assert not run(ROOT / '.gitleaks.toml'), 'only the reviewed checksum/path is excluded'
            control.write_text(line + ' api_key="' + fake + '"\n', encoding='utf-8')
            assert run(ROOT / '.gitleaks.toml'), 'same-line canary remains detectable'
            control.write_text('', encoding='utf-8')
            other.write_text(line + '\n', encoding='utf-8')
            assert run(ROOT / '.gitleaks.toml'), 'public checksum is not globally excluded'
            other.write_text('', encoding='utf-8')
            import re

            changed = re.sub(r'[a-f0-9]{40,64}', fake, line)
            control.write_text(changed + '\n', encoding='utf-8')
            assert run(ROOT / '.gitleaks.toml'), 'changed checksum remains detectable'
            control.write_text('', encoding='utf-8')

    print(f'{5 + 5 * len(checksum_cases)} scanner controls passed: default findings, exact exclusions, same-line canaries, wrong paths, changed tokens.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gitleaks', required=True, type=Path)
    main(parser.parse_args().gitleaks)
