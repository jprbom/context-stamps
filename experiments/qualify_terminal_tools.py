"""Live typed-file probes inside an isolated local Harbor container."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from harbor_sandbox import ResearchSandbox
from terminal_tools import execute


async def qualify(out):
    out.mkdir(parents=True, exist_ok=False)
    sandbox = ResearchSandbox(out / 'sandbox')
    report = {'schema': 1, 'model_calls': 0, 'checks': [], 'source_hashes': {}}
    for name in ('harbor_sandbox.py', 'terminal_tools.py', 'qualify_terminal_tools.py'):
        path = Path(__file__).with_name(name)
        report['source_hashes']['experiments/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        report['profile'] = await sandbox.start()
        content = 'literal shell text: $(whoami) `echo nope`\nUnicode: \u20b9\n'
        cases = [
            ({'tool': 'write_file', 'path': 'nested/file.txt', 'content': content}, True),
            ({'tool': 'read_file', 'path': '/app/nested/file.txt'}, True),
            ({'tool': 'shell', 'command': "ln -s /etc /app/alias && mkfifo /app/pipe"}, True),
            ({'tool': 'write_file', 'path': 'alias/escape', 'content': 'probe'}, False),
            ({'tool': 'read_file', 'path': 'alias/passwd'}, False),
            ({'tool': 'read_file', 'path': 'pipe'}, False),
            ({'tool': 'write_file', 'path': 'json.py', 'content': 'raise RuntimeError("shadow")'}, True),
            ({'tool': 'read_file', 'path': 'nested/file.txt'}, True),
            ({'tool': 'write_file', 'path': 'largest', 'content': '\x00' * 4096}, True),
            ({'tool': 'read_file', 'path': 'largest'}, True),
            ({'tool': 'shell', 'command': 'printf x >> /app/largest'}, True),
            ({'tool': 'read_file', 'path': 'largest'}, False),
        ]
        for action, expected in cases:
            result = await execute(sandbox, action)
            report['checks'].append({'action': action, 'expected_success': expected, 'result': result})
            if (result['return_code'] == 0) != expected or result['boundary_failure'] is not None:
                raise ValueError('Unexpected file-tool result')
            if action['tool'] == 'read_file' and expected:
                value = json.loads(result['stdout'])['content']
                if value != ('\x00' * 4096 if action['path'] == 'largest' else content):
                    raise ValueError('Literal content roundtrip failed')
        report['passed'] = True
    except BaseException as error:
        report['passed'] = False
        report['error'] = type(error).__name__ + ': ' + str(error)[-1000:]
        raise
    finally:
        await sandbox.stop()
        report['containers_remaining'] = sandbox.ids()
        (out / 'qualification.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    asyncio.run(qualify(parser.parse_args().out))
