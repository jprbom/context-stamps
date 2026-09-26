"""Typed-tool ambiguity, confinement and execution-boundary checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import terminal_pilot as pilot
from harbor_sandbox import MAX_COMMAND
from terminal_tools import MAX_FILE, execute, parse_action, relative_path


class ToolContractTests(unittest.TestCase):
    def test_each_valid_variant(self):
        for action in ({"tool": "finish"}, {"tool": "read_file", "path": "foo.py"},
                       {"tool": "write_file", "path": "/app/foo.py", "content": "print(42)\n"},
                       {"tool": "shell", "command": "ls -la"}):
            self.assertEqual(parse_action(json.dumps(action)), action)

    def test_no_ambiguous_mixed_or_duplicate_actions(self):
        for action in ('{"tool":"finish","command":"echo unsafe"}',
                       '{"tool":"finish","tool":"shell"}', '{"tool":"write_file","path":"x","content":null}',
                       '{"tool":"read_file","path":"x","content":"overwrite"}',
                       '{"tool":"shell","command":" "}', '{"tool":{}}', '[]'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                parse_action(action)

    def test_parent_absolute_and_control_paths_refused(self):
        for path in ('../x', '/etc/x', '/app/../x', 'a/../../b', 'a//b', './a', 'a/./b',
                     '/app/', '/app', 'a\\b', 'a\x00b', 'a\nb'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(path)
        self.assertEqual(relative_path('/app/a/b.py'), 'a/b.py')

    def test_utf8_byte_budget_not_character_budget(self):
        with self.assertRaises(ValueError):
            parse_action(json.dumps({'tool': 'write_file', 'path': 'x', 'content': '\u20b9' * (MAX_FILE // 3 + 1)}))

    def test_wire_preserves_tool_discriminant_order_for_all_union_branches(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'{}'
        with patch.object(pilot.urllib.request, 'build_opener', return_value=opener):
            pilot.generate('format probe', interface='typed-files-v3')
        request = opener.open.call_args.args[0]
        body = json.loads(request.data)
        for variant in body['format']['oneOf']:
            self.assertEqual(next(iter(variant['properties'])), 'tool')


class ToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_typed_write_and_finish_flow_reaches_grading(self):
        await self.run_flow('typed-files-v1', False)

    async def test_missing_artifact_rejects_finish_without_exposing_grader(self):
        await self.run_flow('typed-files-v2', True)

    async def test_ordered_interface_keeps_completion_check(self):
        await self.run_flow('typed-files-v3', True)

    async def run_flow(self, interface, premature):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'tasks' / pilot.TASKS[0]
            task.mkdir(parents=True)
            (task / 'instruction.md').write_text('Create the requested fixture artifact.')
            args = SimpleNamespace(source=root, out=root, token_python=None, tokenizer=None)
            sandbox = AsyncMock()
            sandbox.closed = False
            sandbox.start.return_value = {}
            sandbox.execute.return_value = {'return_code': 0, 'boundary_failure': None, 'stdout': '', 'stderr': ''}
            actions = [{'tool': 'write_file', 'path': 'run.py', 'content': 'pass\n'}, {'tool': 'finish'}]
            if premature:
                actions.insert(0, {'tool': 'finish'})
            responses = [{'done': True, 'done_reason': 'stop', 'prompt_eval_count': 1, 'eval_count': 1,
                          'response': json.dumps(a)} for a in actions]
            with (patch.object(pilot, 'ResearchSandbox', return_value=sandbox),
                  patch.object(pilot, 'token_count', return_value={'tokens': 1, 'fixture': True}),
                  patch.object(pilot, 'generate', side_effect=responses) as generate,
                  patch.object(pilot, 'collect_artifact', new_callable=AsyncMock, return_value='cGFzcwo=',
                               side_effect=[None, 'cGFzcwo=', 'cGFzcwo='] if premature else None),
                  patch.object(pilot, 'grade', new_callable=AsyncMock, return_value={'passed': True}) as grade):
                result = await pilot.run_task(args, {'tokenizer': {'fixture': True}, 'verifier_images': {},
                                                    'model': {'name': pilot.MODEL}, 'interface': interface},
                                             pilot.TASKS[0])
            self.assertEqual(result['status'], 'agent_done')
            self.assertEqual(result['calls'], 3 if premature else 2)
            sandbox.execute.assert_awaited_once()
            sandbox.stop.assert_awaited_once()
            self.assertEqual(grade.call_args.kwargs['artifact'], b'pass\n')
            self.assertEqual(generate.call_args.args[2], interface)
            grade.assert_awaited_once()
            if premature:
                feedback_prompt = generate.call_args_list[1].args[0]
                self.assertIn('Cannot finish: required regular artifact /app/run.py', feedback_prompt)
                self.assertNotIn('test_outputs.py', feedback_prompt)

    async def test_invalid_action_never_reaches_sandbox(self):
        sandbox = AsyncMock()
        for action in ({'tool': 'read_file', 'path': '../../host'}, {'tool': 'finish'},
                       {'tool': 'shell', 'command': 'x' * (MAX_COMMAND + 1)}):
            with self.assertRaises(ValueError):
                await execute(sandbox, action)
        sandbox.execute.assert_not_called()

    async def test_file_payload_is_not_interpolated_into_code(self):
        sandbox = AsyncMock()
        value = "`$()';raise RuntimeError('never execute this')\n"
        await execute(sandbox, {'tool': 'write_file', 'path': 'file name.py', 'content': value})
        command = sandbox.execute.call_args.args[0]
        self.assertNotIn(value, command)
        self.assertTrue(command.startswith('python -I -c '))

    async def test_maximum_escaped_payload_fits_windows_command_budget(self):
        sandbox = AsyncMock()
        await execute(sandbox, {'tool': 'write_file', 'path': 'x' * 250, 'content': '\x00' * MAX_FILE})
        self.assertLessEqual(len(sandbox.execute.call_args.args[0].encode()), MAX_COMMAND)


if __name__ == '__main__':
    unittest.main()
