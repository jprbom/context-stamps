"""Data-only replay of typed tools, retained regressions and schema canaries."""

import hashlib
import json
from pathlib import Path

import terminal_tools
from source_evidence import verify_sources
from terminal_pilot import render
from terminal_schema_probe import request_body
from verify_terminal_pilot import read, sha, verify_grader

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/terminal-typed-v1"


def verify_run(directory):
    plan, summary = read(directory / 'plan.json'), read(directory / 'summary.json')
    verify_sources(plan['source_hashes'])
    assert summary['model_unchanged'] and summary['provider_charges_usd'] == 0
    assert read(directory / 'started.json')['plan_sha256'] == sha(directory / 'plan.json')
    controls = read(directory / 'graders-qualified.json')
    assert controls['plan_sha256'] == sha(directory / 'plan.json')
    assert controls['passed'] and controls['model_calls'] == 0 and len(controls['records']) == 4
    for item in controls['records']:
        verify_grader(item['task'], item['result'])
        assert item['result']['passed'] == item['oracle']
    calls, inputs, outputs, passed = 0, 0, 0, 0
    for row in summary['rows']:
        record = read(directory / (row['task'] + '-trajectory.json'))
        grade = read(directory / (row['task'] + '-grade.json'))
        verify_grader(row['task'], grade)
        assert grade['passed'] == row['passed']
        assert record['status'] == row['status']
        assert row['calls'] == len(record['attempts']) <= plan['max_steps']
        messages, errors = record['initial_messages'].copy(), 0
        for attempt in record['attempts']:
            prompt = render(messages)
            assert hashlib.sha256(prompt.encode()).hexdigest() == attempt['prompt_sha256']
            response = attempt['response']
            assert attempt['input_tokens'] == response['prompt_eval_count'] and response['done']
            calls += 1
            inputs += response['prompt_eval_count']
            outputs += response['eval_count']
            if response.get('done_reason') == 'length':
                assert 'tool_result' not in attempt
                continue
            messages.append({'role': 'assistant', 'content': response['response']})
            try:
                action = terminal_tools.parse_action(response['response'])
            except ValueError:
                errors += 1
                assert attempt['format_error'] == 'invalid_action' and 'tool_result' not in attempt
                if errors <= 2:
                    messages.append({'role': 'user', 'content': terminal_tools.REPAIR})
                continue
            if action['tool'] == 'finish' and not attempt.get('finish_rejected'):
                assert record['status'] == 'agent_done' and 'tool_result' not in attempt
                continue
            if attempt.get('finish_rejected'):
                assert action['tool'] == 'finish' and plan['interface'] in ('typed-files-v2', 'typed-files-v3')
                assert attempt['tool_result']['return_code'] == 1 and attempt['finish_check_seconds'] >= 0
            messages.append({'role': 'user', 'content': json.dumps({'tool_result': attempt['tool_result']}, ensure_ascii=False)})
        assert row['input_tokens'] == sum(a['response']['prompt_eval_count'] for a in record['attempts'])
        assert row['output_tokens'] == sum(a['response']['eval_count'] for a in record['attempts'])
        passed += row['passed']
    return {'calls': calls, 'inputs': inputs, 'outputs': outputs, 'passed': passed, 'tasks': len(summary['rows'])}


def verify_probe(directory):
    plan, summary = read(directory / 'plan.json'), read(directory / 'summary.json')
    verify_sources(plan['source_hashes'])
    assert summary['commands_executed'] == 0 and summary['model_unchanged']
    assert summary['provider_charges_usd'] == 0 and summary['calls'] == 12
    counts = dict.fromkeys(plan['modes'], 0)
    for index, target in enumerate(plan['targets']):
        for mode in plan['modes']:
            row = read(directory / (str(index) + '-' + mode + '.json'))
            prompt, wire = request_body(plan['model']['name'], target, mode)
            assert hashlib.sha256(wire).hexdigest() == row['request_sha256']
            assert row['input_tokens'] == row['response']['prompt_eval_count']
            assert row['matches_target'] == (json.loads(row['response']['response']) == target)
            assert json.loads(wire)['prompt'] == prompt
            counts[mode] += row['matches_target']
    assert counts == summary['matches']
    assert counts['sorted_union'] == 0 and counts['ordered_union'] == counts['json_only'] == 3
    return counts


def main():
    manifest = read(EVIDENCE / 'manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(EVIDENCE / name) == digest, name
    verify_sources(manifest['source_hashes'])
    qualification = read(EVIDENCE / 'tool-qualification.json')
    verify_sources(qualification['source_hashes'])
    assert qualification['passed'] and qualification['model_calls'] == 0
    assert not qualification['containers_remaining'] and len(qualification['checks']) == 12
    for case in qualification['checks']:
        assert (case['result']['return_code'] == 0) == case['expected_success']
        assert case['result']['boundary_failure'] is None
    for name, expected in manifest['runs'].items():
        assert verify_run(EVIDENCE / name) == expected, name
    for name in ('schema-small', 'schema-reference'):
        print(name, verify_probe(EVIDENCE / name))
    print('Typed-tool traces, native graders, literal file probes and schema ordering canaries verified.')
    print('Engineering/development controls only; no benchmark training or local-learning qualification.')


if __name__ == '__main__':
    main()
