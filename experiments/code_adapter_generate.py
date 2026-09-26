"""Paired offline code generation; never executes programs or sees hidden tests."""

import argparse
import ast
import contextlib
import gzip
import hashlib
import importlib.metadata
import json
import os
import random
import re
import time
from pathlib import Path

from local_adapter_train import check_base, sha
from mbpp_adapter import SYSTEM
from mbpp_training_data import ROOT, fresh, write_new

DATASET_SHA = '272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101'
CONFIG = {'seed': 7, 'batch_size': 4, 'max_new_tokens': 1024, 'max_prompt_tokens': 2048,
          'do_sample': False, 'attention': 'sdpa', 'dtype': 'bfloat16'}


def prompts(dataset):
    if sha(dataset) != DATASET_SHA:
        raise ValueError('Registered HumanEval+ release required')
    raw = gzip.decompress(dataset.read_bytes())
    if len(raw) > 10 * 1024**2:
        raise ValueError('Dataset expansion bound')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 164 or {r['task_id'] for r in rows} != {'HumanEval/' + str(i) for i in range(164)}:
        raise ValueError('Wrong task inventory')
    # Only public prompt fields cross into generation. The grader owns tests
    # and references; none become a model input or model-selection signal here.
    return [{'task_id': r['task_id'], 'prompt': r['prompt']} for r in
            sorted(rows, key=lambda r: int(r['task_id'].split('/')[1]))]


def extract_program(text):
    if type(text) is not str or len(text.encode()) > 32768:
        return {'code': '', 'valid_format': False, 'reason': 'size_or_type'}
    fences = list(re.finditer(r'^```(?:python)?[ \t]*\r?\n(.*?)^```[ \t]*$', text, re.M | re.S))
    if '```' in text:
        if len(fences) != 1 or text.count('```') != 2:
            return {'code': '', 'valid_format': False, 'reason': 'ambiguous_fences'}
        code = fences[0].group(1)
    else:
        code = text
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return {'code': code, 'valid_format': False, 'reason': 'syntax'}
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body):
        return {'code': code, 'valid_format': False, 'reason': 'missing_function'}
    return {'code': code, 'valid_format': True, 'reason': None}


def run(args):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ[name] = '1'
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = prompts(args.dataset)
    if args.smoke:
        rows = rows[:4]
    base = check_base(args.base, json.loads((args.run / 'base.json').read_text()))
    trained = json.loads((args.run / 'trained.json').read_text())
    adapter = args.run / 'adapter'
    if sha(adapter / 'adapter_model.safetensors') != trained['adapter_sha256'] or trained['activation']:
        raise ValueError('Expected unchanged inactive candidate')
    out = fresh(args.out)
    write_new(out / 'prompts.json', rows)
    names = ('experiments/code_adapter_generate.py', 'experiments/mbpp_adapter.py', 'experiments/local_adapter_train.py')
    write_new(out / 'plan.json', {'schema': 1, 'experiment': 'paired_public_code_generation_v1', 'config': CONFIG,
              'source_hashes': {n: sha(ROOT / n) for n in names}, 'dataset_sha256': DATASET_SHA,
              'base_revision': base['revision'], 'adapter_sha256': trained['adapter_sha256'],
              'adapter_config_sha256': sha(adapter / 'adapter_config.json'),
              'training_plan_sha256': sha(args.run / 'plan.json'), 'task_ids': [r['task_id'] for r in rows],
              'smoke': args.smoke, 'provider_charges_usd': 0, 'activation_allowed': False,
              'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'tokenizers')},
              'comparison': 'Identical BF16 base/backend with trained adapter enabled or disabled; paired randomized order',
              'prompt_protocol': 'Complete implementation instruction with public HumanEval prompt, no hidden tests/references',
              'limitations': ['Custom paired generation protocol, not a leaderboard reproduction',
                              'Four-row smoke is a development subset; full run requires all 164 tasks',
                              'One greedy completion, one seed, one 1.5B base; no frontier or edge-device qualification']})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = {}
    for row in rows:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['prompt']}]
        texts[row['task_id']] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(texts[row['task_id']], add_special_tokens=False)) > CONFIG['max_prompt_tokens']:
            raise ValueError('Prompt exceeds registered limit')
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda'})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    rng = random.Random(CONFIG['seed'])
    calls = 0
    with (out / 'generations.jsonl').open('x', encoding='utf-8', newline='\n') as log:
        for offset in range(0, len(rows), CONFIG['batch_size']):
            group = rows[offset:offset + CONFIG['batch_size']]
            x = tokenizer([texts[r['task_id']] for r in group], padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            arms = ['base', 'adapter']
            rng.shuffle(arms)
            for order, arm in enumerate(arms):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                context = model.disable_adapter() if arm == 'base' else contextlib.nullcontext()
                with context, torch.inference_mode():
                    output = model.generate(**x, max_new_tokens=CONFIG['max_new_tokens'], do_sample=False,
                                            use_cache=True, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                for i, row in enumerate(group):
                    tokens = output[i, x['input_ids'].shape[1]:].tolist()
                    # Count through the first EOS; subsequent batch padding is not generation.
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else eos or []
                    terminal = next((j for j, token in enumerate(tokens) if token in eos), None)
                    count = len(tokens) if terminal is None else terminal + 1
                    text = tokenizer.decode(tokens[:count], skip_special_tokens=True)
                    record = {'task_id': row['task_id'], 'arm': arm, 'order': order,
                              'prompt_sha256': hashlib.sha256(texts[row['task_id']].encode()).hexdigest(),
                              'prompt_tokens': int(x['attention_mask'][i].sum()), 'generated_tokens': count,
                              'reached_token_limit': terminal is None and count >= CONFIG['max_new_tokens'],
                              'batch_seconds': elapsed, 'batch_size': len(group), 'peak_allocated_cuda_bytes': peak,
                              'text': text, **extract_program(text)}
                    log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    calls += 1
                log.flush()
                print(json.dumps({'completed_generations': calls, 'arm': arm, 'batch_seconds': elapsed}), flush=True)
    write_new(out / 'completed.json', {'generations': calls, 'tasks': len(rows), 'activation': False,
                                       'graded': False, 'energy_joules': None, 'provider_charges_usd': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'base', 'run', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
