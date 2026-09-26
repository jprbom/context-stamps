"""Offline candidate-only code SFT from independently executed MBPP references.

Copyright (c) 2026 Prashant Jagtap. MIT License.
MBPP data: Austin et al. (2021), CC BY 4.0; Qwen base: Apache 2.0.
This is an adaptation control, not evidence of a Context Stamps advantage.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

from code_overlap_audit import against_humaneval
from local_adapter_train import REPO, REVISION, check_base, frozen_digest, sha
from mbpp_training_data import ROOT, fresh, training_rows, write_new
from qualify_mbpp_training import parse_result
from source_evidence import verify_sources

CONFIG = {'seed': 7, 'rank': 8, 'alpha': 16, 'target_modules': ['q_proj', 'v_proj'],
          'learning_rate': 0.0001, 'epochs': 2, 'batch_size': 4, 'max_length': 1024,
          'dtype': 'bfloat16', 'attention': 'sdpa', 'weight_decay': 0.01, 'gradient_clip': 1.}
SYSTEM = 'Solve the Python programming task. Return the complete implementation in one Python code block, without explanation.'


def qualified_rows(source, qualification):
    rows, audit = training_rows(source)
    plan = json.loads((qualification / 'plan.json').read_text())
    summary = json.loads((qualification / 'summary.json').read_text())
    verify_sources(plan['source_hashes'])
    if (plan['task_ids'] != list(range(601, 975)) or plan['source_audit'] != audit
            or summary['total'] != 374 or summary['containers_remaining']
            or summary['model_calls'] != 0 or summary['training_runs'] != 0):
        raise ValueError('Complete training-only execution qualification required')
    eligible = []
    record_hashes = {}
    for row in rows:
        path = qualification / (str(row['task_id']) + '.json')
        record = json.loads(path.read_text())
        record_hashes[path.name] = sha(path)
        if record['task_id'] != row['task_id'] or record['containers_remaining']:
            raise ValueError('Wrong task or incomplete cleanup')
        if any(record[k] != v for k, v in audit[str(row['task_id'])].items()):
            raise ValueError('Changed training source')
        for mode in ('empty', 'reference'):
            if mode in record and parse_result(record[mode]['execution']) != record[mode]['native']:
                raise ValueError('Changed native execution result')
        ok = (not record['exact_reserved_overlap'] and 'error' not in record
              and not record.get('empty', {}).get('native', {}).get('passed', True)
              and record.get('reference', {}).get('native', {}).get('passed', False))
        if record['eligible'] != ok:
            raise ValueError('Unverified training admission')
        if ok:
            eligible.append(row)
    if [r['task_id'] for r in eligible] != summary['eligible']:
        raise ValueError('Changed eligibility summary')
    return eligible, record_hashes


def messages(row):
    if type(row['task_id']) is not int or not 601 <= row['task_id'] <= 974:
        raise ValueError('Reserved MBPP data cannot enter SFT')
    # These are TRAINING examples. No reserved benchmark assertions are shown.
    prompt = row['text'] + '\n\nExamples the implementation must satisfy:\n' + '\n'.join(row['test_list'])
    code = '\n'.join(s for s in (row.get('test_setup_code', ''), row['code']) if s)
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': '```python\n' + code.rstrip() + '\n```'}]


def tokenize_training(tokenizer, row):
    chat = messages(row)
    prefix = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=False)
    if full[:len(prefix)] != prefix or not 0 < len(prefix) < len(full) <= CONFIG['max_length']:
        raise ValueError('Invalid assistant boundary or overlong sample; no silent truncation')
    return {'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):],
            'prompt_tokens': len(prefix), 'target_tokens': len(full) - len(prefix)}


def run(args):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ[name] = '1'
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, hashes = qualified_rows(args.source, args.qualification)
    overlap = against_humaneval(rows, args.heldout_source)
    rows = [r for r in rows if r['task_id'] not in overlap['excluded_task_ids']]
    if len(rows) < 200:
        raise ValueError('Insufficient qualified training references')
    base = check_base(args.base, json.loads(args.base_manifest.read_text()))
    out = fresh(args.out)
    write_new(out / 'training-data.json', rows)
    write_new(out / 'base.json', base)
    write_new(out / 'overlap-audit.json', overlap)
    sources = ('experiments/mbpp_adapter.py', 'experiments/mbpp_training_data.py',
               'experiments/qualify_mbpp_training.py', 'experiments/local_adapter_train.py', 'experiments/source_evidence.py',
               'experiments/code_overlap_audit.py')
    write_new(out / 'plan.json', {'schema': 1, 'experiment': 'public_mbpp_code_sft_v1', 'config': CONFIG,
              'source_hashes': {n: sha(ROOT / n) for n in sources}, 'data_sha256': sha(out / 'training-data.json'),
              'qualification_record_hashes': hashes, 'qualification_plan_sha256': sha(args.qualification / 'plan.json'),
              'qualification_summary_sha256': sha(args.qualification / 'summary.json'),
              'base_manifest_sha256': sha(out / 'base.json'), 'python': platform.python_version(),
              'overlap_audit_sha256': sha(out / 'overlap-audit.json'),
              'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors')},
              'provider_charges_usd': 0, 'promotion_allowed': False,
              'selection': 'Fixed final epoch, no evaluation-directed hyperparameter selection',
              'limitations': ['Three native training assertions are an admission filter, not a correctness proof',
                              'Public pretraining contamination and semantic overlap remain unknown',
                              'Generic SFT control; no stamp-specific benefit or edge deployment qualification']})
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Registered experiment requires local BF16 CUDA')
    torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {r['task_id']: tokenize_training(tokenizer, r) for r in rows}
    write_new(out / 'tokenization.json', {str(k): {'prompt': v['prompt_tokens'], 'target': v['target_tokens']}
                                         for k, v in encoded.items()})
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda'})
    config = LoraConfig(r=CONFIG['rank'], lora_alpha=CONFIG['alpha'], lora_dropout=0.,
                        target_modules=CONFIG['target_modules'], task_type='CAUSAL_LM', bias='none')
    model = get_peft_model(model, config)
    model.config.use_cache = False
    before = frozen_digest(model)
    if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
        raise ValueError('Only LoRA weights may change')
    trainable = [p for p in model.parameters() if p.requires_grad]
    write_new(out / 'loaded.json', {'seconds': time.perf_counter() - started,
              'frozen_digest': before, 'gpu': torch.cuda.get_device_name(0),
              'trainable_parameters': sum(p.numel() for p in trainable)})
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    rng = random.Random(CONFIG['seed'])
    with (out / 'training.jsonl').open('x', encoding='utf-8', newline='\n') as log:
        for epoch in range(CONFIG['epochs']):
            order = list(encoded)
            rng.shuffle(order)
            for offset in range(0, len(order), CONFIG['batch_size']):
                tick = time.perf_counter()
                ids = order[offset:offset + CONFIG['batch_size']]
                width = max(len(encoded[k]['input_ids']) for k in ids)
                inputs, masks, labels = [], [], []
                for k in ids:
                    sample = encoded[k]
                    padding = width - len(sample['input_ids'])
                    inputs.append(sample['input_ids'] + [tokenizer.pad_token_id] * padding)
                    masks.append([1] * len(sample['input_ids']) + [0] * padding)
                    labels.append(sample['labels'] + [-100] * padding)
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=torch.tensor(inputs, device='cuda'),
                               attention_mask=torch.tensor(masks, device='cuda'),
                               labels=torch.tensor(labels, device='cuda'), use_cache=False)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(trainable, CONFIG['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                record = {'epoch': epoch, 'offset': offset, 'task_ids': ids, 'loss': loss.item(),
                          'gradient_norm': grad.item(), 'seconds': time.perf_counter() - tick,
                          'target_tokens': sum(encoded[k]['target_tokens'] for k in ids)}
                log.write(json.dumps(record) + '\n')
                log.flush()
                del output, loss
                if offset % 40 == 0:
                    print(json.dumps(record), flush=True)
    seconds = time.perf_counter() - training_started
    peak = torch.cuda.max_memory_allocated()
    after = frozen_digest(model)
    if before != after:
        raise RuntimeError('Frozen base changed')
    model.eval()
    adapter = out / 'adapter'
    adapter.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()},
              adapter / 'adapter_model.safetensors', metadata={'format': 'pt'})
    config.base_model_name_or_path, config.revision = REPO, REVISION
    config.save_pretrained(adapter)
    summary = {'rows': len(rows), 'epochs': CONFIG['epochs'], 'seconds': seconds,
               'peak_allocated_cuda_bytes': peak, 'frozen_base_unchanged': True,
               'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
               'adapter_bytes': (adapter / 'adapter_model.safetensors').stat().st_size,
               'activation': False, 'benchmark_results': None, 'energy_joules': None}
    write_new(out / 'trained.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'qualification', 'base', 'base-manifest', 'heldout-source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
