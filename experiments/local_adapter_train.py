"""Bounded offline LoRA experiment; never activates or executes a candidate.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Authored fixture exercise, not a public benchmark or domain-competence claim.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

from local_adapter_data import ACTIONS, fixtures, validate_rows

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
CONFIG = {'seed': 7, 'rank': 4, 'alpha': 8, 'dropout': 0., 'target_modules': ['q_proj', 'v_proj'],
          'learning_rate': 0.0002, 'epochs': 2, 'batch_size': 4, 'max_length': 768,
          'dtype': 'bfloat16', 'attention': 'sdpa', 'weight_decay': 0.01, 'gradient_clip': 1.}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check_base(base, manifest):
    if manifest['repository'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Registered local base required')
    for name, info in manifest['files'].items():
        path = base / name
        if path.name != name or path.is_symlink() or not path.is_file():
            raise ValueError('Regular base file required')
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Base digest mismatch: ' + name)
    return manifest


def frozen_digest(model):
    import torch
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def score_summary(rows):
    result = {}
    for split in ('calibration', 'test', 'retention'):
        subset = [r for r in rows if r['split'] == split]
        result[split] = {'rows': len(subset), 'clusters': len({r['cluster'] for r in subset})}
        for kind in ('base', 'adapter'):
            result[split][kind + '_correct'] = sum(r[kind]['prediction'] == r['target'] for r in subset)
        result[split]['new_failures'] = sum(r['base']['prediction'] == r['target'] and
                                          r['adapter']['prediction'] != r['target'] for r in subset)
    return result


def run(args):
    # No automatic downloads, implicit credentials, hosted jobs or trackers.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ[name] = '1'
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=False)
    base_manifest = check_base(args.base, json.loads(args.base_manifest.read_text(encoding='utf-8')))
    data = fixtures()
    validate_rows(data)
    write_new(args.out / 'data.json', data)
    write_new(args.out / 'base.json', base_manifest)
    sources = ('experiments/local_adapter_train.py', 'experiments/local_adapter_data.py')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This registered experiment requires local BF16 CUDA')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'tokenizers')}
    plan = {'schema': 1, 'experiment': 'authored_workflow_adapter_engineering_v1', 'config': CONFIG,
            'sources': {n: sha(ROOT / n) for n in sources}, 'data_sha256': sha(args.out / 'data.json'),
            'base_manifest_sha256': sha(args.out / 'base.json'), 'versions': versions,
            'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
            'provider_charges_usd': 0, 'promotion_allowed': False,
            'controls': ['same BF16 base with adapter disabled', 'exact host rule'],
            'evaluation': 'Fixed final epoch; paired batch-order randomization; no evaluation-directed tuning',
            'limitations': ['Authored synthetic states; no public benchmark or full coding tasks',
                            'Shared templates; project IDs are fixture clusters, not real independent deployments',
                            'Six-token constrained classification, not free text or tool execution',
                            'No on-device CPU/NPU, quantized, energy or production qualification']}
    write_new(args.out / 'plan.json', plan)
    torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in ACTIONS]
    if any(len(v) != 1 for v in choice_ids):
        raise ValueError('Single-token choices required')
    choice_ids = [v[0] for v in choice_ids]
    encoded = {}
    for row in data:
        text = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if len(ids) > CONFIG['max_length']:
            raise ValueError('Fixture exceeds registered token limit; never truncate silently')
        encoded[row['id']] = ids
    write_new(args.out / 'tokenization.json', {'choice_ids': choice_ids,
              'input_lengths': {k: len(v) for k, v in encoded.items()},
              'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()})
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    config = LoraConfig(r=CONFIG['rank'], lora_alpha=CONFIG['alpha'], lora_dropout=0.,
                        target_modules=CONFIG['target_modules'], task_type='CAUSAL_LM', bias='none')
    model = get_peft_model(model, config)
    model.config.use_cache = False
    before = frozen_digest(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad and 'lora_' not in n for n, p in model.named_parameters()):
        raise ValueError('Only low-rank parameters may train')
    write_new(args.out / 'loaded.json', {'seconds': time.perf_counter() - started,
              'frozen_digest': before, 'trainable_parameters': sum(p.numel() for p in trainable),
              'total_parameters': sum(p.numel() for p in model.parameters())})

    def batch(rows):
        width = max(len(encoded[r['id']]) for r in rows)
        ids, masks = [], []
        for row in rows:
            v = encoded[row['id']]
            pad = width - len(v)
            ids.append([tokenizer.pad_token_id] * pad + v)
            masks.append([0] * pad + [1] * len(v))
        mask = torch.tensor(masks, device='cuda')
        return {'input_ids': torch.tensor(ids, device='cuda'), 'attention_mask': mask,
                'position_ids': (mask.cumsum(-1) - 1).clamp_min(0)}

    optimizer = torch.optim.AdamW(trainable, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    training = [r for r in data if r['split'] == 'train']
    rng = random.Random(CONFIG['seed'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    with (args.out / 'training.jsonl').open('x', encoding='utf-8', newline='\n') as log:
        for epoch in range(CONFIG['epochs']):
            order = training.copy()
            rng.shuffle(order)
            for offset in range(0, len(order), CONFIG['batch_size']):
                group = order[offset:offset + CONFIG['batch_size']]
                tick = time.perf_counter()
                x = batch(group)
                labels = torch.tensor([choice_ids['ABCDEF'.index(r['target'])] for r in group], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                logits = model(**x, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(trainable, CONFIG['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                record = {'epoch': epoch, 'offset': offset, 'rows': [r['id'] for r in group],
                          'loss': loss.item(), 'gradient_norm': grad.item(), 'seconds': time.perf_counter() - tick}
                log.write(json.dumps(record) + '\n')
                log.flush()
                if offset % 32 == 0:
                    print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - train_started
    training_peak = torch.cuda.max_memory_allocated()
    after = frozen_digest(model)
    if before != after:
        raise RuntimeError('Frozen base changed')
    model.eval()
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    del optimizer
    torch.cuda.empty_cache()
    adapter = args.out / 'adapter'
    adapter.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()},
              adapter / 'adapter_model.safetensors', metadata={'format': 'pt'})
    # PEFT's configuration serializer converts sets/enums to portable JSON.
    config.base_model_name_or_path = REPO
    config.revision = REVISION
    config.save_pretrained(adapter)
    write_new(args.out / 'trained.json', {'seconds': training_seconds, 'peak_allocated_cuda_bytes': training_peak,
              'frozen_base_unchanged': before == after, 'adapter_sha256': sha(adapter / 'adapter_model.safetensors'),
              'adapter_bytes': (adapter / 'adapter_model.safetensors').stat().st_size})

    # Paired matched-backend comparisons after weights are frozen. Evaluation
    # labels do not choose checkpoints, hyperparameters or candidate activation.
    evaluations = [r for r in data if r['split'] != 'train']
    rng.shuffle(evaluations)
    rows = []
    with (args.out / 'evaluation.jsonl').open('x', encoding='utf-8', newline='\n') as out:
        for offset in range(0, len(evaluations), CONFIG['batch_size']):
            group = evaluations[offset:offset + CONFIG['batch_size']]
            records = [{k: r[k] for k in ('id', 'cluster', 'split', 'target', 'kind')} for r in group]
            kinds = ['base', 'adapter']
            rng.shuffle(kinds)
            for kind in kinds:
                scope = model.disable_adapter() if kind == 'base' else contextlib.nullcontext()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                x = batch(group)
                with scope, torch.inference_mode():
                    logits = model(**x, logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
                    values = logits[:, choice_ids].cpu().tolist()
                    unconstrained = logits.argmax(-1).cpu().tolist()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                for record, scores, token in zip(records, values, unconstrained):
                    record[kind] = {'logits': scores, 'prediction': 'ABCDEF'[max(range(6), key=scores.__getitem__)],
                                    'unconstrained_next_token': token, 'batch_seconds': elapsed,
                                    'batch_size': len(group), 'order': kinds.index(kind)}
            for row in records:
                out.write(json.dumps(row) + '\n')
                out.flush()
                rows.append(row)
    result = score_summary(rows)
    result.update({'training_seconds': training_seconds, 'promotion': False, 'provider_charges_usd': 0,
                   'rule_baseline_workflow_correct': sum(r['kind'] == 'workflow' for r in rows),
                   'energy_joules': None,
                   'qualification': 'engineering exercise only; no automatic model-weight activation'})
    if not math.isfinite(training_seconds):
        raise RuntimeError('Invalid measurement')
    write_new(args.out / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        if arguments.out.is_dir() and not (arguments.out / 'failure.json').exists():
            write_new(arguments.out / 'failure.json', {'error': type(error).__name__, 'message': str(error)[:2000]})
        raise
