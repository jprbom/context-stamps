"""Confirm a saved adapter reloads offline and reproduces one registered batch."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path


def run(args):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    training = json.loads((args.run / 'trained.json').read_text())
    weight = args.run / 'adapter/adapter_model.safetensors'
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == training['adapter_sha256']
    records = [json.loads(s) for s in (args.run / 'evaluation.jsonl').read_text().splitlines()][:4]
    data = {r['id']: r for r in json.loads((args.run / 'data.json').read_text(encoding='utf-8'))}
    t = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    t.padding_side = 'left'
    texts = [t.apply_chat_template(data[r['id']]['messages'], tokenize=False, add_generation_prompt=True) for r in records]
    x = t(texts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
    x['position_ids'] = (x['attention_mask'].cumsum(-1) - 1).clamp_min(0)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, args.run / 'adapter', local_files_only=True, is_trainable=False).eval()
    choices = [t.encode(k, add_special_tokens=False)[0] for k in 'ABCDEF']
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'adapter_sha256': training['adapter_sha256'], 'rows': [], 'passed': True}
    for kind in ('base', 'adapter'):
        scope = model.disable_adapter() if kind == 'base' else contextlib.nullcontext()
        with scope, torch.inference_mode():
            scores = model(**x, logits_to_keep=1, use_cache=False).logits[:, -1, choices].float().cpu().tolist()
        for record, values in zip(records, scores):
            prediction = 'ABCDEF'[max(range(6), key=values.__getitem__)]
            delta = max(abs(x - y) for x, y in zip(values, record[kind]['logits']))
            passed = prediction == record[kind]['prediction'] and delta <= .5
            result['passed'] &= passed
            result['rows'].append({'id': record['id'], 'kind': kind, 'prediction': prediction,
                                   'maximum_logit_difference': delta, 'passed': passed})
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Checkpoint round-trip comparison failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
