"""Investigate local BF16 shape/attention parity without training or tools."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def run(args):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    t = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    t.padding_side = 'left'
    raw = Tokenizer.from_file(str(args.base / 'tokenizer.json'))
    data = json.loads((args.run / 'data.json').read_text(encoding='utf-8'))
    records = [json.loads(s) for s in (args.run / 'evaluation.jsonl').read_text().splitlines()]
    # Use exactly the first recorded mixed-length batch and several retention cases.
    ids = [r['id'] for r in records[:4]]
    ids += [r['id'] for r in records if r['split'] == 'retention'][:3]
    rows = [next(r for r in data if r['id'] == name) for name in ids]
    texts = [t.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True) for r in rows]
    assert all(t(s, add_special_tokens=False)['input_ids'] == raw.encode(s, add_special_tokens=False).ids for s in texts)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    choice = [t.encode(k, add_special_tokens=False)[0] for k in 'ABCDEF']
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': [],
              'tokenizer_parity': True, 'training': False}
    for implementation in ('sdpa', 'eager'):
        model.set_attn_implementation(implementation)
        for group in (list(range(4)), *[[i] for i in range(len(rows))]):
            x = t([texts[i] for i in group], padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            x['position_ids'] = (x['attention_mask'].cumsum(-1) - 1).clamp_min(0)
            with torch.inference_mode():
                scores = model(**x, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            for i, logits in zip(group, scores):
                values = logits[choice].cpu().tolist()
                result['rows'].append({'id': rows[i]['id'], 'target': rows[i]['target'], 'attention': implementation,
                    'batch_size': len(group), 'prediction': 'ABCDEF'[max(range(6), key=values.__getitem__)],
                    'choice_logits': values, 'top_token': t.decode([logits.argmax().item()])})
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
