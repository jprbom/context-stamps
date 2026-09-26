"""Short offline BF16/FP32 base-model controls; no fitting or activation."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def run(args):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    t = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    data = json.loads((args.run / 'data.json').read_text(encoding='utf-8'))
    cases = [r for r in data if r['id'] in ('retention-0', 'retention-12', 'retention-23')]
    for i, question in enumerate(('What is 2 + 2? Return only the number.',
                                  'What is 36 + 12? Return only the number.',
                                  'Return exactly the single letter F.')):
        cases.append({'id': f'control-{i}', 'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': question}]})
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    result = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fitting': False, 'rows': []}
    for name, dtype in (('bfloat16', torch.bfloat16), ('float32', torch.float32)):
        if dtype == torch.float32:
            model.float()
        for case in cases:
            x = t.apply_chat_template(case['messages'], tokenize=True, add_generation_prompt=True,
                                      return_tensors='pt', return_dict=True).to('cuda')
            with torch.inference_mode():
                y = model.generate(**x, max_new_tokens=8, do_sample=False, use_cache=True,
                                   pad_token_id=t.pad_token_id, eos_token_id=t.eos_token_id)
            result['rows'].append({'id': case['id'], 'dtype': name,
                    'response': t.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True)})
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
