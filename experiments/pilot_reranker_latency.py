"""Development-only pilot; never selects from regression test outcomes."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_controller_v2 import teacher_model, teacher_scores  # noqa: E402
from run_hybrid_retrieval import read, save  # noqa: E402
from train_controller_v2 import load_partitions  # noqa: E402

from context_stamps.efficient_reranker import EfficientReranker, pair_tokens  # noqa: E402


def main(work):
    torch.set_num_threads(4)
    data = next(d for d in load_partitions(work, "calibration") if d["name"] == "fiqa")
    source = work / "public-validation/fiqa"
    docs = [json.loads(x) for x in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
    queries = {d["_id"]: d["text"] for d in map(json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    protocol = read(ROOT / "evidence/controller-v2/protocol.json")
    output = []
    for mode in ("reference", "autocast_bf16", "fp16", "bf16"):
        tokenizer, model = teacher_model(protocol)
        fast = None if mode == "reference" else EfficientReranker(tokenizer, model, precision=mode)
        rows = []
        for i in (0, 17, 49, 111, 211, 333, 499):
            valid = data["mask"][i]
            passages = [texts[j] for j in data["candidates"][i][valid]]
            query = queries[data["meta"]["query_ids"][i]]
            if fast:
                q = tokenizer(query, add_special_tokens=False)["input_ids"]
                for passage in passages:
                    p = tokenizer(passage, add_special_tokens=False)["input_ids"]
                    ids, types = pair_tokens(q[:512], p[:512], cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id)
                    reference = tokenizer(query, passage, truncation=True, max_length=512)
                    assert ids == reference["input_ids"] and types == reference["token_type_ids"]
            for repeat in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                scores = fast.score(query, passages) if fast else teacher_scores([query] * len(passages), passages, tokenizer, model)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                rows.append(dict(row=i, repeat=repeat, ms=ms, mae=float(np.abs(scores - data["teacher"][i][valid]).mean()),
                                 max_error=float(np.abs(scores - data["teacher"][i][valid]).max())))
        output.append(dict(mode=mode, observations=rows))
        print(mode, "warm_ms", np.median([r["ms"] for r in rows if r["repeat"]]), "mae", np.mean([r["mae"] for r in rows]), flush=True)
        del fast, model
        torch.cuda.empty_cache()
    save(work / "reranker-latency-pilot.json", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    main(p.parse_args().work)
