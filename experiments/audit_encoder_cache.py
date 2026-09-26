"""Check direct mean pooling against the previously pinned query caches."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_hybrid_retrieval import read, save, sha  # noqa: E402


def main(work):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModel, AutoTokenizer
    out = ROOT / "evidence/controller-v1"
    protocol = read(out / "protocol.json")
    kwargs = dict(revision=protocol["revision"], local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(protocol["encoder"], **kwargs)
    model = AutoModel.from_pretrained(protocol["encoder"], use_safetensors=True, **kwargs).cuda().eval()
    rows = []
    for name in ("scifact", "nfcorpus", "arguana", "scidocs"):
        source = work / ("public-validation/scidocs" if name == "scidocs" else "replication-data/" + name)
        queries = {r["_id"]: r["text"] for r in map(json.loads, (source / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
        data = work / "controller-v1-data"
        indexes = read(data / (name + "-test.json"))["query_ids"][:16]
        reference = np.load(data / (name + "-test.npz"), allow_pickle=False)["Q"][:16]
        inputs = tokenizer([queries[q] for q in indexes], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            actual = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
        cosine = (actual * reference).sum(1)
        maximum_error = float(np.max(np.abs(actual - reference)))
        if float(cosine.min()) < .9999 or maximum_error > .001:
            raise ValueError("encoder pooling differs from pinned cache: " + name)
        rows.append(dict(dataset=name, sampled_query_ids=indexes, count=len(indexes),
                         minimum_cosine=float(cosine.min()), maximum_coordinate_error=maximum_error))
    save(out / "encoder-parity.json", dict(source_sha256=sha(Path(__file__)),
         samples=rows, status="Sampled numerical compatibility, not a new retrieval benchmark."))
    checksums = read(out / "checksums.json")
    checksums["encoder-parity.json"] = sha(out / "encoder-parity.json")
    save(out / "checksums.json", checksums)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    main(parser.parse_args().work)
