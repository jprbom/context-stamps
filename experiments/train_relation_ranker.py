"""Train two small support-allocation controls on local public training data.

Copyright (c) 2026 Prashant Jagtap. MIT License.
No evaluation keys are opened until policies and selection are frozen.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from relation_ranker import FEATURES, features, probabilities, rank
from ruler_native import outside_repo, sha, write_new

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tensors(data, partition, device):
    inputs = {r["key"]: r for r in read(data/f"{partition}-inputs.jsonl")}
    keys = read(data/f"{partition}-keys.jsonl")
    selected = [row for row in keys if row["original"]["answerable"]]
    x = np.zeros((len(selected), 32, len(FEATURES)), dtype=np.float32)
    p = np.zeros((len(selected), 32, 32), dtype=np.float32)
    target = np.zeros((len(selected), 32), dtype=np.float32)
    mask = np.zeros((len(selected), 32), dtype=bool)
    for i, row in enumerate(selected):
        case = inputs[row["key"]]
        f, transition, _ = features(case)
        n = len(f)
        x[i, :n], p[i, :n, :n], mask[i, :n] = f, transition, True
        gold = {item["idx"] for item in row["original"]["paragraphs"] if item["is_supporting"]}
        if not gold:
            raise ValueError("answerable training input has no supports")
        target[i, :n] = [float(item["idx"] in gold)/len(gold) for item in case["paragraphs"]]
    return tuple(torch.tensor(a, device=device) for a in (x, p, target, mask)), inputs, selected


def forward(x, p, mask, weights, logit, graph):
    q = (x@weights).masked_fill(~mask, -1e9).softmax(-1)
    alpha = .9*logit.sigmoid() if graph else torch.zeros((), device=x.device)
    h = q
    for _ in range(4):
        h = (1-alpha)*q+alpha*torch.bmm(p.transpose(1, 2), h.unsqueeze(-1)).squeeze(-1)
    return h


def run(data, output):
    data, output = map(outside_repo, (data, output))
    prepared = json.loads((data/"prepared.json").read_bytes())
    if any(sha(data/n) != h for n, h in prepared["files"].items()):
        raise ValueError("prepared files changed")
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise ValueError("registered local CUDA training required")
    torch.manual_seed(71)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    source_hashes = {"experiments/"+n: sha(ROOT/"experiments"/n) for n in
                     ("multihop_data.py", "relation_ranker.py", "train_relation_ranker.py")}
    registration = dict(seed=71, source_hashes=source_hashes, prepared_sha256=sha(data/"prepared.json"),
                        epochs=240, batch=64, optimizer="Adam", lr=.03, weight_l2=.001,
                        checkpoint="lowest validation support cross entropy among epochs 20,40,...,240",
                        validation_scope="calibration partition used for model selection, not probability certification",
                        heldout_keys_opened=False, device=torch.cuda.get_device_name(),
                        packages={n: importlib.metadata.version(n) for n in ("torch", "numpy")})
    write_new(output/"registration.json", registration)
    start = time.perf_counter()
    train, _, train_keys = tensors(data, "training", device)
    validation, val_inputs, val_keys = tensors(data, "calibration", device)
    torch.cuda.synchronize()
    prep_seconds = time.perf_counter()-start
    models = {}
    for name, graph in (("pointwise", False), ("diffusion", True)):
        torch.manual_seed(71)
        w = torch.nn.Parameter(torch.zeros(len(FEATURES), device=device))
        a = torch.nn.Parameter(torch.tensor(-.5, device=device))
        parameters = [w, a] if graph else [w]
        optimizer = torch.optim.Adam(parameters, lr=.03)
        best, history = None, []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        order = list(range(len(train_keys)))
        rng = random.Random(71)
        for epoch in range(1, 241):
            rng.shuffle(order)
            for offset in range(0, len(order), 64):
                ids = torch.tensor(order[offset:offset+64], device=device)
                x, p, target, mask = (t[ids] for t in train)
                optimizer.zero_grad(set_to_none=True)
                h = forward(x, p, mask, w, a, graph)
                loss = -(target*h.clamp_min(1e-12).log()).sum(-1).mean()+.001*(w*w).mean()
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0:
                with torch.no_grad():
                    x, p, target, mask = validation
                    val_loss = float((-(target*forward(x, p, mask, w, a, graph).clamp_min(1e-12).log()).sum(-1)).mean())
                    row = dict(epoch=epoch, validation_loss=val_loss, alpha=float(.9*a.sigmoid()) if graph else 0.)
                    history.append(row)
                    if best is None or val_loss < best["loss"]:
                        best = dict(loss=val_loss, epoch=epoch, weights=w.detach().cpu().tolist(), alpha=row["alpha"])
        torch.cuda.synchronize()
        seconds = time.perf_counter()-tick
        policy = dict(schema=1, name=name, features=list(FEATURES), steps=4,
                      weights=best["weights"], alpha=best["alpha"], epoch=best["epoch"])
        # Independent Python inference must agree with the vectorized trainer.
        errors = []
        with torch.no_grad():
            w.copy_(torch.tensor(policy["weights"], device=device))
            if graph:
                ratio=policy["alpha"]/.9
                a.copy_(torch.tensor(math.log(ratio/(1-ratio)), device=device))
            x, p, _, mask = validation
            predictions = forward(x, p, mask, w, a, graph).cpu().numpy()
        for i, row in enumerate(val_keys):
            f, transition, _ = features(val_inputs[row["key"]])
            inferred = probabilities(f, transition, policy["weights"], policy["alpha"])
            errors.append(float(np.max(np.abs(predictions[i, :len(f)]-np.asarray(inferred)))))
        if max(errors) > 2e-6:
            raise ValueError("Python/CUDA inference parity failed")
        write_new(output/(name+".json"), policy)
        models[name] = dict(parameters=len(parameters[0])+int(graph), training_seconds=seconds,
                            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                            history=history, selected=copy.deepcopy(best), parity_max_abs=max(errors))
        print(json.dumps(dict(model=name, selected_epoch=best["epoch"], alpha=best["alpha"], seconds=seconds)), flush=True)
    comparison = compare(data, "calibration", output)
    # Pick a trained candidate before touching held-out targets. Fixed BM25 and
    # both trained models remain visible even if the trained candidate loses.
    candidate = max(("pointwise", "diffusion"), key=lambda n: (comparison[n]["complete_support_rate"],
                                                              comparison[n]["support_f1"], n == "pointwise"))
    write_new(output/"training.json", dict(preparation_seconds=prep_seconds, training_answerable_tasks=len(train_keys),
                                          models=models, validation=comparison, candidate=candidate,
                                          candidate_active=False, requires_reader_evaluation=True))


def compare(data, partition, output):
    inputs = {r["key"]: r for r in read(data/f"{partition}-inputs.jsonl")}
    keys = [r for r in read(data/f"{partition}-keys.jsonl") if r["original"]["answerable"]]
    results = {}
    for name in ("bm25", "pointwise", "diffusion"):
        model = None if name == "bm25" else json.loads((output/(name+".json")).read_bytes())
        rows = []
        for row in keys:
            selected, _ = rank(inputs[row["key"]], model)
            pred = {p["idx"] for p in selected[:6]}
            gold = {p["idx"] for p in row["original"]["paragraphs"] if p["is_supporting"]}
            rows.append(dict(key=row["key"], selected=sorted(pred), gold=sorted(gold), complete=gold <= pred,
                             f1=2*len(pred&gold)/(len(pred)+len(gold))))
        results[name] = dict(tasks=len(rows), complete_support_rate=sum(r["complete"] for r in rows)/len(rows),
                             support_f1=sum(r["f1"] for r in rows)/len(rows), rows=rows)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
