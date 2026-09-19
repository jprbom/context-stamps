"""Validation-only facet bit allocation within a fixed 32-byte routing code."""

import base64
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps.stamp256 import Stamp256Codec
from context_stamps.training import fit_pairwise
from experiments.run_spherical_public import read, save, sha
from experiments.run_spherical_v2 import counterbalanced
from stamps import Family, HashingEncoder, _planes, stamp_vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/stamp256-v1"
NAMES = ("content", "entity", "intent", "task")
ENCODER = HashingEncoder(64)


def codec(widths, seed):
    return Stamp256Codec({name: Family(ENCODER.identity, 64, width, seed + i)
                          for i, (name, width) in enumerate(zip(NAMES, widths))})


def prepare_codes(cases, seed):
    values = {}
    for i, name in enumerate(NAMES):
        vocabulary = sorted({item[name] for rows in cases.values() for row in rows for item in [row["query"], *row["candidates"]]})
        vectors = np.array([ENCODER.encode(text) for text in vocabulary], dtype=np.float64)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        family = Family(ENCODER.identity, 64, 256, seed + i)
        raw = np.packbits(vectors @ np.array(_planes(family)).T >= 0, axis=1, bitorder="little")
        mapping = {text: int.from_bytes(code.tobytes(), "little") for text, code in zip(vocabulary, raw)}
        for text, vector in zip(vocabulary[:3], vectors[:3]):
            assert mapping[text] == stamp_vector(vector, family).value
        values[name] = mapping
    return values


def features(rows, codes, widths):
    result = []
    for row in rows:
        result.append([[1 - ((codes[n][row["query"][n]] ^ codes[n][candidate[n]]) & ((1 << bits) - 1)).bit_count() / bits
                        for n, bits in zip(NAMES, widths)] for candidate in row["candidates"]])
    return np.array(result)


def evaluate(rows, matrix, weights):
    result = []
    for row, values in zip(rows, matrix):
        scores = values @ weights
        order = np.argsort(-scores, kind="stable").tolist()
        result.append({"case": row["id"], "order": order, "top1": order[0] == row["positive"],
                       "rr": 1 / (order.index(row["positive"]) + 1)})
    return result


def main():
    protocol = read(OUT / "protocol.json")
    old = read(ROOT / "evidence/spherical-v2/fixtures.json")
    cases = {"train": old["train"], "validation": old["validation"],
             "fresh_test": counterbalanced("test", 200, 760123), "fresh_ood": counterbalanced("ood", 200, 760124)}
    trials, models, codebooks = [], {}, {}
    for seed in protocol["seeds"]:
        codes = codebooks[seed] = prepare_codes(cases, seed)
        for widths in protocol["allocations"]:
            train = features(cases["train"], codes, widths)
            pairs = [values[row["positive"]] - value for row, values in zip(cases["train"], train)
                     for j, value in enumerate(values) if j != row["positive"]]
            current = codec(widths, seed)
            template = current.encode({n: ENCODER.encode(cases["train"][0]["query"][n]) for n in NAMES})
            model, history = fit_pairwise(template, pairs, penalty=.001)
            validation = evaluate(cases["validation"], features(cases["validation"], codes, widths), model.coefficients)
            models[(seed, tuple(widths))] = model
            trials.append({"seed": seed, "widths": widths, "validation_top1": statistics.mean(r["top1"] for r in validation),
                           "validation_mrr": statistics.mean(r["rr"] for r in validation), "loss": history,
                           "coefficients": model.coefficients})
    def criterion(widths):
        rows = [r for r in trials if r["widths"] == widths]
        return statistics.mean(r["validation_top1"] for r in rows), statistics.mean(r["validation_mrr"] for r in rows)
    selected = max(protocol["allocations"], key=criterion)
    records, entropy = [], []
    for seed in protocol["seeds"]:
        selected_model = models[(seed, tuple(selected))]
        (OUT / f"model-{seed}.json").write_text(selected_model.to_json() + "\n", encoding="utf-8", newline="\n")
        for split in ("fresh_test", "fresh_ood"):
            for method, widths, model in (("equal_bits_uniform", [64] * 4, None),
                                         ("equal_bits_pairwise", [64] * 4, models[(seed, (64, 64, 64, 64))]),
                                         ("validation_selected_bits_pairwise", selected, selected_model)):
                values = features(cases[split], codebooks[seed], widths)
                weights = np.ones(4) if model is None else model.coefficients
                records.extend({**r, "seed": seed, "split": split, "method": method} for r in evaluate(cases[split], values, weights))
        for name, width in zip(NAMES, selected):
            population = [codebooks[seed][name][item[name]] for row in cases["train"] for item in row["candidates"]]
            entropies = []
            for bit in range(width):
                p = sum((value >> bit) & 1 for value in population) / len(population)
                entropies.append(0 if p in (0, 1) else -p * math.log2(p) - (1-p) * math.log2(1-p))
            entropy.append({"seed": seed, "facet": name, "bits": width, "mean_marginal_bit_entropy": statistics.mean(entropies)})
    summary = []
    for split in ("fresh_test", "fresh_ood"):
        for method in ("equal_bits_uniform", "equal_bits_pairwise", "validation_selected_bits_pairwise"):
            rows = [r for r in records if (r["split"], r["method"]) == (split, method)]
            summary.append({"split": split, "method": method, "query_seed_pairs": len(rows),
                            "top1": statistics.mean(r["top1"] for r in rows), "mrr": statistics.mean(r["rr"] for r in rows)})
    current = codec(selected, 17)
    sample = current.encode({n: ENCODER.encode(cases["fresh_test"][0]["query"][n]) for n in NAMES})
    raw = current.pack(sample)
    assert current.unpack(raw, schema_id=current.schema.identity) == sample
    import zlib
    transport = {"raw_bytes": len(raw), "zlib_bytes": len(zlib.compress(raw)),
                 "base64_bytes": len(base64.b64encode(raw)),
                 "base64_cl100k_tokens": len(tiktoken.get_encoding("cl100k_base").encode(base64.b64encode(raw).decode())),
                 "schema": current.schema.views, "schema_id": current.schema.identity,
                 "note": "Single payload illustration; schema/source/version/authorization are external; zip not used if larger."}
    for name, value in (("fixtures", cases), ("training", trials), ("results", records), ("summary", summary),
                        ("bit-entropy", entropy), ("transport", transport)):
        save(OUT / (name + ".json"), value)
    save(OUT / "manifest.json", {"protocol": protocol, "selected_widths": selected, "numpy": np.__version__,
                                 "exact_field_control": "All fixtures uniquely determined by supplied entity/intent/task; 100%",
                                 "source_sha256": {p: sha(ROOT / p) for p in ("experiments/train_stamp256.py",
                                     "context_stamps/stamp256.py", "context_stamps/training.py", "experiments/run_spherical_v2.py")}})
    save(OUT / "checksums.json", {p.name: sha(p) for p in OUT.glob("*.json") if p.name != "checksums.json"})
    print("Selected allocation", selected, summary, transport, flush=True)


if __name__ == "__main__":
    main()
