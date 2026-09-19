"""Actual local generation parity pilot; does not estimate perceptual quality."""

import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1] / "work"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(WORK / "hf-cache")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel, set_seed

from context_stamps import ContextGraph, ContextNode, Family, HashingEncoder, SphericalStamp, StampSchema
from context_stamps.guarded_activation import activate_constrained

OUT = ROOT / "evidence/multimodal-v1"
MEDIA = WORK / "multimodal-media"
PROMPTS = {
    "image": ["A red ceramic cup on a wooden table, soft daylight.",
              "A yellow bicycle beside a blue wall, clear afternoon light.",
              "A small wooden cabin beside a lake, watercolor painting."],
    "speech": ["The experiment is complete. Please review the recorded measurements.",
               "The library keeps the original context available for the next task.",
               "Check the source version before using the previous result."],
    "video": ["A red ball rolling across a wooden floor.",
              "Gentle waves moving across a blue lake.",
              "A small toy car moving slowly on a table."]}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def routed(prompt, modality, index):
    encoder = HashingEncoder(64)
    names = ("content", "entity", "intent", "task")
    families = {name: Family(encoder.identity, 64, 64, 17 + i) for i, name in enumerate(names)}
    def stamp(facets):
        return SphericalStamp.encode({k: encoder.encode(v) for k, v in facets.items()}, families)
    graph, candidates, metadata, revisions = ContextGraph(), [], [], {}
    for j in range(4):
        fields = {"content": prompt if j == 0 else "Unrelated production context",
                  "entity": f"asset_{index}" if j == 0 else f"other_{j}", "intent": "generate", "task": modality}
        candidates.append(stamp(fields))
        metadata.append(fields)
        # Original conditioning belongs to the dependency; reference node carries routing metadata.
        graph.put(ContextNode(f"asset-{j}", json.dumps(fields), "1", frozenset({"renderer"}), candidates[-1]))
        graph.put(ContextNode(f"prompt-{j}", prompt if j == 0 else "An unrelated context.", "1",
                              frozenset({"renderer"})))
        graph.link(f"asset-{j}", f"prompt-{j}", "depends_on", provenance="original generation brief")
        revisions[f"asset-{j}"] = revisions[f"prompt-{j}"] = "1"
    start = time.perf_counter_ns()
    query = stamp(metadata[0])
    schema = StampSchema.for_stamp(query)
    payload = schema.pack(query)
    query = schema.unpack(payload)
    hits = activate_constrained(query, candidates, metadata=metadata,
                                required={"entity": f"asset_{index}", "task": modality}, threshold=.5, limit=1)
    key = hits[0]["index"] if hits else None
    packet = graph.handoff([f"asset-{key}"], role="renderer", revisions=revisions, budget_bytes=8192)
    if packet.status != "complete" or f"prompt-{key}" not in packet.sources:
        raise ValueError("context resolution failed")
    resolved = graph._nodes[f"prompt-{key}"].text
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return resolved, {"routing_ms": elapsed, "compact_bytes": len(payload),
                      "schema_bytes": len(json.dumps({"id": schema.identity, "views": schema.views})),
                      "resolved_bytes": len(resolved.encode()), "prompt_identity": resolved == prompt}


def model_path(model_id, revision):
    return WORK / "hf-cache" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision


def main():
    MEDIA.mkdir(exist_ok=True)
    protocol = json.loads((OUT / "protocol.json").read_text())
    rows, failures = [], []
    rng = random.Random(95013)
    for modality, (model_id, revision) in protocol["models"].items():
        pipe = tokenizer = None
        try:
            path = model_path(model_id, revision)
            if modality == "speech":
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                pipe = VitsModel.from_pretrained(path, local_files_only=True, use_safetensors=True).to("cuda")
                pipe.eval()
            else:
                from diffusers import StableDiffusionPipeline, TextToVideoSDPipeline
                pipeline = StableDiffusionPipeline if modality == "image" else TextToVideoSDPipeline
                pipe = pipeline.from_pretrained(path, torch_dtype=torch.float16, variant="fp16",
                                                use_safetensors=True, local_files_only=True).to("cuda")
                pipe.set_progress_bar_config(disable=True)
                tokenizer = pipe.tokenizer
            # All measured calls are retained; first-call latency is not silently dropped.
            for index, prompt in enumerate(PROMPTS[modality]):
                for seed in protocol["seeds"]:
                    reference = None
                    pair = []
                    modes = ["direct_oracle_prompt", "spherical_route_and_resolve"]
                    rng.shuffle(modes)
                    for mode in modes:
                        conditioning, routing = (prompt, {"routing_ms": 0, "compact_bytes": 0, "schema_bytes": 0,
                                                        "resolved_bytes": len(prompt.encode()), "prompt_identity": True})
                        if mode == "spherical_route_and_resolve":
                            conditioning, routing = routed(prompt, modality, index)
                        set_seed(seed)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter_ns()
                        with torch.inference_mode():
                            if modality == "speech":
                                inputs = tokenizer(conditioning, return_tensors="pt").to("cuda")
                                output = pipe(**inputs).waveform[0].float().cpu().numpy()
                            else:
                                kwargs = {"prompt": conditioning, "generator": torch.Generator("cuda").manual_seed(seed)}
                                if modality == "image":
                                    result = pipe(**kwargs, num_inference_steps=1, guidance_scale=0,
                                                  height=512, width=512, output_type="np")
                                    output = result.images[0]
                                else:
                                    result = pipe(**kwargs, num_inference_steps=10, guidance_scale=9,
                                                  height=256, width=256, num_frames=8, output_type="np")
                                    output = result.frames[0]
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - t0) / 1e6
                        output = np.asarray(output, dtype=np.float32)
                        digest = hashlib.sha256(output.tobytes()).hexdigest()
                        filename = f"{modality}-{index}-{seed}-{mode}.npy"
                        np.save(MEDIA / filename, output)
                        row = {"modality": modality, "case": index, "seed": seed, "condition": mode,
                               "prompt": conditioning, "shape": list(output.shape), "sha256": digest,
                               "finite_nonempty": bool(output.size and np.isfinite(output).all()),
                               "nonconstant": bool(output.size and output.max() > output.min()),
                               "generation_ms": elapsed, "end_to_end_ms": elapsed + routing["routing_ms"],
                               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "input_tokens": len(tokenizer.encode(conditioning)), **routing}
                        if reference is None:
                            reference = output.copy()
                        pair.append(row)
                        print(modality, index, seed, mode, round(elapsed, 1), flush=True)
                    delta = float(np.max(np.abs(reference - output))) if reference.shape == output.shape else None
                    for row in pair:
                        row["paired_max_absolute_error"] = delta
                        row["paired_hash_equal"] = pair[0]["sha256"] == pair[1]["sha256"]
                    rows.extend(pair)
                    save(OUT / "results.json", rows)
        except Exception as exc:
            failures.append({"modality": modality, "error_type": type(exc).__name__, "error": str(exc)[:4000]})
            print("FAILED", modality, type(exc).__name__, str(exc)[:300], flush=True)
        finally:
            del pipe, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    save(OUT / "failures.json", failures)
    save(OUT / "results.json", rows)
    sources = ["experiments/run_multimodal.py", "context_stamps/guarded_activation.py", "context_stamps/activation.py",
               "context_stamps/spherical.py", "context_stamps/workflow.py"]
    save(OUT / "manifest.json", {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(), "models": protocol["models"], "media_storage": "local work directory only",
         "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}})
    save(OUT / "checksums.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in OUT.glob("*.json") if p.name != "checksums.json"})


if __name__ == "__main__":
    main()
