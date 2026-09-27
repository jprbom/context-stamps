"""Local paired direct/cited reader runs; no scoring keys are opened here.

Copyright (c) 2026 Prashant Jagtap. MIT License.
Each response is persisted before parsing. Ambiguous failed requests are never
retried automatically. Re-running resumes saved responses and untouched calls.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import techqa_generation as reader
from local_eval import local_api, model_identity
from ruler_native import outside_repo, sha, write_new
from techqa_context import digest, locate_answer, rank_windows, windows

from context_stamps.citations import check_citations, prepare_citations
from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(temperature=0, seed=83, num_ctx=8192, num_predict=1024)
MODELS = ("small", "modern")
PHASES = ("fit", "calibration", "development")
PROMPT_BUDGET = 7000


def source_hashes():
    names = ("experiments/techqa_reader.py", "experiments/techqa_context.py", "experiments/techqa_generation.py", "experiments/cited_reader.py",
             "experiments/local_eval.py", "experiments/ruler_native.py", "context_stamps/citations.py",
             "context_stamps/context_state.py", "context_stamps/security.py")
    return {name: sha(ROOT/name) for name in names}


def make_packet(question, selected):
    state = ContextState(tenant="techqa-public", policy_revision="public-v1", clock=lambda: 1)
    for i, row in enumerate(selected):
        state.put(CanonicalNode(key=f"window-{i:02d}", revision=digest(row["text"]), tenant="techqa-public",
                                text=row["text"], kind="OBSERVATION", temporal=TemporalScope(1, 1),
                                roles=("reader",), provenance="TechQA CDLA-Permissive-1.0"))
    snapshot = state.snapshot(AccessScope("techqa-public", "local-reader", "public-v1", ("reader",)), at=1, known_at=1)
    return state, prepare_citations(state, snapshot, question)


def compile_item(item, docs, tokenizers):
    tick = time.perf_counter()
    if len(item["question"].encode("utf-8")) > 16384:
        raise ValueError("complete question exceeds citation packet byte contract")
    if any(len(tokenizers[model].encode(reader.render(item["question"], [], model=model, mode=mode),
                                        add_special_tokens=False).ids) > PROMPT_BUDGET
           for model in MODELS for mode in ("direct", "cited")):
        raise ValueError("complete question already exceeds prompt budget")
    ranked = rank_windows(item["question"], windows(item, docs))
    selected = []
    # Same text, window order and budget for both reader identities and modes.
    for row in ranked:
        proposed = selected+[row]
        sources = [dict(id=f"s{i}", text=r["text"], kind="OBSERVATION") for i, r in enumerate(proposed)]
        fits = all(len(tokenizers[model].encode(reader.render(item["question"], sources, model=model, mode=mode),
                                                add_special_tokens=False).ids) <= PROMPT_BUDGET
                   for model in MODELS for mode in ("direct", "cited"))
        if fits:
            selected = proposed
        if len(selected) == 8:
            break
    if not selected:
        raise ValueError("no window fits the complete question and prompt budget")
    state, packet = make_packet(item["question"], selected)
    sources = json.loads(packet.payload)["sources"]
    prompts = {model: {mode: reader.render(item["question"], sources, model=model, mode=mode)
                       for mode in ("direct", "cited")} for model in MODELS}
    if not packet.is_current(state):
        raise ValueError("new source binding failed")
    return dict(id=item["id"], question=item["question"], selected=selected, candidate_windows=len(ranked),
                prompts=prompts, input_tokens={model: {mode: len(tokenizers[model].encode(prompt, add_special_tokens=False).ids)
                                                       for mode, prompt in modes.items()}
                                              for model, modes in prompts.items()},
                compile_seconds=time.perf_counter()-tick, compile_error=None)


def prepare(data, output):
    data, output = outside_repo(data), outside_repo(output)
    manifest = json.loads((data/"manifest.json").read_bytes())
    source = Path(manifest["data"])
    if sha(source/"training_dev_technotes.json") != manifest["documents_sha256"]:
        raise ValueError("document bytes changed")
    docs = json.loads((source/"training_dev_technotes.json").read_bytes())
    tokenizers = {model: reader.tokenizer_for(model) for model in MODELS}
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(models={model: model_identity(reader.NAMES[model]) for model in MODELS},
                options=OPTIONS, tokenizer_hashes={model: sha(ROOT.parent/reader.TOKENIZERS[model]/"tokenizer.json") for model in MODELS},
                source_hashes=source_hashes(), data_manifest=sha(data/"manifest.json"),
                policy_source_hashes={name: sha(ROOT/"experiments"/name) for name in
                                      ("techqa_policy.py", "train_techqa_policy.py", "techqa_native.py", "eval_techqa.py")},
                input_hashes={phase: manifest["files"][phase+".inputs.json"] for phase in PHASES},
                ollama_version=local_api("/api/version"), prompt_budget=PROMPT_BUDGET,
                windows=dict(width_words=200, stride_words=150, maximum_selected=8),
                modes=["direct", "cited"], key_files_opened=False,
                hypothesis="source binding and learned abstention may reduce unsupported outputs; preserve positive answer quality",
                controls=["direct exact-span", "cited raw", "cited checked", "abstain all"],
                phase_calls={phase: manifest["statistics"][phase]["questions"]*2 for phase in PHASES},
                primary_training_model="modern", no_candidate_activation=True,
                resource_scope="shared CPU compilation, response processing, actual loopback request latency and tokens; no energy/edge-device claim")
    write_new(output/"registration.json", plan)
    for phase in PHASES:
        path = data/(phase+".inputs.json")
        if sha(path) != plan["input_hashes"][phase]:
            raise ValueError("input projection changed")
        items = json.loads(path.read_bytes())
        rows = []
        for item in items:
            tick = time.perf_counter()
            try:
                result = compile_item(item, docs, tokenizers)
            except (ValueError, TypeError, KeyError) as exc:
                result = dict(id=item["id"], compile_error=f"{type(exc).__name__}: {exc}",
                              compile_seconds=time.perf_counter()-tick)
            result["order"] = ["direct", "cited"]
            random.Random("techqa-modes-83\0"+item["id"]).shuffle(result["order"])
            rows.append(result)
            if len(rows) % 50 == 0:
                print(json.dumps(dict(phase=phase, prepared=len(rows), total=len(items))), flush=True)
        write_new(output/(phase+".compiled.json"), rows)
    write_new(output/"prepared.json", dict(registration=sha(output/"registration.json"),
                                            compiled={phase: sha(output/(phase+".compiled.json")) for phase in PHASES}))


def run(output, phase, model, trained=None):
    output = outside_repo(output)
    plan = json.loads((output/"registration.json").read_bytes())
    prepared = json.loads((output/"prepared.json").read_bytes())
    if (plan["source_hashes"] != source_hashes() or plan["options"] != OPTIONS
            or plan["models"][model] != model_identity(reader.NAMES[model])
            or plan["ollama_version"] != local_api("/api/version")
            or prepared["registration"] != sha(output/"registration.json")
            or prepared["compiled"][phase] != sha(output/(phase+".compiled.json"))):
        raise ValueError("frozen plan, source, input, server or model changed")
    if local_api("/api/ps").get("models"):
        raise ValueError("wait for resident workload; no model eviction")
    if phase == "development":
        if trained is None:
            raise ValueError("freeze trained policy selection before development predictions")
        trained = outside_repo(trained)
        training = json.loads((trained/"training.json").read_bytes())
        if training["development_keys_opened"] or training["selection_sha256"] != sha(trained/"selection.json"):
            raise ValueError("fresh frozen policy selection required")
        freeze = dict(selection=sha(trained/"selection.json"), training=sha(trained/"training.json"),
                      registration=sha(trained/"registration.json"))
        freeze_path = output/"development-policy-freeze.json"
        if freeze_path.exists():
            if json.loads(freeze_path.read_bytes()) != freeze:
                raise ValueError("development policy selection changed")
        else:
            write_new(freeze_path, freeze)
    for name in MODELS:
        if sha(ROOT.parent/reader.TOKENIZERS[name]/"tokenizer.json") != plan["tokenizer_hashes"][name]:
            raise ValueError("tokenizer changed")
    run_dir = output/(phase+"-"+model)
    run_dir.mkdir(exist_ok=True)
    if (run_dir/"complete.json").exists():
        raise ValueError("completed run is immutable")
    reader.OPTIONS = dict(OPTIONS)
    started = time.perf_counter()
    # One writer at a time. A stale lock is retained for explicit crash review.
    with (run_dir/"run.lock").open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    warm_path = run_dir/"warmup.json"
    if not warm_path.exists():
        tick = time.perf_counter()
        warm = reader.generate(reader.render("Reply OK", [dict(id="s0", text="OK", kind="OBSERVATION")], model=model, mode="direct"), model, mode="direct")
        write_new(warm_path, dict(response=warm, wall_seconds=time.perf_counter()-tick, resident=local_api("/api/ps")))
    rows = json.loads((output/(phase+".compiled.json")).read_bytes())
    count = 0
    for row in rows:
        key = digest(row["id"])
        for mode in row["order"]:
            stem = key+"-"+mode
            raw_path, result_path = run_dir/(stem+".raw.json"), run_dir/(stem+".json")
            if result_path.exists():
                count += 1
                continue
            if row["compile_error"]:
                write_new(result_path, dict(id=row["id"], mode=mode, model=model, error=row["compile_error"],
                                             prediction=None, check=None, answer=None, wall_seconds=0, called=False))
                count += 1
                continue
            prompt = row["prompts"][model][mode]
            if not raw_path.exists():
                tick = time.perf_counter()
                try:
                    response = reader.generate(prompt, model, mode=mode)
                    raw = dict(response=response, wall_seconds=time.perf_counter()-tick, error=None)
                except Exception as exc:
                    raw = dict(response=None, wall_seconds=time.perf_counter()-tick, error=f"{type(exc).__name__}: {exc}")
                write_new(raw_path, raw)
            else:
                raw = json.loads(raw_path.read_bytes())
            if raw["error"]:
                raise ValueError("saved request failure; review it without retrying")
            response = raw["response"]
            if response.get("prompt_eval_count") != row["input_tokens"][model][mode] or not response.get("done"):
                raise ValueError("native token count or request completion failed; raw response retained")
            tick = time.perf_counter()
            state, packet = make_packet(row["question"], row["selected"])
            check = check_citations(state, packet, response.get("response", "")) if mode == "cited" else None
            error, answer, prediction = None, None, None
            try:
                parsed = json.loads(response["response"])
                if (type(parsed) is not dict or set(parsed) != ({"answer"} if mode == "direct" else {"answer", "citations"})
                        or (mode == "cited" and not isinstance(parsed["citations"], list))
                        or (parsed["answer"] is not None and not isinstance(parsed["answer"], str))):
                    raise ValueError("invalid response schema")
                answer = parsed["answer"]
                prediction = locate_answer(answer, row["selected"])
            except (ValueError, TypeError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                parsed = None
            bound_prediction = None
            if check is not None and check.status == "source_bound":
                bound_prediction = locate_answer(check.answer, row["selected"], citations=parsed["citations"])
            result = dict(id=row["id"], model=model, mode=mode, answer=answer, prediction=prediction,
                          bound_prediction=bound_prediction, check=None if check is None else check.to_dict(),
                          error=error, called=True, raw_sha256=sha(raw_path), wall_seconds=raw["wall_seconds"],
                          process_seconds=time.perf_counter()-tick, input_tokens=response["prompt_eval_count"],
                          output_tokens=response.get("eval_count"), truncated=response.get("done_reason") == "length",
                          timings={k: response.get(k) for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
                          final_scope_check=packet.is_current(state))
            write_new(result_path, result)
            count += 1
        if count % 40 == 0:
            print(json.dumps(dict(phase=phase, model=model, records=count, total=len(rows)*2)), flush=True)
    write_new(run_dir/"complete.json", dict(records=count, invocation_seconds=time.perf_counter()-started,
                                            source_hashes=source_hashes(), model_unchanged=plan["models"][model] == model_identity(reader.NAMES[model]),
                                            files={p.name: sha(p) for p in run_dir.glob("*.json")}, resident=local_api("/api/ps")))
    local_api("/api/generate", dict(model=reader.NAMES[model], keep_alive=0))
    (run_dir/"run.lock").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--trained", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data, args.output)
    else:
        if args.phase is None or args.model is None:
            parser.error("run requires --phase and --model")
        run(args.output, args.phase, args.model, args.trained)
