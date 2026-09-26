"""Fresh local routing-policy development, with exact-result and fixed-rule controls.

Copyright (c) 2026 Prashant Jagtap. MIT License.
RULER-specific input contracts, not a general semantic truth verifier. Training
updates a four-cell CPU policy, never the reader weights. No paid/cloud calls.
"""

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from local_eval import local_api, model_identity
from ruler_context import frame, native_score, runtime_context, select, verify_direct
from ruler_local_eval import MODEL, MODEL_DIGEST, OPTIONS, chat, generate, public_rows, sources, tokenizer_at
from ruler_native import outside_repo, sha, write_new

from context_stamps import ContextExpert, ContextNode, ContextRuntime, RuntimeBudget, Verification
from context_stamps.decisions.calibration import error_upper_bound
from context_stamps.local_learning import (
    EvaluationCohort,
    LearningLimits,
    LocalLearningRegistry,
    PairedOutcome,
    revision,
)
from context_stamps.local_policy import CellPolicy, PolicyObservation, fit_cell_policy

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reader", "learned", "fixed")


def code_sources():
    return sources() | {"experiments/ruler_local_learning.py": sha(__file__)}


def exact_result(view, selection):
    """Bind every input character and the full question to a verified receipt."""
    if selection.answer_values is None:
        return dict(status="abstained", result="", reused=False)
    runtime = ContextRuntime(tenant="ruler-local", principal="evaluator", role="reader", policy="public-v2")
    pieces = tuple(view.context[i:i+8000] for i in range(0, len(view.context), 8000)) or ("",)
    keys = tuple(f"source-{i:04d}" for i in range(len(pieces)))
    for key, text in zip(keys, pieces):
        runtime.put(ContextNode(key, text, selection.full_source_sha256, frozenset({"reader"})))
    prepared = runtime.prepare_context(view.suffix, eligible=keys,
        experts=[ContextExpert("full-source", lambda request: list(keys), estimated_ms=0)],
        baseline="full-source", scope="ruler-public", limit=len(keys),
        verifier=lambda packet: Verification(set(packet.sources) == set(keys)),
        budget=RuntimeBudget(bytes=1048576, milliseconds=10000, iterations=1))
    if prepared.status != "complete":
        return dict(status="abstained", result="", reused=False, reason=prepared.reason)
    result = runtime.run_verified(prepared, request_digest=hashlib.sha256(view.render(view.context).encode()).hexdigest(),
        model="exact-cpu-v2", prompt="ruler-json-list-v1", tool="ruler-contract-v2", verifier_revision="full-source-check-v1",
        compute=lambda packet: json.dumps(selection.answer_values, ensure_ascii=False),
        verify=lambda value, packet: verify_direct(view, value))
    return result | dict(receipt_hex=prepared.receipt.hex(), receipt_bytes=len(prepared.receipt),
                         source_count=len(keys), full_source_sha256=selection.full_source_sha256)


def strict_score(prediction, references, mode, truncated, error):
    """A stricter diagnostic beside the official permissive substring score.

    Exact tasks: comma/newline/whitespace separated reference atoms, each once,
    no extra prose. QA: casefolded whitespace-normalized equality to one alias.
    This is a custom metric, not a general factuality assessment.
    """
    import re
    if error or truncated:
        return False
    normalized = " ".join(prediction.casefold().split()).strip()
    refs = tuple(" ".join(r.casefold().split()).strip() for r in references)
    if mode == "part":
        return normalized in refs
    try:
        parsed = json.loads(prediction)
    except ValueError:
        parsed = None
    if isinstance(parsed, list) and all(type(v) is str for v in parsed):
        values = [" ".join(v.casefold().split()).strip() for v in parsed]
        return len(set(values)) == len(values) and set(values) == set(refs)
    # Match longest aliases first so a shorter answer cannot consume its prefix.
    remaining = normalized
    for term in sorted(set(refs), key=len, reverse=True):
        pattern = r"(?<![\w-])" + re.escape(term) + r"(?![\w-])"
        matches = list(re.finditer(pattern, remaining))
        if len(matches) != 1:
            return False
        match = matches[0]
        remaining = remaining[:match.start()] + remaining[match.end():]
    return re.fullmatch(r"[\s,;\[\]{}\"'.:-]*", remaining) is not None


def execute(row, arm, policy, binding, tok):
    start = time.perf_counter()
    view = frame(row["question"])
    selected = select(view)
    cell = selected.kind
    if arm == "reader":
        action = "reader"
    elif arm == "direct":
        action = "direct"
    elif arm == "fixed":
        action = "direct" if selected.answer_values is not None else "reader"
    else:
        action = policy.choose(cell, binding=binding, allowed_actions=("reader", "direct"))
        if action is None:
            raise ValueError("Frozen policy binding or action mismatch")
    record = dict(id=row["id"], task=row["task"], length=row["length"], arm=arm, action=action, cell=cell,
        prediction="", error=None, verified=False, truncated=False, input_tokens=0, output_tokens=0,
        model_calls=0, prompt_sha256=None, receipt=None)
    try:
        if action == "direct":
            result = exact_result(view, selected)
            record.update(prediction=result["result"], verified=result["status"] == "verified", receipt=result)
            if not record["verified"] and arm != "direct":
                action = "reader"
                record["action"] = "reader_after_abstention"
        if action == "reader":
            def count(text):
                return len(tok.encode(text, add_special_tokens=False).ids)
            text, receipt = runtime_context(view, selected, count)
            prompt = chat(view.render(text))
            tokens = count(prompt)
            if tokens > 30000:
                raise ValueError("Full rendered input exceeds limit; truncation refused")
            record.update(input_tokens=tokens, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          receipt=receipt, model_calls=1)
            response = generate(prompt)
            record.update(prediction=response.get("response", ""), output_tokens=response.get("eval_count", 0),
                          truncated=response.get("done_reason") == "length",
                          model_timings={k: response.get(k) for k in
                                         ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")})
            if not response.get("done") or response.get("prompt_eval_count") != tokens:
                raise ValueError("Incomplete response or server/tokenizer token mismatch")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.perf_counter()-start
    record["peak_ram_bytes"] = None  # No trustworthy whole-pipeline peak measurement.
    return record


def load_keys(data):
    """Called only after that phase's generation log is closed and hashed."""
    keys = {}
    for entry in json.loads((data/"inventory.json").read_bytes()):
        path = data/str(entry["length"])/entry["task"]/"keys.json"
        if sha(path) != entry["keys_sha256"]:
            raise ValueError("Scoring keys changed")
        for row in json.loads(path.read_bytes()):
            keys[row["id"]] = row
    return keys


def grade(records, keys):
    results = []
    for row in records:
        key = keys[row["id"]]
        value = 0. if row["error"] else native_score(row["prediction"], key["expected_answer"], key["match_type"])
        correct = strict_score(row["prediction"], key["expected_answer"], key["match_type"], row["truncated"], row["error"])
        results.append(row | dict(native_score=value, strict_correct=correct))
    return results


def phase(rows, arms, policy, binding, tok, output, name, frozen):
    if code_sources() != frozen:
        raise ValueError("Source changed after registration")
    records = []
    with (output/f"{name}-generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            order = list(arms)
            random.Random(name+":"+row["id"]+":97").shuffle(order)
            for arm in order:
                record = execute(row, arm, policy, binding, tok)
                stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                stream.flush()
                records.append(record)
            print(json.dumps(dict(phase=name, id=row["id"], completed=len(records))), flush=True)
    write_new(output/f"{name}-completed.json", dict(records=len(records), sha256=sha(output/f"{name}-generations.jsonl")))
    return records


def summarize(rows):
    return dict(samples=len(rows), native_mean=statistics.mean(r["native_score"] for r in rows),
        native_complete=sum(r["native_score"] == 1 for r in rows), strict_correct=sum(r["strict_correct"] for r in rows),
        verified=sum(r["verified"] for r in rows), errors=sum(r["error"] is not None for r in rows),
        truncations=sum(r["truncated"] for r in rows), model_calls=sum(r["model_calls"] for r in rows),
        input_tokens=sum(r["input_tokens"] for r in rows), output_tokens=sum(r["output_tokens"] for r in rows),
        wall_seconds=sum(r["wall_seconds"] for r in rows))


def run(train, holdout, tokenizer, output):
    output = outside_repo(output)
    output.mkdir(parents=True, exist_ok=False)
    train_rows, hold_rows = public_rows(train), public_rows(holdout)
    identity = model_identity(MODEL)
    if identity["digest"] != MODEL_DIGEST:
        raise ValueError("Pinned local reader required")
    frozen = code_sources()
    binding = revision(dict(sources=frozen, model=identity, options=OPTIONS, device=platform.platform()))
    # Identity derives from the complete public input, not a reused row number.
    train_ids = {r["id"]: hashlib.sha256(r["question"].encode()).hexdigest() for r in train_rows}
    hold_ids = {r["id"]: hashlib.sha256(r["question"].encode()).hexdigest() for r in hold_rows}
    if (len(set(train_ids.values())) != len(train_rows) or len(set(hold_ids.values())) != len(hold_rows)
            or set(train_ids.values()) & set(hold_ids.values())):
        raise ValueError("Exact public-input overlap in training/evaluation")
    limits = LearningLimits("total_tokens", 31000., 0., 0., 10000, deadline_ms=1000.)
    registration = dict(schema=1, model=identity, options=OPTIONS, sources=frozen, binding=binding,
        tokenizer_sha256=sha(tokenizer/"tokenizer.json"), ollama=local_api("/api/version"),
        training=dict(plan_sha256=sha(train/"plan.json"), inventory_sha256=sha(train/"inventory.json"), samples=len(train_rows)),
        holdout=dict(plan_sha256=sha(holdout/"plan.json"), inventory_sha256=sha(holdout/"inventory.json"), samples=len(hold_rows)),
        fitting=dict(cost_cap=31000., failure_penalty=20., minimum_tasks=20, cells=["exact", "dependency", "statistic", "full"]),
        limits=asdict(limits), sample_identity="SHA-256 of entire public question; exact overlap checked",
        qualification="Development only: source-level IID and true pipeline peak RAM are not established; cannot activate",
        cohorts={"retention": "fresh 4096-length tasks from training families", "adaptation": "fresh 16384-length tasks"},
        limitations=["Same benchmark grammar, not broad domain retention; public corpora/background reused",
                     "CPU four-cell policy fitting, no base-model weight update or attention modification",
                     "Local warm Ollama with native caching; randomized arm order, no isolated cold-latency claim",
                     "Exact operation plus independent grammar verifier can avoid model calls; QA is not truth-verified",
                     "Training tokens are amortized over a declared 10000 tasks; time/energy/bytes not token cost",
                     "Native substring score and custom strict output score reported separately",
                     "Reader uses raw ChatML/256 output cap; custom RULER backend, no official ranking"])
    write_new(output/"registration.json", registration)
    tok = tokenizer_at(tokenizer)
    started = time.perf_counter()
    warm = generate(chat("Reply with OK."))
    write_new(output/"warmup.json", dict(wall_seconds=time.perf_counter()-started,
        response={k: v for k, v in warm.items() if k != "context"}, resident=local_api("/api/ps")))
    raw_train = phase(train_rows, ("reader", "direct"), None, binding, tok, output, "train", frozen)
    train_keys = load_keys(train)
    scored_train = grade(raw_train, train_keys)
    write_new(output/"train-scored.json", scored_train)
    observations = tuple(PolicyObservation(train_ids[r["id"]], r["cell"], r["arm"], r["strict_correct"],
                                          float(r["input_tokens"]+r["output_tokens"])) for r in scored_train)
    started = time.perf_counter()
    fitting = {k: v for k, v in registration["fitting"].items() if k != "cells"}
    policy = fit_cell_policy(observations, binding=binding, baseline_action="reader", **fitting)
    fit_seconds = time.perf_counter()-started
    write_new(output/"policy.json", dict(policy=asdict(policy), revision=policy.revision, fitting_seconds=fit_seconds,
                                         observations=[asdict(r) for r in observations]))
    baseline = CellPolicy(binding, "reader", ())
    update_tokens = sum(r["input_tokens"]+r["output_tokens"] for r in scored_train)
    limits = LearningLimits(**(asdict(limits) | {"update_cost": float(update_tokens)}))
    registry = LocalLearningRegistry(str(output/"registry.sqlite"), scope="ruler-development", binding=binding,
                                     baseline=baseline.revision)
    cohorts = tuple(EvaluationCohort(name, tuple(hold_ids[r["id"]] for r in hold_rows if r["length"] == length))
                    for name, length in (("retention", 4096), ("adaptation", 16384)))
    try:
        plan = registry.register(policy.revision, evaluation_clusters=tuple(hold_ids.values()),
                                 training_clusters=tuple(train_ids.values()), limits=limits, cohorts=cohorts)
        write_new(output/"registered-trial.json", plan)
        raw_hold = phase(hold_rows, ARMS, policy, binding, tok, output, "holdout", frozen)
        hold_keys = load_keys(holdout)
        scored_hold = grade(raw_hold, hold_keys)
        write_new(output/"holdout-scored.json", scored_hold)
        lookup = {(r["id"], r["arm"]): r for r in scored_hold}
        paired = []
        for source in hold_rows:
            old, new = (lookup[source["id"], arm] for arm in ("reader", "learned"))
            paired.append(PairedOutcome(hold_ids[source["id"]], revision([old, new]), old["strict_correct"],
                new["strict_correct"], float(old["input_tokens"]+old["output_tokens"]),
                float(new["input_tokens"]+new["output_tokens"]), new["wall_seconds"]*1000., None))
        report = registry.finish(plan, tuple(paired))
        if report["eligible"] or registry.active_revision(binding=binding) != baseline.revision:
            raise ValueError("Unqualified development candidate must never activate")
        write_new(output/"promotion.json", dict(report=report, observations=[asdict(r) for r in paired],
            active_revision=registry.active_revision(binding=binding), baseline_revision=baseline.revision))
    finally:
        registry.close()
    summary = dict(schema=1, training={arm: summarize([r for r in scored_train if r["arm"] == arm]) for arm in ("reader", "direct")},
        holdout={arm: summarize([r for r in scored_hold if r["arm"] == arm]) for arm in ARMS},
        cells=[dict(length=length, task=task, arms={arm: summarize([r for r in scored_hold if
            (r["length"], r["task"], r["arm"]) == (length, task, arm)]) for arm in ARMS})
            for length, task in sorted({(r["length"], r["task"]) for r in scored_hold})],
        policy_choices=policy.choices, policy_revision=policy.revision, update_tokens=update_tokens,
        fitting_seconds=fit_seconds, promoted=False, promotion_reasons=report["reasons"],
        zero_new_failure_upper_if_iid={c.name: error_upper_bound(0, len(c.clusters), plan["round_alpha"]/7) for c in cohorts},
        zero_failure_bound_caveat="Sample-size diagnostic only: no source-cluster IID qualification")
    write_new(output/"summary.json", summary)
    write_new(output/"references.json", dict(training=list(train_keys.values()), holdout=list(hold_keys.values())))
    write_new(output/"input-identities.json", dict(training=[{k: v for k, v in r.items() if k != "question"} |
        dict(input_sha256=train_ids[r["id"]]) for r in train_rows], holdout=[{k: v for k, v in r.items() if k != "question"} |
        dict(input_sha256=hold_ids[r["id"]]) for r in hold_rows]))
    print(json.dumps(summary["holdout"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.train, args.holdout, args.tokenizer, args.output)
