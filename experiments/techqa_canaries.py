"""Retain all fictional interface checks before new benchmark predictions."""

import argparse
import json
import time
from pathlib import Path

import techqa_generation as reader
from citation_canaries import CASES
from local_eval import local_api, model_identity
from ruler_native import outside_repo, sha, write_new

from context_stamps.citations import check_citations, prepare_citations
from context_stamps.context_state import AccessScope, CanonicalNode, ContextState, TemporalScope


def run(model, output):
    output = outside_repo(output)
    if local_api("/api/ps").get("models"):
        raise ValueError("wait for existing local model")
    identity = model_identity(reader.NAMES[model])
    names = ("experiments/techqa_canaries.py", "experiments/techqa_generation.py", "experiments/cited_reader.py",
             "experiments/citation_canaries.py", "context_stamps/citations.py", "context_stamps/context_state.py",
             "context_stamps/security.py", "experiments/local_eval.py", "experiments/ruler_native.py")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"registration.json", dict(model=identity, options=reader.OPTIONS,
                                               tokenizer_sha256=sha(reader.ROOT.parent/reader.TOKENIZERS[model]/"tokenizer.json"),
                                               source_hashes={name: sha(reader.ROOT/name) for name in names},
                                               purpose="plumbing after direct-schema failure; adapted fictional canaries, not quality evaluation"))
    tok = reader.tokenizer_for(model)
    warm = reader.generate(reader.render("Reply OK", [dict(id="s0", text="OK", kind="OBSERVATION")], model=model, mode="direct"), model, mode="direct")
    write_new(output/"warmup.json", warm)
    rows = []
    for name, question, texts, expected in CASES:
        state = ContextState(tenant="fixture", policy_revision="p1", clock=lambda: 1)
        for i, text in enumerate(texts):
            state.put(CanonicalNode(key=f"doc{i}", revision="1", tenant="fixture", text=text, kind="OBSERVATION",
                                    temporal=TemporalScope(1, 1), roles=("reader",), provenance="fiction"))
        snap = state.snapshot(AccessScope("fixture", "tester", "p1", ("reader",)), at=1, known_at=1)
        packet = prepare_citations(state, snap, question)
        for mode in ("direct", "cited"):
            prompt = reader.render(question, json.loads(packet.payload)["sources"], model=model, mode=mode)
            tick = time.perf_counter()
            response = reader.generate(prompt, model, mode=mode)
            wall = time.perf_counter()-tick
            write_new(output/(name+"-"+mode+".raw.json"), dict(response=response, wall_seconds=wall))
            count = len(tok.encode(prompt, add_special_tokens=False).ids)
            if count != response["prompt_eval_count"] or not response.get("done"):
                raise ValueError("native prompt-token parity or completion failed")
            parsed = json.loads(response["response"])
            check = check_citations(state, packet, response["response"]) if mode == "cited" else None
            row = dict(case=name, mode=mode, expected=expected, output=parsed,
                       exact_fixture_match=parsed["answer"] == expected, expected_prompt_tokens=count,
                       source_check=None if check is None else check.to_dict(), wall_seconds=wall,
                       output_tokens=response["eval_count"])
            rows.append(row)
            print(json.dumps(dict(case=name, mode=mode, output=parsed), ensure_ascii=True), flush=True)
    write_new(output/"complete.json", dict(rows=rows, model_unchanged=model_identity(reader.NAMES[model]) == identity,
                                           resident=local_api("/api/ps")))
    local_api("/api/generate", dict(model=reader.NAMES[model], keep_alive=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(reader.NAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.output)
