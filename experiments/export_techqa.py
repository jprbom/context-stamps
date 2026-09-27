"""Export complete local TechQA evidence without prompts or source documents.

Copyright (c) 2026 Prashant Jagtap. MIT License for code.
Published native target projections retain TechQA's CDLA-Permissive-1.0 terms.
"""

import argparse
import base64
import gzip
import json
from pathlib import Path

from ruler_native import sha, write_new
from techqa_context import digest, terms
from techqa_native import canaries, load_native
from techqa_policy import features

ROOT = Path(__file__).resolve().parents[1]
PHASES = {"fit": 400, "calibration": 177, "development": 310}


def save(path, value):
    if path.suffix == ".gz":
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with path.open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0))
    else:
        write_new(path, value)


def load(path):
    return json.loads(path.read_bytes())


def export(data, runs, policy, analysis, output):
    data, runs, policy, analysis, output = map(lambda p: Path(p).resolve(), (data, runs, policy, analysis, output))
    if output.exists():
        raise ValueError("new evidence directory required")
    plan, prepared, data_plan = load(runs/"registration.json"), load(runs/"prepared.json"), load(data/"manifest.json")
    summary = load(analysis/"summary.json")
    if (prepared["registration"] != sha(runs/"registration.json") or plan["data_manifest"] != sha(data/"manifest.json")
            or summary["evidence_hashes"]["completed_run"] != sha(runs/"development-modern"/"complete.json")
            or summary["evidence_hashes"]["trained"] != sha(policy/"training.json")
            or summary["policy_activation"] or summary["base_weights_changed"]):
        raise ValueError("frozen completed study required")
    snapshots = load(runs/"source-snapshot.json")
    content = {}
    for name, expected in snapshots.items():
        p = runs/"source"/name
        if sha(p) != expected or sha(ROOT/name) != expected:
            raise ValueError("measured source bytes changed: "+name)
        content[expected] = base64.b64encode(p.read_bytes()).decode()
    records, contexts, targets, completions, warmups = [], {}, {}, {}, {}
    for phase, count in PHASES.items():
        folder = runs/(phase+"-modern")
        complete = load(folder/"complete.json")
        if complete["records"] != count*2 or not complete["model_unchanged"] or complete["source_hashes"] != plan["source_hashes"]:
            raise ValueError("complete paired phase required")
        for name, expected in complete["files"].items():
            if sha(folder/name) != expected:
                raise ValueError("reader artifact changed: "+name)
        if sha(runs/(phase+".compiled.json")) != prepared["compiled"][phase]:
            raise ValueError("compiled inputs changed")
        if sha(data/(phase+".keys.json")) != data_plan["files"][phase+".keys.json"]:
            raise ValueError("reference projection changed")
        compiled = {r["id"]: r for r in load(runs/(phase+".compiled.json"))}
        keys = load(data/(phase+".keys.json"))
        if len(keys) != count or set(keys) != set(compiled):
            raise ValueError("declared phase coverage differs")
        targets[phase] = {qid: {k: key[k] for k in ("ANSWERABLE", "DOCUMENT", "START_OFFSET", "END_OFFSET")} for qid, key in keys.items()}
        contexts[phase] = {qid: dict(cluster=digest(" ".join(terms(row["question"]))),
            compile_error=row["compile_error"], compile_seconds=row["compile_seconds"],
            input_tokens=row["input_tokens"]["modern"], candidate_windows=row["candidate_windows"],
            selected=[{k: w[k] for k in ("doc_id", "start", "end", "score", "query_coverage")} for w in row["selected"]])
            for qid, row in compiled.items()}
        seen = set()
        for name in sorted(complete["files"]):
            if not name.endswith(("-direct.json", "-cited.json")):
                continue
            row = load(folder/name)
            key = (row["mode"], row["id"])
            if key in seen or row["id"] not in compiled or row["model"] != "modern":
                raise ValueError("duplicate or foreign reader result")
            seen.add(key)
            answer = row.get("answer")
            checked = row.get("check")
            check = None if checked is None else {k: v for k, v in checked.items() if k != "answer"}
            public = {k: row.get(k) for k in ("id", "mode", "model", "prediction", "bound_prediction", "error",
                "called", "raw_sha256", "wall_seconds", "process_seconds", "input_tokens", "output_tokens",
                "truncated", "timings", "final_scope_check")}
            public.update(phase=phase, answer_present=answer is not None,
                answer_sha256=None if answer is None else digest(answer), check=check,
                observable_features=features(compiled[row["id"]], row), record_sha256=sha(folder/name))
            records.append(public)
        if seen != {(mode, qid) for mode in ("direct", "cited") for qid in keys}:
            raise ValueError("missing declared reader control")
        completions[phase] = dict(records=complete["records"], completion_sha256=sha(folder/"complete.json"),
                                  invocation_seconds=complete["invocation_seconds"])
        warm = load(folder/"warmup.json")
        warmups[phase] = dict(wall_seconds=warm["wall_seconds"],
            input_tokens=warm["response"].get("prompt_eval_count"), output_tokens=warm["response"].get("eval_count"),
            record_sha256=sha(folder/"warmup.json"))
    native = load_native(Path(data_plan["data"])/"techqa_evaluation.py")
    output.mkdir(parents=True)
    save(output/"targets.json.gz", targets)
    save(output/"reader-records.json.gz", records)
    save(output/"contexts.json.gz", contexts)
    save(output/"historical-sources.json.gz", dict(hashes=snapshots, content_by_sha256=content))
    save(output/"reader-registration.json", plan)
    save(output/"training.json", load(policy/"training.json"))
    save(output/"selection.json", load(policy/"selection.json"))
    save(output/"policy-registration.json", load(policy/"registration.json"))
    public_summary = dict(summary)
    public_summary["evidence_hashes"] = dict(summary["evidence_hashes"])
    public_summary["evidence_hashes"]["reference_projection_sha256"] = public_summary["evidence_hashes"].pop("keys")
    public_summary["original_summary_sha256"] = sha(analysis/"summary.json")
    save(output/"summary.json", public_summary)
    save(output/"predictions.json.gz", load(analysis/"predictions.json"))
    save(output/"per-question.json.gz", load(analysis/"per-question.json"))
    save(output/"native-canaries.json", dict(source_sha256=sha(Path(data_plan["data"])/"techqa_evaluation.py"), cases=canaries(native)))
    for name in ("ridge-1.json", "ridge-10.json", "ridge-100.json"):
        save(output/name, load(policy/name))
    public_data = {k: v for k, v in data_plan.items() if k not in ("data", "files")}
    public_data["files"] = [dict(file=name, sha256=value) for name, value in data_plan["files"].items()]
    public_data["original_manifest_sha256"] = sha(data/"manifest.json")
    save(output/"data-protocol.json", public_data)
    save(output/"execution.json", dict(completions=completions, warmups=warmups, measured_requests=len(records),
        actual_warmup_requests=len(warmups), source_hashes=snapshots,
        scope="Source-span predictions, typed check outcomes, fit features and costs. Raw prompts/replies and corpus remain external.",
        source_binding_replay="Requires separately downloaded original corpus and raw local requests; not certified by offset replay.",
        no_energy_or_edge_measurement=True, provider_charges_usd=0))
    (output/"CDLA-Permissive-v1.0.pdf").write_bytes((Path(data_plan["data"])/"CDLA-Permissive-v1.0.pdf").read_bytes())
    # Exact upstream README bytes accompany the derived reference projection.
    (output/"DATA_README.txt").write_bytes((Path(data_plan["data"])/"README.txt").read_bytes())
    print(json.dumps(dict(exported_requests=len(records), development_questions=len(targets["development"]),
                         candidate_active=False, destination=str(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "policy", "analysis", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    export(args.data, args.runs, args.policy, args.analysis, args.output)
