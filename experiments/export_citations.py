"""Export all fictional citation attempts, including interface failures."""

import json
import shutil
from pathlib import Path

from citation_canaries import CASES
from ruler_native import sha, write_new

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/"evidence/citations-v1"
    target.mkdir(parents=True, exist_ok=False)
    paths = dict(
        original_small=ROOT.parent/"citation-canaries-small-v1",
        original_modern=ROOT.parent/"citation-canaries-modern-v1",
        revised_small=ROOT.parent/"techqa-canaries-small-v2",
        revised_modern=ROOT.parent/"techqa-canaries-modern-v2",
    )
    cases = {name: dict(question=q, texts=list(texts), expected=expected) for name, q, texts, expected in CASES}
    runs = []
    for label, directory in paths.items():
        registration = json.loads((directory/"registration.json").read_bytes())
        for name, digest in registration["source_hashes"].items():
            if sha(ROOT/name) != digest:
                raise ValueError("measured source changed before export")
        saved = target/label
        saved.mkdir()
        files = {}
        for path in directory.glob("*.json"):
            shutil.copyfile(path, saved/path.name)
            files[path.name] = sha(saved/path.name)
        complete = json.loads((directory/"complete.json").read_bytes())
        rows = []
        for row in complete["rows"]:
            raw = row.get("response")
            if raw is None:
                raw = json.loads((directory/(row["case"]+"-"+row["mode"]+".raw.json")).read_bytes())["response"]
            rows.append(dict(case=row["case"], mode=row["mode"], raw_reply=raw["response"],
                             parsed=json.loads(raw["response"]), input_tokens=raw["prompt_eval_count"],
                             expected_prompt_tokens=row["expected_prompt_tokens"], output_tokens=raw["eval_count"],
                             done=raw.get("done"), wall_seconds=row["wall_seconds"],
                             check=row["source_check"], expected=cases[row["case"]]["expected"]))
        runs.append(dict(name=label, model=registration["model"], files=files, rows=rows))
    write_new(target/"calls.json", dict(cases=cases, runs=runs, scored_calls=48, warmups=4,
                                         evidence_class="adapted fictional interface checks; not model-quality evaluation"))
    write_new(target/"manifest.json", dict(exporter=sha(__file__),
                                            files={str(p.relative_to(target).as_posix()): sha(p) for p in target.rglob("*.json")}))


if __name__ == "__main__":
    main()
