"""Record full test outcomes and synthetic compiler costs; no model accuracy claim."""

import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_stamps.context_compiler import (  # noqa: E402
    CompileBudget,
    ContextCompiler,
    ContextTask,
    ModelProfile,
    default_render,
)
from context_stamps.context_state import (  # noqa: E402
    AccessScope,
    CanonicalNode,
    ContextClaim,
    ContextState,
    EvidenceRequirement,
    TemporalScope,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        outcome = "passed"
        for name, cases in (("failed", self.failures), ("error", self.errors), ("skipped", self.skipped)):
            if any(item[0] is test for item in cases):
                outcome = name
        self.rows.append(dict(test=test.id(), outcome=outcome, seconds=time.perf_counter() - self.started))
        super().stopTest(test)


def fixture(size):
    state = ContextState(tenant="lab", policy_revision="p1", clock=lambda: 1000)
    scope = AccessScope("lab", "scientist", "p1", ("reader",))
    for i in range(size):
        state.put(CanonicalNode(
            key=f"record-{i:04d}", revision="v1", tenant="lab", text="Reviewed experiment note. " * 20,
            kind="FACT", temporal=TemporalScope(0, 900), roles=("reader",), provenance="fictional-fixture",
            claims=(ContextClaim("batch_limit" if i == 0 else f"unrelated-{i}", "128"),),
            validation_revision="fixture-check-v1",
        ))
    task = ContextTask("compile", "What is the reviewed batch limit?", (EvidenceRequirement("batch_limit"),))
    profile = ModelProfile("exact-fixture-v1")
    return state, scope, task, profile


def measure():
    rows = []
    for size in (4, 16, 64, 256):
        state, scope, task, profile = fixture(size)
        compiler = ContextCompiler(state)
        for repeat in range(21):
            # Alternate ordering; identical renderer, authorization and temporal view.
            order = ("full", "compiled") if repeat % 2 == 0 else ("compiled", "full")
            result = dict(records=size, repeat=repeat, order=order)
            for method in order:
                start = time.perf_counter()
                if method == "full":
                    snapshot = state.snapshot(scope, at=1000, known_at=1000)
                    text = default_render(task, profile, snapshot.records)
                    selected = len(snapshot.records)
                else:
                    packet = compiler.compile(task, scope=scope, at=1000, known_at=1000, profile=profile,
                                              budget=CompileBudget(milliseconds=5000))
                    if packet.status != "complete" or not packet.optimal:
                        raise RuntimeError("fixture must compile exhaustively")
                    text, selected = packet.text, len(packet.snapshot.records)
                result[f"{method}_ms"] = (time.perf_counter() - start) * 1000
                result[f"{method}_bytes"] = len(text.encode())
                result[f"{method}_selected"] = selected
                payload = json.loads(text)
                values = {c["value"] for r in payload["evidence"] for c in r["claims"] if c["name"] == "batch_limit"}
                if values != {"128"}:
                    raise RuntimeError("required fact must remain exact")
            if repeat > 0:  # Warmup is excluded for both paths; no model or tokenizer is invoked.
                rows.append(result)
    summary = []
    for size in (4, 16, 64, 256):
        selected = [r for r in rows if r["records"] == size]
        item = dict(records=size, trials=len(selected))
        for method in ("full", "compiled"):
            durations = sorted(r[f"{method}_ms"] for r in selected)
            item[f"{method}_median_ms"] = statistics.median(durations)
            item[f"{method}_p95_ms"] = durations[18]  # nearest rank, ceil(.95 * 20) - 1
        item["byte_reduction_fraction"] = 1 - sum(r["compiled_bytes"] for r in selected) / sum(r["full_bytes"] for r in selected)
        summary.append(item)
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new evidence directory required; failed runs must remain inspectable")
    args.out.mkdir(parents=True)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(suite)
    with (args.out / "tests.log").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(log.getvalue())
    sources = [*ROOT.glob("context_stamps/**/*.py"), *ROOT.glob("tests/*.py"), Path(__file__),
               ROOT / "examples/enterprise_decision.py"]
    report = dict(schema=1, python=sys.version, platform=platform.platform(),
        source_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)},
        cpu=os.environ.get("PROCESSOR_IDENTIFIER"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        tests=result.rows, discovered=result.testsRun, passed=result.wasSuccessful(),
        test_log_sha256=sha(args.out / "tests.log"),
        limits=["Fictional deterministic facts, explicitly named requirements and exact checks; no trained model.",
                "CPU snapshot/compilation timing only; excludes ingestion, tokenization, inference and network.",
                "Byte reduction is not model-token savings. Fixtures have one relevant record amid irrelevant records.",
                "One machine and process, 20 sequential trials per size, no concurrency or production qualification."])
    if result.wasSuccessful():
        report["compiler_fixture"] = measure()
    with (args.out / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"tests": result.testsRun, "passed": result.wasSuccessful(),
                      "skipped": len(result.skipped), "summary": report.get("compiler_fixture", {}).get("summary")}))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
