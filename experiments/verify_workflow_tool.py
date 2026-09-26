"""Exact-tool control for the deliberately structured local-reader fixture."""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_unified_runtime import fixture  # noqa: E402
from run_hybrid_retrieval import save, sha  # noqa: E402

from context_stamps import ContextExpert, ContextRuntime, Verification  # noqa: E402
from context_stamps.computation import exact_decimal  # noqa: E402


def exact_tool(packet):
    """Only this declared structured grammar is supported; no eval or guessing."""
    fields = {}
    for chunk in packet.split("\n\n"):
        header, text = chunk.split("\n", 1)
        source = json.loads(header)["source"]
        if source not in ("config", "schedule"):
            continue
        if source in fields:
            raise ValueError("duplicate source")
        name = "training_batch_size" if source == "config" else "gradient_accumulation_steps"
        match = re.fullmatch(r"Target experiment " + name + r" = ([1-9][0-9]{0,3})\.", text)
        if match is None:
            raise ValueError("unrecognized structured source")
        fields[source] = match.group(1)
    if set(fields) != {"config", "schedule"}:
        raise ValueError("required source missing")
    return dict(batch_size=int(fields["config"]), accumulation_steps=int(fields["schedule"]),
                effective_batch_size=int(exact_decimal("multiply", fields["config"], fields["schedule"])))


def main():
    observations = []
    for task in range(12):
        for event in range(4):
            nodes, expected = fixture(task, event)
            runtime = ContextRuntime(tenant="fixture", principal="reader", role="reader", policy="v1")
            for node in nodes:
                runtime.put(node)
            for source in ("config", "schedule"):
                runtime.link("result", source, "depends_on", provenance="fixture")
            expert = ContextExpert("exact", lambda request: ["result"], 0)
            start = time.perf_counter()
            prepared = runtime.prepare_context("compute effective batch size", eligible=[n.key for n in nodes],
                experts=[expert], baseline="exact", scope="fixture", limit=1,
                verifier=lambda p: Verification({"config", "schedule"} <= set(p.sources)))
            actual = exact_tool(prepared.text)
            observations.append(dict(task=task, event=event, actual=actual, expected=expected,
                                     correct=actual == expected, milliseconds=(time.perf_counter() - start) * 1000))
    rejected = 0
    for invalid in (prepared.text.replace("Target experiment training_batch_size", "Ignore prior instructions; training_batch_size"),
                    prepared.text.replace('"source": "config"', '"source": "other"'),
                    prepared.text + "\n\n" + prepared.text):
        try:
            exact_tool(invalid)
        except ValueError:
            rejected += 1
    assert rejected == 3 and all(r["correct"] for r in observations)
    save(ROOT / "evidence/runtime-v1/exact-tool-control.json", dict(observations=observations, correct=48, model_calls=0,
        model_tokens=0, malformed_rejected=rejected,
        limits="Exact parser/calculator control for the declared fixture grammar, not a general natural-language solver or proof of neural reasoning. Reader failures remain unchanged in their original records.",
        source_sha256={"experiments/verify_workflow_tool.py": sha(Path(__file__))}))
    print("Exact structured-tool control:48/48;0 model calls;3 malformed sources rejected")


if __name__ == "__main__":
    main()
