"""One-command changing-source demonstration; all files are temporary and fictional."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_stamps import ContextMemory, content_digest
from context_stamps.selection import select_evidence
from context_stamps.sources import explain_versions, observe_files


def main():
    with tempfile.TemporaryDirectory() as folder, ContextMemory() as memory:
        root = Path(folder)
        source = root / "transfer.py"
        source.write_text("if balance >= amount: transfer(amount)", encoding="utf-8")
        observed = observe_files(memory, root, ["transfer.py"])
        memory.add(
            "Test that exact-balance transfers succeed.", source="plan", dependencies=observed["revisions"]
        )
        old_plan = memory.get("plan")["digest"]
        source.write_text("if balance > amount: transfer(amount)", encoding="utf-8")
        changed = observe_files(memory, root, ["transfer.py"])
        revisions = {**changed["revisions"], "plan": old_plan}
        explanation = explain_versions(memory, "plan", revisions)
        packet = select_evidence(
            memory, "transfer balance", revisions=revisions, required=["transfer.py", "plan"], budget=1024
        )
        assert explanation["status"] == "stale"
        assert packet.status == "insufficient_evidence" and not packet.text
        memory.add(
            "Test that exact-balance transfers are rejected.",
            source="plan",
            dependencies=changed["revisions"],
        )
        revisions["plan"] = content_digest(memory.get("plan")["text"])
        repaired = select_evidence(
            memory, "transfer balance", revisions=revisions, required=["transfer.py", "plan"], budget=1024
        )
        assert repaired.status == "current"
        print(
            json.dumps(
                {
                    "changed": changed,
                    "why_plan_is_stale": explanation,
                    "before_refresh": packet.to_dict(),
                    "after_refresh": repaired.to_dict(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
