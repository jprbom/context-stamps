"""Claims come from trusted application structure, not model-inferred truth."""

import json

from context_stamps import Claim, ContextMemory, Requirement, select_structured


def main():
    with ContextMemory() as memory:
        # Real integrations can construct claims from typed configuration records.
        config = {"subject": "worker-alpha", "timeout_ms": "250", "retries": "3"}
        row = memory.add(json.dumps(config), source="worker.json")
        claims = [
            Claim("worker.json", row["digest"], config["subject"], key, config[key])
            for key in ["timeout_ms", "retries"]
        ]
        packet = select_structured(
            memory,
            "worker-alpha timeout and retries",
            requirements=[Requirement("worker-alpha", "timeout_ms"), Requirement("worker-alpha", "retries")],
            claims=claims,
            revisions={"worker.json": row["digest"]},
            budget_bytes=512,
        )
        assert packet.status == "current"
        print(json.dumps(packet.to_dict(), indent=2))


if __name__ == "__main__":
    main()
