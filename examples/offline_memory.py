"""Complete CPU example; run after installing the repository."""

import json

from context_stamps import ContextMemory, content_digest


def main():
    with ContextMemory() as memory:
        old = "if balance >= amount: transfer(amount)"
        new = "if balance > amount: transfer(amount)"
        memory.add(old, source="ledger.py")
        memory.add(old, source="duplicate-read")
        memory.add(
            "Transfer tests must cover an exact balance.",
            source="test-plan",
            dependencies={"ledger.py": content_digest(old)},
        )
        memory.add(new, source="ledger.py")
        packet = memory.pack(
            "balance transfer tests", token_budget=1500, revisions={"ledger.py": content_digest(new)}
        )
        print(json.dumps(packet.to_dict(), indent=2))
        assert new in packet.text
        assert old not in packet.text
        assert memory.get("test-plan")["stale"]


if __name__ == "__main__":
    main()
