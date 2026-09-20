"""Verify the committed v0.4 capsule control record."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "evidence" / "capsule-controls-v1" / "results.json").read_text(encoding="utf-8"))
assert data["format"] == "capsule-controls-v1"
assert data["capsule_bits"] == 256
assert data["facet_cases_passed"] == len(data["facet_cases"]) == 8
assert data["roundtrips_passed"] == data["roundtrips_total"] == 100
assert data["router_controls_passed"] is True
assert data["router_routes"] == ["exact", "abstain", "precise", "precise", "compact"]
print("capsule control evidence verified")
