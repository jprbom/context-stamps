"""Verify the published local Harbor installation/audit evidence without Harbor."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = ROOT / "evidence/enterprise-evaluation-v1"
    original = json.loads((directory / "harbor-preflight.json").read_text())
    installed = json.loads((directory / "harbor-installed-packages.json").read_text())
    reproduced = json.loads((directory / "harbor-reproducibility.json").read_text())
    audit_path = directory / "harbor-dependency-audit.json"
    audit = json.loads(audit_path.read_text())
    requirements = ROOT / "experiments/harbor-requirements.txt"
    assert sha(directory / "harbor-preflight.json") == installed["original_preflight_sha256"]
    assert sha(requirements) == installed["requirements_sha256"] == reproduced["requirements_sha256"]
    assert sha(audit_path) == reproduced["audit_sha256"]
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()  # noqa: E731
    expected = {normalize(k): v for k, v in installed["installed_packages"].items()}
    specified = dict(line.split("==", 1) for line in requirements.read_text().splitlines()
                     if line and not line.startswith("#"))
    assert {normalize(k): v for k, v in specified.items()} == expected
    assert {normalize(d["name"]): d["version"] for d in audit["dependencies"]} == expected
    assert len(expected) == len(audit["dependencies"]) == installed["installed_count"] == 90
    assert not any(d.get("skip_reason") or d.get("vulns") for d in audit["dependencies"])
    assert reproduced["known_findings"] == reproduced["skipped_packages"] == 0
    assert reproduced["fresh_offline_install_exit_code"] == reproduced["fresh_cli_version_exit_code"] == 0
    assert reproduced["fresh_cli_version"] == original["harbor_version"] == "0.23.0"
    assert original["agent_trials"] == original["model_calls"] == 0
    difference = set(original["packages"]) - set(installed["installed_packages"])
    assert difference == {"cortex-context-stamps"}
    print("Harbor: 90 installed packages, clean-install record and dated advisory audit verified; zero agent trials.")


if __name__ == "__main__":
    main()
