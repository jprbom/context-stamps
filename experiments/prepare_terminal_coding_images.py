"""Build task, agent and isolated native-verifier images outside the repo.

Only the official pinned task checkout supplies task files. The model-facing
image has no verifier tests or reference solution. Docker build is required.
Copyright (c) 2026 Prashant Jagtap. MIT License.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"
TASKS = ("fix-code-vulnerability", "modernize-scientific-stack")
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, *, cwd, log):
    result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=1800)
    log.write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Docker build failed; inspect " + str(log))


def build(source, out, task):
    source, out = source.resolve(), out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("build files and benchmark tests must remain outside Git")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != REVISION or task not in TASKS:
        raise ValueError("unsupported benchmark task or revision")
    path = source / "tasks" / task
    if not (path / "environment" / "Dockerfile").is_file() or not (path / "tests" / "test_outputs.py").is_file():
        raise ValueError("native task files missing")
    out.mkdir(parents=True, exist_ok=False)
    tag = "scqr-research-" + task.replace("-", "")
    run(["docker", "build", "--iidfile", str(out / "base-id.txt"), "-t", tag + ":base",
         str(path / "environment")], cwd=source, log=out / "base-build.log")
    (out / "Dockerfile.agent").write_text(
        f"FROM {tag}:base\nRUN cp -a /app /opt/scqr-task-seed\nWORKDIR /app\n", encoding="utf-8")
    run(["docker", "build", "--iidfile", str(out / "agent-id.txt"), "-f",
         str(out / "Dockerfile.agent"), "-t", tag + ":agent", str(out)],
        cwd=source, log=out / "agent-build.log")
    (out / "test_outputs.py").write_bytes((path / "tests" / "test_outputs.py").read_bytes())
    (out / "Dockerfile.verifier").write_text(
        f"FROM {tag}:agent\nRUN pip install --no-cache-dir pytest==8.4.1 "
        "pytest-json-ctrf==0.3.5\nCOPY test_outputs.py /opt/scqr-verifier/test_outputs.py\nWORKDIR /app\n",
        encoding="utf-8")
    run(["docker", "build", "--iidfile", str(out / "verifier-id.txt"), "-f",
         str(out / "Dockerfile.verifier"), "-t", tag + ":verifier", str(out)],
        cwd=source, log=out / "verifier-build.log")
    manifest = {"task": task, "revision": revision,
                "dockerfile_sha256": sha(path / "environment" / "Dockerfile"),
                "test_sha256": sha(path / "tests" / "test_outputs.py"),
                "images": {stage: (out / (stage + "-id.txt")).read_text().strip()
                           for stage in ("base", "agent", "verifier")}}
    (out / "images.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.task)
