"""Fetch the pinned public tasks or build offline verifier images outside Git.

Never executes an upstream solution on the host. Task execution belongs to the
separate qualified sandbox. Downloaded evaluation data is not training data.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from harbor_sandbox import PYTHON_IMAGE
from terminal_pilot import TASKS, check_source

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/terminal-pilot-v1"


def fresh_out(path):
    path = path.resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Keep downloaded tasks and Docker build contexts outside the repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def fetch(out):
    manifest_path = EVIDENCE / "upstream/source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["repository"] != "harbor-framework/terminal-bench-2-1":
        raise ValueError("Unexpected upstream")
    from terminal_pilot import UPSTREAM
    if manifest["revision"] != UPSTREAM:
        raise ValueError("Unexpected revision")
    out = fresh_out(out)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError("Unexpected task download redirect")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    for name, record in manifest["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name:
            raise ValueError("Invalid task path")
        url = "https://raw.githubusercontent.com/" + manifest["repository"] + "/" + UPSTREAM + "/" + name
        with opener.open(url, timeout=45) as response:
            raw = response.read(record["bytes"] + 1)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Downloaded task differs from the pinned evidence")
        target = out / parts
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    shutil.copyfile(manifest_path, out / "source-manifest.json")
    check_source(out)
    print(f"Verified {len(manifest['files'])} pinned files; evaluation only.")


def build(source, out):
    source = source.resolve()
    check_source(source)
    out = fresh_out(out)
    requirements = EVIDENCE / "verifier-requirements.txt"
    record = json.loads((EVIDENCE / "manifest.json").read_text())
    if hashlib.sha256(requirements.read_bytes()).hexdigest() != record["files"][requirements.name]:
        raise ValueError("Verifier requirements changed")
    images = {}
    for task in TASKS:
        context = out / task
        context.mkdir()
        shutil.copyfile(requirements, context / "requirements.txt")
        shutil.copytree(source / "tasks" / task / "tests", context / "tests")
        (context / "Dockerfile").write_text(
            f"FROM {PYTHON_IMAGE}\nCOPY requirements.txt /requirements.txt\n"
            "RUN python -m pip install --no-cache-dir --require-hashes -r /requirements.txt\n"
            "COPY tests /tests\nWORKDIR /app\n", encoding="utf-8", newline="\n")
        tag = "scqr-" + task + "-verifier:" + uuid.uuid4().hex[:12]
        result = subprocess.run(["docker", "build", "--iidfile", str(context / "image-id.txt"),
                                 "--tag", tag, str(context)], capture_output=True, timeout=900)
        (context / "build.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Verifier image build failed; inspect retained build.log")
        images[task] = (context / "image-id.txt").read_text().strip()
    (out / "images.json").write_text(json.dumps(images, indent=2) + "\n", encoding="utf-8")
    print("Built both isolated verifier images; qualify their native graders before inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fetch", "build"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.out)
    else:
        if args.source is None:
            parser.error("build requires --source")
        build(args.source, args.out)
