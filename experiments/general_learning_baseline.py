"""Capture a local foundation baseline; no model training or benchmark claims."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return {"command": ["python" if v == sys.executable else v for v in command],
            "returncode": proc.returncode, "seconds": time.perf_counter() - started,
            "stdout": proc.stdout, "stderr": proc.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    manifest = {
        "schema": 1, "profile": args.profile, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": run(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "source_status": run(["git", "status", "--porcelain"])["stdout"].splitlines(),
        "python": platform.python_version(), "platform": platform.platform(),
        "isolated": sys.prefix != sys.base_prefix and "include-system-site-packages = false" in
                    (Path(sys.prefix) / "pyvenv.cfg").read_text().lower(),
        "packages": dict(sorted(packages.items())),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,"
                    "temperature.gpu,power.draw", "--format=csv"]),
        "limits": ["Tests and recorded-evidence replay; no fresh GPU benchmark or independent labels.",
                   "CPU profile intentionally excludes torch. CUDA profile must execute optional tests."],
    }
    if args.profile == "cuda":
        import torch
        manifest["cuda"] = {"torch": torch.__version__, "runtime": torch.version.cuda,
                            "available": torch.cuda.is_available()}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile requires a working CUDA device")
        torch.manual_seed(7)
        x = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
        torch.cuda.synchronize()
        manifest["cuda"]["matmul_parity"] = bool(torch.equal((x @ x.T).cpu(), x.cpu() @ x.cpu().T))
        manifest["cuda"]["device"] = torch.cuda.get_device_name(0)
        if not manifest["cuda"]["matmul_parity"]:
            raise RuntimeError("CUDA smoke parity failed")
        del x
        torch.cuda.empty_cache()
    stream = io.StringIO()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    manifest["tests"] = {"discovered": result.testsRun,
        "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
        "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
        "failures": [t.id() for t, _ in result.failures], "errors": [t.id() for t, _ in result.errors],
        "seconds": time.perf_counter() - started}
    log = stream.getvalue().replace(str(ROOT), "<repository>").replace(str(Path.home()), "<home>")
    (args.out / "tests.log").write_text(log, encoding="utf-8")
    manifest["checks"] = [run([sys.executable, p]) for p in (
        "experiments/verify_controller_v2.py", "experiments/verify_runtime.py", "examples/unified_context.py")]
    manifest["source_hashes"] = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
        for base in ("context_stamps", "tests") for p in sorted((ROOT / base).rglob("*.py"))}
    manifest["source_hashes"]["experiments/general_learning_baseline.py"] = digest(Path(__file__))
    manifest["test_log_sha256"] = digest(args.out / "tests.log")
    manifest["passed"] = (result.wasSuccessful() and all(c["returncode"] == 0 for c in manifest["checks"])
                           and manifest["isolated"] and (args.profile != "cuda" or not result.skipped))
    content = json.dumps(manifest, indent=2).replace(str(ROOT).replace("\\", "\\\\"), "<repository>")
    content = content.replace(str(Path.home()).replace("\\", "\\\\"), "<home>")
    (args.out / "manifest.json").write_text(content + "\n", encoding="utf-8")
    (args.out / "requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in sorted(packages.items())
        if k != "cortex-context-stamps"), encoding="utf-8")
    print(json.dumps({"profile": args.profile, "tests": manifest["tests"], "passed": manifest["passed"]}, indent=2))
    raise SystemExit(0 if manifest["passed"] else 1)


if __name__ == "__main__":
    main()
