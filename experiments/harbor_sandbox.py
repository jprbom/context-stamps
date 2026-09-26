"""Bounded local Harbor Docker environment for isolated research commands.

No host mounts, model credentials or task graders enter an agent environment.
Harbor remains optional and outside the installable standard-library core.
"""

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

PYTHON_IMAGE = "python@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26"
MAX_OUTPUT = 65536
MAX_COMMAND = 12000  # Includes headroom for encoded argv on Windows.

# Runs inside the Linux container. Data from the model is one argv value, never a
# host-shell fragment. On any boundary failure the owner destroys the container.
EXEC_WRAPPER = r'''
import base64,json,os,selectors,signal,subprocess,sys,time
command=base64.b64decode(sys.argv[1]).decode('utf-8')
limit=float(sys.argv[2]); cap=int(sys.argv[3]); started=time.monotonic()
p=subprocess.Popen(['/bin/bash','-lc',command],cwd='/app',stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,start_new_session=True)
sel=selectors.DefaultSelector()
sel.register(p.stdout,selectors.EVENT_READ,'stdout');sel.register(p.stderr,selectors.EVENT_READ,'stderr')
data={'stdout':bytearray(),'stderr':bytearray()};reason=None;total=0
try:
    while sel.get_map():
        if time.monotonic()-started>limit: reason='timeout';break
        for key,_ in sel.select(.05):
            chunk=os.read(key.fileobj.fileno(),4096)
            if not chunk: sel.unregister(key.fileobj);continue
            remaining=cap-total
            data[key.data].extend(chunk[:remaining]);total+=len(chunk)
            if total>cap: reason='output_limit';break
        if reason: break
    if not reason:
        try: p.wait(timeout=max(.001,limit-(time.monotonic()-started)))
        except subprocess.TimeoutExpired: reason='timeout'
finally:
    try: os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    p.wait();sel.close()
print(json.dumps({'stdout':data['stdout'].decode('utf-8',errors='replace'),
                  'stderr':data['stderr'].decode('utf-8',errors='replace'),
                  'return_code':p.returncode,'boundary_failure':reason,
                  'command_seconds':time.monotonic()-started}))
'''


def docker(args, *, timeout=60):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("Docker operation failed: " + result.stderr[-2000:])
    return result.stdout


class ResearchSandbox:
    def __init__(self, directory, *, image=PYTHON_IMAGE, memory_mb=2048):
        if os.environ.get("HARBOR_TELEMETRY") != "off":
            raise ValueError("Set HARBOR_TELEMETRY=off before importing or starting Harbor")
        from harbor.environments.docker.docker import DockerEnvironment
        from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
        from harbor.models.trial.config import ResourceMode
        from harbor.models.trial.paths import TrialPaths

        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.session = "scqr-eval-" + uuid.uuid4().hex[:16]
        self.closed = False
        self.started = False
        self.failed = False
        self.memory_mb = memory_mb
        self.image = image
        definition = self.directory / "environment"
        definition.mkdir()
        # Explicit network_mode:none also prevents DNS/ICMP; native controlled
        # TCP egress alone is not sufficient for this pilot's offline boundary.
        overlay = {"services": {"main": {"network_mode": "none", "user": "1000:1000",
                    "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
                    "read_only": True, "pids_limit": 64, "working_dir": "/app",
                    "tmpfs": ["/app:rw,nosuid,nodev,size=128m,mode=1777",
                              "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
                              "/logs:rw,nosuid,nodev,size=16m,mode=1777"],
                    "environment": {"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp"}}}}
        (definition / "docker-compose.yaml").write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        paths = TrialPaths(trial_dir=self.directory / "trial")
        paths.mkdir()
        self.env = DockerEnvironment(environment_dir=definition, environment_name="scqr-local-eval",
                                     session_id=self.session, trial_paths=paths,
                                     task_env_config=EnvironmentConfig(docker_image=image, cpus=1,
                                                                       memory_mb=memory_mb, gpus=0),
                                     mounts=[], persistent_env={},
                                     cpu_enforcement_policy=ResourceMode.LIMIT,
                                     memory_enforcement_policy=ResourceMode.LIMIT,
                                     network_policy=NetworkPolicy(network_mode="no-network"))

    def ids(self, service=None):
        args = ["ps", "-aq", "--filter", "label=com.docker.compose.project=" + self.session]
        if service:
            args += ["--filter", "label=com.docker.compose.service=" + service]
        return docker(args).split()

    async def start(self):
        try:
            await asyncio.wait_for(self.env.start(force_build=False), timeout=300)
            self.started = True
            ids = self.ids("main")
            if len(ids) != 1:
                raise ValueError("Expected one isolated main container")
            record = json.loads(docker(["inspect", ids[0]]))[0]
            cfg, host = record["Config"], record["HostConfig"]
            if (host["NetworkMode"] != "none" or host["Privileged"] or host.get("Binds")
                    or record["Mounts"] or cfg["User"] != "1000:1000" or not host["ReadonlyRootfs"]
                    or host["PidsLimit"] != 64 or host["Memory"] != self.memory_mb * 1024**2
                    or host["NanoCpus"] != 1000000000 or host.get("Devices") or host.get("DeviceRequests")
                    or "ALL" not in host["CapDrop"] or host.get("CapAdd")
                    or not any(s in host["SecurityOpt"] for s in ("no-new-privileges", "no-new-privileges:true"))):
                raise ValueError("Docker isolation differs from the registered profile")
            return {"image_id": record["Image"], "user": cfg["User"], "network_mode": host["NetworkMode"],
                    "read_only": host["ReadonlyRootfs"], "pids_limit": host["PidsLimit"],
                    "memory_bytes": host["Memory"], "nano_cpus": host["NanoCpus"],
                    "cap_drop": host["CapDrop"], "security_opt": host["SecurityOpt"], "mounts": record["Mounts"]}
        except BaseException:
            await self.stop()
            raise

    async def execute(self, command, *, timeout=30):
        if self.closed or not self.started or self.failed:
            raise ValueError("Sandbox is not in an executable state")
        if not isinstance(command, str) or not 1 <= len(command.encode()) <= MAX_COMMAND or "\x00" in command:
            raise ValueError("Bounded nonempty command required")
        if type(timeout) not in (int, float) or not 0 < timeout <= 180:
            raise ValueError("Command deadline must be within 180 seconds")
        encoded = base64.b64encode(command.encode()).decode()
        # Model-created modules in /app must not shadow the trusted supervisor's
        # stdlib imports. Isolated mode excludes cwd, user site and PYTHONPATH.
        wrapped = "python -I -c " + shlex.quote(EXEC_WRAPPER) + " " + shlex.quote(encoded) + f" {timeout} {MAX_OUTPUT}"
        try:
            raw = await self.env.exec(wrapped, cwd="/app", timeout_sec=timeout + 10, user="1000:1000")
            if raw.return_code != 0 or raw.stderr or len(raw.stdout or "") > 8 * MAX_OUTPUT:
                raise ValueError("Command supervisor failed")
            result = json.loads(raw.stdout)
            if result["boundary_failure"]:
                self.failed = True
                await self.stop()
            return result
        except BaseException:
            self.failed = True
            await self.stop()
            raise

    async def stop(self):
        if self.closed:
            return
        try:
            await asyncio.wait_for(self.env.stop(delete=True), timeout=45)
        finally:
            remaining = self.ids()
            # Only exact containers belonging to this randomly named task project.
            if remaining:
                docker(["rm", "-f", *remaining])
            self.closed = True
            if self.ids():
                raise RuntimeError("Task-owned containers remain after teardown")


async def qualify(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "image": PYTHON_IMAGE, "checks": {}, "model_calls": 0,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "harbor_version": importlib.metadata.version("harbor"),
              "docker_version": docker(["version", "--format", "{{.Server.Version}}"]).strip()}
    try:
        for label, command, timeout, expected in (
            ("ordinary", "python -c 'print(6*7)'", 10, None),
            ("output_limit", "python -c 'print(\"x\"*100000)'", 10, "output_limit"),
            ("deadline", "python -c 'import subprocess,time;subprocess.Popen([\"sleep\",\"60\"],start_new_session=True);time.sleep(60)'",
             .5, "timeout"),
        ):
            sandbox = ResearchSandbox(directory / label)
            try:
                profile = await sandbox.start()
                result = await sandbox.execute(command, timeout=timeout)
                if result["boundary_failure"] != expected:
                    raise ValueError("Unexpected supervisor result")
                if label == "ordinary":
                    check = """import json,os,socket,pathlib
out={'uid':os.getuid(),'no_grader':not pathlib.Path('/tests').exists(),
     'no_docker_socket':not pathlib.Path('/var/run/docker.sock').exists(),
     'only_loopback':sorted(p.name for p in pathlib.Path('/sys/class/net').iterdir())==['lo']}
for name,addr in [('internet',('1.1.1.1',443)),('host_service',('192.168.65.254',11434))]:
    try:
        with socket.create_connection(addr,timeout=.5): out[name+'_denied']=False
    except OSError: out[name+'_denied']=True
try: pathlib.Path('/etc/scqr-test').write_text('probe');out['root_readonly']=False
except OSError: out['root_readonly']=True
status=pathlib.Path('/proc/self/status').read_text(encoding="utf-8")
out['no_new_privileges']='NoNewPrivs:\\t1' in status
out['zero_capabilities']='CapEff:\\t0000000000000000' in status
print(json.dumps(out))"""
                    probe = await sandbox.execute("python -I -c " + shlex.quote(check))
                    checks = json.loads(probe["stdout"])
                    if checks.pop("uid") != 1000 or not all(checks.values()) or result["stdout"] != "42\n":
                        raise ValueError("Sandbox boundary probe failed")
                    report["boundary_checks"] = checks
                    shadow = "from pathlib import Path;[(Path('/app')/name).write_text('raise RuntimeError(\"shadow module imported\")') for name in ('json.py','pathlib.py','selectors.py','subprocess.py','base64.py')]"
                    await sandbox.execute("python -I -c " + shlex.quote(shadow))
                    isolated = await sandbox.execute("python -I -c 'import json,pathlib;print(json.dumps({\"ok\":True}))'")
                    shadow_ok = isolated["return_code"] == 0 and json.loads(isolated["stdout"]) == {"ok": True}
                    report["boundary_checks"]["supervisor_module_shadowing_denied"] = shadow_ok
                    if not shadow_ok:
                        raise ValueError("Model-created modules affected a trusted helper")
                report["checks"][label] = {"profile": profile, "result": result}
            finally:
                await sandbox.stop()
                report["checks"].setdefault(label, {})["containers_remaining"] = sandbox.ids()
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = type(error).__name__ + ": " + str(error)[-2000:]
        raise
    finally:
        (directory / "qualification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(qualify(args.out))
