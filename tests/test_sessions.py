"""Persistent sandbox sessions (secure_container.sessions) against a fake Docker client.

No real Docker here: containers.run / list / get, exec_run, networks and the browser server are
fakes (the browser server is a tiny unix-socket server thread speaking the JSON-lines protocol),
/proc/meminfo is replaced through ``sessions.read_meminfo`` and iptables through the egress runner.
The exec wrapper script itself runs in the local bash when coreutils are there.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sessions = importlib.import_module("secure_container.sessions")
cm_module = importlib.import_module("secure_container.container_manager")
ti_module = importlib.import_module("secure_container.tool_integrator")

ExecResult = namedtuple("ExecResult", "exit_code output")
GB_KB = 1024 * 1024  # kB in a GiB
APP_DIR = Path(__file__).resolve().parents[1] / "app"


# --------------------------------------------------------------------------- fakes


class FakeAPIError(Exception):
    def __init__(self, status_code, message="docker says no"):
        super().__init__(message)
        self.status_code = status_code


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _bind_unix(sock, path: Path):
    """bind() that also works for paths longer than sockaddr_un allows."""
    fd = None
    try:
        target = str(path)
        if len(os.fsencode(os.path.abspath(target))) >= 100:
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            target = f"/proc/self/fd/{fd}/{path.name}"
        sock.bind(target)
    finally:
        if fd is not None:
            os.close(fd)


class FakeBrowserServer:
    """The browser_server.py protocol in miniature: one JSON line in, one JSON line out."""

    def __init__(self, path: Path, mode="ok"):
        self.path = Path(path)
        self.mode = mode          # ok | die (close without answering) | hang (never answer)
        self.requests: list[dict] = []
        self.stopped = threading.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(self.path):
            os.unlink(self.path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _bind_unix(self.sock, self.path)
        self.sock.listen(8)
        self.sock.settimeout(0.05)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def actions(self):
        return [r.get("action") for r in self.requests]

    def _serve(self):
        while not self.stopped.is_set():
            try:
                conn, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                self._handle(conn)

    def _handle(self, conn):
        conn.settimeout(5)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
        request = json.loads(buf.split(b"\n", 1)[0])
        self.requests.append(request)
        action = request.get("action")
        if self.mode == "die" and action not in ("_ping", "_shutdown"):
            return  # closes the connection without an answer
        if self.mode == "hang" and action not in ("_ping", "_shutdown"):
            self.stopped.wait(3)
            return
        meta = {"tab_id": "tab-1", "title": "Example", "url": "https://example.com/"}
        if action == "_ping" or action == "_touch":
            answer = {"ok": True}
        elif action == "_peek":
            answer = {"ok": True, "image_b64": "/9j/AAAA", "media_type": "image/jpeg", "meta": meta}
        elif action == "_status":
            answer = {"ok": True, "tabs": [dict(meta, active=True)], "started": 1, "last_request": 2}
        elif action == "_shutdown":
            answer = {"ok": True}
        else:
            answer = {"ok": True, "content": [{"type": "text", "text": f"did {action}"}], "meta": meta,
                      "sensitive_done": False}
        conn.sendall((json.dumps(answer) + "\n").encode())
        if action == "_shutdown":
            self.stop()

    def stop(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


class FakeContainer:
    def __init__(self, docker, image, command, kwargs):
        self.docker = docker
        self.image, self.command, self.kwargs = image, command, kwargs
        self.name = kwargs.get("name")
        self.labels = dict(kwargs.get("labels") or {})
        self.status = "running"
        self.removed = False
        self.execs: list[tuple[list, dict]] = []
        self.server: FakeBrowserServer | None = None

    def exec_run(self, cmd, **kwargs):
        if self.removed:
            raise FakeAPIError(404, "No such container")
        if self.status != "running":
            raise FakeAPIError(409, "container is not running")
        self.execs.append((cmd, kwargs))
        return ExecResult(*self.docker.exec_handler(self, cmd, kwargs))

    def commands(self):
        """The user commands of the execs ($2 of the wrapper)."""
        return [cmd[5] for cmd, _ in self.execs]

    def reload(self):
        if self.removed:
            raise FakeAPIError(404, "No such container")

    def wait(self, timeout=None):
        if self.server is not None:
            self.server.stopped.wait(timeout or 0)
        return {"StatusCode": 0}

    def logs(self, tail=None, **kw):
        return b"browser log line"

    def stats(self, stream=True, one_shot=None):
        return {"memory_stats": {"usage": 300 * 1024 * 1024, "stats": {"inactive_file": 100 * 1024 * 1024}}}

    def remove(self, force=False):
        self.removed = True
        self.status = "removed"
        self.docker.removed.append(self.name)
        if self.server is not None:
            self.server.stop()


def default_exec(container, cmd, kwargs):
    return 0, f"ran: {cmd[5]}".encode()


class FakeContainers:
    def __init__(self, docker):
        self.docker = docker

    def run(self, image, command=None, **kwargs):
        if self.docker.run_error is not None:
            raise self.docker.run_error
        container = FakeContainer(self.docker, image, command, kwargs)
        self.docker.created.append(container)
        if container.labels.get(sessions.LABEL_KIND) == sessions.BROWSER and self.docker.browser_mode != "absent":
            mount = next(host for host, spec in kwargs["volumes"].items() if spec["bind"] == sessions.BROWSER_MOUNT)
            container.server = FakeBrowserServer(Path(mount) / "ctl" / "browser.sock", mode=self.docker.browser_mode)
            self.docker.servers.append(container.server)
        return container

    def list(self, all=False, filters=None):
        wanted = (filters or {}).get("label")
        wanted = [wanted] if isinstance(wanted, str) else list(wanted or [])
        found = []
        for container in self.docker.created + self.docker.external:
            if container.removed:
                continue
            ok = True
            for item in wanted:
                key, _, value = item.partition("=")
                if key not in container.labels or (value and container.labels[key] != value):
                    ok = False
            if ok:
                found.append(container)
        return found

    def get(self, name):
        for container in self.docker.created + self.docker.external:
            if container.name == name and not container.removed:
                return container
        raise FakeAPIError(404, "No such container")


class FakeNetworks:
    def __init__(self):
        self.existing: list[SimpleNamespace] = []
        self.created: list[tuple[str, dict]] = []

    def list(self, names=None):
        return [n for n in self.existing if any(name in n.name for name in (names or [n.name]))]

    def create(self, name, **kwargs):
        self.created.append((name, kwargs))
        network = SimpleNamespace(name=name, attrs={})
        self.existing.append(network)
        return network


class FakeDocker:
    def __init__(self):
        self.containers = FakeContainers(self)
        self.networks = FakeNetworks()
        self.images = SimpleNamespace(get=lambda ref: SimpleNamespace(id="sha256:x", tags=[ref]))
        self.created: list[FakeContainer] = []
        self.external: list[FakeContainer] = []
        self.removed: list[str] = []
        self.servers: list[FakeBrowserServer] = []
        self.exec_handler = default_exec
        self.browser_mode = "ok"
        self.run_error = None

    def of_kind(self, kind):
        return [c for c in self.created if c.labels.get(sessions.LABEL_KIND) == kind]

    def running(self, kind):
        return [c for c in self.of_kind(kind) if not c.removed]


class FakeIptables:
    """iptables in miniature: chains, a rule set, -S / -C / -I and their exit codes."""

    def __init__(self, binaries=("iptables",), chains=("DOCKER-USER", "FORWARD", "INPUT"), fail_insert=False):
        self.binaries, self.chains, self.fail_insert = binaries, chains, fail_insert
        self.rules: set[tuple] = set()
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[0] not in self.binaries:
            return -1
        op, chain = argv[3], argv[4]
        if chain not in self.chains:
            return 1
        if op == "-S":
            return 0
        rule = tuple(argv[4:])
        if op == "-C":
            return 0 if rule in self.rules else 1
        if op == "-I":
            if self.fail_insert:
                return 4
            self.rules.add(rule)
            return 0
        return 2

    def inserts(self):
        return [c for c in self.calls if c[3] == "-I"]


def meminfo(total_mb, available_mb):
    return lambda: {"MemTotal": total_mb * 1024, "MemAvailable": available_mb * 1024}


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Plenty of memory, automatic caps, no real iptables, no process-wide manager left behind."""
    monkeypatch.setattr(sessions, "read_meminfo", meminfo(16 * 1024, 12 * 1024))
    monkeypatch.setattr(sessions, "MAX_SESSIONS_SETTING", 0)
    monkeypatch.setattr(sessions, "MAX_BROWSERS_SETTING", 0)
    monkeypatch.setattr(sessions, "EGRESS_FILTER", "best_effort")
    monkeypatch.setattr(sessions, "SESSIONS_ENABLED", True)
    sessions.reset_manager()
    yield
    sessions.reset_manager()


@pytest.fixture
def short_dir():
    """A short base directory: unix socket paths must stay under ~100 bytes for the fake server."""
    path = Path(tempfile.mkdtemp(prefix="gs"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def env(short_dir, monkeypatch):
    base = short_dir / "data"
    base.mkdir()
    script = short_dir / "browser_server.py"
    script.write_text("# the real server is app/secure_container/scripts/browser_server.py\nprint('hi')\n")
    monkeypatch.setattr(sessions, "BROWSER_SERVER_SCRIPT", script)
    docker = FakeDocker()
    clock = Clock()
    iptables = FakeIptables()
    manager = sessions.SessionManager(docker, "secure-container:abc", base, base, clock=clock, egress_runner=iptables)
    ns = SimpleNamespace(m=manager, docker=docker, clock=clock, base=base, script=script, iptables=iptables)
    yield ns
    manager.shutdown_all()
    for server in docker.servers:
        server.stop()


# --------------------------------------------------------------------------- configuration


def _settings_in_fresh_interpreter(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SANDBOX_", "BROWSER_"))}
    env.update(extra_env)
    code = ("import json, secure_container.sessions as s; print(json.dumps([s.SESSION_IDLE_SECONDS, "
            "s.SESSION_WARN_BEFORE_SECONDS, s.SESSION_MEM_MB, s.BROWSER_MEM_MB, s.MEMORY_RESERVE_MB, "
            "s.MAX_SESSIONS_SETTING, s.MAX_BROWSERS_SETTING, s.NETWORK_NAME, s.SUBNET, s.EGRESS_FILTER, "
            "s.SESSIONS_ENABLED, s.DNS_SERVERS]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=APP_DIR, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_and_env_overrides():
    assert _settings_in_fresh_interpreter({}) == [
        1800, 300, 768, 1536, 1536, 0, 0, "generall-sandbox", "172.30.240.0/24", "best_effort", True,
        ["1.1.1.1", "8.8.8.8"],
    ]
    assert _settings_in_fresh_interpreter({
        "SANDBOX_IDLE_SECONDS": "600", "SANDBOX_WARN_BEFORE_SECONDS": "60", "SANDBOX_SESSION_MEM_MB": "1024",
        "BROWSER_MEM_MB": "2048", "SANDBOX_MEMORY_RESERVE_MB": "512", "SANDBOX_MAX_SESSIONS": "3",
        "BROWSER_MAX_SESSIONS": "2", "SANDBOX_NETWORK": "x-net", "SANDBOX_SUBNET": "10.99.0.0/24",
        "SANDBOX_EGRESS_FILTER": "Required", "SANDBOX_SESSIONS": "false", "SANDBOX_DNS": "",
    }) == [600, 60, 1024, 2048, 512, 3, 2, "x-net", "10.99.0.0/24", "required", False, []]
    # nonsense falls back to the defaults
    assert _settings_in_fresh_interpreter({"SANDBOX_IDLE_SECONDS": "soon", "SANDBOX_EGRESS_FILTER": "maybe"})[0::9] == [
        1800, "best_effort"]


def test_meminfo_is_read_from_proc(monkeypatch, tmp_path):
    monkeypatch.undo()  # the real reader
    fake = tmp_path / "meminfo"
    fake.write_text("MemTotal:       16314040 kB\nMemFree:  1 kB\nMemAvailable:    9000000 kB\nHugePages_Total: 0\n")
    monkeypatch.setattr(sessions, "MEMINFO_PATH", str(fake))
    assert sessions.mem_total_mb() == 16314040 // 1024 and sessions.mem_available_mb() == 9000000 // 1024
    monkeypatch.setattr(sessions, "MEMINFO_PATH", str(tmp_path / "missing"))
    assert sessions.mem_total_mb() is None and sessions.mem_available_mb() is None


@pytest.mark.parametrize("total_mb, shells, browsers", [
    (8 * 1024, 8, 4),      # (8192 - 1536) // 768 = 8, // 1536 = 4
    (16 * 1024, 16, 8),    # 19 and 9, clamped to 16 and 8
    (2 * 1024, 1, 1),      # never below one
])
def test_auto_limits_follow_total_memory(env, monkeypatch, total_mb, shells, browsers):
    monkeypatch.setattr(sessions, "read_meminfo", meminfo(total_mb, total_mb // 2))
    assert env.m.max_shells() == shells and env.m.max_browsers() == browsers


def test_limits_from_env_and_unknown_memory(env, monkeypatch):
    monkeypatch.setattr(sessions, "read_meminfo", lambda: {})
    assert (env.m.max_shells(), env.m.max_browsers()) == (4, 2)
    monkeypatch.setattr(sessions, "MAX_SESSIONS_SETTING", 3)
    monkeypatch.setattr(sessions, "MAX_BROWSERS_SETTING", 1)
    assert (env.m.max_shells(), env.m.max_browsers()) == (3, 1)


def test_user_ids_that_could_leave_the_workspace_are_refused(env):
    for bad in ("_sandbox", "../7", "a/b", "", "7 8", "x" * 65):
        with pytest.raises(ValueError):
            env.m.exec(bad, "ls")
    assert sessions.check_user_id(-1001234) == "-1001234"
    assert env.docker.created == []


# --------------------------------------------------------------------------- shell sessions


def test_one_session_serves_every_command(env):
    assert env.m.exec("7", "ls | wc -l", timeout=30) == "ran: ls | wc -l"
    assert env.m.exec("7", "cd sub && pwd") == "ran: cd sub && pwd"
    shells = env.docker.of_kind("shell")
    assert len(shells) == 1
    shell = shells[0]
    assert shell.commands() == ["ls | wc -l", "cd sub && pwd"]
    argv, kwargs = shell.execs[0]
    assert argv[:2] == ["bash", "-c"] and argv[3:5] == ["sandbox-exec", "30"]
    assert 'timeout -k 5 "$1" bash -lc "$2"' in argv[2]
    assert kwargs == {"user": "runner", "workdir": "/home/runner/workspace", "demux": False}
    assert shell.execs[1][0][4] == "60"  # the default timeout
    run = shell.kwargs
    assert shell.image == "secure-container:abc" and shell.command == ["sleep", "infinity"]
    assert run["name"].startswith("sandbox-7-") and run["detach"] is True and run["init"] is True
    assert run["user"] == "runner" and run["network"] == "generall-sandbox"
    assert run["mem_limit"] == run["memswap_limit"] == "768m"
    assert run["oom_score_adj"] == 800 and run["pids_limit"] == 512
    assert run["labels"] == {"generall.sandbox": "shell", "generall.user": "7"}
    assert run["dns"] == ["1.1.1.1", "8.8.8.8"]
    assert "security_opt" not in run  # no-new-privileges would break sudo apt-get
    assert run["volumes"] == {str(env.base / "7"): {"bind": "/home/runner/workspace", "mode": "rw", "propagation": "shared"}}
    assert oct((env.base / "7").stat().st_mode & 0o777) == "0o777"


def test_users_get_separate_sessions_and_commands_run_concurrently(env):
    results = []
    threads = [threading.Thread(target=lambda u=u: results.append(env.m.exec(u, "echo hi"))) for u in ("1", "2", "1", "2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4
    assert sorted(c.labels["generall.user"] for c in env.docker.of_kind("shell")) == ["1", "2"]


def test_exec_result_formats():
    fmt = sessions.format_exec_result
    assert fmt(0, b"hello\n", 30) == "hello\n"
    assert fmt(0, b"", 30) == "(no output)"
    assert fmt(2, b"bad\n", 30) == "Error (exit code 2):\nbad\n"
    timed_out = fmt(124, b"partial", 30)
    assert timed_out.startswith("Error: the command timed out after 30 s (the sandbox session is still running; "
                                "long jobs: start them in the background with nohup")
    assert timed_out.endswith("Output before the timeout:\npartial")
    assert fmt(124, b"", 5) == sessions.timeout_message(5)
    assert fmt(137, b"", 30, elapsed=35.2).startswith("Error: the command timed out after 30 s")  # killed by -k
    killed = fmt(137, b"x", 30, elapsed=2.0, mem_mb=768)
    assert killed.startswith("Error (exit code 137: the process was killed") and "768 MB" in killed
    big = fmt(0, b"a" * (sessions.OUTPUT_CAP + 1), 30)
    assert big == "a" * sessions.OUTPUT_CAP + "\n[output truncated at 1 MB]"
    assert fmt(0, "é".encode() + b"\xff", 30) == "é�"


def test_exec_maps_exit_codes_through_the_session(env):
    outcomes = {"ok": (0, b"fine"), "fail": (3, b"nope"), "slow": (124, b"")}
    env.docker.exec_handler = lambda c, cmd, kw: outcomes[cmd[5]]
    assert env.m.exec("7", "ok") == "fine"
    assert env.m.exec("7", "fail") == "Error (exit code 3):\nnope"
    assert env.m.exec("7", "slow", timeout=10).startswith("Error: the command timed out after 10 s")
    assert env.m.exec("7", "ok", timeout=99999) == "fine"
    assert env.docker.of_kind("shell")[0].execs[-1][0][4] == str(sessions.MAX_EXEC_TIMEOUT)


@pytest.mark.skipif(not all(shutil.which(b) for b in ("bash", "timeout", "head", "mktemp")), reason="needs bash + coreutils")
def test_exec_wrapper_script_in_a_real_bash(tmp_path):
    def run(command, timeout=10):
        started = time.monotonic()
        done = subprocess.run(sessions.exec_argv(command, timeout), cwd=tmp_path, capture_output=True, timeout=60)
        return done.returncode, done.stdout, time.monotonic() - started

    code, out, _ = run("echo one; echo two >&2; ls /nonexistent-x 2>/dev/null | wc -l; exit 3")
    assert code == 3 and out.split() == [b"one", b"two", b"0"]  # stderr is part of the output
    code, out, _ = run("cd / && pwd && echo *bin* | head -c 3")
    assert code == 0 and out.startswith(b"/\n")
    code, _, took = run("sleep 30", timeout=1)
    assert code == 124 and took < 10
    # a background job does not hold the exec open: its output goes to the temp file
    code, out, took = run("sleep 20 & echo started")
    assert code == 0 and out == b"started\n" and took < 10
    code, out, _ = run("head -c 3000000 /dev/zero | tr '\\0' a")
    assert code == 0 and len(out) == sessions.OUTPUT_CAP + 1  # capped inside the container


def test_dead_container_is_replaced_and_the_command_runs_once(env):
    env.m.exec("7", "first")
    first = env.docker.of_kind("shell")[0]
    first.status = "exited"  # OOM-killed or removed behind our back
    assert env.m.exec("7", "second") == "ran: second"
    shells = env.docker.of_kind("shell")
    assert len(shells) == 2 and first.removed and shells[1].commands() == ["second"]


def test_docker_errors_that_are_not_about_the_container_are_reported(env):
    def boom(c, cmd, kw):
        raise FakeAPIError(500, "daemon hiccup")

    env.docker.exec_handler = boom
    assert env.m.exec("7", "ls") == "Error executing command: daemon hiccup"
    assert len(env.docker.of_kind("shell")) == 1  # the session stays


# --------------------------------------------------------------------------- packages


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_saved_packages_are_replayed_once_per_session_start(env):
    _write(env.base / "7" / "installed_packages.txt", "ImageMagick\npython3-numpy\n--bad-option\n")
    _write(env.base / "7" / "installed_python_packages.txt", "# keep\nrequests==2.31.0 rich\n")
    env.m.exec("7", "a")
    env.m.exec("7", "b")
    shell = env.docker.of_kind("shell")[0]
    replay, *rest = shell.commands()
    assert rest == ["a", "b"]  # replayed once, before the first command, not per command
    assert "sudo apt-get update -qq" in replay
    assert "sudo apt-get install -y -qq -o DPkg::Lock::Timeout=120 imagemagick python3-numpy" in replay
    assert "pip install --user --quiet --no-input numpy 2>&1 || true" in replay  # python3-* fallback
    assert "pip install --user --quiet --no-input requests==2.31.0 rich" in replay
    assert "--bad-option" not in replay and "keep" not in replay
    assert shell.execs[0][0][4] == str(sessions.REPLAY_TIMEOUT)
    env.m.close("7")
    env.m.exec("7", "c")
    assert env.docker.of_kind("shell")[1].commands()[0] == replay  # a new session replays again


def test_no_saved_packages_no_replay(env):
    env.m.exec("7", "a")
    assert env.docker.of_kind("shell")[0].commands() == ["a"]


def test_failed_replay_is_told_once(env):
    _write(env.base / "7" / "installed_python_packages.txt", "nosuchpkg\n")
    env.docker.exec_handler = lambda c, cmd, kw: (1, b"ERROR: No matching distribution") if "Reinstalling" in cmd[5] else (0, b"ok")
    first = env.m.exec("7", "a")
    assert first.startswith("ok\n[Note: a new sandbox session started") and "No matching distribution" in first
    assert env.m.exec("7", "b") == "ok"


def test_install_package_apt_and_pip_are_saved(env):
    out = env.m.install_package("7", "ImageMagick jq", "apt")
    shell = env.docker.of_kind("shell")[0]
    assert shell.commands()[-1] == ("sudo apt-get update -qq && "
                                    "sudo apt-get install -y -q -o DPkg::Lock::Timeout=120 imagemagick jq")
    assert shell.execs[-1][0][4] == "300"
    assert "Saved imagemagick, jq to installed_packages.txt" in out
    assert (env.base / "7" / "installed_packages.txt").read_text() == "imagemagick\njq\n"

    out = env.m.install_package("7", "pandas==2.2.2 openpyxl", manager="pip", timeout=120)
    assert shell.commands()[-1] == "pip install --user --no-input pandas==2.2.2 openpyxl"
    assert "installed_python_packages.txt" in out
    pip_file = env.base / "7" / "installed_python_packages.txt"
    assert pip_file.read_text() == "pandas==2.2.2\nopenpyxl\n"
    # a new version replaces the old line instead of asking pip for both
    env.m.install_package("7", "pandas>=2.3,<3", "pip")
    assert pip_file.read_text() == "openpyxl\npandas>=2.3,<3\n"
    out = env.m.install_package("7", "jq", "apt")
    assert "already listed in installed_packages.txt" in out
    assert (env.base / "7" / "installed_packages.txt").read_text() == "imagemagick\njq\n"
    assert sessions.persisted_packages(env.base / "7") == {"apt": ["imagemagick", "jq"], "pip": ["openpyxl", "pandas>=2.3,<3"]}
    # the next session replays exactly these
    env.m.close("7")
    env.m.exec("7", "x")
    replay = env.docker.of_kind("shell")[1].commands()[0]
    assert "imagemagick jq" in replay and "openpyxl 'pandas>=2.3,<3'" in replay


def test_failed_or_invalid_installs_save_nothing(env):
    env.docker.exec_handler = lambda c, cmd, kw: (100, b"E: Unable to locate package nope")
    assert env.m.install_package("7", "nope", "apt") == "Error (exit code 100):\nE: Unable to locate package nope"
    assert not (env.base / "7" / "installed_packages.txt").exists()
    runs = len(env.docker.created)
    for name, manager in (("--allow-unauthenticated", "apt"), ("x; rm -rf ~", "apt"), ("git+https://x/y", "pip"),
                          ("--index-url=http://evil", "pip"), ("", "apt"), ("jq", "brew")):
        assert env.m.install_package("7", name, manager).startswith("Error:"), name
    assert len(env.docker.created) == runs


def test_package_names():
    assert sessions.parse_packages("apt", " ImageMagick  python3-pil=9.4.0-1 libc6:amd64 ") == [
        "imagemagick", "python3-pil=9.4.0-1", "libc6:amd64"]
    assert sessions.parse_packages("pip", "Pillow requests[socks]==2.31.0 numpy>=1.26,<2 Pillow") == [
        "Pillow", "requests[socks]==2.31.0", "numpy>=1.26,<2"]
    with pytest.raises(ValueError):
        sessions.parse_packages("pip", "a " * 51)
    assert sessions.package_key("pip", "Foo_Bar[x]>=1") == sessions.package_key("pip", "foo-bar") == "foo-bar"
    assert sessions.package_key("apt", "libc6:amd64=2.36") == "libc6"


# --------------------------------------------------------------------------- admission


def test_memory_refusal_is_readable(env, monkeypatch):
    monkeypatch.setattr(sessions, "read_meminfo", meminfo(8 * 1024, 2100))  # 2100 - 1536 < 768
    with pytest.raises(sessions.CapacityError) as shell_err:
        env.m.exec("7", "ls")
    assert str(shell_err.value) == ("The server is short on memory (2100 MB free); no new sandbox session can start "
                                    "right now. Try again in a few minutes.")
    with pytest.raises(sessions.CapacityError) as browser_err:
        env.m.browser_call("7", {"action": "navigate", "input": {"url": "example.com"}})
    assert "no new browser can start" in str(browser_err.value) and "(2100 MB free)" in str(browser_err.value)
    assert env.docker.created == []


def test_count_cap_and_lru_eviction(env, monkeypatch):
    monkeypatch.setattr(sessions, "MAX_SESSIONS_SETTING", 2)
    env.m.exec("1", "a")
    env.clock.advance(50)
    env.m.exec("2", "b")
    env.clock.advance(100)  # user 1 idle 150 s, user 2 idle 100 s
    env.m.exec("3", "c")    # evicts user 1 (least recently used, idle >= 120 s)
    running = {c.labels["generall.user"] for c in env.docker.running("shell")}
    assert running == {"2", "3"}
    with pytest.raises(sessions.CapacityError) as err:  # nobody idle long enough now
        env.m.exec("4", "d")
    assert str(err.value).startswith("All 2 sandbox sessions this server can run are in use")
    assert (env.base / "1").is_dir()  # an evicted user's files stay
    env.clock.advance(130)
    env.m.exec("4", "d")  # user 2 (idle 230 s) goes now, not user 3 (idle 130 s, used later)
    assert {c.labels["generall.user"] for c in env.docker.running("shell")} == {"3", "4"}


def test_sessions_with_a_call_in_flight_are_never_evicted(env, monkeypatch):
    monkeypatch.setattr(sessions, "MAX_SESSIONS_SETTING", 1)
    entered, release = threading.Event(), threading.Event()

    def slow(container, cmd, kw):
        if cmd[5] == "slow":
            entered.set()
            release.wait(5)
        return 0, b"done"

    env.docker.exec_handler = slow
    worker = threading.Thread(target=lambda: env.m.exec("1", "slow"))
    worker.start()
    assert entered.wait(5)
    env.clock.advance(10_000)
    try:
        with pytest.raises(sessions.CapacityError):
            env.m.exec("2", "x")
        assert env.m.reap_idle() == []  # busy sessions are not reaped either
    finally:
        release.set()
        worker.join()


def test_memory_eviction_frees_room_for_a_browser(env, monkeypatch):
    env.m.browser_call("1", {"action": "screenshot"})
    env.clock.advance(300)
    state = {"free": 2000}  # 2000 - 1536 < 1536: no room until the idle browser goes
    monkeypatch.setattr(sessions, "read_meminfo", lambda: {"MemTotal": 16 * GB_KB, "MemAvailable": state["free"] * 1024})
    first = env.docker.of_kind("browser")[0]
    original_remove = first.remove

    def remove(force=False):
        state["free"] = 4000
        original_remove(force)

    first.remove = remove
    assert env.m.browser_call("2", {"action": "screenshot"})["ok"] is True
    assert first.removed and "_shutdown" in first.server.actions()  # asked to close cleanly first
    assert [c.labels["generall.user"] for c in env.docker.running("browser")] == ["2"]


def test_required_egress_filter_refuses_sessions(env, monkeypatch):
    monkeypatch.setattr(sessions, "EGRESS_FILTER", "required")
    env.m.egress_state = "failed"
    with pytest.raises(sessions.CapacityError) as err:
        env.m.exec("7", "ls")
    assert "SANDBOX_EGRESS_FILTER=required" in str(err.value)
    env.m.egress_state = "on"
    assert env.m.exec("7", "ls") == "ran: ls"


def test_docker_run_failure_is_a_capacity_error(env):
    env.docker.run_error = FakeAPIError(500, "no space left on device")
    with pytest.raises(sessions.SessionStartError) as err:
        env.m.exec("7", "ls")
    assert isinstance(err.value, sessions.CapacityError)
    assert str(err.value) == "The sandbox could not start: no space left on device"
    env.docker.run_error = None
    assert env.m.exec("7", "ls") == "ran: ls"  # the reservation was given back


# --------------------------------------------------------------------------- idle, warnings, extend


def test_idle_sessions_are_reaped(env, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 1800)
    env.m.exec("1", "a")
    env.m.browser_call("1", {"action": "screenshot"})
    env.clock.advance(1000)
    env.m.exec("2", "b")
    env.clock.advance(799)
    assert env.m.reap_idle() == []
    env.clock.advance(1)
    assert sorted(env.m.reap_idle()) == [("1", "browser"), ("1", "shell")]
    assert env.m.status("1")["shell"]["running"] is False and env.m.status("2")["shell"]["running"] is True
    assert "_shutdown" in env.docker.of_kind("browser")[0].server.actions()
    env.clock.advance(1000)
    assert env.m.reap_idle() == [("2", "shell")]
    assert env.docker.running("shell") == [] and env.docker.running("browser") == []


def test_containers_that_died_by_themselves_are_dropped(env):
    env.m.exec("7", "a")
    env.docker.of_kind("shell")[0].status = "exited"
    assert env.m.reap_idle() == [("7", "shell")]
    assert env.m.status("7")["shell"]["running"] is False


def test_due_warnings_once_per_idle_period_per_user(env, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 1800)
    monkeypatch.setattr(sessions, "SESSION_WARN_BEFORE_SECONDS", 300)
    env.m.exec("1", "a")
    env.clock.advance(600)
    env.m.browser_call("1", {"action": "screenshot"})  # the user's last activity
    last = env.clock.now
    env.m.exec("2", "b")
    env.clock.advance(1499)
    assert env.m.due_warnings() == []
    env.clock.advance(1)
    due = env.m.due_warnings()
    assert len(due) == 2
    one = next(d for d in due if d["user_id"] == "1")
    assert one["kinds"] == ["shell", "browser"]  # one warning covers both kinds
    assert one["closes_at"] == datetime.fromtimestamp(last + 1800, timezone.utc)
    assert one["closes_at"].tzinfo is not None
    assert env.m.status("1")["warned"] is True
    assert env.m.due_warnings() == []  # once per idle period
    env.clock.advance(60)
    env.m.exec("1", "again")  # any activity clears the mark
    assert env.m.status("1")["warned"] is False
    env.clock.advance(1499)
    assert env.m.due_warnings() == []
    env.clock.advance(1)
    assert [d["user_id"] for d in env.m.due_warnings()] == ["1"]


def test_warnings_off_and_nothing_running(env, monkeypatch):
    env.m.exec("1", "a")
    env.clock.advance(5000)
    monkeypatch.setattr(sessions, "SESSION_WARN_BEFORE_SECONDS", 0)
    assert env.m.due_warnings() == []
    monkeypatch.setattr(sessions, "SESSION_WARN_BEFORE_SECONDS", 300)
    env.m.close("1")
    assert env.m.due_warnings() == []


def test_extend_touches_both_kinds_and_the_browser_server(env, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 1800)
    assert env.m.extend("1") is None  # nothing running
    env.m.exec("1", "a")
    env.m.browser_call("1", {"action": "screenshot"})
    env.clock.advance(1600)
    assert [d["user_id"] for d in env.m.due_warnings()] == ["1"]
    closes = env.m.extend("1")
    assert closes == datetime.fromtimestamp(env.clock.now + 1800, timezone.utc)
    server = env.docker.of_kind("browser")[0].server
    assert server.actions()[-1] == "_touch"
    status = env.m.status("1")
    assert status["warned"] is False
    assert status["shell"]["closes_at"] == status["browser"]["closes_at"] == closes
    env.clock.advance(1799)
    assert env.m.reap_idle() == []
    env.clock.advance(1)
    assert sorted(env.m.reap_idle()) == [("1", "browser"), ("1", "shell")]


def test_status_shapes(env, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 1800)
    _write(env.base / "7" / "installed_packages.txt", "jq\n")
    idle = env.m.status("7")
    assert idle == {
        "shell": {"running": False, "started": None, "last_used": None, "idle_seconds_left": None, "closes_at": None},
        "browser": {"running": False, "started": None, "last_used": None, "idle_seconds_left": None,
                    "closes_at": None, "tabs": []},
        "profile_bytes": 0, "packages": {"apt": ["jq"], "pip": []}, "idle_timeout_seconds": 1800,
        "warn_before_seconds": 300, "warned": False,
    }
    assert env.docker.created == []  # status never starts anything
    started = env.clock.now
    env.m.exec("7", "a")
    env.m.browser_call("7", {"action": "screenshot"})
    _write(env.base / "_sandbox" / "7" / "browser" / "profile" / "Default" / "Cookies", "x" * 1000)
    env.clock.advance(100)
    status = env.m.status("7")
    shell = status["shell"]
    assert shell["running"] is True and shell["busy"] is False
    assert shell["started"] == datetime.fromtimestamp(started, timezone.utc)
    assert shell["idle_seconds_left"] == 1700
    assert shell["closes_at"] == datetime.fromtimestamp(started + 1800, timezone.utc)
    assert shell["mem_limit_mb"] == 768 and shell["mem_mb"] == 200
    browser = status["browser"]
    assert browser["running"] is True and browser["remember"] is True and browser["mem_limit_mb"] == 1536
    assert browser["tabs"] == [{"tab_id": "tab-1", "title": "Example", "url": "https://example.com/", "active": True}]
    assert status["profile_bytes"] == 1000
    server = env.m.server_status()
    assert server == {"shells": 1, "max_shells": 16, "browsers": 1, "max_browsers": 8, "mem_available_mb": 12 * 1024,
                      "mem_total_mb": 16 * 1024, "reserve_mb": 1536, "egress_filter": "off"}
    # _status is not activity
    assert env.m.status("7")["browser"]["last_used"] == browser["last_used"]


# --------------------------------------------------------------------------- the browser


def test_browser_start_command_mounts_and_files(env, monkeypatch):
    monkeypatch.setattr(sessions, "SESSION_IDLE_SECONDS", 1800)
    browser_dir = env.base / "_sandbox" / "7" / "browser"
    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        _write(browser_dir / "profile" / name, "stale")
    (browser_dir / "profile" / "Default").mkdir()
    answer = env.m.browser_call("7", {"action": "navigate", "input": {"url": "example.com"}})
    assert answer["ok"] is True and answer["content"] == [{"type": "text", "text": "did navigate"}]
    browser = env.docker.of_kind("browser")[0]
    assert browser.command == [
        "python", "/run/browser/browser_server.py", "--socket", "/run/browser/ctl/browser.sock",
        "--profile", "/run/browser/profile", "--downloads", "/home/runner/workspace/downloads",
        "--idle-seconds", "1860", "--viewport", "1280x800",
    ]
    run = browser.kwargs
    assert run["name"].startswith("browser-7-") and run["init"] is True and run["user"] == "runner"
    assert run["mem_limit"] == run["memswap_limit"] == "1536m"
    assert run["security_opt"] == ["no-new-privileges"] and run["oom_score_adj"] == 800 and run["pids_limit"] == 512
    assert run["labels"] == {"generall.sandbox": "browser", "generall.user": "7"}
    assert run["network"] == "generall-sandbox"
    assert run["volumes"] == {
        str(browser_dir): {"bind": "/run/browser", "mode": "rw"},
        str(env.base / "7" / "downloads"): {"bind": "/home/runner/workspace/downloads", "mode": "rw"},
    }
    # the shell's workspace is not mounted: commands never see the cookies and vice versa
    assert str(env.base / "7") not in run["volumes"]
    assert (browser_dir / "browser_server.py").read_text() == env.script.read_text()
    for sub in ("", "ctl", "profile"):
        assert oct((browser_dir / sub).stat().st_mode & 0o777) == "0o777", sub
    assert oct((env.base / "7" / "downloads").stat().st_mode & 0o777) == "0o777"
    assert sorted(p.name for p in (browser_dir / "profile").iterdir()) == ["Default"]  # stale locks gone
    server = browser.server
    assert server.actions() == ["_ping", "navigate"]
    sent = server.requests[1]
    assert sent["input"] == {"url": "example.com"} and isinstance(sent["id"], str) and sent["id"]


def test_browser_is_reused_and_restarted_when_remember_changes(env):
    env.m.browser_call("7", {"action": "screenshot", "id": "a"})
    env.m.browser_call("7", {"action": "screenshot", "id": "b"})
    assert len(env.docker.of_kind("browser")) == 1
    env.m.browser_call("7", {"action": "screenshot"}, remember=False)
    first, second = env.docker.of_kind("browser")
    assert first.removed and "_shutdown" in first.server.actions()
    assert second.command[second.command.index("--profile") + 1] == ""  # ephemeral profile
    assert env.m.status("7")["browser"]["remember"] is False


def test_peek_and_status_are_not_activity(env):
    assert env.m.browser_peek("7") is None  # no browser running: nothing starts
    assert env.docker.created == []
    env.m.browser_call("7", {"action": "screenshot"})
    used = env.m.status("7")["browser"]["last_used"]
    env.clock.advance(500)
    peek = env.m.browser_peek("7")
    assert peek == {"ok": True, "image_b64": "/9j/AAAA", "media_type": "image/jpeg",
                    "meta": {"tab_id": "tab-1", "title": "Example", "url": "https://example.com/"}}
    env.m.browser_call("7", {"action": "_status"})
    assert env.m.status("7")["browser"]["last_used"] == used
    env.m.browser_call("7", {"action": "left_click", "input": {"coordinate": [1, 2]}})
    assert env.m.status("7")["browser"]["last_used"] > used


def test_dead_browser_server_gives_the_error_and_restarts_next_time(env):
    env.m.browser_call("7", {"action": "screenshot"})
    first = env.docker.of_kind("browser")[0]
    first.server.mode = "die"
    answer = env.m.browser_call("7", {"action": "left_click"})
    assert answer == {"ok": False, "error": "Error: the browser stopped unexpectedly; it will restart on the next action."}
    assert first.removed and env.m.status("7")["browser"]["running"] is False
    assert env.m.browser_call("7", {"action": "screenshot"})["ok"] is True
    assert len(env.docker.of_kind("browser")) == 2


def test_unreachable_browser_is_restarted_transparently(env):
    env.m.browser_call("7", {"action": "screenshot"})
    first = env.docker.of_kind("browser")[0]
    first.server.stop()  # exited by itself: socket gone, nothing was sent
    answer = env.m.browser_call("7", {"action": "navigate", "input": {"url": "https://example.com"}})
    assert answer["ok"] is True
    second = env.docker.of_kind("browser")[1]
    assert first.removed and second.server.actions() == ["_ping", "navigate"]


def test_restart_that_cannot_come_up_is_a_start_error(env, monkeypatch):
    env.m.browser_call("7", {"action": "screenshot"})
    env.docker.browser_mode = "absent"
    env.docker.of_kind("browser")[0].server.stop()
    monkeypatch.setattr(sessions, "BROWSER_START_TIMEOUT", 0.5)
    with pytest.raises(sessions.SessionStartError):  # the restart cannot come up either
        env.m.browser_call("7", {"action": "screenshot"})
    assert all(c.removed for c in env.docker.of_kind("browser"))


def test_unreachable_after_a_restart_is_the_died_error(env, monkeypatch):
    env.m.browser_call("7", {"action": "screenshot"})
    env.docker.of_kind("browser")[0].server.stop()
    calls = []
    real = sessions.socket_request

    def flaky(path, request, timeout):  # the new browser answers its _ping, then is gone again
        calls.append(request["action"])
        if request["action"] == "screenshot":
            raise sessions._NotDelivered("gone")
        return real(path, request, timeout)

    monkeypatch.setattr(sessions, "socket_request", flaky)
    assert env.m.browser_call("7", {"action": "screenshot"}) == {"ok": False, "error": sessions.BROWSER_DIED}
    assert calls.count("screenshot") == 2 and len(env.docker.of_kind("browser")) == 2


def test_browser_timeout_stops_the_browser(env):
    env.m.browser_call("7", {"action": "screenshot"})
    browser = env.docker.of_kind("browser")[0]
    browser.server.mode = "hang"
    answer = env.m.browser_call("7", {"action": "navigate"}, timeout=0.3)
    assert answer["ok"] is False and "did not answer within 0.3 seconds" in answer["error"]
    assert browser.removed


def test_browser_that_never_answers_does_not_start(env, monkeypatch):
    monkeypatch.setattr(sessions, "BROWSER_START_TIMEOUT", 0.5)
    env.docker.browser_mode = "absent"
    with pytest.raises(sessions.SessionStartError) as err:
        env.m.browser_call("7", {"action": "screenshot"})
    assert str(err.value).startswith("The browser could not start: the browser server did not answer within 0.5 s")
    assert env.docker.of_kind("browser")[0].removed
    assert env.m.server_status()["browsers"] == 0


def test_browser_that_exits_at_start_reports_its_logs(env, monkeypatch):
    env.docker.browser_mode = "absent"
    original_run = env.docker.containers.run

    def run(image, command=None, **kwargs):
        container = original_run(image, command, **kwargs)
        container.status = "exited"
        return container

    monkeypatch.setattr(env.docker.containers, "run", run)
    with pytest.raises(sessions.SessionStartError) as err:
        env.m.browser_call("7", {"action": "screenshot"})
    assert "the browser server exited: browser log line" in str(err.value)


def test_filesystem_trouble_at_start_is_a_start_error(env, monkeypatch):
    def refuse(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(sessions.SessionManager, "_writable", staticmethod(refuse))
    with pytest.raises(sessions.SessionStartError) as err:
        env.m.exec("7", "ls")
    assert str(err.value).startswith("The sandbox could not start: [Errno 13] Permission denied")
    with pytest.raises(sessions.SessionStartError):
        env.m.browser_call("7", {"action": "screenshot"})
    assert env.docker.created == [] and env.m.server_status()["shells"] == 0


def test_missing_browser_server_script(env, monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "BROWSER_SERVER_SCRIPT", tmp_path / "nope.py")
    with pytest.raises(sessions.SessionStartError) as err:
        env.m.browser_call("7", {"action": "screenshot"})
    assert "browser_server.py is missing" in str(err.value) and env.docker.created == []


def test_shutdown_request_ends_the_session(env):
    env.m.browser_call("7", {"action": "screenshot"})
    assert env.m.browser_call("7", {"action": "_shutdown"}) == {"ok": True}
    assert env.m.status("7")["browser"]["running"] is False and env.docker.of_kind("browser")[0].removed


def test_forget_browser_profile(env):
    env.m.browser_call("7", {"action": "screenshot"})
    profile = env.base / "_sandbox" / "7" / "browser" / "profile"
    _write(profile / "Default" / "Cookies", "secret")
    assert env.m.forget_browser_profile("7") is True
    assert not profile.exists()
    browser = env.docker.of_kind("browser")[0]
    assert browser.removed and "_shutdown" in browser.server.actions()
    assert env.m.forget_browser_profile("7") is False
    # the next browser gets a fresh, empty profile
    env.m.browser_call("7", {"action": "screenshot"})
    assert profile.is_dir() and list(profile.iterdir()) == []


def test_socket_client_handles_long_paths(tmp_path):
    deep = tmp_path.joinpath(*(["d" * 30] * 4), "ctl")
    path = deep / "browser.sock"
    assert len(str(path)) > 108
    server = FakeBrowserServer(path)
    try:
        assert sessions.socket_request(path, {"id": "1", "action": "_ping"}, 2) == {"ok": True}
    finally:
        server.stop()
    with pytest.raises(sessions._NotDelivered):
        sessions.socket_request(path, {"id": "2", "action": "_ping"}, 1)


# --------------------------------------------------------------------------- close, cleanup, shutdown


def test_close_one_kind_or_both(env):
    env.m.exec("7", "a")
    env.m.browser_call("7", {"action": "screenshot"})
    assert env.m.close("7", "shell") is True
    status = env.m.status("7")
    assert status["shell"]["running"] is False and status["browser"]["running"] is True
    assert env.m.close("7") is True
    assert env.m.close("7") is False
    with pytest.raises(ValueError):
        env.m.close("7", "gpu")


def test_cleanup_stale_removes_only_unknown_labelled_containers(env):
    env.m.exec("7", "a")
    stale_shell = FakeContainer(env.docker, "img", None, {"name": "sandbox-9-old", "labels": {"generall.sandbox": "shell", "generall.user": "9"}})
    stale_browser = FakeContainer(env.docker, "img", None, {"name": "browser-9-old", "labels": {"generall.sandbox": "browser", "generall.user": "9"}})
    other = FakeContainer(env.docker, "img", None, {"name": "unrelated", "labels": {}})
    env.docker.external += [stale_shell, stale_browser, other]
    assert env.m.cleanup_stale() == 2
    assert stale_shell.removed and stale_browser.removed and not other.removed
    assert not env.docker.of_kind("shell")[0].removed  # the live session stays
    assert env.m.cleanup_stale() == 0


def test_leftover_container_of_the_same_user_is_removed_before_a_start(env):
    leftover = FakeContainer(env.docker, "img", None, {"name": "browser-7-old", "labels": {"generall.sandbox": "browser", "generall.user": "7"}})
    env.docker.external.append(leftover)
    env.m.browser_call("7", {"action": "screenshot"})
    assert leftover.removed


def test_shutdown_all_stops_everything_and_refuses_new_sessions(env):
    env.m.exec("1", "a")
    env.m.browser_call("2", {"action": "screenshot"})
    env.m.shutdown_all()
    assert env.docker.running("shell") == [] and env.docker.running("browser") == []
    assert "_shutdown" in env.docker.of_kind("browser")[0].server.actions()
    with pytest.raises(sessions.CapacityError):
        env.m.exec("1", "b")


# --------------------------------------------------------------------------- network and egress filter


def test_egress_rules_are_plain_command_lines():
    rules = sessions.egress_rules("172.30.240.0/24")
    assert rules[0] == ["DOCKER-USER", "-s", "172.30.240.0/24", "-d", "10.0.0.0/8", "-j", "DROP"]
    assert [r[4] for r in rules[:-1]] == list(sessions.BLOCKED_RANGES)
    assert "169.254.0.0/16" in sessions.BLOCKED_RANGES and "127.0.0.0/8" in sessions.BLOCKED_RANGES
    assert rules[-1] == ["INPUT", "-s", "172.30.240.0/24", "-j", "DROP"]
    check, insert = sessions.egress_commands("172.30.240.0/24", "iptables-legacy")[0]
    assert check == ["iptables-legacy", "-w", "5", "-C", "DOCKER-USER", "-s", "172.30.240.0/24", "-d", "10.0.0.0/8", "-j", "DROP"]
    assert insert == ["iptables-legacy", "-w", "5", "-I", "DOCKER-USER", "-s", "172.30.240.0/24", "-d", "10.0.0.0/8", "-j", "DROP"]
    assert sessions.egress_rules("10.9.0.0/24", "FORWARD")[0][0] == "FORWARD"


def test_egress_filter_is_idempotent():
    iptables = FakeIptables()
    assert sessions.apply_egress_filter("172.30.240.0/24", "best_effort", iptables) == "on"
    assert len(iptables.inserts()) == len(sessions.BLOCKED_RANGES) + 1
    assert sessions.apply_egress_filter("172.30.240.0/24", "required", iptables) == "on"
    assert len(iptables.inserts()) == len(sessions.BLOCKED_RANGES) + 1  # nothing inserted twice
    assert len(iptables.rules) == len(sessions.BLOCKED_RANGES) + 1


def test_egress_filter_backends_and_failures():
    legacy = FakeIptables(binaries=("iptables", "iptables-legacy"))
    legacy.chains = ("FORWARD", "INPUT")  # nft has no DOCKER-USER...
    def run(argv):  # ...but the legacy tables (what Docker uses) do
        if argv[0] == "iptables-legacy" and argv[4] == "DOCKER-USER":
            legacy.calls.append(list(argv))
            if argv[3] == "-I":
                legacy.rules.add(tuple(argv[4:]))
            return 0 if argv[3] != "-C" or tuple(argv[4:]) in legacy.rules else 1
        return legacy(argv)
    assert sessions.apply_egress_filter("172.30.240.0/24", "best_effort", run) == "on"
    assert {c[0] for c in legacy.inserts()} == {"iptables-legacy"}

    no_docker_chain = FakeIptables(chains=("FORWARD", "INPUT"))
    assert sessions.apply_egress_filter("172.30.240.0/24", "best_effort", no_docker_chain) == "on"
    assert {c[4] for c in no_docker_chain.inserts()} == {"FORWARD", "INPUT"}

    missing = FakeIptables(binaries=())
    assert sessions.apply_egress_filter("172.30.240.0/24", "best_effort", missing) == "failed"
    refused = FakeIptables(fail_insert=True)
    assert sessions.apply_egress_filter("172.30.240.0/24", "required", refused) == "failed"
    off = FakeIptables()
    assert sessions.apply_egress_filter("172.30.240.0/24", "off", off) == "off" and off.calls == []


def test_setup_creates_the_network_without_icc(env):
    env.m.setup()
    name, kwargs = env.docker.networks.created[0]
    assert name == "generall-sandbox" and kwargs["driver"] == "bridge"
    assert kwargs["options"] == {"com.docker.network.bridge.enable_icc": "false"}
    assert kwargs["ipam"]["Config"][0]["Subnet"] == "172.30.240.0/24"
    assert kwargs["enable_ipv6"] is False
    assert env.m.egress_state == "on" and len(env.iptables.rules) == len(sessions.BLOCKED_RANGES) + 1
    env.m.setup()  # idempotent: the network exists, the rules exist
    assert len(env.docker.networks.created) == 1 and len(env.iptables.inserts()) == len(sessions.BLOCKED_RANGES) + 1


def test_existing_network_with_another_subnet_is_filtered_by_its_real_subnet(env):
    env.docker.networks.existing.append(SimpleNamespace(name="generall-sandbox", attrs={
        "IPAM": {"Config": [{"Subnet": "172.31.9.0/24"}]},
        "Options": {"com.docker.network.bridge.enable_icc": "false"},
    }))
    env.docker.networks.existing.append(SimpleNamespace(name="generall-sandbox-old", attrs={}))
    env.m.setup()
    assert env.docker.networks.created == []
    assert env.m.subnet == "172.31.9.0/24"
    assert all(rule[2] == "172.31.9.0/24" for rule in env.iptables.rules)


def test_init_manager_is_idempotent_and_process_wide(short_dir, monkeypatch):
    monkeypatch.setattr(sessions, "EGRESS_FILTER", "off")
    assert sessions.get_manager() is None
    docker = FakeDocker()
    first = sessions.init_manager(docker, "secure-container:abc", short_dir, short_dir)
    second = sessions.init_manager(FakeDocker(), "other:tag", "/elsewhere", "elsewhere")
    assert first is second is sessions.get_manager()
    assert first.image_ref == "secure-container:abc" and first.egress_state == "off"
    assert len(docker.networks.created) == 1


def test_init_manager_failure_leaves_no_manager(short_dir):
    docker = FakeDocker()
    docker.networks.list = lambda names=None: (_ for _ in ()).throw(FakeAPIError(500, "daemon down"))
    with pytest.raises(FakeAPIError):
        sessions.init_manager(docker, "img", short_dir, short_dir)
    assert sessions.get_manager() is None


# --------------------------------------------------------------------------- ContainerManager integration


@pytest.fixture
def container_manager(tmp_path):
    cm = cm_module.ContainerManager.__new__(cm_module.ContainerManager)
    cm.base_data_path = tmp_path / "data"
    cm.host_data_path = tmp_path / "data"
    cm.image_name, cm.image_tag, cm.container_prefix = "secure-container", "abc", "secure-container-"
    return cm


class FakeSessionManager:
    network_name = "generall-sandbox"

    def __init__(self, error=None):
        self.error, self.calls = error, []

    def exec(self, user_id, command, *, timeout=60):
        self.calls.append(("exec", user_id, command, timeout))
        if self.error:
            raise self.error
        return f"session: {command}"

    def install_package(self, user_id, package, manager="apt", *, timeout=300):
        self.calls.append(("install", user_id, package, manager, timeout))
        if self.error:
            raise self.error
        return f"installed {package} with {manager}"


def test_run_command_goes_to_the_session(container_manager, monkeypatch):
    container_manager.session_manager = FakeSessionManager()
    monkeypatch.setattr(container_manager, "run_one_shot", lambda *a, **k: pytest.fail("no one-off container"))
    assert container_manager.run_command("7", "ls | wc -l", 45, True, "1g") == "session: ls | wc -l"
    assert container_manager.session_manager.calls == [("exec", "7", "ls | wc -l", 45)]


def test_run_command_falls_back_to_a_one_off_container_on_capacity_error(container_manager, monkeypatch):
    container_manager.session_manager = FakeSessionManager(sessions.CapacityError("The server is short on memory (500 MB free); no new sandbox session can start right now. Try again in a few minutes."))
    calls = []
    monkeypatch.setattr(container_manager, "run_one_shot", lambda *a, **k: calls.append((a, k)) or "one-off output")
    out = container_manager.run_command("7", "ls | wc -l", 30, True)
    assert calls == [(("7", "ls | wc -l", 30, True, None), {"shell": True})]  # network_enabled honoured
    assert out.startswith("one-off output\n[Note: this ran in a one-off container")
    assert "short on memory (500 MB free)" in out


def test_run_command_without_sessions_uses_one_off_containers(container_manager, monkeypatch):
    container_manager.session_manager = FakeSessionManager(AssertionError("must not be called"))
    monkeypatch.setattr(sessions, "SESSIONS_ENABLED", False)
    calls = []
    monkeypatch.setattr(container_manager, "run_one_shot", lambda *a, **k: calls.append((a, k)) or "out")
    assert container_manager.run_command("7", "ls", 10, False) == "out"
    assert calls == [(("7", "ls", 10, False, None), {"shell": True})] and container_manager.session_manager.calls == []
    # no manager at all (Docker init failed / never built): the same path
    container_manager.session_manager = None
    monkeypatch.setattr(sessions, "SESSIONS_ENABLED", True)
    assert container_manager.run_command("7", "ls") == "out"


def test_python_code_and_scripts_run_in_the_session(container_manager):
    seen = []

    class Recording(FakeSessionManager):
        def exec(self, user_id, command, *, timeout=60):
            path = container_manager.base_data_path / user_id / Path(command.split()[-1]).name
            seen.append((command, timeout, path.read_text()))
            return "ok"

    container_manager.session_manager = Recording()
    assert container_manager.run_python_code("7", "print(1)", 20) == "ok"
    assert container_manager.run_shell_script("7", "echo hi | cat", 15) == "ok"
    (py_cmd, py_timeout, py_src), (sh_cmd, sh_timeout, sh_src) = seen
    assert py_cmd.startswith("python /home/runner/workspace/temp_code_") and py_timeout == 20 and py_src == "print(1)"
    assert sh_cmd.startswith("/bin/bash /home/runner/workspace/temp_script_") and sh_timeout == 15 and sh_src == "echo hi | cat"
    assert list((container_manager.base_data_path / "7").glob("temp_*")) == []


def test_one_off_fallback_runs_through_bash_on_the_filtered_network(container_manager):
    runs = []
    container = SimpleNamespace(wait=lambda timeout=None: {"StatusCode": 0}, logs=lambda: b"hi", remove=lambda force=False: None)
    container_manager.docker_client = SimpleNamespace(containers=SimpleNamespace(
        run=lambda **kw: runs.append(kw) or container, get=lambda name: container))
    container_manager.session_manager = FakeSessionManager(sessions.CapacityError("full"))
    out = container_manager.run_command("7", "ls | wc -l", 30, True)
    assert out.startswith("hi\n[Note:")
    assert runs[0]["command"] == ["bash", "-lc", "ls | wc -l"]
    assert runs[0]["network"] == "generall-sandbox" and runs[0]["dns"] == ["1.1.1.1", "8.8.8.8"]
    assert "network_mode" not in runs[0]
    container_manager.run_command("7", "ls", 30, False)
    assert runs[1]["network_mode"] == "none"
    # run_script stays a plain one-off command with no bash wrapper and no package replay
    (container_manager.base_data_path / "7" / "installed_packages.txt").write_text("jq\n")
    container_manager.run_script("7", "browse.py", ["https://x"], timeout=75, network_enabled=True, mem_limit="1g")
    assert isinstance(runs[2]["command"], str) and runs[2]["command"].startswith("python -I /home/runner/workspace/temp_tool_")
    assert runs[2]["mem_limit"] == "1g"


def test_install_package_routes_and_falls_back(container_manager, monkeypatch):
    container_manager.session_manager = FakeSessionManager()
    assert container_manager.install_package("7", "jq", 120, manager="pip") == "installed jq with pip"
    assert container_manager.install_python_package("7", "rich") == "installed rich with pip"
    assert container_manager.session_manager.calls == [("install", "7", "jq", "pip", 120), ("install", "7", "rich", "pip", 300)]
    assert container_manager.install_package("7", "x; rm -rf /").startswith("Error: ")
    assert container_manager.install_package("7", "jq", manager="brew") == 'Error: manager must be "apt" or "pip".'

    container_manager.session_manager = FakeSessionManager(sessions.CapacityError("full"))
    calls = []
    monkeypatch.setattr(container_manager, "run_one_shot", lambda *a, **k: calls.append((a, k)) or "Setting up jq")
    out = container_manager.install_package("7", "JQ")
    assert calls == [(("7", sessions.install_command("apt", ["jq"]), 300, True, None), {"shell": True, "replay_packages": False})]
    assert out.startswith("Setting up jq\nSaved jq to installed_packages.txt")
    assert (container_manager.base_data_path / "7" / "installed_packages.txt").read_text() == "jq\n"
    monkeypatch.setattr(container_manager, "run_one_shot", lambda *a, **k: "Error (exit code 100):\nE: nope")
    assert container_manager.install_package("7", "nope").startswith("Error (exit code 100)")
    assert (container_manager.base_data_path / "7" / "installed_packages.txt").read_text() == "jq\n"


# --------------------------------------------------------------------------- the patched tools


class FakeWrapper:
    def __init__(self):
        self.calls = []

    def wrap_terminal_command(self, user_id, command, timeout=60, network_enabled=False):
        self.calls.append(("run", user_id, command, timeout, network_enabled))
        return "ran"

    def wrap_shell_script(self, user_id, script, timeout=60, network_enabled=False):
        self.calls.append(("script", user_id, script, timeout, network_enabled))
        return "script ran"

    def wrap_package_installation(self, user_id, package_name, timeout=300, manager="apt"):
        self.calls.append(("install", user_id, package_name, timeout, manager))
        return "installed"


def _patched_terminal_tools():
    from agents.terminal_tools import TerminalTools

    class Terminal(TerminalTools):  # a subclass: the real class stays unpatched for other tests
        pass

    integrator = ti_module.ToolIntegrator.__new__(ti_module.ToolIntegrator)
    integrator.secure_wrapper = FakeWrapper()
    integrator.patch_terminal_tools(Terminal)
    return Terminal("7"), integrator.secure_wrapper


def test_patched_install_package_takes_a_manager_end_to_end():
    tools, wrapper = _patched_terminal_tools()
    assert tools.execute_tool("install_package", {"package_name": "pandas", "manager": "pip"}) == "installed"
    assert tools.execute_tool("install_package", {"package_name": "jq"}) == "installed"
    assert tools.install_package("ffmpeg", "apt", 60) == "installed"
    assert wrapper.calls == [("install", "7", "pandas", 300, "pip"), ("install", "7", "jq", 300, "apt"),
                             ("install", "7", "ffmpeg", 60, "apt")]
    schema = {t["name"]: t for t in tools.tools_schema}
    props = schema["install_package"]["input_schema"]["properties"]
    assert props["manager"]["enum"] == ["apt", "pip"] and props["manager"]["default"] == "apt"
    assert "installed_python_packages.txt" in schema["install_package"]["description"]
    assert "Ignored in the persistent sandbox session" in schema["run_shell_script"]["input_schema"]["properties"]["network_enabled"]["description"]
    assert "bash" in schema["run_command"]["description"] and "pipes" in schema["run_command"]["description"]
    tools.tools_schema[0]["description"] = "mutated"  # a fresh schema every time
    assert tools.tools_schema[0]["description"] != "mutated"


def test_secure_tool_wrapper_passes_the_manager_by_keyword():
    wrapper_module = importlib.import_module("secure_container.secure_tool_wrapper")
    wrapper = wrapper_module.SecureToolWrapper.__new__(wrapper_module.SecureToolWrapper)
    calls = []
    wrapper.container_manager = SimpleNamespace(install_package=lambda *a, **k: calls.append((a, k)) or "ok")
    assert wrapper.wrap_package_installation("7", "jq", 100) == "ok"
    assert wrapper.wrap_package_installation("7", "rich", manager="pip") == "ok"
    assert wrapper.wrap_python_package_installation("7", "numpy", 50) == "ok"
    assert calls == [(("7", "jq"), {"timeout": 100, "manager": "apt"}), (("7", "rich"), {"timeout": 300, "manager": "pip"}),
                     (("7", "numpy"), {"timeout": 50, "manager": "pip"})]


def test_code_tools_network_flag_is_described_as_ignored():
    from agents.code_tools import CodeTools

    class Code(CodeTools):
        pass

    integrator = ti_module.ToolIntegrator.__new__(ti_module.ToolIntegrator)
    integrator.secure_wrapper = SimpleNamespace(wrap_python_execution=lambda uid, code, timeout, net: (uid, code, timeout, net))
    integrator.patch_code_tools(Code)
    tool = Code("7").tools_schema[0]
    assert "persistent sandbox" in tool["description"]
    assert tool["input_schema"]["properties"]["network_enabled"]["description"] == ti_module.NETWORK_ENABLED_DESCRIPTION
    assert Code("7").execute_tool("execute_python", {"code": "print(1)", "network_enabled": True}) == ("7", "print(1)", 120, True)
