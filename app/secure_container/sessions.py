"""Persistent per-user sandbox sessions: one shell container and one browser container per user.

Before this module every sandboxed tool call ran in a fresh ``docker run`` that was thrown away
afterwards, so nothing but the workspace files survived between two commands. Now ONE process-wide
:class:`SessionManager` keeps, per user and on demand:

* a **shell** container (``sandbox-<uid>-<hex>``): ``sleep infinity`` under Docker's init, the user's
  workspace ``data/<uid>`` mounted at ``/home/runner/workspace``. Every command is a ``docker exec``
  of ``timeout -k 5 <t> bash -lc <command>`` as ``runner``, so shell syntax works and files,
  installed packages and background processes stay while the session runs. The persisted package
  lists (``installed_packages.txt`` / ``installed_python_packages.txt``) are replayed ONCE when the
  session starts, not before every command;
* a **browser** container (``browser-<uid>-<hex>``): ``scripts/browser_server.py`` (copied into
  ``data/_sandbox/<uid>/browser/`` before each start, so a script change needs no image rebuild)
  driving a persistent Chromium profile, reached over a unix socket with one JSON line each way.
  It sees only ``data/_sandbox/<uid>/browser`` (profile, socket, its own ``downloads/``); the bot
  moves each finished download into the workspace's ``downloads/`` (``move_download``) and the
  shell container never sees the profile, so commands cannot read the cookies. No path inside the
  workspace is mounted into the browser: the sandbox controls its links, and Docker resolves a
  bind-mount source on the host.

Both run as ``runner`` with ``init``, ``oom_score_adj`` 800 (the kernel kills sandboxes before the
bot), no swap beyond their memory limit, a pid limit and labels ``generall.sandbox`` /
``generall.user`` (``cleanup_stale`` finds leftovers by them). The browser also gets
``no-new-privileges``; the shell does not, because it would break the image's ``sudo apt-get`` rule.

Network: both join a user-defined bridge (``SANDBOX_NETWORK``, fixed ``SANDBOX_SUBNET``,
inter-container traffic off). Since the bot container is privileged with host networking it
inserts host iptables rules at setup and checks them again every ``EGRESS_RECHECK_SECONDS`` from
``reap_idle`` (``apply_egress_filter``, idempotent; a host firewall reload drops them, and in
``required`` mode a failed check stops every session): traffic from the subnet to
private, loopback, link-local (cloud metadata), CGNAT, multicast and reserved ranges is dropped in
``DOCKER-USER`` (``FORWARD`` when Docker has no such chain), and traffic from the subnet to the
host itself is dropped in ``INPUT`` (the local Bot API server listens there). Sandboxes resolve
names through public resolvers (``SANDBOX_DNS``): Docker's embedded DNS forwards from the
container's namespace, so a private host resolver would be cut off by the filter.

Admission: a session starts only while the count of running sessions of its kind is below the cap
(auto: what the host's total memory can hold), the memory limits of all running and starting
sessions of both kinds plus its own fit in ``MemTotal`` minus ``SANDBOX_MEMORY_RESERVE_MB`` (one
shared budget), and ``MemAvailable`` (``/proc/meminfo``, host-wide even inside the bot container)
minus the reserve covers its limit. Otherwise the least recently used session of that kind that
has been idle for two minutes is stopped (its files and profile stay) and admission is retried
once; then :class:`CapacityError`.

Idle: every call touches the session; ``reap_idle`` (the bot runs it every minute in a worker
thread) stops sessions unused for ``SANDBOX_IDLE_SECONDS``. ``due_warnings`` tells the bot which
users to warn ``SANDBOX_WARN_BEFORE_SECONDS`` before that, once per idle period; ``extend`` is the
user's answer. The browser server also exits by itself ``--idle-seconds`` (idle + 60) after its
last request, so a crashed bot never leaves a browser running for long.

Everything here is synchronous and blocking (docker-py, sockets, subprocess): callers on the event
loop use ``asyncio.to_thread``. The manager is thread-safe: one manager lock guards the session
table (never held across Docker calls) and a lock per (user, kind) serialises starting and
stopping, so a user's concurrent tool calls share one container and never start two.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- configuration


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        value = default
    return max(minimum, value) if minimum is not None else value


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


EGRESS_MODES = ("best_effort", "required", "off")

# Route run_command & co. to the persistent session (false: every call is a one-off container).
SESSIONS_ENABLED = _env_bool("SANDBOX_SESSIONS", True)
# Seconds without use after which a session (shell or browser) is stopped.
SESSION_IDLE_SECONDS = _env_int("SANDBOX_IDLE_SECONDS", 1800, 60)
# How long before that the user is warned (0 = never).
SESSION_WARN_BEFORE_SECONDS = _env_int("SANDBOX_WARN_BEFORE_SECONDS", 300, 0)
# Memory limit of one shell session / one browser (MB); also what admission requires free.
SESSION_MEM_MB = _env_int("SANDBOX_SESSION_MEM_MB", 768, 128)
BROWSER_MEM_MB = _env_int("BROWSER_MEM_MB", 1536, 256)
# Memory kept free for the OS, the bot and the Bot API server.
MEMORY_RESERVE_MB = _env_int("SANDBOX_MEMORY_RESERVE_MB", 1536, 0)
# Caps on running sessions; 0 = derived from the host's total memory (see max_shells/max_browsers).
MAX_SESSIONS_SETTING = _env_int("SANDBOX_MAX_SESSIONS", 0, 0)
MAX_BROWSERS_SETTING = _env_int("BROWSER_MAX_SESSIONS", 0, 0)
NETWORK_NAME = _env_str("SANDBOX_NETWORK", "generall-sandbox")
SUBNET = _env_str("SANDBOX_SUBNET", "172.30.240.0/24")
EGRESS_FILTER = _env_str("SANDBOX_EGRESS_FILTER", "best_effort").lower()
if EGRESS_FILTER not in EGRESS_MODES:
    logger.warning("SANDBOX_EGRESS_FILTER=%r is unknown; using best_effort", EGRESS_FILTER)
    EGRESS_FILTER = "best_effort"
# Resolvers Docker's embedded DNS forwards to for sandbox containers ("" = Docker's default).
DNS_SERVERS = [s for s in re.split(r"[\s,]+", os.getenv("SANDBOX_DNS", "1.1.1.1,8.8.8.8")) if s]

SHELL, BROWSER = "shell", "browser"
KINDS = (SHELL, BROWSER)

AUTO_MAX_SHELLS, AUTO_MAX_BROWSERS = 16, 8            # upper bounds of the automatic caps
UNKNOWN_MEM_MAX_SHELLS, UNKNOWN_MEM_MAX_BROWSERS = 4, 2  # caps when /proc/meminfo is unreadable
EVICT_IDLE_SECONDS = 120          # a session idle this long may be stopped to make room
EGRESS_RECHECK_SECONDS = 300      # reap_idle re-applies the egress rules this often (a firewall reload drops them)
OUTPUT_CAP = 1024 * 1024          # bytes of command output returned (the rest is dropped)
MAX_EXEC_TIMEOUT = 3600
REPLAY_TIMEOUT = 900              # the package replay at session start
OOM_SCORE_ADJ = 800
PIDS_LIMIT = 512
NANO_CPUS = 1_000_000_000         # one CPU per container, like the one-off containers

WORKSPACE = "/home/runner/workspace"
BROWSER_MOUNT = "/run/browser"
BROWSER_SOCKET = f"{BROWSER_MOUNT}/ctl/browser.sock"
BROWSER_PROFILE = f"{BROWSER_MOUNT}/profile"
BROWSER_DOWNLOADS = f"{BROWSER_MOUNT}/downloads"  # the browser's own; finished files are moved into the workspace
DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024           # a larger download is refused (deleted) instead of moved
BROWSER_VIEWPORT = "1280x800"
BROWSER_SERVER_SCRIPT = Path(__file__).parent / "scripts" / "browser_server.py"
BROWSER_START_TIMEOUT = 40.0      # seconds to wait for the server's _ping after docker run
BROWSER_CONTROL_TIMEOUT = 5.0     # _status / _touch / _shutdown
BROWSER_PEEK_TIMEOUT = 10.0
BROWSER_MAX_RESPONSE = 64 * 1024 * 1024
SANDBOX_DIR = "_sandbox"          # data/_sandbox/<uid>/browser: outside every user's workspace

LABEL_KIND = "generall.sandbox"
LABEL_USER = "generall.user"

APT_FILE = "installed_packages.txt"
PIP_FILE = "installed_python_packages.txt"
PACKAGE_FILES = {"apt": APT_FILE, "pip": PIP_FILE}

BROWSER_DIED = "Error: the browser stopped unexpectedly; it will restart on the next action."
CLOSED_WHILE_RUNNING = "Error: the sandbox was closed while the command was running; it did not finish."
EXIT_UNKNOWN = "Error: the command's exit status is unknown (the sandbox may have been closed)"

# Host ranges a sandbox must never reach (cloud metadata lives in 169.254.0.0/16).
BLOCKED_RANGES = (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10",
    "127.0.0.0/8", "0.0.0.0/8", "224.0.0.0/4", "240.0.0.0/4",
)
IPTABLES_BINARIES = ("iptables", "iptables-legacy", "iptables-nft")


class CapacityError(RuntimeError):
    """No session can start right now. ``str()`` is a readable reason for the model and the user."""


class SessionStartError(CapacityError):
    """Docker could not start the container (or the browser server never answered)."""


class _NotDelivered(OSError):
    """The browser server could not be reached at all: the request was never sent."""


class _ExecLost(Exception):
    """Docker failed after a shell command had started: it may have run (partly), so it is never
    run again. ``error`` is docker's exception."""

    def __init__(self, error: BaseException):
        super().__init__(str(error))
        self.error = error


# --------------------------------------------------------------------------- memory


MEMINFO_PATH = "/proc/meminfo"


def read_meminfo() -> dict[str, int]:
    """``/proc/meminfo`` as {field: kB}; {} when unreadable (tests replace this function)."""
    try:
        text = Path(MEMINFO_PATH).read_text()
    except OSError:
        return {}
    info: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            info[key.strip()] = int(parts[0])
    return info


def mem_available_mb() -> int | None:
    value = read_meminfo().get("MemAvailable")
    return value // 1024 if value is not None else None


def mem_total_mb() -> int | None:
    value = read_meminfo().get("MemTotal")
    return value // 1024 if value is not None else None


def auto_limit(total_mb: int | None, per_session_mb: int, upper: int, unknown: int) -> int:
    """How many sessions of ``per_session_mb`` the host's memory holds next to the reserve (1..upper)."""
    if total_mb is None:
        return unknown
    return max(1, min(upper, (total_mb - MEMORY_RESERVE_MB) // max(1, per_session_mb)))


# --------------------------------------------------------------------------- packages


_APT_NAME = re.compile(r"^[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?(?:=[A-Za-z0-9.+:~_-]+)?$")
_PIP_VERSION = r"(?:===|==|!=|<=|>=|~=|<|>)[A-Za-z0-9.*+!_-]+"
_PIP_NAME = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    rf"(?:{_PIP_VERSION}(?:,{_PIP_VERSION})*)?$"
)
_PACKAGE_PATTERNS = {"apt": _APT_NAME, "pip": _PIP_NAME}
# pip installs a requirement ending like an archive from that file in the cwd (the workspace),
# not from PyPI: such names are refused, never saved and never replayed.
_PIP_ARCHIVE_SUFFIXES = (".whl", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz",
                         ".tlz", ".tar.lz", ".tar.lzma", ".egg")
MAX_PACKAGES_AT_ONCE = 50
_package_locks: dict[str, threading.Lock] = {}
_package_locks_guard = threading.Lock()


def valid_package(manager: str, token: str) -> bool:
    pattern = _PACKAGE_PATTERNS.get(manager)
    if not (pattern and len(token) <= 200 and pattern.match(token)):
        return False
    return not (manager == "pip" and token.lower().endswith(_PIP_ARCHIVE_SUFFIXES))


def parse_packages(manager: str, text: Any) -> list[str]:
    """Package names from the tool's ``package_name`` (several separated by spaces).

    apt names are lower-cased (``name``, ``name:arch``, ``name=version``); pip takes requirement
    specifiers (``name``, ``name[extra]``, ``name==1.2``, ``name>=1,<2``). Anything else (options,
    URLs, paths, archive file names such as ``x.whl`` or ``x.tar.gz``) raises ValueError with a
    readable reason: install_package never passes free text to apt or pip."""
    if manager not in PACKAGE_FILES:
        raise ValueError('manager must be "apt" or "pip"')
    tokens = str(text or "").split()
    if not tokens:
        raise ValueError("no package name given")
    if len(tokens) > MAX_PACKAGES_AT_ONCE:
        raise ValueError(f"at most {MAX_PACKAGES_AT_ONCE} packages at once")
    packages: list[str] = []
    for token in tokens:
        token = token.lower() if manager == "apt" else token
        if not valid_package(manager, token):
            hint = (" (install from a URL or a file with run_command instead; that is not saved)"
                    if manager == "pip" else "")
            raise ValueError(f"{token!r} is not a valid {manager} package name{hint}")
        if token not in packages:
            packages.append(token)
    return packages


def package_key(manager: str, token: str) -> str:
    """The package a line is about, without version or extras (pip names normalised like PyPI)."""
    if manager == "apt":
        return token.split("=", 1)[0].split(":", 1)[0].lower()
    name = re.split(r"[\[=<>!~;,]", token, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------- link-safe file helpers
# The bot runs as root in a privileged container with the Docker socket, while the sandbox (the
# workspace data/<uid>) and the browser (data/_sandbox/<uid>/browser) can put a symlink anywhere
# they write. Everything the bot does inside those trees goes through these helpers, which never
# follow a link: a chmod, copy or read through one would act on the link's target, as root.

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def real_dir(path: Path, mode: int | None = 0o777) -> Path:
    """Make ``path`` a real directory (a link or file in its place is unlinked) and set ``mode``
    through an ``O_NOFOLLOW`` descriptor. The parents must be trusted (bot-created)."""
    path = Path(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        st = None
    if st is not None and not stat.S_ISDIR(st.st_mode):
        os.unlink(path)
        st = None
    if st is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
    fd = os.open(path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        if mode is not None:
            os.fchmod(fd, mode)
    finally:
        os.close(fd)
    return path


def read_regular(path: Path, limit: int = 1024 * 1024) -> str | None:
    """The text of ``path`` when it is a regular file (never through a link, never a FIFO)."""
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return None
    finally:
        os.close(fd)


def write_new_file(directory: Path, name: str, data: bytes, mode: int) -> None:
    """Replace ``directory/name`` with a new regular file holding ``data``; whatever was there (a
    link, a file, a directory) is removed first and nothing is ever followed."""
    dir_fd = os.open(directory, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        try:
            os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            shutil.rmtree(Path(directory) / name)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, mode, dir_fd=dir_fd)
        try:
            _write_all(fd, data)
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def move_download(src_dir: Path, dst_dir: Path, name: str, max_bytes: int | None = None) -> str:
    """Move a finished download from the browser's own folder into the workspace's downloads/.

    Neither side is trusted: the source must be a regular file (opened ``O_NOFOLLOW``), the target
    is created ``O_EXCL | O_NOFOLLOW`` relative to a descriptor of the real ``dst_dir``. A taken
    name becomes ``stem-2.ext``, ``stem-3.ext`` ... Returns the final name; raises ``OSError`` or
    ``ValueError`` with a readable reason (the source is deleted either way)."""
    max_bytes = DOWNLOAD_MAX_BYTES if max_bytes is None else max_bytes
    too_big = (f"the file is larger than {max_bytes // (1024 * 1024)} MB" if max_bytes >= 1024 * 1024
               else f"the file is larger than {max_bytes} bytes")
    name = os.path.basename(str(name or ""))
    if not name or name in (".", ".."):
        raise ValueError("no file name")
    src_fd = os.open(src_dir, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        try:
            fd_in = os.open(name, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=src_fd)
        except OSError as e:
            try:
                os.unlink(name, dir_fd=src_fd)  # a link planted in its place goes too
            except OSError:
                pass
            if e.errno == errno.ELOOP:
                raise ValueError("the download is not a regular file") from e
            raise ValueError(f"the downloaded file is missing ({e.strerror or e})") from e
        try:
            st = os.fstat(fd_in)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError("the download is not a regular file")
            if st.st_size > max_bytes:
                raise ValueError(too_big)
            real_dir(dst_dir, 0o777)
            dst_fd = os.open(dst_dir, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
            try:
                final, fd_out = _create_unique(dst_fd, name)
                try:
                    copied = 0
                    while True:
                        chunk = os.read(fd_in, 1 << 20)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > max_bytes:
                            raise ValueError(too_big)
                        _write_all(fd_out, chunk)
                    os.fchmod(fd_out, 0o666)
                except BaseException:
                    os.close(fd_out)
                    fd_out = -1
                    try:
                        os.unlink(final, dir_fd=dst_fd)
                    except OSError:
                        pass
                    raise
                finally:
                    if fd_out >= 0:
                        os.close(fd_out)
            finally:
                os.close(dst_fd)
        finally:
            os.close(fd_in)
            try:
                os.unlink(name, dir_fd=src_fd)
            except OSError:
                pass
    finally:
        os.close(src_fd)
    return final


def _create_unique(dir_fd: int, name: str) -> tuple[str, int]:
    stem, ext = os.path.splitext(name)
    for n in range(1, 1000):
        candidate = name if n == 1 else f"{stem}-{n}{ext}"
        try:
            return candidate, os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o666, dir_fd=dir_fd)
        except FileExistsError:
            continue
    raise ValueError("no free file name in downloads/")


def _package_tokens(path: Path) -> list[str]:
    text = read_regular(path)
    if text is None:
        return []
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        for token in line.split():
            if token not in tokens:
                tokens.append(token)
    return tokens


def persisted_packages(user_dir: Path) -> dict[str, list[str]]:
    """The saved apt / pip packages of a workspace that are valid (what a session start reinstalls)."""
    result: dict[str, list[str]] = {}
    for manager, name in PACKAGE_FILES.items():
        tokens = _package_tokens(Path(user_dir) / name)
        if manager == "apt":
            tokens = [t.lower() for t in tokens]
        valid = [t for t in tokens if valid_package(manager, t)]
        skipped = [t for t in tokens if t not in valid]
        if skipped:
            logger.warning("Skipping invalid %s package names in %s: %s", manager, Path(user_dir) / name, skipped)
        result[manager] = valid
    return result


def add_persisted_packages(user_dir: Path, manager: str, packages: Iterable[str]) -> list[str]:
    """Save ``packages`` to the workspace's package file (a line each); returns the ones added.

    A new line replaces earlier lines about the same package (``numpy`` then ``numpy==2.0``), so
    the replay never asks pip for two conflicting versions. Other lines, comments included, stay
    as they are. Atomic write; the file is left world-writable so the sandbox can edit it too."""
    path = Path(user_dir) / PACKAGE_FILES[manager]
    with _package_locks_guard:
        lock = _package_locks.setdefault(str(path), threading.Lock())
    with lock:
        lines = (read_regular(path) or "").splitlines()
        existing = set(_package_tokens(path))
        added = [p for p in packages if p not in existing]
        if not added:
            return []
        keys = {package_key(manager, p) for p in added}

        def keep(line: str) -> bool:
            tokens = line.split("#", 1)[0].split()
            return not tokens or not all(package_key(manager, t) in keys for t in tokens)

        lines = [line for line in lines if keep(line)] + added
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        # mode set on the new file itself before it takes the name: a chmod by path afterwards
        # would follow a link the sandbox swapped in meanwhile
        write_new_file(path.parent, tmp.name, ("\n".join(lines) + "\n").encode("utf-8"), 0o666)
        os.replace(tmp, path)
        return added


def persisted_note(manager: str, packages: list[str], added: list[str]) -> str:
    name = PACKAGE_FILES[manager]
    if added:
        return (f"Saved {', '.join(added)} to {name}: reinstalled automatically whenever a new sandbox "
                f"session starts (remove the line from {name} to stop that).")
    return f"{', '.join(packages)} already listed in {name} (reinstalled when a new sandbox session starts)."


def install_command(manager: str, packages: list[str]) -> str:
    quoted = " ".join(shlex.quote(p) for p in packages)
    if manager == "apt":
        return ("sudo apt-get update -qq && "
                f"sudo apt-get install -y -q -o DPkg::Lock::Timeout=120 {quoted}")
    return f"pip install --user --no-input {quoted}"


def replay_script(apt: list[str], pip: list[str]) -> str:
    """The bash script a new shell session runs once to reinstall the saved packages.

    Same logic as the old per-command prelude: apt via sudo (the image's sudoers rule), a
    best-effort pip install of the PyPI name of every ``python3-*`` apt package (Debian's builds
    target Debian's Python, not the image's 3.12), then the saved pip packages with ``--user``.
    Exits non-zero when an install failed."""
    lines = ["status=0"]
    if apt:
        quoted = " ".join(shlex.quote(p) for p in apt)
        lines += [
            f"echo Reinstalling apt packages: {quoted}",
            "sudo apt-get update -qq || status=$?",
            f"sudo apt-get install -y -qq -o DPkg::Lock::Timeout=120 {quoted} > /dev/null || status=$?",
        ]
        from_apt = [p[len("python3-"):].split("=", 1)[0] for p in apt if p.startswith("python3-") and len(p) > 8]
        if from_apt:
            quoted_pip = " ".join(shlex.quote(p) for p in from_apt)
            lines.append(f"pip install --user --quiet --no-input {quoted_pip} 2>&1 || true")
    if pip:
        quoted = " ".join(shlex.quote(p) for p in pip)
        lines += [
            f"echo Reinstalling pip packages: {quoted}",
            f"pip install --user --quiet --no-input {quoted} || status=$?",
        ]
    lines.append('exit "$status"')
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- command execution


# Runs inside the shell container. The command's stdout+stderr go to a temp file, not to the exec
# stream: a background job (``cmd &``) inherits the file, so the exec ends when bash does instead
# of when the job does, and only the first OUTPUT_CAP + 1 bytes ever reach the bot.
_EXEC_SCRIPT = (
    'out=$(mktemp /tmp/.sandbox-exec.XXXXXX) || exit 125\n'
    'timeout -k 5 "$1" bash -lc "$2" >"$out" 2>&1 </dev/null\n'
    'rc=$?\n'
    f'head -c {OUTPUT_CAP + 1} "$out"\n'
    'rm -f "$out"\n'
    'exit "$rc"\n'
)


def exec_argv(command: str, timeout: int) -> list[str]:
    """argv of one command: ``timeout -k 5 <timeout> bash -lc <command>`` inside the capture wrapper."""
    return ["bash", "-c", _EXEC_SCRIPT, "sandbox-exec", str(int(timeout)), command]


def clamp_timeout(timeout: Any, default: int = 60) -> int:
    try:
        value = int(timeout)
    except (TypeError, ValueError):
        value = default
    return max(1, min(MAX_EXEC_TIMEOUT, value))


def timeout_message(timeout: int) -> str:
    return (f"Error: the command timed out after {timeout} s (the sandbox session is still running; "
            "long jobs: start them in the background with nohup <command> > job.log 2>&1 & "
            "and check job.log later)")


def format_exec_result(exit_code: int | None, output: bytes | str | None, timeout: int,
                       elapsed: float = 0.0, mem_mb: int | None = None) -> str:
    """The tool result of one command: its output, or one of the error formats.

    0 → the output (``(no output)`` when empty); 124 or 137 once the time was up (``elapsed`` ≥
    ``timeout``) → the timeout text (plus what was printed before); 137 earlier → killed (memory);
    None (docker reported no exit status) → an error saying so, never success; other codes, 124
    from the command itself included → ``Error (exit code N):\\n<output>``, the format of the
    one-off containers."""
    if isinstance(output, str):
        output = output.encode("utf-8", errors="replace")
    data = output or b""
    text = data[:OUTPUT_CAP].decode("utf-8", errors="replace")
    if len(data) > OUTPUT_CAP:
        text += "\n[output truncated at 1 MB]"
    if exit_code is None:
        return f"{EXIT_UNKNOWN}:\n{text}"
    code = int(exit_code)
    if code == 0:
        return text if text else "(no output)"
    if code in (124, 137) and elapsed >= timeout:
        message = timeout_message(timeout)
        return message + (f"\nOutput before the timeout:\n{text}" if text.strip() else "")
    if code == 137:
        limit = f" of the sandbox's {mem_mb} MB" if mem_mb else ""
        return f"Error (exit code 137: the process was killed, probably for using more than the memory{limit}):\n{text}"
    return f"Error (exit code {code}):\n{text}"


def _container_gone(error: BaseException) -> bool:
    """A docker error meaning the container no longer runs (removed, exited, OOM-killed)."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status in (404, 409) or type(error).__name__ == "NotFound"


# --------------------------------------------------------------------------- egress filter


def egress_rules(subnet: str, forward_chain: str = "DOCKER-USER") -> list[list[str]]:
    """The iptables rules (chain first) that keep the sandbox subnet off private networks and the host."""
    rules = [[forward_chain, "-s", subnet, "-d", cidr, "-j", "DROP"] for cidr in BLOCKED_RANGES]
    rules.append(["INPUT", "-s", subnet, "-j", "DROP"])
    return rules


def egress_commands(subnet: str, binary: str = "iptables",
                    forward_chain: str = "DOCKER-USER") -> list[tuple[list[str], list[str]]]:
    """(check, insert) argv pairs: ``-C`` first, ``-I`` only when the check fails (idempotent)."""
    pairs = []
    for chain, *spec in egress_rules(subnet, forward_chain):
        pairs.append(([binary, "-w", "5", "-C", chain, *spec], [binary, "-w", "5", "-I", chain, *spec]))
    return pairs


def _run_quiet(argv: list[str]) -> int:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("%s failed: %s", argv[0], e)
        return -1
    if done.returncode != 0:
        logger.debug("%s exited %s: %s", " ".join(argv), done.returncode, (done.stderr or "").strip())
    return done.returncode


def pick_iptables(run: Callable[[list[str]], int]) -> tuple[str | None, str | None]:
    """The iptables binary Docker uses (the one whose tables have DOCKER-USER) and the forward chain."""
    for binary in IPTABLES_BINARIES:
        if run([binary, "-w", "5", "-S", "DOCKER-USER"]) == 0:
            return binary, "DOCKER-USER"
    for binary in IPTABLES_BINARIES:
        if run([binary, "-w", "5", "-S", "FORWARD"]) == 0:
            return binary, "FORWARD"
    return None, None


_egress_warned = False


def apply_egress_filter(subnet: str, mode: str | None = None,
                        run: Callable[[list[str]], int] | None = None, *, recheck: bool = False) -> str:
    """Insert the egress rules for ``subnet`` unless present: "on", "off" (mode off) or "failed".

    Never raises: without iptables or the rights to use it (a local run) the failure is logged
    once as a WARNING and the bot goes on; whether sessions may still start is the caller's call
    (``required`` mode refuses them). ``recheck`` (the periodic check of a running manager) logs
    only rules that had to be inserted again, not the "active" line."""
    global _egress_warned
    mode = (mode or EGRESS_FILTER).lower()
    if mode == "off":
        return "off"
    run = run or _run_quiet
    binary, chain = pick_iptables(run)
    problem = None
    inserted = 0
    if binary is None:
        problem = "iptables is not available (or not permitted) in the bot container"
    else:
        for check, insert in egress_commands(subnet, binary, chain):
            if run(check) == 0:
                continue
            if run(insert) != 0:
                problem = f"'{' '.join(insert)}' failed"
                break
            inserted += 1
    if problem is None:
        if not recheck:
            logger.info("Sandbox egress filter active for %s (%s, chain %s)", subnet, binary, chain)
        elif inserted:
            logger.warning("Sandbox egress filter: %s rule(s) for %s were missing (a firewall reload?) and were "
                           "inserted again (%s, chain %s)", inserted, subnet, binary, chain)
        return "on"
    if recheck:
        logger.debug("Sandbox egress filter check failed: %s", problem)
    elif not _egress_warned:
        _egress_warned = True
        logger.warning("Sandbox egress filter NOT active (%s): sandboxes can reach private networks "
                       "and the host. SANDBOX_EGRESS_FILTER=%s.", problem, mode)
    return "failed"


# --------------------------------------------------------------------------- unix socket client


def _connect_unix(path: Path, timeout: float) -> socket.socket:
    """Connect to the unix socket ``path`` without following a link (the browser can write ctl/).

    The directory is opened ``O_NOFOLLOW``, the entry must be a socket (checked without following
    it) and the connect goes through ``/proc/self/fd``, which also keeps a deep path under the
    sockaddr_un limit. Where ``O_PATH`` exists the entry itself is opened ``O_PATH | O_NOFOLLOW``
    and connected through its descriptor, so a link swapped in after the check is not followed
    either. A refused entry raises ``ConnectionRefusedError`` (the server counts as unreachable)."""
    path = Path(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    dir_fd = entry_fd = None
    o_path = getattr(os, "O_PATH", 0)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        if o_path:
            entry_fd = os.open(path.name, o_path | _NOFOLLOW, dir_fd=dir_fd)
            mode = os.fstat(entry_fd).st_mode
            target = f"/proc/self/fd/{entry_fd}"
        else:
            mode = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False).st_mode
            target = f"/proc/self/fd/{dir_fd}/{path.name}"
        if not stat.S_ISSOCK(mode):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "not a socket", str(path))
        sock.connect(target)
    except BaseException:
        sock.close()
        raise
    finally:
        for fd in (entry_fd, dir_fd):
            if fd is not None:
                os.close(fd)
    return sock


def socket_request(path: Path, request: dict, timeout: float) -> dict:
    """One request to the browser server: connect, one JSON line out, one JSON line in.

    Raises _NotDelivered when the server cannot be reached (nothing ran), TimeoutError when no
    answer arrived in ``timeout`` seconds, ConnectionError / ValueError when the server went away
    or answered garbage after the request was sent."""
    deadline = time.monotonic() + max(0.1, timeout)
    try:
        sock = _connect_unix(Path(path), min(5.0, max(0.1, timeout)))
    except OSError as e:
        raise _NotDelivered(str(e)) from e
    with sock:
        try:
            sock.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("ascii"))
        except (BrokenPipeError, ConnectionResetError) as e:
            raise _NotDelivered(str(e)) from e
        buf = bytearray()
        while b"\n" not in buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no answer from the browser server")
            sock.settimeout(remaining)
            chunk = sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("the browser server closed the connection without an answer")
            buf += chunk
            if len(buf) > BROWSER_MAX_RESPONSE:
                raise ValueError("the browser server's answer is too large")
    line = bytes(buf).split(b"\n", 1)[0]
    data = json.loads(line.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("the browser server's answer is not an object")
    return data


def browser_command(remember: bool, idle_seconds: int | None = None) -> list[str]:
    """The browser container's command (the image's entrypoint execs it)."""
    idle = SESSION_IDLE_SECONDS if idle_seconds is None else idle_seconds
    return [
        "python", f"{BROWSER_MOUNT}/browser_server.py",
        "--socket", BROWSER_SOCKET,
        "--profile", BROWSER_PROFILE if remember else "",
        "--downloads", BROWSER_DOWNLOADS,
        "--idle-seconds", str(int(idle) + 60),
        "--viewport", BROWSER_VIEWPORT,
    ]


def _utc(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, timezone.utc) if ts is not None else None


def _dir_size(path: Path) -> int:
    """Bytes of the files under ``path``; 0 unless ``path`` is a real directory. Never follows a
    link (the browser can write there): ``os.fwalk`` from an ``O_NOFOLLOW`` descriptor, sizes by
    lstat relative to each directory's descriptor."""
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            return 0
        top = os.open(path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return 0
    total = 0
    try:
        for _root, _dirs, files, root_fd in os.fwalk(".", dir_fd=top, follow_symlinks=False):
            for name in files:
                try:
                    total += os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_size
                except OSError:
                    pass
    except OSError:
        pass
    finally:
        os.close(top)
    return total


_UID = re.compile(r"^[A-Za-z0-9-][A-Za-z0-9_-]{0,63}$")


def check_user_id(user_id: Any) -> str:
    """The user id as a path/name component; refuses anything that could leave data/<uid>.

    A leading underscore is refused too: ``_sandbox`` is where the browser profiles live."""
    uid = str(user_id).strip()
    if not _UID.match(uid):
        raise ValueError(f"invalid user id for a sandbox session: {user_id!r}")
    return uid


# --------------------------------------------------------------------------- the manager


@dataclass(eq=False)
class Session:
    user_id: str
    kind: str
    container: Any
    name: str
    started: float
    last_used: float
    mem_mb: int
    remember: bool = True
    in_flight: int = 0
    note: str = ""   # told once with the next command's result (a failed package replay)


class SessionManager:
    """The per-user shell and browser containers of this bot process (see the module docstring)."""

    def __init__(self, docker_client: Any, image_ref: str, host_data_path: Any, base_data_path: Any = "data",
                 *, clock: Callable[[], float] | None = None,
                 egress_runner: Callable[[list[str]], int] | None = None):
        self.docker = docker_client
        self.image_ref = image_ref
        self.host_data_path = Path(host_data_path)
        self.base_data_path = Path(base_data_path)
        self.network_name = NETWORK_NAME
        self.subnet = SUBNET
        self.egress_state = "off"
        self._clock = clock or time.time
        self._egress_runner = egress_runner
        self._egress_checked: float | None = None  # when setup / the last re-check applied the rules
        self._lock = threading.Lock()
        self._sessions: dict[tuple[str, str], Session] = {}
        self._user_locks: dict[tuple[str, str], threading.Lock] = {}
        self._starting = {SHELL: 0, BROWSER: 0}
        self._warned: dict[str, float] = {}
        self._closed = False

    # ---- setup -----------------------------------------------------------------------------

    def setup(self) -> None:
        """Create the sandbox network when missing and apply the egress filter (init_manager);
        ``reap_idle`` re-applies the filter every EGRESS_RECHECK_SECONDS afterwards."""
        self._ensure_network()
        self.egress_state = apply_egress_filter(self.subnet, EGRESS_FILTER, self._egress_runner)
        self._egress_checked = self._clock()

    def _ensure_network(self) -> Any:
        name = self.network_name
        existing = [n for n in self.docker.networks.list(names=[name]) if getattr(n, "name", None) == name]
        if existing:
            network = existing[0]
            attrs = getattr(network, "attrs", None) or {}
            configs = (attrs.get("IPAM") or {}).get("Config") or []
            subnet = (configs[0] or {}).get("Subnet") if configs else None
            if subnet and subnet != self.subnet:
                logger.warning("Sandbox network %s uses %s, not SANDBOX_SUBNET=%s; filtering %s",
                               name, subnet, self.subnet, subnet)
                self.subnet = subnet
            options = attrs.get("Options") or {}
            if attrs and options.get("com.docker.network.bridge.enable_icc") != "false":
                logger.warning("Sandbox network %s lets containers talk to each other; remove it "
                               "(docker network rm %s) so the bot recreates it with enable_icc=false", name, name)
            return network
        from docker.types import IPAMConfig, IPAMPool

        logger.info("Creating sandbox network %s (%s)", name, self.subnet)
        return self.docker.networks.create(
            name,
            driver="bridge",
            options={"com.docker.network.bridge.enable_icc": "false"},
            ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=self.subnet)]),
            enable_ipv6=False,
            labels={LABEL_KIND: "network"},
        )

    # ---- limits ----------------------------------------------------------------------------

    def max_shells(self) -> int:
        if MAX_SESSIONS_SETTING > 0:
            return MAX_SESSIONS_SETTING
        return auto_limit(mem_total_mb(), SESSION_MEM_MB, AUTO_MAX_SHELLS, UNKNOWN_MEM_MAX_SHELLS)

    def max_browsers(self) -> int:
        if MAX_BROWSERS_SETTING > 0:
            return MAX_BROWSERS_SETTING
        return auto_limit(mem_total_mb(), BROWSER_MEM_MB, AUTO_MAX_BROWSERS, UNKNOWN_MEM_MAX_BROWSERS)

    @staticmethod
    def _mem_of(kind: str) -> int:
        return SESSION_MEM_MB if kind == SHELL else BROWSER_MEM_MB

    def _refusal_locked(self, kind: str) -> str | None:
        """Why a new session of ``kind`` cannot start now (None = it can). Manager lock held."""
        running = sum(1 for (_u, k) in self._sessions if k == kind) + self._starting[kind]
        limit = self.max_shells() if kind == SHELL else self.max_browsers()
        what = "sandbox session" if kind == SHELL else "browser"
        if running >= limit:
            noun = "sandbox sessions" if kind == SHELL else "browsers"
            return (f"All {limit} {noun} this server can run are in use; no new {what} can start right now. "
                    "Try again in a few minutes.")
        pending = sum(self._mem_of(k) * n for k, n in self._starting.items())
        total = mem_total_mb()
        if total is not None:
            # one budget for both kinds: the memory limits of everything running or starting, plus
            # this one, must fit next to the reserve (each kind's count cap alone assumes all of it)
            budget = total - MEMORY_RESERVE_MB
            committed = sum(s.mem_mb for s in self._sessions.values()) + pending
            if committed + self._mem_of(kind) > budget:
                return (f"The sandboxes already hold {committed} of the {max(0, budget)} MB this server gives them; "
                        f"no new {what} can start right now. Try again in a few minutes.")
        available = mem_available_mb()
        if available is not None:
            if available - MEMORY_RESERVE_MB - pending < self._mem_of(kind):
                return (f"The server is short on memory ({available} MB free); no new {what} can start "
                        "right now. Try again in a few minutes.")
        return None

    def _pick_victim_locked(self, kind: str) -> tuple[Session, threading.Lock | None] | None:
        """The least recently used idle session of ``kind`` that may be stopped; removed from the table."""
        now = self._clock()
        candidates = sorted((s for (_u, k), s in self._sessions.items() if k == kind), key=lambda s: s.last_used)
        for session in candidates:
            if session.in_flight or now - session.last_used < EVICT_IDLE_SECONDS:
                continue
            lock = self._user_locks.get((session.user_id, kind))
            if lock is not None and not lock.acquire(blocking=False):
                continue  # being started or stopped right now
            del self._sessions[(session.user_id, kind)]
            return session, lock
        return None

    def _reserve(self, kind: str) -> None:
        """Admit one start of ``kind`` (counted until _unreserve) or raise CapacityError."""
        reason = None
        for attempt in range(2):
            with self._lock:
                reason = self._refusal_locked(kind)
                if reason is None:
                    self._starting[kind] += 1
                    return
                victim = self._pick_victim_locked(kind) if attempt == 0 else None
            if victim is None:
                break
            session, lock = victim
            logger.info("Stopping the idle %s session of %s to make room", kind, session.user_id)
            try:
                self._stop(session)
            finally:
                if lock is not None:
                    lock.release()
        raise CapacityError(reason or "No sandbox can start right now. Try again in a few minutes.")

    def _unreserve(self, kind: str) -> None:
        with self._lock:
            self._starting[kind] = max(0, self._starting[kind] - 1)

    # ---- the session table -------------------------------------------------------------------

    def _user_lock(self, uid: str, kind: str) -> threading.Lock:
        with self._lock:
            return self._user_locks.setdefault((uid, kind), threading.Lock())

    def _touch_locked(self, session: Session) -> None:
        session.last_used = self._clock()
        self._warned.pop(session.user_id, None)

    def _start_refusal(self) -> str | None:
        """Why no session may start or be kept from a start in flight (None = they may)."""
        if self._closed:
            return "The sandbox is shutting down."
        if EGRESS_FILTER == "required" and self.egress_state != "on":
            return ("The sandbox network filter is not active on this server, so sandbox sessions "
                    "and the browser are disabled (SANDBOX_EGRESS_FILTER=required).")
        return None

    def _check_startable(self) -> None:
        reason = self._start_refusal()
        if reason is not None:
            raise CapacityError(reason)

    def _acquire(self, user_id: Any, kind: str, *, remember: bool | None = None, touch: bool = True) -> Session:
        """The user's running session of ``kind`` (started when needed), marked as in use.

        Every caller pairs it with :meth:`_release`. CapacityError when none can start."""
        uid = check_user_id(user_id)
        with self._user_lock(uid, kind):
            with self._lock:
                session = self._sessions.get((uid, kind))
                if session is not None and (remember is None or session.remember == remember):
                    session.in_flight += 1
                    if touch:
                        self._touch_locked(session)
                    return session
                if session is not None:  # the browser must restart with the other profile mode
                    del self._sessions[(uid, kind)]
            if session is not None:
                self._stop(session)
            self._check_startable()
            self._reserve(kind)
            try:
                session = self._start(uid, kind, True if remember is None else bool(remember))
            finally:
                self._unreserve(kind)
            with self._lock:
                # shutdown_all (or a failed egress re-check in required mode) may have emptied the
                # table while docker was starting this one: it must not outlive that
                refusal = self._start_refusal()
                if refusal is None:
                    self._sessions[(uid, kind)] = session
                    session.in_flight += 1
                    self._touch_locked(session)
            if refusal is not None:
                logger.info("Stopping the %s session %s: it came up after starts were refused (%s)",
                            kind, session.name, refusal)
                self._stop(session, graceful=False)
                raise CapacityError(refusal)
            return session

    def _release(self, session: Session, touch: bool = True) -> None:
        with self._lock:
            session.in_flight = max(0, session.in_flight - 1)
            if touch and self._sessions.get((session.user_id, session.kind)) is session:
                self._touch_locked(session)

    def _discard(self, session: Session, graceful: bool = False) -> None:
        """Forget a session that broke and remove its container (best effort)."""
        with self._lock:
            if self._sessions.get((session.user_id, session.kind)) is session:
                del self._sessions[(session.user_id, session.kind)]
        self._stop(session, graceful=graceful)

    def _running(self, uid: str, kind: str) -> Session | None:
        with self._lock:
            return self._sessions.get((uid, kind))

    # ---- paths -----------------------------------------------------------------------------

    def _user_dir(self, uid: str) -> Path:
        return self.base_data_path / uid

    def _browser_dir(self, uid: str) -> Path:
        return self.base_data_path / SANDBOX_DIR / uid / "browser"

    def _socket_path(self, uid: str) -> Path:
        return self._browser_dir(uid) / "ctl" / "browser.sock"

    def _host(self, *parts: str) -> str:
        return str(self.host_data_path.joinpath(*parts).absolute())

    @staticmethod
    def _writable(path: Path) -> Path:
        """A real directory the containers' non-root runner can write (never through a link)."""
        try:
            real_dir(path, 0o777)
        except OSError as e:
            logger.warning("Could not prepare %s: %s", path, e)
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

    # ---- starting and stopping containers ----------------------------------------------------

    def _run_kwargs(self, kind: str, uid: str, mem_mb: int) -> dict:
        kwargs = dict(
            detach=True,
            user="runner",
            init=True,
            network=self.network_name,
            mem_limit=f"{mem_mb}m",
            memswap_limit=f"{mem_mb}m",
            oom_score_adj=OOM_SCORE_ADJ,
            pids_limit=PIDS_LIMIT,
            nano_cpus=NANO_CPUS,
            labels={LABEL_KIND: kind, LABEL_USER: uid},
            working_dir=WORKSPACE,
        )
        if DNS_SERVERS:
            kwargs["dns"] = list(DNS_SERVERS)
        return kwargs

    def _start(self, uid: str, kind: str, remember: bool) -> Session:
        self._remove_orphans(uid, kind)
        try:
            if kind == SHELL:
                return self._start_shell(uid)
            return self._start_browser(uid, remember)
        except OSError as e:  # preparing the directories or copying the server script failed
            logger.error("Could not prepare the %s session of %s: %s", kind, uid, e)
            what = "sandbox" if kind == SHELL else "browser"
            raise SessionStartError(f"The {what} could not start: {e}") from e

    def _start_shell(self, uid: str) -> Session:
        user_dir = self._writable(self._user_dir(uid))
        name = f"sandbox-{uid}-{uuid.uuid4().hex[:8]}"
        try:
            container = self.docker.containers.run(
                self.image_ref,
                command=["sleep", "infinity"],
                name=name,
                volumes={self._host(uid): {"bind": WORKSPACE, "mode": "rw", "propagation": "shared"}},
                **self._run_kwargs(SHELL, uid, SESSION_MEM_MB),
            )
        except Exception as e:  # noqa: BLE001 - docker errors become a readable refusal
            logger.error("Could not start sandbox session %s: %s", name, e)
            raise SessionStartError(f"The sandbox could not start: {e}") from e
        now = self._clock()
        session = Session(uid, SHELL, container, name, now, now, SESSION_MEM_MB)
        logger.info("Started sandbox session %s", name)
        try:
            self._replay_packages(session, user_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("Sandbox session %s broke during the package replay: %s", name, e)
            self._stop(session)
            raise SessionStartError(f"The sandbox could not start: {e}") from e
        return session

    def _replay_packages(self, session: Session, user_dir: Path) -> None:
        """Reinstall the saved packages once, before the session's first command."""
        packages = persisted_packages(user_dir)
        if not packages["apt"] and not packages["pip"]:
            return
        logger.info("Replaying saved packages in %s: %s", session.name, packages)
        started = time.monotonic()
        code, output = self._exec_raw(session, replay_script(packages["apt"], packages["pip"]), REPLAY_TIMEOUT)
        logger.info("Package replay in %s exited %s after %.0f s", session.name, code, time.monotonic() - started)
        if code != 0:
            tail = "\n".join(output.decode("utf-8", errors="replace").strip().splitlines()[-10:])[-1500:]
            status = "exit status unknown" if code is None else f"exit code {code}"
            session.note = (f"\n[Note: a new sandbox session started for this command and reinstalling the saved "
                            f"packages ({APT_FILE} / {PIP_FILE}) failed ({status}). Last output:\n{tail}]")

    def _start_browser(self, uid: str, remember: bool) -> Session:
        if not BROWSER_SERVER_SCRIPT.is_file():
            raise SessionStartError("The browser is not available on this server (browser_server.py is missing).")
        browser_dir = self._writable(self._browser_dir(uid))
        self._writable(browser_dir / "ctl")
        profile = self._writable(browser_dir / "profile")
        self._writable(browser_dir / "downloads")
        self._sweep_downloads(uid)  # files finished but never reported before the last browser closed
        try:
            write_new_file(browser_dir, "browser_server.py", BROWSER_SERVER_SCRIPT.read_bytes(), 0o644)
        except OSError as e:
            raise SessionStartError(f"The browser could not start: {e}") from e
        stale = [self._socket_path(uid)] + list(profile.glob("Singleton*"))
        for path in stale:  # a killed Chromium leaves its profile locks behind
            try:
                if os.path.lexists(path):
                    os.unlink(path)
            except OSError as e:
                logger.warning("Could not remove %s: %s", path, e)
        name = f"browser-{uid}-{uuid.uuid4().hex[:8]}"
        try:
            container = self.docker.containers.run(
                self.image_ref,
                command=browser_command(remember, SESSION_IDLE_SECONDS),
                name=name,
                # only its own folder: never a path inside the workspace, whose links the sandbox
                # controls and Docker would resolve on the host
                volumes={self._host(SANDBOX_DIR, uid, "browser"): {"bind": BROWSER_MOUNT, "mode": "rw"}},
                security_opt=["no-new-privileges"],
                **self._run_kwargs(BROWSER, uid, BROWSER_MEM_MB),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Could not start browser %s: %s", name, e)
            raise SessionStartError(f"The browser could not start: {e}") from e
        now = self._clock()
        session = Session(uid, BROWSER, container, name, now, now, BROWSER_MEM_MB, remember=remember)
        try:
            self._wait_for_browser(session)
        except Exception as e:  # noqa: BLE001
            logger.error("Browser %s did not come up: %s", name, e)
            self._stop(session, graceful=False)
            raise SessionStartError(f"The browser could not start: {e}") from e
        logger.info("Started browser %s (%s profile)", name, "persistent" if remember else "ephemeral")
        return session

    def _wait_for_browser(self, session: Session) -> None:
        deadline = time.monotonic() + BROWSER_START_TIMEOUT
        next_check = 0.0
        last_error = "no answer"
        while time.monotonic() < deadline:
            try:
                answer = socket_request(self._socket_path(session.user_id),
                                        {"id": "start", "action": "_ping"}, min(3.0, BROWSER_START_TIMEOUT))
                if answer.get("ok"):
                    return
                last_error = str(answer.get("error") or answer)
            except (OSError, ValueError) as e:
                last_error = str(e)
            if time.monotonic() >= next_check:
                next_check = time.monotonic() + 1.0
                status = self._container_status(session)
                if status in ("exited", "dead"):
                    raise RuntimeError(f"the browser server exited: {self._logs_tail(session)}")
            time.sleep(0.2)
        raise RuntimeError(f"the browser server did not answer within {BROWSER_START_TIMEOUT:g} s ({last_error})")

    @staticmethod
    def _container_status(session: Session) -> str | None:
        try:
            session.container.reload()
            return getattr(session.container, "status", None)
        except Exception as e:  # noqa: BLE001
            return "dead" if _container_gone(e) else None

    @staticmethod
    def _logs_tail(session: Session, lines: int = 15) -> str:
        try:
            text = session.container.logs(tail=lines)
            text = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else str(text)
            return text.strip()[-1500:] or "no output"
        except Exception as e:  # noqa: BLE001
            return f"no logs ({e})"

    def _stop(self, session: Session, graceful: bool = True) -> None:
        """Stop and remove a session's container. A browser is first asked to close (it flushes cookies)."""
        if session.kind == BROWSER and graceful:
            try:
                socket_request(self._socket_path(session.user_id),
                               {"id": "stop", "action": "_shutdown"}, BROWSER_CONTROL_TIMEOUT)
                session.container.wait(timeout=10)
            except Exception as e:  # noqa: BLE001 - a dead browser cannot be asked
                logger.debug("Browser %s did not shut down cleanly: %s", session.name, e)
        try:
            session.container.remove(force=True)
            logger.info("Stopped %s session %s", session.kind, session.name)
        except Exception as e:  # noqa: BLE001
            if not _container_gone(e):
                logger.warning("Could not remove container %s: %s", session.name, e)

    def _remove_orphans(self, uid: str, kind: str) -> None:
        """Containers of this user and kind the table does not know (a crashed bot's) go first:
        two browsers on one profile would corrupt it."""
        try:
            found = self.docker.containers.list(all=True, filters={"label": [f"{LABEL_KIND}={kind}", f"{LABEL_USER}={uid}"]})
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not list leftover %s containers of %s: %s", kind, uid, e)
            return
        with self._lock:
            known = {s.name for s in self._sessions.values()}
        for container in found:
            if getattr(container, "name", None) in known:
                continue
            try:
                container.remove(force=True)
                logger.info("Removed leftover container %s", getattr(container, "name", "?"))
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not remove leftover container %s: %s", getattr(container, "name", "?"), e)

    # ---- the shell ---------------------------------------------------------------------------

    def _exec_raw(self, session: Session, command: str, timeout: int) -> tuple[int | None, bytes]:
        """One ``docker exec`` of ``command`` in the session: (exit code, output), the exit code None
        when docker reports none.

        docker-py's ``exec_run`` calls, made one by one so a failure can be placed: errors of
        ``exec_create`` and ``exec_start`` propagate as is (Docker answers 404/409 for a container
        that is gone BEFORE it starts the stream, so such an error means nothing ran); once
        ``exec_start`` returned, a failing ``exec_inspect`` (a 404 after the container was removed
        mid-command: the stream just ends) raises :class:`_ExecLost`."""
        container = session.container
        api = container.client.api
        created = api.exec_create(container.id, exec_argv(command, timeout), stdout=True, stderr=True,
                                  user="runner", workdir=WORKSPACE)
        exec_id = created["Id"]
        output = api.exec_start(exec_id, demux=False)
        try:
            exit_code = (api.exec_inspect(exec_id) or {}).get("ExitCode")
        except Exception as e:  # noqa: BLE001 - placed by the caller
            raise _ExecLost(e) from e
        return (int(exit_code) if exit_code is not None else None), (output or b"")

    def _run(self, user_id: Any, command: str, timeout: Any) -> tuple[str, int | None]:
        """Run ``command`` in the user's shell session: (tool result text, exit code or None).

        A session found gone before the command started is replaced and the command run there
        once; one that went away while it ran (closed by the user, the reaper, a shutdown, or
        died) gives CLOSED_WHILE_RUNNING and never runs it twice."""
        seconds = clamp_timeout(timeout)
        for attempt in range(2):
            session = self._acquire(user_id, SHELL)
            started = time.monotonic()
            try:
                code, output = self._exec_raw(session, command, seconds)
            except _ExecLost as lost:
                if _container_gone(lost.error):
                    logger.warning("Sandbox session %s went away while a command ran (%s)", session.name, lost.error)
                    self._discard(session)
                    return CLOSED_WHILE_RUNNING, None
                logger.error("Command in %s failed after it started: %s", session.name, lost.error)
                return f"Error executing command: {lost.error}", None
            except Exception as e:  # noqa: BLE001
                if _container_gone(e):
                    logger.warning("Sandbox session %s is gone (%s)", session.name, e)
                    self._discard(session)
                    if attempt == 0:
                        continue  # the command never started: run it in a fresh session
                    return "Error: the sandbox session stopped unexpectedly; try again.", None
                logger.error("Command in %s failed: %s", session.name, e)
                return f"Error executing command: {e}", None
            finally:
                self._release(session)
            text = format_exec_result(code, output, seconds, time.monotonic() - started, session.mem_mb)
            if session.note:
                text, session.note = text + session.note, ""
            return text, code
        return "Error: the sandbox session stopped unexpectedly; try again.", None

    def exec(self, user_id: Any, command: str, *, timeout: int = 60) -> str:
        """Run one command in the user's persistent shell session (started when needed).

        The result is the output, or ``Error (exit code N):\\n<output>`` / the timeout text.
        Raises CapacityError when no session can start (ContainerManager then falls back to a
        one-off container)."""
        return self._run(user_id, str(command), timeout)[0]

    def install_package(self, user_id: Any, package: str, manager: str = "apt", *, timeout: int = 300) -> str:
        """Install apt (``sudo apt-get``) or pip (``pip install --user``) packages in the session and,
        when that worked, save them to installed_packages.txt / installed_python_packages.txt."""
        manager = str(manager or "apt").strip().lower()
        try:
            packages = parse_packages(manager, package)
        except ValueError as e:
            return f"Error: {e}"
        uid = check_user_id(user_id)
        text, code = self._run(uid, install_command(manager, packages), timeout)
        if code != 0:
            return text
        added = add_persisted_packages(self._writable(self._user_dir(uid)), manager, packages)
        return f"{text.rstrip()}\n{persisted_note(manager, packages, added)}"

    # ---- the browser -------------------------------------------------------------------------

    def browser_call(self, user_id: Any, request: dict, *, remember: bool = True, timeout: float = 75) -> dict:
        """Send one request to the user's browser server (started when needed) and return its answer.

        CapacityError when no browser can start. A server that is gone answers
        ``{"ok": false, "error": BROWSER_DIED}`` and the session is dropped (the next action starts a
        new browser); one that cannot even be reached is restarted and asked again once, because
        nothing ran. ``_peek`` / ``_status`` / ``_ping`` do not count as activity."""
        uid = check_user_id(user_id)
        request = dict(request or {})
        request.setdefault("id", uuid.uuid4().hex[:12])
        action = str(request.get("action") or "")
        activity = not action.startswith("_") or action == "_touch"
        for attempt in range(2):
            session = self._acquire(uid, BROWSER, remember=bool(remember), touch=activity)
            try:
                answer = socket_request(self._socket_path(uid), request, timeout)
            except _NotDelivered as e:
                logger.warning("Browser %s unreachable (%s)", session.name, e)
                self._discard(session)
                if attempt == 0:
                    continue
                return {"ok": False, "error": BROWSER_DIED}
            except TimeoutError:
                logger.warning("Browser %s did not answer %s within %s s; stopping it", session.name, action, timeout)
                self._discard(session)
                return {"ok": False, "error": f"Error: the browser did not answer within {timeout:g} seconds and was "
                                              "stopped; it will restart on the next action."}
            except (OSError, ValueError) as e:
                logger.warning("Browser %s stopped during %s: %s", session.name, action, e)
                self._discard(session)
                return {"ok": False, "error": BROWSER_DIED}
            finally:
                self._release(session, touch=activity)
            if action == "_shutdown":
                self._discard(session)
            if isinstance(answer, dict) and answer.get("ok"):
                self._collect_downloads(uid, answer)
            return answer
        return {"ok": False, "error": BROWSER_DIED}

    def _collect_downloads(self, uid: str, answer: dict) -> None:
        """Move the downloads an answer reports as completed into the workspace's downloads/ and
        fix their ``path`` (a taken name gets a suffix); one that cannot be moved is reported as
        ``download_failed``. Never raises."""
        content = answer.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "browser_state":
                continue
            for change in block.get("state_changes") or []:
                if not isinstance(change, dict) or change.get("type") != "download_completed":
                    continue
                name = os.path.basename(str(change.get("path") or ""))
                try:
                    final = move_download(self._browser_dir(uid) / "downloads", self._user_dir(uid) / "downloads", name)
                except (OSError, ValueError) as e:
                    logger.warning("Could not move the download %r of %s: %s", name, uid, e)
                    reason = getattr(e, "strerror", None) or str(e)
                    error = f"the download could not be saved: {reason}"[:500]
                    kept = {k: change[k] for k in ("download_id", "url") if k in change}
                    change.clear()
                    change.update({"type": "download_failed", **kept, "error": error})
                    continue
                change["path"] = f"downloads/{final}"

    def _sweep_downloads(self, uid: str) -> None:
        """Move leftovers of an earlier browser (finished, never reported) into the workspace."""
        private = self._browser_dir(uid) / "downloads"
        try:
            names = os.listdir(private)
        except OSError:
            return
        for name in names:
            try:
                move_download(private, self._user_dir(uid) / "downloads", name)
            except (OSError, ValueError) as e:
                logger.info("Dropped the leftover download %r of %s: %s", name, uid, e)

    def _control(self, session: Session, action: str, timeout: float) -> dict | None:
        """A control request to a running browser (no start, no activity); None when it failed.
        An unreachable server is dropped from the table."""
        with self._lock:
            if self._sessions.get((session.user_id, BROWSER)) is not session:
                return None
            session.in_flight += 1
        try:
            return socket_request(self._socket_path(session.user_id), {"id": action.strip("_"), "action": action}, timeout)
        except _NotDelivered:
            self._discard(session)
            return None
        except (OSError, ValueError) as e:
            logger.debug("Browser %s: %s failed: %s", session.name, action, e)
            return None
        finally:
            self._release(session, touch=False)

    def browser_peek(self, user_id: Any) -> dict | None:
        """The current viewport as a JPEG for the chat (``_peek``), or None when no browser runs."""
        session = self._running(check_user_id(user_id), BROWSER)
        if session is None:
            return None
        return self._control(session, "_peek", BROWSER_PEEK_TIMEOUT)

    # ---- state for the UI and the agent --------------------------------------------------------

    def _mem_usage_mb(self, session: Session) -> int | None:
        try:
            try:
                stats = session.container.stats(stream=False, one_shot=True)
            except TypeError:
                stats = session.container.stats(stream=False)
            memory = (stats or {}).get("memory_stats") or {}
            usage = memory.get("usage")
            if usage is None:
                return None
            details = memory.get("stats") or {}
            usage -= details.get("inactive_file", details.get("total_inactive_file", 0)) or 0
            return max(0, int(usage) // (1024 * 1024))
        except Exception:  # noqa: BLE001 - optional detail
            return None

    def _view(self, session: Session | None, now: float) -> dict:
        if session is None:
            return {"running": False, "started": None, "last_used": None, "idle_seconds_left": None,
                    "closes_at": None}
        closes = session.last_used + SESSION_IDLE_SECONDS
        view = {
            "running": True,
            "started": _utc(session.started),
            "last_used": _utc(session.last_used),
            "idle_seconds_left": max(0, int(closes - now)),
            "closes_at": _utc(closes),
            "busy": session.in_flight > 0,
            "mem_limit_mb": session.mem_mb,
        }
        usage = self._mem_usage_mb(session)
        if usage is not None:
            view["mem_mb"] = usage
        return view

    def status(self, user_id: Any) -> dict:
        """The user's sandbox state for /sandbox and the sandbox_status tool (never starts anything)."""
        uid = check_user_id(user_id)
        browser = self._running(uid, BROWSER)
        tabs: list = []
        if browser is not None:
            answer = self._control(browser, "_status", BROWSER_CONTROL_TIMEOUT)
            if answer and answer.get("ok"):
                tabs = [t for t in answer.get("tabs") or [] if isinstance(t, dict)]
        now = self._clock()
        with self._lock:
            shell = self._sessions.get((uid, SHELL))
            browser = self._sessions.get((uid, BROWSER))
            warned = uid in self._warned
        browser_view = self._view(browser, now)
        browser_view["tabs"] = tabs if browser is not None else []
        if browser is not None:
            browser_view["remember"] = browser.remember
        profile = self._browser_dir(uid) / "profile"
        return {
            "shell": self._view(shell, now),
            "browser": browser_view,
            "profile_bytes": _dir_size(profile),  # 0 for a link: the browser can plant one there
            "packages": persisted_packages(self._user_dir(uid)),
            "idle_timeout_seconds": SESSION_IDLE_SECONDS,
            "warn_before_seconds": SESSION_WARN_BEFORE_SECONDS,
            "warned": warned,
        }

    def server_status(self) -> dict:
        with self._lock:
            shells = sum(1 for (_u, k) in self._sessions if k == SHELL)
            browsers = sum(1 for (_u, k) in self._sessions if k == BROWSER)
        return {
            "shells": shells,
            "max_shells": self.max_shells(),
            "browsers": browsers,
            "max_browsers": self.max_browsers(),
            "mem_available_mb": mem_available_mb(),
            "mem_total_mb": mem_total_mb(),
            "reserve_mb": MEMORY_RESERVE_MB,
            "egress_filter": self.egress_state,
        }

    # ---- closing ---------------------------------------------------------------------------

    def close(self, user_id: Any, kind: str | None = None) -> bool:
        """Stop the user's session of ``kind`` (both when None); True when something was running.
        Files and the browser profile stay."""
        uid = check_user_id(user_id)
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown session kind {kind!r}")
        stopped = False
        for k in ([kind] if kind else list(KINDS)):
            with self._user_lock(uid, k):
                with self._lock:
                    session = self._sessions.pop((uid, k), None)
                if session is not None:
                    self._stop(session)
                    stopped = True
        self._forget_warning_if_idle(uid)
        return stopped

    def forget_browser_profile(self, user_id: Any) -> bool:
        """Close the browser and delete its profile (cookies, logins). True when a profile was deleted."""
        uid = check_user_id(user_id)
        with self._user_lock(uid, BROWSER):
            with self._lock:
                session = self._sessions.pop((uid, BROWSER), None)
            if session is not None:
                self._stop(session)
            profile = self._browser_dir(uid) / "profile"
            if not os.path.lexists(profile):
                deleted = False
            else:
                try:
                    if profile.is_symlink() or not profile.is_dir():
                        profile.unlink()
                    else:
                        shutil.rmtree(profile)
                    deleted = True
                except OSError as e:
                    logger.error("Could not delete the browser profile of %s: %s", uid, e)
                    deleted = False
        self._forget_warning_if_idle(uid)
        return deleted

    def _forget_warning_if_idle(self, uid: str) -> None:
        with self._lock:
            if not any(u == uid for (u, _k) in self._sessions):
                self._warned.pop(uid, None)

    def _recheck_egress(self) -> list[Session]:
        """Re-apply the egress rules when EGRESS_RECHECK_SECONDS passed since setup or the last check
        (idempotent: ``-C`` first, ``-I`` only for a missing rule) and update ``egress_state``. In
        required mode a failed check stops every session; returns the sessions stopped."""
        now = self._clock()
        if (self._egress_checked is None or EGRESS_FILTER == "off"
                or now - self._egress_checked < EGRESS_RECHECK_SECONDS):
            return []
        self._egress_checked = now
        before = self.egress_state
        try:
            state = apply_egress_filter(self.subnet, EGRESS_FILTER, self._egress_runner, recheck=True)
        except Exception as e:  # noqa: BLE001 - a custom runner may raise
            logger.warning("Sandbox egress filter check failed: %s", e)
            state = "failed"
        self.egress_state = state
        if state == "on":
            if before != "on":
                logger.info("Sandbox egress filter active again for %s", self.subnet)
            return []
        if EGRESS_FILTER == "required":
            with self._lock:
                running = bool(self._sessions)
            if running or before == "on":
                logger.error("Sandbox egress filter is no longer active (its iptables rules could not be checked or "
                             "inserted again); SANDBOX_EGRESS_FILTER=required: stopping every sandbox session")
            return self._stop_all()
        if before == "on":
            logger.warning("Sandbox egress filter is no longer active (its iptables rules could not be checked or "
                           "inserted again): sandboxes can reach private networks and the host")
        return []

    def reap_idle(self) -> list[tuple[str, str]]:
        """Stop sessions unused for SESSION_IDLE_SECONDS with nothing in flight, and drop sessions
        whose container died by itself. Every EGRESS_RECHECK_SECONDS it first re-applies the
        egress filter (``_recheck_egress``; required mode: a failed check stops every session).
        Returns the (user_id, kind) pairs that were stopped."""
        rechecked = [(s.user_id, s.kind) for s in self._recheck_egress()]
        now = self._clock()
        idle: list[tuple[Session, threading.Lock | None]] = []
        with self._lock:
            for key, session in list(self._sessions.items()):
                if session.in_flight or now - session.last_used < SESSION_IDLE_SECONDS:
                    continue
                lock = self._user_locks.get(key)
                if lock is not None and not lock.acquire(blocking=False):
                    continue
                del self._sessions[key]
                idle.append((session, lock))
            others = [s for s in self._sessions.values() if not s.in_flight]
        stopped: list[tuple[str, str]] = rechecked
        for session, lock in idle:
            try:
                logger.info("Closing the idle %s session %s", session.kind, session.name)
                self._stop(session)
            finally:
                if lock is not None:
                    lock.release()
            stopped.append((session.user_id, session.kind))
        for session in others:
            if self._container_status(session) in ("exited", "dead"):
                logger.info("The %s session %s ended by itself", session.kind, session.name)
                with self._lock:
                    if self._sessions.get((session.user_id, session.kind)) is not session or session.in_flight:
                        continue
                    del self._sessions[(session.user_id, session.kind)]
                self._stop(session, graceful=False)
                stopped.append((session.user_id, session.kind))
        for uid in {u for u, _k in stopped}:
            self._forget_warning_if_idle(uid)
        return stopped

    def due_warnings(self) -> list[dict]:
        """Users to warn that their sandbox closes soon, each once per idle period.

        A user is due when their last activity across both kinds is at least idle − warn_before
        seconds ago, a session of theirs runs with nothing in flight, and they were not warned since
        that activity. Returns ``{"user_id", "kinds", "closes_at"}`` (aware UTC) and marks them
        warned; any activity (or :meth:`extend`) clears the mark."""
        warn_before = min(SESSION_WARN_BEFORE_SECONDS, SESSION_IDLE_SECONDS)
        if warn_before <= 0:
            return []
        now = self._clock()
        due: list[dict] = []
        with self._lock:
            by_user: dict[str, list[Session]] = {}
            for (uid, _kind), session in self._sessions.items():
                by_user.setdefault(uid, []).append(session)
            for uid, sessions in by_user.items():
                if uid in self._warned or any(s.in_flight for s in sessions):
                    continue
                last = max(s.last_used for s in sessions)
                if now - last < SESSION_IDLE_SECONDS - warn_before:
                    continue
                self._warned[uid] = now
                kinds = [k for k in KINDS if any(s.kind == k for s in sessions)]
                due.append({"user_id": uid, "kinds": kinds, "closes_at": _utc(last + SESSION_IDLE_SECONDS)})
        return due

    def extend(self, user_id: Any) -> datetime | None:
        """The user's answer to a warning: counts as activity for every running session (the browser
        server's own idle timer too, via ``_touch``). The new closing time, or None when nothing runs."""
        uid = check_user_id(user_id)
        with self._lock:
            sessions = [s for (u, _k), s in self._sessions.items() if u == uid]
            for session in sessions:
                self._touch_locked(session)
            browser = self._sessions.get((uid, BROWSER))
        if not sessions:
            return None
        if browser is not None:
            self._control(browser, "_touch", BROWSER_CONTROL_TIMEOUT)
        with self._lock:
            remaining = [s for (u, _k), s in self._sessions.items() if u == uid]
            if not remaining:
                return None
            return _utc(max(s.last_used for s in remaining) + SESSION_IDLE_SECONDS)

    def cleanup_stale(self) -> int:
        """Remove labelled sandbox containers this manager does not track (left by an earlier run)."""
        try:
            found = self.docker.containers.list(all=True, filters={"label": LABEL_KIND})
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not list leftover sandbox containers: %s", e)
            return 0
        with self._lock:
            known = {s.name for s in self._sessions.values()}
        removed = 0
        for container in found:
            name = getattr(container, "name", "?")
            if name in known:
                continue
            labels = getattr(container, "labels", None) or {}
            uid = labels.get(LABEL_USER)
            if labels.get(LABEL_KIND) == BROWSER and uid:
                try:  # let Chromium write its cookies before the container goes
                    socket_request(self._socket_path(check_user_id(uid)), {"id": "stale", "action": "_shutdown"}, 2.0)
                    container.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            try:
                container.remove(force=True)
                removed += 1
                logger.info("Removed leftover sandbox container %s", name)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not remove leftover sandbox container %s: %s", name, e)
        return removed

    def shutdown_all(self) -> None:
        """Stop every session (bot shutdown); no new session starts afterwards (a start already in
        flight is stopped by ``_acquire`` when it comes up)."""
        with self._lock:
            self._closed = True
        self._stop_all()
        self.cleanup_stale()

    def _stop_all(self) -> list[Session]:
        """Take every session out of the table and stop it, browsers asked to save their cookies
        first (all at once). Returns the sessions stopped."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._warned.clear()
        browsers = [s for s in sessions if s.kind == BROWSER]
        for session in browsers:  # ask all browsers first, then wait for them together
            try:
                socket_request(self._socket_path(session.user_id), {"id": "stop", "action": "_shutdown"}, 2.0)
            except Exception:  # noqa: BLE001
                pass
        deadline = time.monotonic() + 10
        for session in browsers:
            try:
                session.container.wait(timeout=max(0.1, deadline - time.monotonic()))
            except Exception:  # noqa: BLE001
                pass
        for session in sessions:
            self._stop(session, graceful=False)
        return sessions


# --------------------------------------------------------------------------- process-wide instance


_manager: SessionManager | None = None
_manager_lock = threading.Lock()


def init_manager(docker_client: Any, image_ref: str, host_data_path: Any, base_data_path: Any = "data") -> SessionManager:
    """The process-wide SessionManager, created (network + egress filter) on the first call.

    Idempotent: ContainerManager is constructed more than once at startup and every instance gets
    the same manager. Raises when the sandbox network cannot be set up (sessions stay off)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            manager = SessionManager(docker_client, image_ref, host_data_path, base_data_path)
            manager.setup()
            _manager = manager
            logger.info("Sandbox sessions ready: network %s (%s), egress filter %s, up to %s shells and %s browsers, "
                        "idle timeout %s s", manager.network_name, manager.subnet, manager.egress_state,
                        manager.max_shells(), manager.max_browsers(), SESSION_IDLE_SECONDS)
        return _manager


def get_manager() -> SessionManager | None:
    """The process-wide manager, or None when it was never initialised (tests, Docker down)."""
    return _manager


def reset_manager() -> None:
    """Forget the process-wide manager (tests)."""
    global _manager
    with _manager_lock:
        _manager = None
