"""MCP server configuration and the cached tool lists.

Two sources of servers, both read by ``agents.mcp_tools.MCPTools`` at turn start without any network:

* the user's own servers in ``data/<uid>/mcp.json``::

      {"servers": [{"name": "github", "url": "https://.../mcp", "transport": "http", "headers": {"Authorization": "Bearer ..."},
                    "enabled": true, "tools": [{"name", "description", "input_schema"}], "updated": "<iso>", "error": null}]}

* the admin's servers, offered to every user and read-only in ``/mcp``: the same shape in the file named
  by ``MCP_SERVERS_FILE`` (default ``data/mcp_servers.json``), hand-edited and never rewritten by the
  bot. Their tool cache lives next to it in ``data/mcp_servers.cache.json`` (``{"servers": [{"name",
  "tools", "updated", "error"}]}``), filled at startup and by an admin's ``/mcp refresh``. Records
  returned from here carry ``"admin": True``.

Validation: a name is ``^[a-z0-9][a-z0-9_-]{0,31}$`` (lower-cased first); a user's URL must be
``https://`` to a public host (no localhost, no loopback/private/link-local/reserved literal IP; a
DNS name is resolved and checked by ``check_public_host`` before every connection); the admin may
use ``http://`` and local or private hosts. A token is kept as
``headers["Authorization"] = "Bearer <token>"`` and never rendered back: ``describe_headers`` says
"token set", ``display_url`` drops user info, query and fragment.

OAuth servers (``"auth": "oauth"``, signed in through ``bot.oauth_flow``) keep their sign-in in the user's
file, encrypted with ``token_cipher()`` (Fernet; the key is ``MCP_TOKEN_KEY``, else a generated
``data/.mcp_token_key``, plus the decrypt-only ``MCP_TOKEN_KEY_OLD`` keys):

* a user's server: ``record["oauth"] = {"tokens": "<fernet>", "client": "<fernet>", "account": null,
  "updated": "<iso>"[, "needs_signin": true]}``;
* an admin server (each user signs in with their own account): the same entry in the user's file under
  the top-level ``"oauth_admin": {"<server name>": {...}}``.

Each encrypted value is bound to the user id, the server name, its URL and whether it is an admin
server (they are inside the plaintext and checked on decryption): ``data/<uid>/mcp.json`` is in the
user's sandbox workspace, and a value moved to another record (say, one pointing at a server the
sandbox controls), a changed URL, a tampered or undecryptable value (key lost or changed) all read as
"signed out", never as an error. ``read_oauth`` / ``write_oauth`` / ``clear_oauth`` / ``seal_oauth`` work
on decrypted dicts; nothing here renders a token: ``describe_headers`` says "🔐 signed in" or
"🔐 sign in needed" (``all_servers(uid, with_state=True)`` puts the per-user ``oauth_state`` on the
records; it is never stored).

All functions are synchronous file I/O (callers use ``asyncio.to_thread``): one ``threading.Lock``
per user and one for the admin cache, atomic writes (tmp + ``os.replace``). A missing or corrupt
file is an empty server list, logged, never an error.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import stat
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bot.config import config

logger = logging.getLogger(__name__)

USER_FILE = "mcp.json"
DEFAULT_ADMIN_FILE = "mcp_servers.json"
ENV_ADMIN_FILE = "MCP_SERVERS_FILE"
TRANSPORTS = ("http", "sse")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
NOT_PUBLIC = "The server must be on a public address (not localhost or a private network)."
MAX_ERROR_CHARS = 300
AUTH_OAUTH = "oauth"
SIGN_IN_NEEDED = "sign in needed"  # the record's error (and the admin cache's) while a sign-in is missing
STATE_SIGNED_IN = "signed_in"
STATE_SIGN_IN_NEEDED = "signin_needed"
RUNTIME_KEYS = ("admin", "oauth_state")  # put on loaded records, never stored
ENV_TOKEN_KEY = "MCP_TOKEN_KEY"
ENV_TOKEN_KEY_OLD = "MCP_TOKEN_KEY_OLD"
KEY_FILE = ".mcp_token_key"
OAUTH_FIELDS = ("tokens", "client")

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_admin_lock = threading.Lock()


# ---- paths ---------------------------------------------------------------------
def user_path(user_id: str) -> Path:
    return Path(config.data_dir) / str(user_id) / USER_FILE


def admin_path() -> Path:
    """The hand-edited admin file; the env var is read at call time so a test can point elsewhere."""
    return Path(os.getenv(ENV_ADMIN_FILE) or os.path.join(config.data_dir, DEFAULT_ADMIN_FILE))


def admin_cache_path() -> Path:
    """``mcp_servers.json`` → ``mcp_servers.cache.json``, next to the admin file."""
    path = admin_path()
    return path.with_name(f"{path.stem}.cache.json")


def _lock(user_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(user_id), threading.Lock())


# ---- validation -------------------------------------------------------------------
def validate_name(name: str) -> str:
    """The lower-cased name, or ValueError with a message fit for the user."""
    name = str(name or "").strip().lower()
    if not NAME_RE.match(name):
        raise ValueError("The name must be 1-32 characters: lowercase letters, digits, '-' or '_', starting with a letter or digit.")
    return name


def _is_public_address(address: str) -> bool:
    """True for a globally routable unicast IP; False for loopback, private, link-local, reserved,
    shared, unspecified and multicast ones (an IPv4-mapped IPv6 address is judged as IPv4)."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_global and not ip.is_multicast


def _is_local_host(host: str) -> bool:
    """A name or literal IP that is not public: localhost and friends, or a non-public address."""
    host = host.rstrip(".")
    if host in LOCAL_HOSTS or host.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # a DNS name: check_public_host resolves it before every connection
    return not _is_public_address(host)


def validate_url(url: str, *, admin: bool = False) -> str:
    """The URL as given, or ValueError. Users need https:// to a public host; the admin may use
    http:// and local or private hosts (the bot runs on the host network, so "localhost" is the
    server itself: a user must not reach its internal services through an MCP server)."""
    url = str(url or "").strip()
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https") or not host:
        raise ValueError("The URL must start with https:// and name a host.")
    if admin:
        return url
    if parts.scheme != "https":
        raise ValueError("Only https:// URLs are accepted.")
    if _is_local_host(host):
        raise ValueError(NOT_PUBLIC)
    return url


def check_public_host(url: str) -> None:
    """ValueError when the URL's host is, or resolves to, an address that is not public.

    Blocking (DNS): call in a worker thread. ``agents.mcp_tools`` runs it before every connection
    to a user's server, so a name that resolves (or later re-resolves) to a local address and a
    record written straight into ``mcp.json`` are refused too. A name that does not resolve
    passes: the connection fails on its own.
    """
    parts = urlsplit(str(url or ""))
    host = (parts.hostname or "").lower()
    if not host or _is_local_host(host):
        raise ValueError(NOT_PUBLIC)
    try:
        port = parts.port or (80 if parts.scheme == "http" else 443)
    except ValueError:
        port = 443
    try:
        addresses = _resolve(host, port)
    except (OSError, UnicodeError):
        return
    if any(not _is_public_address(address) for address in addresses):
        raise ValueError(NOT_PUBLIC)


def _resolve(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (tests replace this). Blocking."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def validate_transport(transport: str | None) -> str:
    transport = str(transport or "http").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError("The transport must be 'http' (streamable HTTP, the default) or 'sse'.")
    return transport


def new_server(name: str, url: str, token: str | None = None, transport: str | None = "http", *, admin: bool = False) -> dict:
    """A validated server record without tools yet; the token becomes a bearer Authorization header."""
    headers: dict[str, str] = {}
    token = str(token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return {
        "name": validate_name(name),
        "url": validate_url(url, admin=admin),
        "transport": validate_transport(transport),
        "headers": headers,
        "enabled": True,
        "tools": [],
        "updated": None,
        "error": None,
    }


def normalize(raw: Any, *, admin: bool = False) -> dict | None:
    """A record with every key present, or None when the entry has no usable name or URL.

    An invalid name, URL or transport is logged and the entry skipped, everything else gets its
    default. The https policy applied when a server is added (``new_server``); a stored URL only has
    to be an http(s) URL with a host, so the admin's own plain-http servers survive a reload."""
    if not isinstance(raw, dict):
        return None
    try:
        name = validate_name(raw.get("name", ""))
        url = validate_url(raw.get("url", ""), admin=True)
        transport = validate_transport(raw.get("transport"))
    except ValueError as e:
        logger.warning("Skipping MCP server entry %r: %s", raw.get("name"), e)
        return None
    headers = raw.get("headers")
    tools = raw.get("tools")
    record = {
        "name": name,
        "url": url,
        "transport": transport,
        "headers": {str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {},
        "enabled": bool(raw.get("enabled", True)),
        "tools": [t for t in tools if isinstance(t, dict) and t.get("name")] if isinstance(tools, list) else [],
        "updated": raw.get("updated") or None,
        "error": (str(raw["error"])[:MAX_ERROR_CHARS] if raw.get("error") else None),
    }
    if raw.get("auth") == AUTH_OAUTH:
        record["auth"] = AUTH_OAUTH
        entry = _clean_entry(raw.get("oauth"))
        if entry is not None and not admin:  # an admin server's sign-ins live in each user's file
            record["oauth"] = entry
    if admin:
        record["admin"] = True
    return record


def _clean_entry(raw: Any) -> dict | None:
    """A stored sign-in entry with only its known keys and value types, or None: the file is in the
    user's sandbox workspace, so anything odd is dropped instead of trusted."""
    if not isinstance(raw, dict):
        return None
    entry: dict[str, Any] = {}
    for key in OAUTH_FIELDS:
        value = raw.get(key)
        entry[key] = value if isinstance(value, str) and value else None
    account = raw.get("account")
    entry["account"] = str(account)[:200] if isinstance(account, str) and account else None
    entry["updated"] = str(raw["updated"])[:40] if isinstance(raw.get("updated"), str) else None
    if raw.get("needs_signin") is True:
        entry["needs_signin"] = True
    return entry


# ---- file I/O ---------------------------------------------------------------------
def _read_json(path: Path, *, no_follow: bool = False) -> Any:
    """The parsed file, or None when it is missing or broken (logged). ``no_follow``: a link or a
    non-regular file reads as missing (the user's file lives in the sandbox workspace)."""
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if no_follow:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise OSError("not a regular file")
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 - a broken file must not break every turn
        logger.error("Could not read MCP servers from %s: %s", path, e)
        return None


def _servers_of(data: Any, *, admin: bool = False) -> list[dict]:
    entries = data.get("servers") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    servers: list[dict] = []
    seen: set[str] = set()
    for raw in entries:
        record = normalize(raw, admin=admin)
        if record is not None and record["name"] not in seen:
            seen.add(record["name"])
            servers.append(record)
    return servers


def _read_servers(path: Path, *, admin: bool = False) -> list[dict]:
    return _servers_of(_read_json(path), admin=admin)


def _read_user_file(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """The user's servers and their sign-ins to admin servers (``oauth_admin``)."""
    data = _read_json(path, no_follow=True)
    raw = data.get("oauth_admin") if isinstance(data, dict) else None
    oauth_admin: dict[str, dict] = {}
    if isinstance(raw, dict):
        for name, entry in raw.items():
            cleaned = _clean_entry(entry)
            if isinstance(name, str) and NAME_RE.match(name) and cleaned is not None:
                oauth_admin[name] = cleaned
    return _servers_of(data), oauth_admin


def _write(path: Path, data: dict) -> None:
    """Atomic: a fresh temp file (``mkstemp``: O_EXCL, never through a link planted under its name,
    mode 0600) beside the target, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _storable(server: dict) -> dict:
    return {k: v for k, v in server.items() if k not in RUNTIME_KEYS}


def _write_user_file(path: Path, servers: list[dict], oauth_admin: dict[str, dict]) -> None:
    data: dict[str, Any] = {"servers": [_storable(s) for s in servers]}
    if oauth_admin:
        data["oauth_admin"] = oauth_admin
    _write(path, data)


def load_user(user_id: str) -> list[dict]:
    with _lock(user_id):
        return _read_user_file(user_path(user_id))[0]


def save_user(user_id: str, servers: list[dict]) -> None:
    with _lock(user_id):
        path = user_path(user_id)
        _, oauth_admin = _read_user_file(path)
        _write_user_file(path, servers, oauth_admin)


def load_admin() -> list[dict]:
    """The admin servers with their cached tools merged in by name (``admin: True`` on each); the
    cache's ``auth`` ("oauth", found by a discovery) applies when the file names none."""
    with _admin_lock:
        servers = _read_servers(admin_path(), admin=True)
        if not servers:
            return []
        cache = _read_cache(admin_cache_path())
    for server in servers:
        cached = cache.get(server["name"])
        if cached:
            server["tools"] = cached.get("tools") or []
            server["updated"] = cached.get("updated")
            server["error"] = cached.get("error")
            if cached.get("auth") == AUTH_OAUTH:
                server["auth"] = AUTH_OAUTH
    return servers


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("servers") if isinstance(data, dict) else []
        return {str(e["name"]): e for e in entries if isinstance(e, dict) and e.get("name")}
    except Exception as e:  # noqa: BLE001
        logger.error("Could not read the MCP tool cache %s: %s", path, e)
        return {}


def save_admin_cache(servers: list[dict]) -> None:
    """Only the discovered state of the admin servers; the admin file itself stays hand-edited."""
    entries = []
    for s in servers:
        entry = {"name": s["name"], "tools": s.get("tools") or [], "updated": s.get("updated"), "error": s.get("error")}
        if s.get("auth") == AUTH_OAUTH:
            entry["auth"] = AUTH_OAUTH
        entries.append(entry)
    with _admin_lock:
        _write(admin_cache_path(), {"servers": entries})


def all_servers(user_id: str, *, with_state: bool = False) -> list[dict]:
    """What the agent sees: the user's servers, then the admin's. An admin server the user has
    signed in to counts as an OAuth server. ``with_state``: every OAuth record gets ``oauth_state``
    (``signed_in`` / ``signin_needed``, decided by decrypting the stored tokens) for ``/mcp``."""
    with _lock(user_id):
        servers, oauth_admin = _read_user_file(user_path(user_id))
    admin = load_admin()
    for server in admin:
        if server["name"] in oauth_admin:
            server["auth"] = AUTH_OAUTH
    if with_state:
        for server in servers:
            if server.get("auth") == AUTH_OAUTH:
                server["oauth_state"] = _state_of(user_id, server, server.get("oauth"))
        for server in admin:
            if server.get("auth") == AUTH_OAUTH:
                server["oauth_state"] = _state_of(user_id, server, oauth_admin.get(server["name"]))
    return servers + admin


# ---- helpers on a list -------------------------------------------------------------
def find(servers: list[dict], name: str) -> dict | None:
    name = str(name or "").strip().lower()
    return next((s for s in servers if s.get("name") == name), None)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def with_tools(server: dict, tools: list[dict] | None, error: str | None = None) -> dict:
    """The record after a discovery: tools replaced on success, kept on failure; error and time set."""
    updated = dict(server)
    if error is None:
        updated["tools"] = list(tools or [])
    updated["error"] = str(error)[:MAX_ERROR_CHARS] if error else None
    updated["updated"] = now_iso()
    return updated


# ---- mutations of the user's file ----------------------------------------------------
def add_user_server(user_id: str, server: dict) -> dict:
    """Insert, or replace the server of the same name (re-adding is how a token is changed; the old
    record's sign-in goes with it)."""
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        record = _storable(server)
        for i, existing in enumerate(servers):
            if existing["name"] == record["name"]:
                servers[i] = record
                break
        else:
            servers.append(record)
        _write_user_file(path, servers, oauth_admin)
    return record


def remove_user_server(user_id: str, name: str) -> bool:
    """Forget the server and, with its record, its stored sign-in."""
    name = str(name or "").strip().lower()
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        kept = [s for s in servers if s["name"] != name]
        if len(kept) == len(servers):
            return False
        _write_user_file(path, kept, oauth_admin)
    return True


def toggle_user_server(user_id: str, name: str) -> bool | None:
    """Flip ``enabled``; the new state, or None when there is no such server."""
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        server = find(servers, name)
        if server is None:
            return None
        server["enabled"] = not server.get("enabled", True)
        _write_user_file(path, servers, oauth_admin)
        return server["enabled"]


def update_user_tools(user_id: str, name: str, tools: list[dict] | None, error: str | None = None,
                      *, auth: str | None = None) -> dict | None:
    """Store a discovery result (``with_tools``); the updated record, or None when the server is gone.
    ``auth="oauth"`` marks a server whose discovery asked for a sign-in."""
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        server = find(servers, name)
        if server is None:
            return None
        server.update(with_tools(server, tools, error))
        if auth == AUTH_OAUTH:
            server["auth"] = AUTH_OAUTH
        _write_user_file(path, servers, oauth_admin)
        return dict(server)


def record_error(user_id: str, server: dict, error: str | None) -> None:
    """Remember the last error of a server after a tool call (None clears it). Best effort: a failure
    to write is logged, the tool result is already on its way."""
    try:
        error = str(error)[:MAX_ERROR_CHARS] if error else None
        if server.get("admin"):
            with _admin_lock:
                cache = _read_cache(admin_cache_path())
                entry = cache.get(server["name"]) or {"name": server["name"], "tools": server.get("tools") or [], "updated": server.get("updated")}
                if entry.get("error") == error:
                    return
                entry["error"] = error
                cache[server["name"]] = entry
                _write(admin_cache_path(), {"servers": list(cache.values())})
            return
        with _lock(user_id):
            path = user_path(user_id)
            servers, oauth_admin = _read_user_file(path)
            stored = find(servers, server.get("name", ""))
            if stored is None or stored.get("error") == error:
                return
            stored["error"] = error
            _write_user_file(path, servers, oauth_admin)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record the MCP error of %s for %s: %s", server.get("name"), user_id, e)


# ---- the encryption key ------------------------------------------------------------------
_cipher: Any = None
_cipher_lock = threading.Lock()
_warned: set[tuple] = set()


def key_path() -> Path:
    return Path(config.data_dir) / KEY_FILE


def token_cipher():
    """The ``MultiFernet`` for stored sign-ins, built once per process: ``MCP_TOKEN_KEY`` encrypts,
    the key file (when one exists besides the env key) and ``MCP_TOKEN_KEY_OLD`` (comma-separated)
    only decrypt, so sign-ins sealed under them stay readable and move to the new key on their next
    write. Without ``MCP_TOKEN_KEY`` the key is ``data/.mcp_token_key``, generated once (0600, O_EXCL,
    atomic) with a WARNING. The key is never logged. Blocking file I/O on first use: called from the
    store functions, which run in worker threads."""
    global _cipher
    with _cipher_lock:
        if _cipher is None:
            _cipher = _load_cipher()
        return _cipher


def reset_cipher() -> None:
    """Forget the cached key (tests, after changing the environment)."""
    global _cipher
    with _cipher_lock:
        _cipher = None
    _warned.clear()


def _fernet(key: str | bytes, label: str):
    from cryptography.fernet import Fernet

    try:
        return Fernet(key.encode("ascii") if isinstance(key, str) else key)
    except (ValueError, TypeError, UnicodeError):
        logger.error("%s is not a valid Fernet key (the urlsafe base64 of 32 bytes); it is ignored", label)
        return None


def _load_cipher():
    from cryptography.fernet import Fernet, MultiFernet

    keys = []
    env_key = (os.getenv(ENV_TOKEN_KEY) or "").strip()
    file_key = _read_key_file()
    primary = _fernet(env_key, ENV_TOKEN_KEY) if env_key else None
    if primary is not None:
        keys.append(primary)
        if file_key and file_key != env_key:
            extra = _fernet(file_key, str(key_path()))
            if extra is not None:
                keys.append(extra)  # sign-ins sealed before MCP_TOKEN_KEY was set
    else:
        if file_key is None:
            file_key = _create_key_file()
            if file_key is not None:
                logger.warning("%s is not set: generated the key for stored MCP sign-ins in %s. Set %s (a secret "
                               "that survives deploys) to that file's content, or a new key, to keep sign-ins "
                               "when data/ is lost.", ENV_TOKEN_KEY, key_path(), ENV_TOKEN_KEY)
        else:
            logger.warning("%s is not set: stored MCP sign-ins use the key in %s", ENV_TOKEN_KEY, key_path())
        key = _fernet(file_key, str(key_path())) if file_key else None
        if key is None:
            logger.error("No usable key for stored MCP sign-ins: using a temporary one, sign-ins last until the restart")
            key = Fernet(Fernet.generate_key())
        keys.append(key)
    for i, old in enumerate(k.strip() for k in (os.getenv(ENV_TOKEN_KEY_OLD) or "").split(",")):
        if old:
            key = _fernet(old, f"{ENV_TOKEN_KEY_OLD} #{i + 1}")
            if key is not None:
                keys.append(key)
    return MultiFernet(keys)


def _read_key_file() -> str | None:
    path = key_path()
    try:
        return path.read_text(encoding="ascii").strip() or None
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as e:
        logger.error("Could not read %s: %s", path, e)
        return None


def _create_key_file() -> str | None:
    """Generate the key and store it at ``key_path()``: a temp file created with O_EXCL and mode 0600,
    linked into place (atomic; refuses an existing file, whose key then wins)."""
    from cryptography.fernet import Fernet

    path = key_path()
    key = Fernet.generate_key()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(key + b"\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                return _read_key_file()
        finally:
            os.unlink(tmp)
    except OSError as e:
        logger.error("Could not store a generated key in %s: %s", path, e)
        return None
    return key.decode("ascii")


# ---- stored sign-ins (OAuth) -------------------------------------------------------------
def _binding(user_id: str, server: dict) -> dict:
    return {"user": str(user_id), "server": str(server.get("name") or ""), "url": str(server.get("url") or ""),
            "admin": bool(server.get("admin"))}


def _seal(user_id: str, server: dict, kind: str, data: Any) -> str:
    payload = {"v": 1, "kind": kind, **_binding(user_id, server), "data": data}
    return token_cipher().encrypt(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")


def _warn_once(key: tuple, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _unseal(user_id: str, server: dict, kind: str, value: Any) -> Any:
    """The data sealed for exactly this user, server, URL and kind, or None (signed out): missing,
    undecryptable (key lost or changed, tampered) or sealed for something else."""
    if not isinstance(value, str) or not value:
        return None
    from cryptography.fernet import InvalidToken

    try:
        payload = json.loads(token_cipher().decrypt(value.encode("ascii")))
    except (InvalidToken, ValueError, TypeError, UnicodeError):
        _warn_once((str(user_id), server.get("name"), kind, "decrypt"),
                   "The stored MCP sign-in (%s) of %s for %s cannot be decrypted (key changed or value tampered); "
                   "treated as signed out", kind, user_id, server.get("name"))
        return None
    if not isinstance(payload, dict) or payload.get("kind") != kind or any(
            payload.get(k) != v for k, v in _binding(user_id, server).items()):
        _warn_once((str(user_id), server.get("name"), kind, "binding"),
                   "The stored MCP sign-in (%s) of %s for %s belongs to another server or URL; treated as signed out",
                   kind, user_id, server.get("name"))
        return None
    return payload.get("data")


def _rotate(value: Any) -> Any:
    """``value`` re-encrypted under the current key when an older key sealed it (else unchanged)."""
    if not isinstance(value, str) or not value:
        return value
    from cryptography.fernet import InvalidToken

    try:
        return token_cipher().rotate(value.encode("ascii")).decode("ascii")
    except (InvalidToken, ValueError, TypeError, UnicodeError):
        return value


def _open_entry(user_id: str, server: dict, entry: dict | None) -> dict:
    out: dict[str, Any] = {"tokens": None, "expires_at": None, "metadata": None, "client": None,
                           "needs_signin": bool(entry and entry.get("needs_signin")),
                           "account": (entry or {}).get("account"), "updated": (entry or {}).get("updated")}
    if not entry:
        return out
    sealed = _unseal(user_id, server, "tokens", entry.get("tokens"))
    if isinstance(sealed, dict) and isinstance(sealed.get("tokens"), dict) and sealed["tokens"].get("access_token") is not None:
        out["tokens"] = sealed["tokens"]
        expires_at = sealed.get("expires_at")
        out["expires_at"] = float(expires_at) if isinstance(expires_at, (int, float)) else None
        out["metadata"] = sealed.get("metadata") if isinstance(sealed.get("metadata"), dict) else None
    client = _unseal(user_id, server, "client", entry.get("client"))
    if isinstance(client, dict) and client.get("client_id"):
        out["client"] = client
    return out


def _state_of(user_id: str, server: dict, entry: dict | None) -> str:
    if not entry or entry.get("needs_signin") or (server.get("error") == SIGN_IN_NEEDED and not server.get("admin")):
        return STATE_SIGN_IN_NEEDED
    return STATE_SIGNED_IN if _open_entry(user_id, server, entry)["tokens"] else STATE_SIGN_IN_NEEDED


def _locate(user_id: str, server: dict, servers: list[dict], oauth_admin: dict[str, dict]) -> tuple[dict | None, dict | None]:
    """(the stored record or None for an admin server, the server dict to bind against). A user's
    server binds against its record as stored: a URL changed in the file no longer matches."""
    if server.get("admin"):
        return None, server
    stored = find(servers, server.get("name", ""))
    return stored, stored


def read_oauth(user_id: str, server: dict) -> dict:
    """The decrypted sign-in of ``user_id`` to ``server``: ``{"tokens": dict|None, "expires_at": float|None,
    "metadata": dict|None, "client": dict|None, "needs_signin": bool, "account", "updated"}``. Never raises."""
    try:
        with _lock(user_id):
            servers, oauth_admin = _read_user_file(user_path(user_id))
        stored, bind = _locate(user_id, server, servers, oauth_admin)
        if bind is None:
            return _open_entry(user_id, server, None)
        entry = oauth_admin.get(server["name"]) if server.get("admin") else (stored or {}).get("oauth")
        return _open_entry(user_id, bind, entry)
    except Exception as e:  # noqa: BLE001 - signed out, never an error for the user
        logger.warning("Could not read the MCP sign-in of %s for %s: %s", user_id, server.get("name"), e)
        return _open_entry(user_id, server, None)


_UNSET: Any = object()


def write_oauth(user_id: str, server: dict, *, tokens: Any = _UNSET, client: Any = _UNSET,
                needs_signin: bool | None = None) -> bool:
    """Change the stored sign-in: ``tokens`` = ``{"tokens", "expires_at", "metadata"}`` or None to drop
    them, ``client`` = the registration dict or None; ``needs_signin`` sets or clears the flag. A user
    server's record becomes ``auth: "oauth"``. False when the user's server is gone."""
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        stored, bind = _locate(user_id, server, servers, oauth_admin)
        if bind is None:
            return False
        if server.get("admin"):
            entry = oauth_admin.setdefault(server["name"], _clean_entry({}) or {})
        else:
            stored["auth"] = AUTH_OAUTH
            entry = stored.setdefault("oauth", _clean_entry({}) or {})
        for field, value in (("tokens", tokens), ("client", client)):
            if value is _UNSET:
                entry[field] = _rotate(entry.get(field))
            elif value is None:
                entry[field] = None
            else:
                entry[field] = _seal(user_id, bind, field, value)
        if needs_signin is True:
            entry["needs_signin"] = True
        elif needs_signin is False:
            entry.pop("needs_signin", None)
        entry["updated"] = now_iso()
        _write_user_file(path, servers, oauth_admin)
    return True


def clear_oauth(user_id: str, server: dict, *, keep_client: bool = True) -> bool:
    """Sign out: the stored tokens are deleted (the client registration stays unless ``keep_client``
    is False); the record stays. False when there was nothing to clear."""
    with _lock(user_id):
        path = user_path(user_id)
        servers, oauth_admin = _read_user_file(path)
        if server.get("admin"):
            entry = oauth_admin.get(server["name"])
            if entry is None:
                return False
            if keep_client and entry.get("client"):
                oauth_admin[server["name"]] = {**entry, "tokens": None, "needs_signin": True, "updated": now_iso()}
            else:
                del oauth_admin[server["name"]]
        else:
            stored = find(servers, server.get("name", ""))
            if stored is None or not stored.get("oauth"):
                return False
            if keep_client and stored["oauth"].get("client"):
                stored["oauth"] = {**stored["oauth"], "tokens": None, "needs_signin": True, "updated": now_iso()}
            else:
                stored.pop("oauth", None)
        _write_user_file(path, servers, oauth_admin)
    return True


def seal_oauth(user_id: str, server: dict, secrets_: dict) -> dict:
    """A new record's ``oauth`` entry from decrypted pieces (``{"tokens": ..., "client": ...}``, like
    ``write_oauth``), bound to ``server`` as it will be stored."""
    entry = _clean_entry({}) or {}
    for field in OAUTH_FIELDS:
        if secrets_.get(field):
            entry[field] = _seal(user_id, server, field, secrets_[field])
    if not secrets_.get("tokens"):
        entry["needs_signin"] = True
    entry["updated"] = now_iso()
    return entry


def mark_signin_needed(user_id: str, server: dict) -> None:
    """A call or discovery found the sign-in unusable: the per-user flag (and a user server's error)
    say so, so ``/mcp`` offers "Sign in". An admin server's shared cache is not touched: the state is
    the user's. Best effort, never raises."""
    try:
        if not server.get("admin"):
            record_error(user_id, server, SIGN_IN_NEEDED)
        write_oauth(user_id, server, needs_signin=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not mark the MCP sign-in of %s for %s as needed: %s", user_id, server.get("name"), e)


# ---- rendering helpers (no secrets) -------------------------------------------------------
def describe_headers(server: dict) -> str:
    """What to show for the stored credentials: "token set", "token set, 2 more headers", "" when none;
    an OAuth server's sign-in state ("🔐 signed in" / "🔐 sign in needed", from ``oauth_state``) first."""
    parts = []
    if server.get("auth") == AUTH_OAUTH:
        parts.append("🔐 signed in" if server.get("oauth_state") == STATE_SIGNED_IN else "🔐 sign in needed")
    headers_text = _describe_header_names(server)
    return ", ".join(p for p in (*parts, headers_text) if p)


def _describe_header_names(server: dict) -> str:
    headers = server.get("headers") or {}
    has_token = any(k.lower() == "authorization" for k in headers)
    others = sum(1 for k in headers if k.lower() != "authorization")
    parts = []
    if has_token:
        parts.append("token set")
    if others:
        parts.append(f"{others} more header{'s' if others != 1 else ''}" if has_token else f"{others} header{'s' if others != 1 else ''}")
    return ", ".join(parts)


def display_url(url: str) -> str:
    """The URL without user info, query or fragment: nothing secret can hide in what is shown."""
    try:
        parts = urlsplit(str(url or ""))
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid url>"
