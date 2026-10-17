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
import socket
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
    if admin:
        record["admin"] = True
    return record


# ---- file I/O ---------------------------------------------------------------------
def _read_servers(path: Path, *, admin: bool = False) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - a broken file must not break every turn
        logger.error("Could not read MCP servers from %s: %s", path, e)
        return []
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


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _strip_admin(server: dict) -> dict:
    return {k: v for k, v in server.items() if k != "admin"}


def load_user(user_id: str) -> list[dict]:
    with _lock(user_id):
        return _read_servers(user_path(user_id))


def save_user(user_id: str, servers: list[dict]) -> None:
    with _lock(user_id):
        _write(user_path(user_id), {"servers": [_strip_admin(s) for s in servers]})


def load_admin() -> list[dict]:
    """The admin servers with their cached tools merged in by name (``admin: True`` on each)."""
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
    entries = [
        {"name": s["name"], "tools": s.get("tools") or [], "updated": s.get("updated"), "error": s.get("error")}
        for s in servers
    ]
    with _admin_lock:
        _write(admin_cache_path(), {"servers": entries})


def all_servers(user_id: str) -> list[dict]:
    """What the agent sees: the user's servers, then the admin's."""
    return load_user(user_id) + load_admin()


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
    """Insert, or replace the server of the same name (re-adding is how a token is changed)."""
    with _lock(user_id):
        servers = _read_servers(user_path(user_id))
        record = _strip_admin(server)
        for i, existing in enumerate(servers):
            if existing["name"] == record["name"]:
                servers[i] = record
                break
        else:
            servers.append(record)
        _write(user_path(user_id), {"servers": servers})
    return record


def remove_user_server(user_id: str, name: str) -> bool:
    name = str(name or "").strip().lower()
    with _lock(user_id):
        servers = _read_servers(user_path(user_id))
        kept = [s for s in servers if s["name"] != name]
        if len(kept) == len(servers):
            return False
        _write(user_path(user_id), {"servers": kept})
    return True


def toggle_user_server(user_id: str, name: str) -> bool | None:
    """Flip ``enabled``; the new state, or None when there is no such server."""
    with _lock(user_id):
        servers = _read_servers(user_path(user_id))
        server = find(servers, name)
        if server is None:
            return None
        server["enabled"] = not server.get("enabled", True)
        _write(user_path(user_id), {"servers": servers})
        return server["enabled"]


def update_user_tools(user_id: str, name: str, tools: list[dict] | None, error: str | None = None) -> dict | None:
    """Store a discovery result (``with_tools``); the updated record, or None when the server is gone."""
    with _lock(user_id):
        servers = _read_servers(user_path(user_id))
        server = find(servers, name)
        if server is None:
            return None
        server.update(with_tools(server, tools, error))
        _write(user_path(user_id), {"servers": servers})
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
            servers = _read_servers(user_path(user_id))
            stored = find(servers, server.get("name", ""))
            if stored is None or stored.get("error") == error:
                return
            stored["error"] = error
            _write(user_path(user_id), {"servers": servers})
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record the MCP error of %s for %s: %s", server.get("name"), user_id, e)


# ---- rendering helpers (no secrets) -------------------------------------------------------
def describe_headers(server: dict) -> str:
    """What to show for the stored headers: "token set", "token set, 2 more headers", "" when none."""
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
