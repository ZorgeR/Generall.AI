#!/usr/bin/env python3
"""Persistent per-user browser for Anthropic's client-side browser toolset (browser_toolset_20260801).

Runs INSIDE the user's browser container (the sandbox image: Python 3.12, Playwright with Chromium,
trafilatura) as the non-root ``runner`` user. It drives ONE persistent Chromium profile and serves
the toolset's member calls over a unix socket, one JSON object per line each way::

    -> {"id": "7", "action": "left_click", "input": {"target": {"type": "ref", "ref": "ref_2"}},
        "allow_sensitive": false, "force_state": false}
    <- {"id": "7", "ok": true, "content": [...API-ready blocks...], "meta": {...}, "sensitive_done": false}
    <- {"id": "7", "ok": false, "error": "Error: ...", "needs_confirmation": false, "meta": {...}}

``content`` is what the bot puts into the member's ``tool_result`` unchanged: only ``text``,
``image`` (base64 JPEG) and ``browser_state`` blocks, following the toolset rules (tab-management
members answer with exactly one ``browser_state`` block, ``new_tab`` carries exactly one
``tab_opened`` for the active tab, no ``browser_state`` on errors, never an empty
``state_changes``, sanitized tab fields). Other members add ``browser_state`` when the tabs changed
since the last report, or always with ``force_state`` (the bot sets it on a turn's first call: the
last report may have gone to another conversation). Internal actions: ``_ping`` (health check), ``_peek``
(viewport JPEG for the chat's live view), ``_status`` (tabs and timestamps), ``_touch`` (counts as
activity, so the idle exit restarts) and ``_shutdown``. Only member calls and ``_touch`` count as
activity; after ``--idle-seconds`` without any the server closes the browser and exits by itself,
so a crashed bot never leaves a browser running.

Element references: ``read_page`` and ``find`` tag elements ``ref_N``. An injected script keeps a
WeakMap element -> ref in the page and a ref -> WeakRef(element) map, so a ref is stable while the
document lives; ONE counter for the whole server never restarts, so a ref handed out before a
navigation or in another tab can never silently point at a different element (it is reported
stale). Ref targets are scrolled into view and clicked at the centre of their visible box with
the real mouse, like coordinate targets.

Safety (best effort, defence in depth next to the container's egress filter):
- Chromium sends every request through ``EgressProxy``, a small forward proxy inside this process
  on 127.0.0.1 (plain HTTP absolute-URI requests and CONNECT tunnels for https/wss; loopback is
  not bypassed, WebRTC may not send non-proxied UDP). It resolves each host once (cached 60 s),
  refuses it when any answer is private, loopback, link-local, CGNAT, reserved or multicast (and
  when DNS fails: closed), and connects to the address it checked. Redirects, popups, workers and
  every other request Chromium makes are covered this way. Second layer: the route filter
  (``**/*`` and websockets) aborts such requests early; ``navigate`` refuses such hosts and every
  scheme but http(s). ``--allow-private`` (tests only) turns the address check off;
- clicks (including middle clicks, mouse down/up and drags), ``form_input`` on non-text controls,
  Enter and Space (``key``/``hold_key``/``type``, re-checked on the element that has the focus
  right before each one) on things that look like purchases, payments, sending, publishing,
  deleting or account/security changes are refused with ``needs_confirmation`` unless the request
  carries ``allow_sensitive: true``; then ``sensitive_done`` tells the bot the allowance was used.
  The element under a click point and the focused element are found through the DevTools
  protocol, so closed shadow roots and cross-origin frames are looked into. A page's own confirm()
  dialog that looks consequential is answered Cancel unless allowed, and so is going back,
  forward or reloading onto a page that would send a form again (POST).

Downloads land in ``--downloads`` under a sanitized unique name and are reported as
``download_started`` / ``download_completed`` (path ``downloads/<name>``) / ``download_failed``
state changes on the call during which they happened, or on the next successful one.

Cookies: Chromium keeps only cookies with an expiry in the profile. With a persistent profile the
session cookies (no expiry: many logins) are written to ``SESSION_COOKIES_FILE`` in the profile at
shutdown (mode 0600, atomically) and added back, then the file deleted, at the next start.

Usage: browser_server.py --socket PATH [--profile DIR|""] [--downloads DIR] [--idle-seconds N]
       [--viewport 1280x800] [--allow-private] [--executable PATH]

The module imports without Playwright: the pure helpers (key mapping, URL policy, private
addresses, sensitive names, content assembly, read_page line formatting) are unit-tested on the
host.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import socket
import sys
import tempfile
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

try:  # the sandbox image has Playwright; the host tests import the pure helpers without it
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:  # pragma: no cover - exercised only where Playwright is missing
    class PlaywrightError(Exception):  # type: ignore[no-redef]
        pass

    class PlaywrightTimeoutError(PlaywrightError):  # type: ignore[no-redef]
        pass


log = logging.getLogger("browser_server")

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

DEFAULT_VIEWPORT = (1280, 800)
SCREENSHOT_QUALITY = 80
PEEK_QUALITY = 70
ZOOM_MAX_SIDE = 1568           # the API does not downscale toolset images; stay under its limit
ZOOM_MAX_SCALE = 10.0
MAX_TABS = 6                   # new_tab refuses beyond this
MAX_POPUP_TABS = 10            # pages a site opens itself beyond this are closed again
READ_PAGE_CAP = 50_000
PAGE_TEXT_CAP = 60_000
FIND_LIMIT = 20
FIND_CANDIDATES = 4000
NAV_TIMEOUT_MS = 30_000
NETWORK_IDLE_MS = 5_000
SETTLE_SECONDS = 3.0
DOWNLOAD_WAIT_SECONDS = 5.0
CALL_TIMEOUT_SECONDS = 70.0    # the bot waits 75 s for an answer
DNS_CACHE_SECONDS = 60.0
DNS_TIMEOUT_SECONDS = 5.0
FIELD_LIMIT = 4096             # tab_id / title / url / download fields
MAX_STATE_TABS = 100
MAX_STATE_CHANGES = 200
MAX_TYPE_CHARS = 20_000
TYPE_INSERT_THRESHOLD = 400    # longer text without newlines is inserted, only its tail typed
TYPE_TAIL_CHARS = 20
SCROLL_NOTCH_PX = 100
DOWNLOADS_LABEL = "downloads"  # where the user's other tools see the downloads dir (workspace-relative)
CHROMIUM_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                 "--disable-blink-features=AutomationControlled",
                 # WebRTC may only use the proxy (no direct UDP to any address)
                 "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
FALLBACK_CHROME_MAJOR = "141"
MAX_FRAME_DEPTH = 6            # nested frames the guard follows into
SESSION_COOKIES_FILE = "gall-session-cookies.json"
SESSION_COOKIES_MAX_BYTES = 4 * 1024 * 1024

# the egress proxy (see EgressProxy)
PROXY_HEAD_LIMIT = 256 * 1024          # request / response head
PROXY_HEAD_TIMEOUT = 30.0              # a client must send its request head within this
PROXY_CONNECT_TIMEOUT = 15.0           # per address
PROXY_MAX_ADDRESSES = 4                # addresses tried per host
PROXY_IDLE_TIMEOUT = 300.0             # a connection with no bytes either way for this long is closed
PROXY_MAX_CONNECTIONS = 256
PROXY_CHUNK = 64 * 1024
DNS_FAILURE_CACHE_SECONDS = 5.0

TAB_MEMBERS = frozenset({"new_tab", "list_tabs", "switch_tab", "close_tab"})
MEMBERS = (
    "navigate", "screenshot", "zoom", "left_click", "right_click", "middle_click", "double_click",
    "triple_click", "hover", "left_click_drag", "left_mouse_down", "left_mouse_up", "mouse_move",
    "scroll", "scroll_to", "type", "key", "hold_key", "wait", "read_page", "find", "get_page_text",
    "form_input", "new_tab", "list_tabs", "switch_tab", "close_tab",
)
DISABLED_MEMBERS = frozenset({"file_upload", "read_console", "read_network", "javascript_exec"})
INTERNAL_ACTIONS = frozenset({"_ping", "_peek", "_status", "_touch", "_shutdown"})

REF_RE = re.compile(r"ref_[1-9][0-9]{0,8}")


class ActionError(Exception):
    """An expected failure: its message is the "Error: ..." text the model reads."""

    def __init__(self, message: str, needs_confirmation: bool = False):
        super().__init__(message)
        self.message = message
        self.needs_confirmation = needs_confirmation


# ---------------------------------------------------------------------------------------------
# Pure helpers: text and fields
# ---------------------------------------------------------------------------------------------

def clean_field(value, limit: int = FIELD_LIMIT) -> str:
    """A tab/download field the API accepts: no control characters (newlines included), no
    U+2028/U+2029, no leading/trailing or double spaces, at most ``limit`` characters.
    Idempotent: ``validate_content`` checks every tab field by cleaning it again."""
    text = "" if value is None else str(value)
    out = []
    for ch in text:
        if ch in "\u2028\u2029" or unicodedata.category(ch) in ("Cc", "Cs"):
            out.append(" ")
        else:
            out.append(ch)
    text = re.sub(r" {2,}", " ", "".join(out)).strip()
    return text[:limit].rstrip()  # a cut may end on a space


def clean_text(value, limit: int | None = None) -> str:
    """Collapse whitespace and drop control characters for names quoted in text lines."""
    text = clean_field(value, limit=10**9)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[: max(0, limit - 1)] + "\u2026"
    return text


def quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def display_url(url: str | None) -> str:
    """A URL as the model sees it: user info (user:password@) removed."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return clean_field(url)
    if parts.scheme in ("http", "https", "ws", "wss") and ("@" in parts.netloc):
        host = parts.netloc.rsplit("@", 1)[1]
        url = urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return clean_field(url)


def download_url(url: str | None) -> str:
    """Download URLs often carry signed credentials in the query: keep scheme, host and path."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return clean_field(url)
    if parts.scheme in ("http", "https"):
        host = parts.netloc.rsplit("@", 1)[-1]
        return clean_field(urlunsplit((parts.scheme, host, parts.path, "", "")))
    if parts.scheme in ("data", "blob"):
        return clean_field(url[:80])
    return clean_field(url)


def context_lost(exc: BaseException) -> bool:
    """A page script failed because the document went away (a navigation in between)."""
    text = str(exc)
    return any(s in text for s in ("Execution context was destroyed", "Cannot find context",
                                   "most likely because of a navigation"))


def short_error(exc: BaseException, limit: int = 300) -> str:
    """The first line of a Playwright error without its "Page.goto: " prefix."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    text = re.sub(r"^[A-Za-z]+\.[A-Za-z_]+: ", "", text)
    return clean_text(text, limit)


def sanitize_filename(name: str | None) -> str:
    """A safe single file name: no directories, no control or shell-hostile characters."""
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = "".join(ch if (ch.isalnum() or ch in "._-()+ ,=@") else "_" for ch in base)
    base = re.sub(r"\s+", " ", base).strip(" .")
    base = re.sub(r"_{2,}", "_", base)
    if not base or base in (".", ".."):
        base = "download"
    if len(base) > 120:
        stem, ext = os.path.splitext(base)
        base = stem[: 120 - len(ext[:20])] + ext[:20]
    return base


def unique_name(directory: str, name: str, reserved: set[str] | None = None) -> str:
    """``name`` or ``stem-2.ext``, ``stem-3.ext`` ... not yet present in ``directory``."""
    reserved = reserved if reserved is not None else set()
    stem, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate in reserved or os.path.exists(os.path.join(directory, candidate)):
        n += 1
        candidate = f"{stem}-{n}{ext}"
    return candidate


def parse_viewport(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d{2,5})\s*[xX]\s*(\d{2,5})\s*", text or "")
    if not match:
        raise ValueError(f"invalid viewport {text!r}; expected WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if not (200 <= width <= 4000 and 200 <= height <= 4000):
        raise ValueError("viewport out of range")
    return width, height


# ---------------------------------------------------------------------------------------------
# Pure helpers: URL policy and private addresses
# ---------------------------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
_KNOWN_NON_WEB_SCHEMES = {
    "javascript", "data", "file", "about", "chrome", "chrome-extension", "chrome-search",
    "chrome-untrusted", "devtools", "blob", "mailto", "tel", "sms", "ftp", "ftps", "ws", "wss",
    "view-source", "vbscript", "filesystem", "intent", "market", "edge", "opera", "brave",
    "content", "resource", "moz-extension", "ms-settings", "steam", "magnet", "irc", "news",
    "nntp", "git", "ssh", "smb", "sftp", "gopher", "jar", "x-javascript", "livescript",
}
NAV_SCHEME_ERROR = "Error: Navigation refused. Only http and https URLs are allowed."


def normalize_navigation_url(url: str) -> tuple[str | None, str | None]:
    """(absolute http(s) URL, None) or (None, "Error: ..."). A bare host becomes https://host."""
    text = (url or "").strip()
    if not text:
        return None, "Error: navigate needs a url."
    if any(unicodedata.category(ch) == "Cc" for ch in text):
        return None, "Error: The URL contains control characters."
    match = _SCHEME_RE.match(text)
    if "://" in text.split("?", 1)[0].split("#", 1)[0]:
        scheme = text.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            return None, NAV_SCHEME_ERROR
    elif match and match.group(1).lower() in _KNOWN_NON_WEB_SCHEMES:
        return None, NAV_SCHEME_ERROR
    elif match and not re.match(r"^[^:/?#]+:\d+(?:[/?#]|$)", text):
        # "something:..." that is not host:port: an unknown scheme, refuse rather than guess
        if not re.match(r"^[\w.\-]+\.[a-zA-Z]{2,}:", text):
            return None, NAV_SCHEME_ERROR
        text = "https://" + text
    else:
        text = "https://" + text.lstrip("/")
    try:
        parts = urlsplit(text)
    except ValueError:
        return None, "Error: The URL is not valid."
    if parts.scheme not in ("http", "https"):
        return None, NAV_SCHEME_ERROR
    if not parts.hostname:
        return None, "Error: The URL has no host."
    return text, None


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def is_private_address(value) -> bool:
    """True for every address the browser must not reach: private, loopback, link-local,
    CGNAT, reserved, multicast, unspecified, documentation ranges (anything not global).

    IPv4-mapped (::ffff:a.b.c.d) and NAT64 (64:ff9b::a.b.c.d) addresses are judged by the IPv4
    address inside them only (Python 3.12 and 3.13 disagree about them otherwise); 6to4 and
    Teredo addresses are refused when the IPv4 address they carry is private."""
    try:
        ip = ipaddress.ip_address(str(value).split("%", 1)[0].strip("[]"))
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_private_address(ip.ipv4_mapped)
        if ip in _NAT64:
            return is_private_address(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        carried = [ip.sixtofour] + ([ip.teredo[1]] if ip.teredo else [])
        if any(inner is not None and is_private_address(inner) for inner in carried):
            return True
    if ip.is_multicast or ip.is_unspecified or ip.is_loopback or ip.is_link_local:
        return True
    if ip.is_private or ip.is_reserved:
        return True
    return not ip.is_global


def host_is_literal_private(host: str) -> bool | None:
    """True/False for an IP literal or a localhost name, None when DNS has to decide."""
    name = (host or "").strip().strip("[]").rstrip(".").lower()
    if not name:
        return None
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return None
    return is_private_address(name)


class HostPolicy:
    """Decides whether a host may be contacted; DNS answers are cached ``ttl`` seconds (a failed
    lookup ``DNS_FAILURE_CACHE_SECONDS``), and the egress proxy connects to exactly the addresses
    it checked."""

    def __init__(self, allow_private: bool = False, ttl: float = DNS_CACHE_SECONDS, resolver=None):
        self.allow_private = allow_private
        self.ttl = ttl
        self._cache: dict[str, tuple[float, tuple[str, ...] | None]] = {}
        self._resolver = resolver  # async (host) -> list[str]; tests inject one

    async def _resolve(self, host: str) -> list[str]:
        if self._resolver is not None:
            return await self._resolver(host)
        loop = asyncio.get_running_loop()
        infos = await asyncio.wait_for(loop.getaddrinfo(host, None, type=socket.SOCK_STREAM),
                                       DNS_TIMEOUT_SECONDS)
        return [info[4][0] for info in infos]

    async def addresses(self, host: str) -> list[str]:
        """The host's addresses (cached); OSError when it does not resolve."""
        name = (host or "").strip().rstrip(".").lower()
        if not name:
            raise OSError("no host name")
        now = time.monotonic()
        cached = self._cache.get(name)
        if cached and cached[0] > now:
            if cached[1] is None:
                raise OSError("the host name could not be resolved")
            return list(cached[1])
        try:
            found = tuple(dict.fromkeys(str(a).split("%", 1)[0] for a in await self._resolve(name)))
        except Exception:  # noqa: BLE001 - any resolver failure (timeout, NXDOMAIN, a test's KeyError)
            found = ()
        if len(self._cache) > 4096:
            self._cache.clear()
        if not found:
            self._cache[name] = (now + min(self.ttl, DNS_FAILURE_CACHE_SECONDS), None)
            raise OSError("the host name could not be resolved")
        self._cache[name] = (now + self.ttl, found)
        return list(found)

    async def is_blocked(self, host: str | None) -> bool:
        """The route filter's and navigate's verdict. An unresolvable host is not "blocked" here
        (the proxy refuses it anyway, and the error then says it did not resolve)."""
        if self.allow_private or not host:
            return False
        literal = host_is_literal_private(host)
        if literal is not None:
            return literal
        try:
            return any(is_private_address(a) for a in await self.addresses(host))
        except OSError:
            return False

    async def egress_addresses(self, host: str) -> list[str]:
        """The proxy's check: the addresses to connect to, or ``EgressRefused``. Fails CLOSED: a
        host that does not resolve is refused, and so is one with ANY private answer."""
        literal = host_is_literal_private(host)
        if literal is not None and not self.allow_private:
            if literal:
                raise EgressRefused(host, "private")
            return [host.strip().strip("[]").split("%", 1)[0]]
        try:
            ipaddress.ip_address(host.strip().strip("[]").split("%", 1)[0])
            return [host.strip().strip("[]").split("%", 1)[0]]  # a literal (allow_private)
        except ValueError:
            pass
        try:
            found = await self.addresses(host)
        except OSError as e:
            raise EgressRefused(host, "dns") from e
        if not self.allow_private and any(is_private_address(a) for a in found):
            raise EgressRefused(host, "private")
        return found

    async def check_url(self, url: str) -> str | None:
        """None when the request may go out, else the blocked host (or scheme) for the log."""
        try:
            parts = urlsplit(url)
        except ValueError:
            return url[:80]
        scheme = parts.scheme.lower()
        if scheme in ("data", "blob", "about"):
            return None
        if scheme not in ("http", "https", "ws", "wss"):
            return scheme + ":"
        host = parts.hostname or ""
        if await self.is_blocked(host):
            return host
        return None


class EgressRefused(Exception):
    """The proxy does not connect to ``host``: ``reason`` is "private", "dns" or "connect"."""

    REASONS = {
        "private": "is a private or local address",
        "dns": "could not be resolved",
        "connect": "could not be reached",
    }

    def __init__(self, host: str, reason: str):
        super().__init__(f"{host} {self.REASONS.get(reason, reason)}")
        self.host = host
        self.reason = reason


# ---------------------------------------------------------------------------------------------
# The egress proxy: Chromium's only way out
# ---------------------------------------------------------------------------------------------

_METHOD_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,32}")
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,256}")
_HOP_HEADERS = {"connection", "keep-alive", "proxy-connection", "proxy-authorization",
                "proxy-authenticate", "te", "trailer", "upgrade"}


class ProxyBadRequest(Exception):
    """A request head the proxy does not take (answered 400)."""


@dataclass
class ProxyRequest:
    method: str
    host: str          # without brackets
    port: int
    connect: bool      # a CONNECT tunnel (https, wss) or one plain HTTP request
    head: bytes = b""  # the rewritten head for the origin server (plain HTTP only)


def _split_head(head: bytes) -> tuple[str, list[tuple[str, str]]]:
    text = head.decode("latin-1")
    if not text.endswith("\r\n\r\n"):
        raise ProxyBadRequest("incomplete head")
    lines = text[:-4].split("\r\n")
    headers = []
    for line in lines[1:]:
        if not line or line[0] in " \t" or ":" not in line:
            raise ProxyBadRequest("malformed header line")
        name, value = line.split(":", 1)
        if not _HEADER_NAME_RE.fullmatch(name):
            raise ProxyBadRequest("malformed header name")
        headers.append((name, value.strip()))
    return lines[0], headers


def parse_proxy_request(head: bytes) -> ProxyRequest:
    """A client's request head -> what to connect to and, for plain HTTP, the head to send
    upstream: origin-form request line, hop-by-hop headers dropped, ``Connection: close`` (one
    request per connection, so every request is checked; ``Connection: Upgrade`` is kept for an
    upgrade request)."""
    line, headers = _split_head(head)
    parts = line.split(" ")
    if len(parts) != 3 or not _METHOD_RE.fullmatch(parts[0]) or parts[2] not in ("HTTP/1.1", "HTTP/1.0"):
        raise ProxyBadRequest("malformed request line")
    method, target, version = parts
    if method == "CONNECT":
        try:
            split = urlsplit("//" + target)
            host, port = split.hostname, split.port
        except ValueError as e:
            raise ProxyBadRequest("bad CONNECT target") from e
        if not host or not port or split.path or split.query or split.username:
            raise ProxyBadRequest("bad CONNECT target")
        return ProxyRequest(method, host, port, True)
    try:
        url = urlsplit(target)
        host, port = url.hostname, url.port or 80
    except ValueError as e:
        raise ProxyBadRequest("bad request target") from e
    if url.scheme.lower() != "http" or not host or url.username is not None or url.password is not None:
        raise ProxyBadRequest("only absolute http:// targets are proxied")
    path = (url.path or "/") + (f"?{url.query}" if url.query else "")
    connection = {t.strip().lower() for name, value in headers if name.lower() == "connection"
                  for t in value.split(",")}
    upgrade = [value for name, value in headers if name.lower() == "upgrade"]
    is_upgrade = bool(upgrade) and "upgrade" in connection
    dropped = _HOP_HEADERS | (connection - {"close", "keep-alive", "upgrade", ""})
    out = [f"{method} {path} {version}"]
    if not any(name.lower() == "host" for name, _ in headers):
        shown = f"[{host}]" if ":" in host else host
        out.append(f"Host: {shown}" + (f":{port}" if port != 80 else ""))
    out += [f"{name}: {value}" for name, value in headers if name.lower() not in dropped]
    if is_upgrade:
        out += [f"Upgrade: {upgrade[0]}", "Connection: Upgrade"]
    else:
        out.append("Connection: close")
    return ProxyRequest(method, host, port, False, ("\r\n".join(out) + "\r\n\r\n").encode("latin-1"))


def rewrite_response_head(head: bytes) -> tuple[bytes, int]:
    """An origin server's response head -> (head for Chromium, status). Final responses get
    ``Connection: close`` so Chromium never sends a second request (maybe for another host) down
    this connection; 1xx heads pass unchanged. Lenient about the header lines (Chromium judges
    them), strict about the status line."""
    text = head.decode("latin-1")
    if not text.endswith("\r\n\r\n"):
        raise ProxyBadRequest("incomplete head")
    lines = text[:-4].split("\r\n")
    match = re.fullmatch(r"HTTP/1\.[01] (\d{3})(?: .*)?", lines[0])
    if not match:
        raise ProxyBadRequest("malformed status line")
    status = int(match.group(1))
    if 100 <= status < 200:
        return head, status
    out = [lines[0]]
    dropping = False
    for line in lines[1:]:
        if line[:1] in (" ", "\t"):          # an obsolete folded continuation of the line before
            if not dropping:
                out.append(line)
            continue
        name = line.split(":", 1)[0].strip().lower()
        dropping = name in ("connection", "keep-alive", "proxy-connection")
        if not dropping:
            out.append(line)
    out += ["Connection: close", "Proxy-Connection: close"]
    return ("\r\n".join(out) + "\r\n\r\n").encode("latin-1"), status


def proxy_error_response(status: int, text: str) -> bytes:
    reason = {400: "Bad Request", 403: "Forbidden", 502: "Bad Gateway", 503: "Service Unavailable"}.get(status, "Error")
    body = (clean_text(text, 500) + "\n").encode("utf-8")
    return (f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\nCache-Control: no-store\r\nConnection: close\r\n"
            "Proxy-Connection: close\r\n\r\n").encode("ascii") + body


class EgressProxy:
    """A forward proxy on 127.0.0.1 for Chromium (``--proxy-server``, loopback not bypassed).

    Plain HTTP: one absolute-URI request per connection, sent to the origin in origin-form with
    ``Connection: close``. CONNECT (https, wss): a raw tunnel after ``200``. Before connecting,
    ``HostPolicy.egress_addresses`` resolves the host once and refuses private answers and DNS
    failures (403 / 502); the connection goes to an address it checked, never to a name, so DNS
    rebinding cannot slip in. Bounded: head size and time, connect time, idle time, number of
    connections; bodies are streamed in chunks with back-pressure and never logged."""

    def __init__(self, policy: HostPolicy, on_refused=None):
        self.policy = policy
        self.on_refused = on_refused  # callable(EgressRefused) for the current call's messages
        self.server = None
        self.port = 0
        self._active = 0
        self._writers: set = set()

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=PROXY_HEAD_LIMIT,
                                                 backlog=128)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
        for writer in list(self._writers):
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    def _refused(self, refusal: EgressRefused) -> None:
        if refusal.reason == "private":
            log.info("proxy refused %s: private or local address", clean_text(refusal.host, 200))
        if self.on_refused is not None:
            try:
                self.on_refused(refusal)
            except Exception:  # noqa: BLE001
                pass

    async def _connect_one(self, address: str, port: int):
        return await asyncio.wait_for(
            asyncio.open_connection(address, port, limit=PROXY_HEAD_LIMIT), PROXY_CONNECT_TIMEOUT)

    async def _open_upstream(self, host: str, port: int):
        addresses = await self.policy.egress_addresses(host)
        last = None
        for address in addresses[:PROXY_MAX_ADDRESSES]:
            try:
                return await self._connect_one(address, port)
            except (OSError, asyncio.TimeoutError) as e:
                last = e
        raise EgressRefused(host, "connect") from last

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._active >= PROXY_MAX_CONNECTIONS:
            try:
                writer.write(proxy_error_response(503, "Too many connections through the sandbox proxy."))
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        self._active += 1
        self._writers.add(writer)
        upstream = None
        try:
            try:
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), PROXY_HEAD_TIMEOUT)
                request = parse_proxy_request(head)
            except (ProxyBadRequest, asyncio.LimitOverrunError, ValueError) as e:
                await self._reply(writer, 400, f"The sandbox proxy cannot handle this request ({e}).")
                return
            if not 0 < request.port < 65536:
                await self._reply(writer, 400, "Invalid port.")
                return
            try:
                up_reader, up_writer = await self._open_upstream(request.host, request.port)
            except EgressRefused as refusal:
                self._refused(refusal)
                status = 403 if refusal.reason == "private" else 502
                prefix = "Blocked by the sandbox network policy: " if status == 403 else ""
                await self._reply(writer, status, f"{prefix}{refusal}.")
                return
            upstream = up_writer
            self._writers.add(up_writer)
            if request.connect:
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await self._relay(reader, writer, up_reader, up_writer, http=False)
            else:
                up_writer.write(request.head)
                await up_writer.drain()
                await self._relay(reader, writer, up_reader, up_writer, http=True)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError, OSError):
            pass
        except Exception as e:  # noqa: BLE001 - never let one connection take the server down
            log.info("proxy connection failed: %s", type(e).__name__)
        finally:
            self._active -= 1
            for w in (writer, upstream):
                if w is None:
                    continue
                self._writers.discard(w)
                try:
                    w.close()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    async def _reply(writer, status: int, text: str) -> None:
        writer.write(proxy_error_response(status, text))
        try:
            await asyncio.wait_for(writer.drain(), 5)
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass

    async def _relay(self, reader, writer, up_reader, up_writer, http: bool) -> None:
        """Copy bytes both ways until either side closes or nothing moved for the idle time.
        For plain HTTP the response heads are rewritten on the way (``rewrite_response_head``)."""
        last = [time.monotonic()]

        async def pump(src, dst):
            while True:
                data = await src.read(PROXY_CHUNK)
                if not data:
                    return
                last[0] = time.monotonic()
                dst.write(data)
                await dst.drain()

        async def response():
            while http:
                head = await up_reader.readuntil(b"\r\n\r\n")
                last[0] = time.monotonic()
                try:
                    head, status = rewrite_response_head(head)
                except ProxyBadRequest:
                    writer.write(proxy_error_response(502, "The server sent an invalid response."))
                    return
                writer.write(head)
                await writer.drain()
                if not 100 <= status < 200 or status == 101:
                    break
            await pump(up_reader, writer)

        async def watchdog():
            while True:
                await asyncio.sleep(min(10.0, PROXY_IDLE_TIMEOUT))
                if time.monotonic() - last[0] > PROXY_IDLE_TIMEOUT:
                    return

        tasks = [asyncio.ensure_future(pump(reader, up_writer)), asyncio.ensure_future(response()),
                 asyncio.ensure_future(watchdog())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------------------------
# Pure helpers: keys
# ---------------------------------------------------------------------------------------------

_KEY_ALIASES = {
    "enter": "Enter", "return": "Enter", "kpenter": "Enter", "numpadenter": "NumpadEnter",
    "tab": "Tab", "esc": "Escape", "escape": "Escape",
    "backspace": "Backspace", "bksp": "Backspace", "delete": "Delete", "del": "Delete",
    "insert": "Insert", "ins": "Insert",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "uparrow": "ArrowUp", "downarrow": "ArrowDown", "leftarrow": "ArrowLeft", "rightarrow": "ArrowRight",
    "home": "Home", "end": "End", "pageup": "PageUp", "pgup": "PageUp", "pagedown": "PageDown",
    "pgdn": "PageDown", "pgdown": "PageDown",
    "space": "Space", "spacebar": "Space",
    "ctrl": "Control", "control": "Control", "alt": "Alt", "option": "Alt", "opt": "Alt",
    "shift": "Shift", "super": "Meta", "cmd": "Meta", "command": "Meta", "meta": "Meta",
    "win": "Meta", "windows": "Meta",
    "capslock": "CapsLock", "numlock": "NumLock", "scrolllock": "ScrollLock",
    "printscreen": "PrintScreen", "pause": "Pause", "contextmenu": "ContextMenu", "menu": "ContextMenu",
    "plus": "+", "minus": "-",
}
MODIFIER_KEYS = ("Alt", "Control", "Meta", "Shift")
ENTER_KEYS = frozenset({"Enter", "NumpadEnter"})


def activation_kind(keys: list[str]) -> str | None:
    """Keys pressed together that may activate the focused element: "enter" (Enter submits a
    form or activates a button/link; with Shift alone, "shift_enter": a new line in a multi-line
    field), "space" (Space clicks a focused button, link or checkbox), else None."""
    if any(k in ENTER_KEYS for k in keys):
        others = {k for k in keys if k not in ENTER_KEYS}
        return "shift_enter" if others == {"Shift"} else "enter"
    if any(k in ("Space", " ") for k in keys):
        return "space"
    return None


def map_key_name(name: str) -> str:
    """One key name in the toolset's vocabulary -> Playwright's key name."""
    raw = (name or "").strip()
    if not raw:
        raise ActionError("Error: Empty key name.")
    if len(raw) == 1:
        return raw
    norm = raw.lower().replace("_", "").replace("-", "")
    if norm in _KEY_ALIASES:
        return _KEY_ALIASES[norm]
    match = re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", norm)
    if match:
        return "F" + match.group(1)
    return raw  # Playwright knows "KeyA", "Digit1", "NumpadAdd", ... and rejects the rest


def parse_chord(chord: str) -> list[str]:
    """"ctrl+shift+a" -> ["Control", "Shift", "a"]; "+" and "ctrl++" keep the plus key."""
    text = chord.strip()
    if text == "+":
        parts = ["+"]
    elif text.endswith("++"):
        parts = [p for p in text[:-2].split("+")] + ["+"]
    else:
        parts = text.split("+")
    if any(not p.strip() for p in parts):
        raise ActionError(f"Error: Invalid key chord {quote(clean_text(chord, 60))}.")
    return [map_key_name(p) for p in parts]


def parse_key_sequence(text: str) -> list[list[str]]:
    """The ``key`` member's text: space-separated chords -> list of key lists."""
    if not isinstance(text, str) or not text.strip():
        raise ActionError("Error: key needs a key name such as \"Enter\" or \"ctrl+a\".")
    return [parse_chord(chunk) for chunk in text.split()]


def chord_to_playwright(keys: list[str]) -> str:
    return "+".join(keys)


def parse_modifiers(value) -> list[str]:
    """The click members' ``modifiers`` ("shift", "ctrl+shift") -> Playwright modifier names."""
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        raise ActionError("Error: modifiers must be a string such as \"shift\" or \"ctrl+shift\".")
    out: list[str] = []
    for part in re.split(r"[+\s]+", value.strip()):
        if not part:
            continue
        key = map_key_name(part)
        if key not in MODIFIER_KEYS:
            raise ActionError(f"Error: {quote(clean_text(part, 40))} is not a modifier key "
                              "(use shift, ctrl, alt or meta).")
        if key not in out:
            out.append(key)
    return out


# ---------------------------------------------------------------------------------------------
# Pure helpers: input validation
# ---------------------------------------------------------------------------------------------

def _as_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise ActionError(f"Error: {label} must be a number.")
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        return int(round(float(value)))
    raise ActionError(f"Error: {label} must be a number.")


def _as_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise ActionError(f"Error: {label} must be a number.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
        return float(value)
    raise ActionError(f"Error: {label} must be a number.")


def parse_coordinate(target, width: int, height: int, label: str = "target") -> tuple[int, int]:
    """A CoordinateTarget inside the viewport -> (x, y)."""
    if not isinstance(target, dict) or target.get("type", "coordinate") != "coordinate":
        raise ActionError(f"Error: {label} must be a coordinate target "
                          "{\"type\": \"coordinate\", \"x\": ..., \"y\": ...}.")
    x = _as_int(target.get("x"), f"{label}.x")
    y = _as_int(target.get("y"), f"{label}.y")
    if not (0 <= x < width and 0 <= y < height):
        raise ActionError(f"Error: Coordinate ({x}, {y}) is outside the viewport ({width}x{height}).")
    return x, y


def parse_ref(target, label: str = "target") -> str:
    """A RefTarget -> "ref_N"."""
    if not isinstance(target, dict) or target.get("type") != "ref":
        raise ActionError(f"Error: {label} must be a ref target {{\"type\": \"ref\", \"ref\": \"ref_1\"}}.")
    ref = target.get("ref")
    if not isinstance(ref, str) or not REF_RE.fullmatch(ref.strip()):
        shown = clean_text(ref if isinstance(ref, str) else repr(ref), 40)
        raise ActionError(f"Error: {quote(shown)} is not an element reference; references look like ref_12.")
    return ref.strip()


def stale_ref_error(ref: str) -> str:
    return (f"Error: {ref} is stale or not found on the current page. "
            "Re-read the page to get fresh references.")


def check_tab_id(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionError("Error: tab_id must be a non-empty string such as \"tab-1\".")
    if len(value) > FIELD_LIMIT or clean_field(value) != value:
        raise ActionError("Error: tab_id is not valid.")
    return value


# ---------------------------------------------------------------------------------------------
# Pure helpers: the sensitive-action guard
# ---------------------------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    # purchases and payments
    r"\bpay\b", r"\bbuy\b", r"\bpurchase\b", r"\bcheck\s?-?out\b",
    r"\bplace\s+(?:your\s+|the\s+|my\s+)?order\b", r"\border\s+now\b",
    r"\bconfirm\s+(?:your\s+|the\s+|my\s+)?(?:order|purchase|payment|booking|transfer)\b",
    r"\bcomplete\s+(?:your\s+|the\s+|my\s+)?(?:order|purchase|payment|booking)\b",
    r"\bsubmit\s+(?:your\s+|the\s+|my\s+)?(?:order|payment|purchase|transfer)\b",
    r"\bmake\s+(?:a\s+|the\s+|your\s+|my\s+)?(?:payment|purchase|transfer|donation)\b",
    r"\bsubscribe\b", r"\bdonate\b", r"\btransfer\b", r"\btop[\s-]?up\b",
    r"\bоплат\w*", r"\bкупить\b", r"\bприобрести\b", r"\bзаказ\w*", r"\bоформить\b", r"\bперевести\b",
    r"\bподписаться\b", r"\bпополнить\b", r"\bпожертвовать\b",
    r"\b(?:подтвердить|выполнить|сделать|совершить|отправить)\s+(?:перевод|плат[её]ж|покупку)\w*",
    # "Перевод" as a money transfer, not "Перевод на английский" / "Перевод страницы" (translation)
    r"\bперевод\b(?!\s+(?:страниц|текст|сайт|документ|слов|фраз|на\s+\w+ий\b))",
    # sending and publishing ("Отправить" alone is any form's Submit in Russian: see send_in_context)
    r"\bsend\b", r"\bpost\b", r"\bpublish\b", r"\btweet\b", r"\breply\b", r"\bcomment\b",
    r"\bотправить\s+(?:сообщени|письм|комментари|ответ|отзыв|пост|публикаци)\w*",
    r"\bопубликовать\b", r"\bответить\b",
    # deleting data, account and security changes
    r"\bdelete\b", r"\bremove\s+(?:my\s+|the\s+|this\s+|your\s+)?account\b", r"\bdeactivate\b",
    r"\bclose\s+(?:my\s+|the\s+|this\s+|your\s+)?account\b",
    r"\b(?:change|update)\s+(?:my\s+|the\s+|your\s+)?(?:password|e-?mail(?:\s+address)?|phone\s+number)\b",
    r"\b(?:disable|turn\s+off|remove|reset)\s+(?:my\s+|the\s+|your\s+)?"
    r"(?:2fa|mfa|two[\s-]?factor|2[\s-]?step|two[\s-]?step|multi[\s-]?factor|authenticator|passkeys?|security\s+keys?)",
    r"\bудалить\b", r"\b(?:сменить|изменить|поменять)\s+(?:пароль|e-?mail|email|почту|адрес\s+почты|"
    r"электронную\s+почту|номер\s+телефона|телефон)\b",
    r"\b(?:отключить|выключить)\s+(?:двухфакторн\w*|двухэтапн\w*|2fa)",
]
_SENSITIVE_RE = re.compile("|".join(f"(?:{p})" for p in _SENSITIVE_PATTERNS), re.IGNORECASE)

# Phrases removed before matching: registration and login steps are never consequential.
_SAFE_PHRASES_RE = re.compile("|".join(f"(?:{p})" for p in [
    r"\b(?:re)?send\s+(?:me\s+)?(?:a\s+|the\s+|an?\s+|new\s+)*(?:verification\s+|confirmation\s+|security\s+|"
    r"login\s+|log[\s-]?in\s+|sign[\s-]?in\s+|magic\s+|one[\s-]time\s+|activation\s+|reset\s+)?"
    r"(?:code|link|otp|sms|pin|email)\b",
    r"\b(?:re)?send\s+(?:it\s+)?again\b",                                       # resend a one-time code
    r"\bотправить\s+(?:ещё\s+|еще\s+|повторно\s+)?(?:код|ссылк|смс|sms|письмо)\w*",
    r"\bотправить\s+(?:ещё\s+раз|еще\s+раз|повторно|снова)\b",
    r"\bмои\s+заказы\b", r"\bистори\w*\s+заказ\w*", r"\bстатус\w*\s+заказ\w*", r"\bотслежива\w*\s+заказ\w*",
]), re.IGNORECASE)

_SEND_WORD_RE = re.compile(r"\bотправить\b", re.IGNORECASE)
_COMPOSER_RE = re.compile(
    r"\bmessage|\breply|\bcomment|\bwrite\b|\bchat\b|сообщени|комментар|напиш|\bответить|\bчат\b",
    re.IGNORECASE)
_SEARCH_RE = re.compile(r"search|поиск|найти", re.IGNORECASE)
# a cross-origin frame whose content the guard cannot read: payment-looking URL, title or name
_PAYMENT_FRAME_RE = re.compile(
    r"(?<![a-z])(?:pay|payments?|checkout|billing|cards?|stripe|braintree|adyen|klarna|paypal|applepay|"
    r"googlepay|gpay|3ds|threeds|3-?d\s*secure|acs|cardinal(?:commerce)?|оплат\w*|плат[её]ж\w*|"
    r"карт[аыу])(?![a-z])", re.IGNORECASE)


def _guard_text(name: str | None) -> str:
    text = clean_text(name or "")
    return _SAFE_PHRASES_RE.sub(" ", text) if text else ""


def matches_sensitive(name: str | None) -> bool:
    """Does an accessible name read like a purchase, payment, send/post or deletion?"""
    text = _guard_text(name)
    return bool(text) and bool(_SENSITIVE_RE.search(text))


def mentions_send(name: str | None) -> bool:
    """A bare "Отправить" (Russian forms' generic Submit): consequential only in a message context."""
    text = _guard_text(name)
    return bool(text) and bool(_SEND_WORD_RE.search(text))


def looks_like_composer(*texts: str | None) -> bool:
    joined = " ".join(clean_text(t or "") for t in texts)
    return bool(joined.strip()) and bool(_COMPOSER_RE.search(joined)) and not _SEARCH_RE.search(joined)


def send_in_context(desc: dict) -> bool:
    """Is the form (or the container) around this control a message context: a message box,
    chat, comment, reply or e-mail form? Registration, login and contact-details forms are not."""
    fields = desc.get("ctxFields") or []
    if any(looks_like_composer(f) for f in fields):
        return True
    return bool(desc.get("ctxMultiline")) and not desc.get("ctxPassword")


def looks_like_payment_frame(desc: dict) -> bool:
    frame = desc.get("frame") or {}
    texts = [frame.get("url"), frame.get("title"), frame.get("ownerName"), frame.get("ownerSrc")]
    if desc.get("opaqueFrame"):
        texts += [desc.get("name")]
    return any(_PAYMENT_FRAME_RE.search(clean_text(t or "")) for t in texts)


def describe_label(role: str | None, name: str | None) -> str:
    role = clean_text(role or "element", 30) or "element"
    name = clean_text(name or "", 80)
    return f"{role} {quote(name)}" if name else role


def _name_label(desc: dict, role: str, name: str) -> str | None:
    if matches_sensitive(name) or (mentions_send(name) and send_in_context(desc)):
        return describe_label(role, name)
    return None


def sensitive_label(desc: dict | None, kind: str) -> str | None:
    """The guard's verdict: a label such as 'button "Place order"' when the action looks
    consequential, else None. ``kind`` is "click", "form_input", "enter", "shift_enter" (Enter
    with Shift only: a new line in a multi-line field) or "space" (Space activates a focused
    button, link or checkbox like a click; in a text field it types a space)."""
    if not desc:
        return None
    role = desc.get("role") or "element"
    name = desc.get("name") or ""
    if desc.get("opaqueFrame"):
        # a cross-origin frame the guard could not look into: only payment-looking ones count
        if kind in ("click", "enter", "shift_enter", "space") and looks_like_payment_frame(desc):
            return f"{describe_label('frame', name or (desc.get('frame') or {}).get('title'))} (looks like a payment form)"
        return None
    if desc.get("actionable") is False and not desc.get("textEntry"):
        return None  # plain text, a heading, an empty area: clicking it does nothing consequential
    if kind in ("click", "space"):
        if desc.get("textEntry"):
            return None  # clicking into (or typing a space in) a text field sends nothing by itself
        return _name_label(desc, role, name)
    if kind == "form_input":
        if desc.get("textEntry"):
            return None  # filling a text field sends nothing by itself
        return _name_label(desc, role, name)
    # Enter: on a button or link it activates it; in a field it submits the form or a composer
    if not desc.get("textEntry"):
        return _name_label(desc, role, name)
    if kind == "shift_enter" and desc.get("multiline"):
        return None
    if not desc.get("multiline"):
        for button in desc.get("submit") or []:
            if matches_sensitive(button) or (mentions_send(button) and send_in_context(desc)):
                return describe_label("button", button)
    if desc.get("inSearch"):
        return None
    if looks_like_composer(name, desc.get("placeholder"), desc.get("label")):
        shown = name or desc.get("placeholder") or desc.get("label") or ""
        return f"Enter in the text field {quote(clean_text(shown, 80))}"
    if not desc.get("multiline") and desc.get("frame") and looks_like_payment_frame(desc):
        shown = name or desc.get("placeholder") or desc.get("label") or ""
        return f"Enter in the text field {quote(clean_text(shown, 80))} of a payment form"
    return None


def first_sensitive_label(descs, kind: str) -> str | None:
    for desc in descs:
        label = sensitive_label(desc, kind)
        if label:
            return label
    return None


def confirmation_error(label: str) -> str:
    return (f"Error: This looks like a consequential action ({label}). "
            "Ask the user first with browser_confirm, then repeat the action.")


# ---------------------------------------------------------------------------------------------
# Pure helpers: content blocks and browser_state
# ---------------------------------------------------------------------------------------------

def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(data_b64: str, media_type: str = "image/jpeg") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data_b64}}


def make_tab_entries(tabs: list[tuple[str, str, str]], active_id: str | None) -> list[dict]:
    """[(tab_id, title, url)] -> browser_state tab entries with exactly one active tab."""
    entries = []
    seen = set()
    for tab_id, title, url in tabs[:MAX_STATE_TABS]:
        if not tab_id or tab_id in seen:
            continue
        seen.add(tab_id)
        entries.append({"tab_id": clean_field(tab_id), "title": clean_field(title),
                        "url": clean_field(url), "active": False})
    if entries:
        index = next((i for i, e in enumerate(entries) if e["tab_id"] == active_id), len(entries) - 1)
        entries[index]["active"] = True
    return entries


def state_signature(entries: list[dict]) -> tuple:
    return tuple((e["tab_id"], e["title"], e["url"], bool(e.get("active"))) for e in entries)


def merge_download_events(events: list[dict]) -> list[dict]:
    """At most one state change per download_id: a later completed/failed replaces started."""
    order: list[str] = []
    latest: dict[str, dict] = {}
    for event in events:
        did = event.get("download_id")
        if not did:
            continue
        if did not in latest:
            order.append(did)
        latest[did] = event
    return [latest[did] for did in order]


def clean_state_change(change: dict) -> dict:
    out = {}
    for key, value in change.items():
        if isinstance(value, str):
            out[key] = clean_field(value)
        else:
            out[key] = value
    return out


def make_state_block(entries: list[dict], changes: list[dict] | None = None) -> dict:
    """A browser_state block; ``state_changes`` only when there is something to report, and a
    tab_opened only for tabs that are in the inventory."""
    block: dict = {"type": "browser_state", "tabs": entries}
    ids = {e["tab_id"] for e in entries}
    kept = []
    for change in changes or []:
        if change.get("type") == "tab_opened" and change.get("tab_id") not in ids:
            continue
        kept.append(clean_state_change(change))
    if kept:
        block["state_changes"] = kept[:MAX_STATE_CHANGES]
    return block


def assemble_content(member: str, *, entries: list[dict], text: str | None = None,
                     image_b64: str | None = None, changes: list[dict] | None = None,
                     previous_signature: tuple | None = None,
                     force_state: bool = False) -> tuple[list[dict], tuple | None]:
    """The success content for one member call and the signature of the state now reported.

    Tab-management members answer with exactly one browser_state block. Every other member
    answers with its text and/or image and adds browser_state only when the tab set, the active
    tab, a title or a URL changed since the last reported state, or there are state changes, or
    ``force_state`` asks for it (the last report may have gone to another conversation).
    """
    changes = [c for c in (changes or []) if c]
    signature = state_signature(entries)
    if member in TAB_MEMBERS:
        return [make_state_block(entries, changes)], signature
    blocks: list[dict] = []
    if text:
        blocks.append(text_block(text))
    if image_b64:
        blocks.append(image_block(image_b64))
    include = bool(changes) or force_state or signature != previous_signature
    if include:
        state = make_state_block(entries, changes)
        if entries or state.get("state_changes"):
            blocks.append(state)
    if not blocks:
        blocks.append(text_block("Done."))
    return blocks, (signature if include else previous_signature)


def validate_content(member: str, content: list[dict]) -> list[str]:
    """Problems that would make the Messages API reject this tool_result content ([] = valid)."""
    problems: list[str] = []
    if not isinstance(content, list) or not content:
        return ["content must be a non-empty list"]
    states = [b for b in content if isinstance(b, dict) and b.get("type") == "browser_state"]
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind not in ("text", "image", "browser_state"):
            problems.append(f"block type {kind!r} not allowed")
        if kind == "text" and not isinstance(block.get("text"), str):
            problems.append("text block without text")
        if kind == "image":
            source = block.get("source") or {}
            if source.get("type") != "base64" or not source.get("data") or not source.get("media_type"):
                problems.append("image block without base64 data")
    if len(states) > 1:
        problems.append("more than one browser_state block")
    if member in TAB_MEMBERS and (len(content) != 1 or len(states) != 1):
        problems.append(f"{member} must answer with exactly one browser_state block")
    for state in states:
        tabs = state.get("tabs")
        if not isinstance(tabs, list) or len(tabs) > MAX_STATE_TABS:
            problems.append("tabs must be a list of at most 100 entries")
            continue
        ids = [t.get("tab_id") for t in tabs]
        if len(ids) != len(set(ids)):
            problems.append("duplicate tab_id")
        if tabs and sum(1 for t in tabs if t.get("active")) != 1:
            problems.append("exactly one tab must be active")
        for tab in tabs:
            if not tab.get("tab_id"):
                problems.append("empty tab_id")
            for key in ("tab_id", "title", "url"):
                value = tab.get(key)
                if not isinstance(value, str) or len(value) > FIELD_LIMIT or clean_field(value) != value:
                    problems.append(f"tab {key} not clean: {value!r:.80}")
        if "state_changes" in state:
            changes = state["state_changes"]
            if not isinstance(changes, list) or not changes or len(changes) > MAX_STATE_CHANGES:
                problems.append("state_changes must be omitted or a non-empty list")
                changes = []
            download_ids = [c.get("download_id") for c in changes if c.get("download_id")]
            if len(download_ids) != len(set(download_ids)):
                problems.append("more than one state change per download_id")
            for change in changes:
                kind = change.get("type")
                if kind == "tab_opened":
                    if change.get("tab_id") not in ids:
                        problems.append("tab_opened for a tab not in tabs")
                elif kind in ("download_started", "download_completed", "download_failed"):
                    if not change.get("download_id") or not isinstance(change.get("url"), str):
                        problems.append(f"{kind} without download_id/url")
                else:
                    problems.append(f"unknown state change {kind!r}")
        if member == "new_tab":
            opened = [c for c in state.get("state_changes") or [] if c.get("type") == "tab_opened"]
            active = [t.get("tab_id") for t in tabs if t.get("active")]
            if len(opened) != 1 or not active or opened[0].get("tab_id") != active[0]:
                problems.append("new_tab needs exactly one tab_opened matching the active tab")
    return problems


# ---------------------------------------------------------------------------------------------
# Pure helpers: read_page / find formatting
# ---------------------------------------------------------------------------------------------

def format_props(props: dict) -> list[str]:
    out: list[str] = []
    if props.get("level"):
        out.append(f"level={int(props['level'])}")
    if props.get("type"):
        out.append(f"type={clean_text(props['type'], 20)}")
    if props.get("value") not in (None, ""):
        out.append(f"value={quote(clean_text(props['value'], 150))}")
    if props.get("placeholder"):
        out.append(f"placeholder={quote(clean_text(props['placeholder'], 100))}")
    if props.get("href"):
        out.append(f"href={quote(clean_text(props['href'], 150))}")
    checked = props.get("checked")
    if checked == "mixed":
        out.append("checked=mixed")
    elif checked is True:
        out.append("checked")
    if props.get("selected"):
        out.append("selected")
    if props.get("expanded") is True:
        out.append("expanded")
    elif props.get("expanded") is False:
        out.append("collapsed")
    if props.get("disabled"):
        out.append("disabled")
    if props.get("focused"):
        out.append("focused")
    return out


def format_node_line(node: dict, indent: bool = True) -> str:
    """One read_page line: ``  link "Documentation" [ref_1] href="/docs"``."""
    level = max(0, int(node.get("level") or 0)) if indent else 0
    role = clean_text(node.get("role") or "generic", 40) or "generic"
    parts = ["  " * level + role]
    name = clean_text(node.get("name") or "", 300)
    if name or role == "text":
        parts.append(quote(name))
    if node.get("ref"):
        parts.append(f"[{node['ref']}]")
    parts.extend(format_props(node.get("props") or {}))
    return " ".join(parts)


def truncation_notice(cap: int) -> str:
    return (f"[Output truncated at {cap:,} characters. Read a part of the page with a smaller "
            "depth or a ref.]")


def render_tree(nodes: list[dict], cap: int = READ_PAGE_CAP, truncated: bool = False,
                notice_cap: int | None = None) -> str:
    """read_page lines joined, cut at a line boundary so that the text WITH the notice saying so
    stays within ``cap`` characters (the notice names ``notice_cap``, default ``cap``)."""
    notice = truncation_notice(cap if notice_cap is None else notice_cap)
    lines = [format_node_line(node) for node in nodes]
    if not truncated and sum(len(line) + 1 for line in lines) - 1 <= cap:
        return "\n".join(lines)
    budget = max(0, cap - len(notice) - 1)
    kept: list[str] = []
    used = -1
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    text = "\n".join(kept)
    return (text + "\n" + notice) if text else notice[:cap]


_STOPWORDS = {"the", "a", "an", "to", "of", "for", "in", "on", "with", "and", "or", "at", "by",
              "this", "that", "page", "element", "named", "called", "labeled", "labelled"}
_ROLE_WORDS = {
    "button": {"button"}, "btn": {"button"}, "link": {"link"}, "links": {"link"},
    "field": {"textbox", "searchbox", "combobox", "spinbutton"},
    "input": {"textbox", "searchbox", "combobox", "spinbutton", "checkbox", "radio"},
    "box": {"textbox", "searchbox", "combobox", "checkbox"}, "textbox": {"textbox", "searchbox"},
    "textarea": {"textbox"}, "text": {"textbox", "text"}, "search": {"searchbox"},
    "checkbox": {"checkbox", "switch"}, "check": {"checkbox"}, "toggle": {"switch", "checkbox"},
    "switch": {"switch"}, "radio": {"radio"}, "dropdown": {"combobox", "listbox"},
    "select": {"combobox", "listbox"}, "menu": {"menu", "menuitem", "combobox"},
    "option": {"option"}, "tab": {"tab"}, "image": {"img"}, "picture": {"img"}, "icon": {"img"},
    "heading": {"heading"}, "title": {"heading"}, "header": {"heading", "banner"},
    "slider": {"slider"}, "form": {"form"}, "dialog": {"dialog"}, "popup": {"dialog"},
    "navigation": {"navigation"}, "nav": {"navigation"}, "table": {"table"},
}


def find_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", (text or "").lower()) if t not in _STOPWORDS]


def score_find_candidate(query: str, candidate: dict) -> float:
    """Token overlap of the query with the candidate's role / name / placeholder / aria-label /
    title (plus small bonuses for a phrase match and being in the viewport)."""
    q_tokens = find_tokens(query)
    if not q_tokens:
        return 0.0
    fields = " ".join(str(candidate.get(k) or "") for k in ("name", "placeholder", "label", "title", "id"))
    f_tokens = set(find_tokens(fields))
    role = candidate.get("role") or ""
    score = 0.0
    matched = False
    for token in q_tokens:
        if token in f_tokens:
            score += 2.0
            matched = True
        elif len(token) >= 3 and any(len(f) >= 3 and (f.startswith(token) or token.startswith(f))
                                     for f in f_tokens):
            score += 1.0
            matched = True
        if role in _ROLE_WORDS.get(token, ()) or token == role:
            score += 1.5
            matched = True
    if not matched:
        return 0.0
    phrase = " ".join(q_tokens)
    name = " ".join(find_tokens(candidate.get("name") or ""))
    if phrase and name and phrase in name:
        score += 3.0
    if phrase and name == phrase:
        score += 1.0
    if candidate.get("inView"):
        score += 0.25
    return score


def rank_find_candidates(query: str, candidates: list[dict], limit: int = FIND_LIMIT) -> list[int]:
    """Indices into ``candidates`` of the best matches, best first, ties in document order."""
    scored = []
    for position, candidate in enumerate(candidates):
        score = score_find_candidate(query, candidate)
        if score > 0:
            scored.append((-score, position))
    scored.sort()
    return [position for _, position in scored[:limit]]


def choose_page_text(extracted: str | None, body: str | None) -> str:
    """trafilatura's main text when it found a real article, else the page's visible text."""
    extracted = (extracted or "").strip()
    body = (body or "").strip()
    if extracted and (len(extracted) >= 500 or len(extracted) >= 0.25 * len(body)):
        return extracted
    return body or extracted


def cap_text(text: str, cap: int = PAGE_TEXT_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + f"\n\n[Text truncated at {cap:,} characters.]"


def default_user_agent() -> str:
    """A current desktop Chrome UA matching the bundled Chromium's major version (the headless
    default says "HeadlessChrome", which many sites block)."""
    major = FALLBACK_CHROME_MAJOR
    try:
        import playwright  # noqa: WPS433 - optional at import time

        path = os.path.join(os.path.dirname(playwright.__file__), "driver", "package", "browsers.json")
        with open(path, encoding="utf-8") as handle:
            for browser in json.load(handle).get("browsers", []):
                if browser.get("name") == "chromium" and browser.get("browserVersion"):
                    major = str(browser["browserVersion"]).split(".", 1)[0]
                    break
    except Exception:  # noqa: BLE001
        pass
    return (f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.0.0 Safari/537.36")


# ---------------------------------------------------------------------------------------------
# The script injected into pages (accessibility tree, refs, hit testing)
# ---------------------------------------------------------------------------------------------

AGENT_JS = r"""
(args) => {
  const KEY = '__gallBrowserAgentV1';
  let A = window[KEY];
  if (!A || !A.refs) {
    A = {refs: new WeakMap(), byRef: new Map(), found: []};
    try { Object.defineProperty(window, KEY, {value: A, enumerable: false, configurable: true}); }
    catch (e) { window[KEY] = A; }
  }
  let nextRef = args.nextRef || 1;
  const VW = window.innerWidth, VH = window.innerHeight;
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'HEAD', 'META', 'LINK', 'TITLE',
                        'BASE', 'PARAM', 'SOURCE', 'TRACK', 'DATALIST']);
  const INTERACTIVE = new Set(['button', 'link', 'textbox', 'searchbox', 'combobox', 'listbox',
    'checkbox', 'radio', 'switch', 'slider', 'spinbutton', 'menuitem', 'menuitemcheckbox',
    'menuitemradio', 'option', 'tab', 'treeitem', 'clickable']);
  const STRUCTURE = new Set(['heading', 'paragraph', 'img', 'navigation', 'main', 'banner',
    'contentinfo', 'complementary', 'form', 'search', 'region', 'dialog', 'alertdialog', 'alert',
    'article', 'table', 'row', 'cell', 'columnheader', 'rowheader', 'gridcell', 'menu', 'menubar',
    'tablist', 'tabpanel', 'toolbar', 'tree', 'grid', 'group', 'radiogroup', 'iframe', 'figure',
    'status', 'progressbar', 'meter', 'tooltip']);
  const NEEDS_NAME = new Set(['group', 'region', 'figure', 'status']);
  const NAME_FROM_CONTENT = new Set(['button', 'link', 'heading', 'tab', 'menuitem',
    'menuitemcheckbox', 'menuitemradio', 'option', 'treeitem', 'cell', 'gridcell', 'columnheader',
    'rowheader', 'paragraph', 'checkbox', 'radio', 'switch', 'clickable', 'tooltip', 'alert', 'status']);
  const LONG_NAME = new Set(['paragraph', 'cell', 'gridcell', 'alert', 'status', 'clickable']);
  const INLINE = new Set(['B', 'I', 'EM', 'STRONG', 'SPAN', 'SMALL', 'CODE', 'BR', 'SUP', 'SUB',
    'MARK', 'U', 'S', 'ABBR', 'TIME', 'KBD', 'Q', 'CITE', 'WBR', 'FONT', 'BDI', 'BDO', 'DATA',
    'VAR', 'DEL', 'INS', 'DFN', 'SAMP', 'TT', 'BIG', 'STRIKE']);
  const collapse = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
  const cut = (s, n) => (s.length > n ? s.slice(0, n - 1) + '\u2026' : s);
  const styles = new WeakMap();
  const styleOf = (el) => {
    let st = styles.get(el);
    if (st === undefined) {
      try { st = el.ownerDocument.defaultView.getComputedStyle(el); } catch (e) { st = null; }
      styles.set(el, st);
    }
    return st;
  };

  function refOf(el) {
    let r = A.refs.get(el);
    if (!r) {
      r = 'ref_' + (nextRef++);
      A.refs.set(el, r);
      A.byRef.set(r, new WeakRef(el));
    }
    return r;
  }
  function lookup(ref) {
    const w = A.byRef.get(ref);
    const el = w && w.deref();
    return el && el.isConnected ? el : null;
  }

  function implicitRole(el) {
    const tag = el.tagName.toLowerCase();
    switch (tag) {
      case 'a': case 'area': return el.hasAttribute('href') ? 'link' : null;
      case 'button': case 'summary': return 'button';
      case 'input': {
        const t = (el.getAttribute('type') || 'text').toLowerCase();
        if (t === 'hidden') return null;
        if (['button', 'submit', 'reset', 'image', 'file', 'color'].includes(t)) return 'button';
        if (t === 'checkbox') return 'checkbox';
        if (t === 'radio') return 'radio';
        if (t === 'range') return 'slider';
        if (t === 'number') return 'spinbutton';
        if (t === 'search') return el.hasAttribute('list') ? 'combobox' : 'searchbox';
        return el.hasAttribute('list') ? 'combobox' : 'textbox';
      }
      case 'textarea': return 'textbox';
      case 'select': return (el.multiple || el.size > 1) ? 'listbox' : 'combobox';
      case 'option': return 'option';
      case 'img': return el.getAttribute('alt') === '' ? null : 'img';
      case 'svg': return (el.getAttribute('aria-label') || el.querySelector(':scope > title')) ? 'img' : null;
      case 'h1': case 'h2': case 'h3': case 'h4': case 'h5': case 'h6': return 'heading';
      case 'p': return 'paragraph';
      case 'nav': return 'navigation';
      case 'main': return 'main';
      case 'header': case 'footer': {
        const p = el.parentElement;
        if (p && p.closest('article, aside, main, nav, section')) return null;
        return tag === 'header' ? 'banner' : 'contentinfo';
      }
      case 'aside': return 'complementary';
      case 'form': return 'form';
      case 'search': return 'search';
      case 'section': return (el.hasAttribute('aria-label') || el.hasAttribute('aria-labelledby')) ? 'region' : null;
      case 'article': return 'article';
      case 'table': return 'table';
      case 'tr': return 'row';
      case 'td': return 'cell';
      case 'th': return 'columnheader';
      case 'dialog': return 'dialog';
      case 'fieldset': return 'group';
      case 'iframe': case 'frame': return 'iframe';
      case 'figure': return 'figure';
      case 'progress': return 'progressbar';
      case 'meter': return 'meter';
    }
    if (el.isContentEditable && !(el.parentElement && el.parentElement.isContentEditable)) return 'textbox';
    return null;
  }
  function roleOf(el) {
    const raw = (el.getAttribute('role') || '').trim().toLowerCase().split(/\s+/)[0];
    if (raw === 'presentation' || raw === 'none' || raw === 'generic') return null;
    return raw || implicitRole(el);
  }
  function isClickable(el, st) {
    if (el === el.ownerDocument.body || el === el.ownerDocument.documentElement) return false;
    if (el.hasAttribute('onclick')) return true;
    if (el.hasAttribute('tabindex') && el.tabIndex >= 0) return true;
    if (st && st.cursor === 'pointer') {
      const p = el.parentElement;
      if (!p) return true;
      const ps = styleOf(p);
      return !ps || ps.cursor !== 'pointer';
    }
    return false;
  }
  function textOfIds(el, ids) {
    const doc = el.ownerDocument;
    return ids.split(/\s+/).map((id) => {
      const t = id && doc.getElementById(id);
      return t ? collapse(t.innerText || t.textContent) : '';
    }).join(' ');
  }
  function accName(el, role) {
    let n = '';
    const tag = el.tagName;
    const lb = el.getAttribute('aria-labelledby');
    if (lb) n = textOfIds(el, lb);
    if (!n) n = el.getAttribute('aria-label') || '';
    if (!n && (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'METER' || tag === 'PROGRESS')) {
      const t = (el.getAttribute('type') || '').toLowerCase();
      if (tag === 'INPUT' && (t === 'button' || t === 'submit' || t === 'reset')) {
        n = el.value || (t === 'submit' ? 'Submit' : (t === 'reset' ? 'Reset' : ''));
      } else if (tag === 'INPUT' && t === 'image') {
        n = el.getAttribute('alt') || el.value || 'Submit';
      } else if (el.labels && el.labels.length) {
        n = Array.from(el.labels).map((l) => collapse(l.innerText || l.textContent)).join(' ');
      }
    }
    if (!n && tag === 'IMG') n = el.getAttribute('alt') || '';
    if (!n && tag === 'FIELDSET') { const lg = el.querySelector(':scope > legend'); if (lg) n = lg.innerText; }
    if (!n && tag === 'TABLE' && el.caption) n = el.caption.innerText;
    if (!n && tag === 'FIGURE') { const fc = el.querySelector(':scope > figcaption'); if (fc) n = fc.innerText; }
    if (!n && tag.toLowerCase() === 'svg') { const t = el.querySelector(':scope > title'); if (t) n = t.textContent; }
    if (!n && (tag === 'IFRAME' || tag === 'FRAME')) n = el.getAttribute('title') || el.getAttribute('name') || '';
    if (!n && NAME_FROM_CONTENT.has(role) && !el.isContentEditable && tag !== 'INPUT' && tag !== 'SELECT' && tag !== 'TEXTAREA') {
      n = el.innerText || el.textContent || '';
      if (!collapse(n)) {
        const img = el.querySelector('img[alt]:not([alt=""]), [aria-label], svg > title');
        if (img) n = img.getAttribute('alt') || img.getAttribute('aria-label') || img.textContent || '';
      }
    }
    if (!n) n = el.getAttribute('title') || '';
    if (!n && (tag === 'INPUT' || tag === 'TEXTAREA')) n = el.getAttribute('placeholder') || '';
    if (!n && el.isContentEditable) n = el.getAttribute('aria-placeholder') || el.getAttribute('data-placeholder') || '';
    return cut(collapse(n), LONG_NAME.has(role) ? 300 : 150);
  }
  function deepActive(doc) {
    let a = doc.activeElement;
    while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
    if (a && (a.tagName === 'IFRAME' || a.tagName === 'FRAME')) {
      try { const d = a.contentDocument; if (d && d.activeElement && d.activeElement !== d.body) return deepActive(d); } catch (e) {}
    }
    return a;
  }
  const FOCUSED = deepActive(document);
  function propsOf(el, role) {
    const p = {};
    const tag = el.tagName;
    if (role === 'heading') {
      const m = /^H([1-6])$/.exec(tag);
      const lvl = parseInt(el.getAttribute('aria-level') || (m ? m[1] : '0'), 10);
      if (lvl) p.level = lvl;
    }
    if (role === 'link') {
      const h = el.getAttribute('href');
      if (h) p.href = /^\s*javascript:/i.test(h) ? 'javascript:' : cut(h.trim(), 150);
    }
    if (tag === 'INPUT') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (!['text', 'search', 'checkbox', 'radio', 'button', 'submit', 'reset', 'image', 'number', 'range'].includes(t)) p.type = t;
      if (!['checkbox', 'radio', 'button', 'submit', 'reset', 'image', 'file'].includes(t) && el.value) {
        p.value = t === 'password' ? '\u2022\u2022\u2022\u2022' : cut(el.value, 150);
      }
      if (t === 'checkbox' || t === 'radio') { if (el.indeterminate) p.checked = 'mixed'; else if (el.checked) p.checked = true; }
    } else if (tag === 'TEXTAREA') {
      if (el.value) p.value = cut(el.value, 150);
    } else if (tag === 'SELECT') {
      const sel = Array.from(el.selectedOptions || []).map((o) => collapse(o.label || o.text)).join(', ');
      if (sel) p.value = cut(sel, 150);
    } else if (tag === 'OPTION') {
      if (el.selected) p.selected = true;
    } else if (el.isContentEditable && role === 'textbox') {
      const v = collapse(el.innerText);
      if (v) p.value = cut(v, 150);
    }
    const ph = el.getAttribute('placeholder') || el.getAttribute('aria-placeholder');
    if (ph && (tag === 'INPUT' || tag === 'TEXTAREA' || role === 'textbox' || role === 'searchbox' || role === 'combobox')) p.placeholder = cut(collapse(ph), 100);
    const ac = el.getAttribute('aria-checked');
    if (ac === 'true') p.checked = true; else if (ac === 'mixed') p.checked = 'mixed';
    if (el.getAttribute('aria-selected') === 'true') p.selected = true;
    const ae = el.getAttribute('aria-expanded');
    if (ae === 'true') p.expanded = true; else if (ae === 'false') p.expanded = false;
    if (tag === 'DETAILS') p.expanded = el.open;
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') p.disabled = true;
    if (el === FOCUSED && el !== document.body) p.focused = true;
    return p;
  }
  function boxInfo(el, st, ctx) {
    if (!st || st.display === 'contents') return {visible: false, inView: false};
    if (st.visibility !== 'visible') return {visible: false, inView: false};
    if (st.opacity === '0' && el.tagName !== 'INPUT') return {visible: false, inView: false};
    const rects = el.getClientRects();
    let visible = false, inView = false;
    for (const r of rects) {
      if (r.width > 0 && r.height > 0 && !(r.width <= 1 && r.height <= 1)) {  // 1x1 = screen-reader only
        visible = true;
        const top = r.top + ctx.oy, left = r.left + ctx.ox;
        if (top + r.height > 0 && left + r.width > 0 && top < VH && left < VW) { inView = true; break; }
      }
    }
    return {visible, inView};
  }
  function textInfo(node, ctx) {
    const p = node.parentElement;
    if (!p) return {visible: false, inView: false};
    const st = styleOf(p);
    if (!st || st.visibility !== 'visible' || st.opacity === '0') return {visible: false, inView: false};
    const range = node.ownerDocument.createRange();
    range.selectNodeContents(node);
    let visible = false, inView = false;
    for (const r of range.getClientRects()) {
      if (r.width > 0 && r.height > 0) {
        visible = true;
        const top = r.top + ctx.oy, left = r.left + ctx.ox;
        if (top + r.height > 0 && left + r.width > 0 && top < VH && left < VW) { inView = true; break; }
      }
    }
    return {visible, inView};
  }
  function isTextLeaf(el) {
    if (el.shadowRoot || el.isContentEditable) return false;
    const all = el.getElementsByTagName('*');
    if (all.length > 40) return false;
    for (const d of all) {
      if (!INLINE.has(d.tagName) || d.shadowRoot || d.hasAttribute('role') || d.hasAttribute('tabindex') || d.hasAttribute('onclick')) return false;
    }
    return true;
  }
  function childrenOf(el) {
    if (el.shadowRoot) return Array.from(el.shadowRoot.childNodes);
    if (el.tagName === 'SLOT') {
      const a = el.assignedNodes({flatten: true});
      if (a.length) return a;
    }
    return Array.from(el.childNodes);
  }

  // Generic walk; sink(el, role, level, info) returns false to stop descending into el.
  function walk(root, opts, sink) {
    const state = {visits: 0, stop: false};
    function visitChildren(parent, level, ctx) {
      for (const k of childrenOf(parent)) {
        if (state.stop) return;
        visit(k, level, ctx);
      }
    }
    function frameCtx(el, ctx) {
      const r = el.getBoundingClientRect();
      return Object.assign({}, ctx, {ox: ctx.ox + r.left + el.clientLeft, oy: ctx.oy + r.top + el.clientTop});
    }
    function visit(node, level, ctx) {
      if (++state.visits > 60000) { state.stop = true; opts.onTruncate(); return; }
      if (node.nodeType === 3) {
        if (opts.filter === 'interactive' || ctx.suppressText || !opts.text) return;
        const t = collapse(node.data);
        if (!t) return;
        const ti = textInfo(node, ctx);
        if (!ti.visible || (opts.viewportOnly && !ti.inView)) return;
        if (sink.text(t, level) === false) state.stop = true;
        return;
      }
      if (node.nodeType !== 1) return;
      const el = node;
      if (SKIP.has(el.tagName)) return;
      if (el.getAttribute('aria-hidden') === 'true') return;
      const st = styleOf(el);
      if (!st || st.display === 'none') return;
      if (el.tagName === 'INPUT' && (el.getAttribute('type') || '').toLowerCase() === 'hidden') return;
      let role = roleOf(el);
      if (!role && isClickable(el, st)) role = 'clickable';
      let emit = false;
      let box = null;
      const known = !!role && (INTERACTIVE.has(role) || STRUCTURE.has(role));
      if (known) {
        emit = opts.filter === 'interactive' ? (INTERACTIVE.has(role) || (el.isContentEditable && role === 'textbox')) : true;
        if (emit) {
          box = boxInfo(el, st, ctx);
          if (!box.visible || (opts.viewportOnly && !box.inView)) emit = false;
        }
        if (emit && NEEDS_NAME.has(role) && !accName(el, role)) emit = false;
        if (emit && role === 'paragraph' && !collapse(el.innerText)) emit = false;
      }
      if (emit) {
        const more = sink.element(el, role, level, box);
        if (more === null) { state.stop = true; return; }
        if (more === false) return;
        if (level + 1 >= opts.maxDepth) return;
        if (el.tagName === 'SELECT') { sink.options(el, level + 1); return; }
        const childCtx = NAME_FROM_CONTENT.has(role) ? Object.assign({}, ctx, {suppressText: true}) : ctx;
        if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
          try { const d = el.contentDocument; if (d && d.body) visitChildren(d.body, level + 1, frameCtx(el, childCtx)); } catch (e) {}
          return;
        }
        visitChildren(el, level + 1, childCtx);
        return;
      }
      if (!known && el.tagName === 'LABEL' && el.control && !ctx.suppressText) {
        // the control it labels carries this text as its name already
        visitChildren(el, level, Object.assign({}, ctx, {suppressText: true}));
        return;
      }
      if (!known && opts.text && opts.filter !== 'interactive' && !ctx.suppressText &&
          st.display !== 'contents' && st.visibility === 'visible' && isTextLeaf(el)) {
        // a block of plain (inline-formatted) text: one line instead of a line per text node
        const t = collapse(el.innerText);
        if (t) {
          const b = boxInfo(el, st, ctx);
          if (b.visible && (!opts.viewportOnly || b.inView) && sink.text(t, level) === false) state.stop = true;
        }
        return;
      }
      if (el.tagName === 'IFRAME' || el.tagName === 'FRAME') {
        try { const d = el.contentDocument; if (d && d.body) visitChildren(d.body, level, frameCtx(el, ctx)); } catch (e) {}
        return;
      }
      visitChildren(el, level, ctx);
    }
    visit(root, 0, {ox: 0, oy: 0, suppressText: false});
  }

  function rootChildrenWalk(opts, sink) {
    // the body has no role and is never clickable: its children start at level 0
    const body = document.body || document.documentElement;
    if (body) walk(body, opts, sink);
  }

  // ancestors across shadow boundaries (a node in a shadow root continues at its host)
  function hostOf(node) {
    const r = node && node.getRootNode ? node.getRootNode() : null;
    return r && r !== node && r.host ? r.host : null;
  }
  function parentDeep(el) { return el.parentElement || hostOf(el); }
  function closestDeep(el, sel) {
    let cur = el;
    for (let i = 0; cur && i < 30; i++) {
      const m = cur.closest ? cur.closest(sel) : null;
      if (m) return m;
      cur = hostOf(cur);
    }
    return null;
  }
  const FIELDS = 'textarea, input, select, [contenteditable=""], [contenteditable="true"], [role=textbox]';
  // the fields around a control: its form, else the nearest ancestor (6 levels) holding fields
  function contextOf(target) {
    const out = {ctxFields: [], ctxMultiline: false, ctxPassword: false};
    let scope = target.form || closestDeep(target, 'form');
    if (!scope) {
      let cur = parentDeep(target);
      for (let i = 0; i < 6 && cur && cur.nodeType === 1 && cur !== cur.ownerDocument.body; i++, cur = parentDeep(cur)) {
        if (cur.querySelector && cur.querySelector(FIELDS)) { scope = cur; break; }
      }
    }
    if (!scope || !scope.querySelectorAll) return out;
    for (const f of Array.from(scope.querySelectorAll(FIELDS)).slice(0, 40)) {
      const ft = (f.getAttribute('type') || '').toLowerCase();
      if (f.tagName === 'INPUT' && ['hidden', 'submit', 'button', 'reset', 'image', 'checkbox', 'radio', 'file', 'range', 'color'].includes(ft)) continue;
      if (ft === 'password') out.ctxPassword = true;
      if (f.tagName === 'TEXTAREA' || f.isContentEditable || f.getAttribute('aria-multiline') === 'true') out.ctxMultiline = true;
      const n = collapse([accName(f, 'textbox'), f.getAttribute('placeholder') || '', f.getAttribute('name') || ''].join(' '));
      if (n && out.ctxFields.length < 12) out.ctxFields.push(cut(n, 120));
    }
    return out;
  }

  function describe(el) {
    if (el && el.nodeType !== 1) el = el.parentElement || hostOf(el);
    if (!el || el.nodeType !== 1) return null;
    const ACTIONABLE = 'a[href],button,input,select,textarea,summary,label,[role=button],[role=link],' +
      '[role=menuitem],[role=menuitemcheckbox],[role=menuitemradio],[role=tab],[role=checkbox],' +
      '[role=switch],[role=radio],[role=option],[role=treeitem],[role=textbox],[role=searchbox],' +
      '[role=combobox],[onclick],[contenteditable=""],[contenteditable="true"]';
    let a = closestDeep(el, ACTIONABLE);
    if (a && a.tagName === 'LABEL') a = a.control || a;
    if (!a && el.isContentEditable) a = el;
    if (!a) {  // a div/span "button" without a role: the nearest cursor:pointer / onclick ancestor
      let cur = el;
      for (let i = 0; i < 8 && cur && cur.nodeType === 1 && cur !== cur.ownerDocument.body; i++, cur = parentDeep(cur)) {
        if (isClickable(cur, styleOf(cur))) { a = cur; break; }
      }
    }
    const target = a || el;
    let role = roleOf(target);
    if (!role) role = isClickable(target, styleOf(target)) ? 'clickable' : (target.tagName === 'LABEL' ? 'label' : 'generic');
    let name = accName(target, role);
    // plain text or an empty area (even <body>) has no name: its text is not what the click does
    if (role === 'generic') name = cut(collapse(target.getAttribute('aria-label') || target.getAttribute('title') || ''), 150);
    if (!name && role === 'label') name = cut(collapse(target.innerText || target.textContent), 150);
    const tag = target.tagName.toLowerCase();
    const type = (target.getAttribute('type') || '').toLowerCase();
    const textEntry = tag === 'textarea' || target.isContentEditable ||
      (tag === 'input' && !['button', 'submit', 'reset', 'image', 'checkbox', 'radio', 'range', 'color', 'file', 'hidden'].includes(type)) ||
      ['textbox', 'searchbox'].includes(role);
    const placeholder = target.getAttribute('placeholder') || target.getAttribute('aria-placeholder') || target.getAttribute('data-placeholder') || '';
    const label = target.getAttribute('aria-label') || '';
    const multiline = tag === 'textarea' || target.isContentEditable || target.getAttribute('aria-multiline') === 'true';
    let submit = [];
    let inSearch = role === 'searchbox' || type === 'search';
    const form = target.form || closestDeep(target, 'form');
    if (form) {
      const els = Array.from(form.elements || []).filter((b) =>
        (b.tagName === 'BUTTON' && (!b.getAttribute('type') || b.getAttribute('type').toLowerCase() === 'submit')) ||
        (b.tagName === 'INPUT' && ['submit', 'image'].includes((b.getAttribute('type') || '').toLowerCase())));
      submit = els.slice(0, 3).map((b) => accName(b, 'button')).filter((s) => s);
      if ((form.getAttribute('role') || '').toLowerCase() === 'search' || /search|поиск/i.test((form.getAttribute('action') || '') + ' ' + (form.getAttribute('aria-label') || '') + ' ' + (form.id || '') + ' ' + (form.className || ''))) inSearch = true;
    }
    if (closestDeep(target, '[role=search], search')) inSearch = true;
    const out = {role, name, tag, type, textEntry, multiline, placeholder: cut(collapse(placeholder), 100),
                 label: cut(collapse(label), 100), submit, inSearch, actionable: !!a};
    if (tag === 'iframe' || tag === 'frame') {
      out.frameOwner = true;
      let open = false;
      try { open = !!target.contentDocument; } catch (e) {}
      out.frameOpen = open;
      out.ownerSrc = cut(collapse(target.getAttribute('src') || ''), 300);
    }
    Object.assign(out, contextOf(target));
    return out;
  }
  function deepPoint(doc, x, y) {
    let el = doc.elementFromPoint(x, y);
    while (el && el.shadowRoot) {
      const inner = el.shadowRoot.elementFromPoint(x, y);
      if (!inner || inner === el) break;
      el = inner;
    }
    if (el && (el.tagName === 'IFRAME' || el.tagName === 'FRAME')) {
      try {
        const r = el.getBoundingClientRect();
        const d = el.contentDocument;
        if (d) return deepPoint(d, x - r.left - el.clientLeft, y - r.top - el.clientTop) || el;
      } catch (e) {}
    }
    return el;
  }

  const op = args.op;
  if (op === 'get') return lookup(args.ref);
  if (op === 'describe') {
    let el = null;
    if (args.node !== undefined) el = args.node;  // a node the DevTools protocol found (Python side)
    else if (args.ref) { el = lookup(args.ref); if (!el) return {stale: true}; }
    else if (args.focused) { el = FOCUSED; if (el === document.body) el = null; }
    else el = deepPoint(document, args.x, args.y);
    const d = describe(el);
    if (d && d.frameOwner && !d.frameOpen && args.node === undefined) {
      // the fallback without DevTools: a cross-origin frame this script cannot look into
      d.opaqueFrame = true;
      d.frame = {url: d.ownerSrc, title: '', ownerName: d.name, ownerSrc: d.ownerSrc};
    }
    return d;
  }
  if (op === 'scroll') {
    const de = document.documentElement || {};
    return {x: Math.round(window.scrollX), y: Math.round(window.scrollY), w: de.scrollWidth || 0, h: de.scrollHeight || 0, vw: VW, vh: VH};
  }
  if (op === 'refs') {
    const out = (args.indices || []).map((i) => { const el = A.found[i]; return el && el.isConnected ? refOf(el) : null; });
    A.found = [];
    return {refs: out, nextRef};
  }
  const filter = args.filter || null;
  const maxDepth = Math.max(1, args.depth || 15);
  const cap = args.cap || 50000;
  if (op === 'tree') {
    const nodes = [];
    let chars = 0, truncated = false;
    const opts = {filter, maxDepth, text: true, viewportOnly: !filter && !args.ref, onTruncate: () => { truncated = true; }};
    const add = (node, extra) => {
      if (chars > cap) { truncated = true; return false; }
      nodes.push(node);
      chars += node.level * 2 + (node.role || '').length + (node.name || '').length + 16 + extra;
      return true;
    };
    const sink = {
      element: (el, role, level) => {
        const props = propsOf(el, role);
        return add({level, role, name: accName(el, role), ref: refOf(el), props}, JSON.stringify(props).length) ? true : null;
      },
      text: (t, level) => add({level, role: 'text', name: cut(t, 300)}, 0),
      options: (sel, level) => {
        const opts2 = Array.from(sel.options || []);
        for (const o of opts2.slice(0, 30)) {
          if (!add({level, role: 'option', name: cut(collapse(o.label || o.text), 150), ref: refOf(o), props: o.selected ? {selected: true} : {}}, 10)) return;
        }
        if (opts2.length > 30) add({level, role: 'text', name: '\u2026 ' + (opts2.length - 30) + ' more options'}, 0);
      },
    };
    if (args.ref) {
      const root = lookup(args.ref);
      if (!root) return {stale: true};
      walk(root, opts, sink);
    } else {
      rootChildrenWalk(opts, sink);
    }
    const de = document.documentElement || {};
    return {nodes, truncated, nextRef, scroll: {y: Math.round(window.scrollY), h: de.scrollHeight || 0, vh: VH}};
  }
  if (op === 'find') {
    A.found = [];
    const cands = [];
    const opts = {filter: 'all', maxDepth: 1000, text: false, viewportOnly: false, onTruncate: () => {}};
    const sink = {
      element: (el, role, level, box) => {
        if (cands.length >= args.limit) return null;
        A.found.push(el);
        cands.push({i: A.found.length - 1, role, name: accName(el, role), props: propsOf(el, role),
                    placeholder: el.getAttribute('placeholder') || '', label: el.getAttribute('aria-label') || '',
                    title: el.getAttribute('title') || '', id: el.id || '', inView: !!(box && box.inView)});
        return true;
      },
      text: () => true,
      options: (sel) => {
        for (const o of Array.from(sel.options || []).slice(0, 200)) {
          if (cands.length >= args.limit) return;
          A.found.push(o);
          cands.push({i: A.found.length - 1, role: 'option', name: cut(collapse(o.label || o.text), 150), props: o.selected ? {selected: true} : {}, placeholder: '', label: '', title: '', id: o.id || '', inView: false});
        }
      },
    };
    rootChildrenWalk(opts, sink);
    return {cands};
  }
  return null;
}
"""

# Run through the DevTools protocol (Runtime.callFunctionOn) on a node it found, in that node's own
# frame: closed shadow roots and cross-origin frames included, which the page script cannot reach.
DESCRIBE_NODE_FN = ("function(args) { return (" + AGENT_JS.strip() +
                    ")(Object.assign({}, args || {}, {op: 'describe', node: this})); }")
# From a document, a shadow root or an element: the element that has the focus, through open
# shadow roots and same-origin frames (closed roots and other frames: the Python side goes on).
FOCUS_WALK_FN = """function() {
  let a = (this.nodeType === 9 || this.nodeType === 11) ? this.activeElement : this;
  for (let i = 0; i < 50 && a; i++) {
    if (a.shadowRoot && a.shadowRoot.activeElement) { a = a.shadowRoot.activeElement; continue; }
    if (a.tagName === 'IFRAME' || a.tagName === 'FRAME') {
      let d = null;
      try { d = a.contentDocument; } catch (e) {}
      if (d && d.activeElement) { a = d.activeElement; continue; }
    }
    break;
  }
  return a || null;
}"""
PROBE_GROUP = "gall-guard"


# ---------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    text: str | None = None
    image: str | None = None
    sensitive: bool = False


@dataclass(eq=False)
class Tab:
    id: str
    page: object
    title: str = ""
    cdp: object = None
    pending_navs: dict = field(default_factory=dict)  # navigation request -> monotonic start
    frame_sessions: dict = field(default_factory=dict)  # DevTools frame id -> (CDP session, Frame)
    in_process: set = field(default_factory=set)        # (id(frame), url) without a session of its own


class BrowserServer:
    def __init__(self, args: argparse.Namespace):
        self.socket_path = args.socket
        self.profile_arg = args.profile
        self.downloads_dir = os.path.abspath(args.downloads)
        self.idle_seconds = float(args.idle_seconds)
        self.width, self.height = parse_viewport(args.viewport)
        self.allow_private = bool(args.allow_private)
        self.executable = args.executable or None
        self.policy = HostPolicy(allow_private=self.allow_private)
        self.proxy = EgressProxy(self.policy, on_refused=self._on_proxy_refused)

        self.playwright = None
        self.context = None
        self.server = None
        self.profile_dir = ""
        self.ephemeral = False

        self.tabs: dict[str, Tab] = {}
        self.active_id: str | None = None
        self._activation: list[str] = []
        self._tab_counter = 0
        self.next_ref = 1  # ONE counter for every tab: a ref never means an element of another tab

        self.lock = asyncio.Lock()
        self.started = time.time()
        self.last_request = time.time()
        self._last_activity = time.monotonic()
        self._in_flight = 0
        self._stopping = False
        self._done = asyncio.Event()
        self.exit_code = 0

        self._in_call = False
        self._call_started = 0.0
        self._call_allow_sensitive = False
        self._call_dialog_sensitive = False
        self._call_dialog_refused: str | None = None
        self._opened_in_call: list[str] = []
        self._notes: list[str] = []
        self._events: list[dict] = []
        self._call_downloads: set[asyncio.Task] = set()
        self._download_tasks: set[asyncio.Task] = set()
        self._reserved_names: set[str] = set()
        self._blocked_navs: dict[int, str] = {}
        self._proxy_refusals: list[EgressRefused] = []
        self._history_page = None             # the page of a running back/forward/reload
        self._resubmit_refused: str | None = None
        self._resubmitted = False
        self._mouse_pressed: tuple | None = None  # (tab id, x, y) of a guarded left_mouse_down
        self._last_reported: tuple | None = None
        self._background: set[asyncio.Task] = set()

        self.members = {name: getattr(self, "m_" + name) for name in MEMBERS}

    # -- lifecycle -------------------------------------------------------------------------

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        os.makedirs(self.downloads_dir, exist_ok=True)
        if self.profile_arg:
            self.profile_dir = os.path.abspath(self.profile_arg)
            os.makedirs(self.profile_dir, exist_ok=True)
            for lock in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
                try:
                    os.unlink(os.path.join(self.profile_dir, lock))
                except FileNotFoundError:
                    pass
                except OSError as e:
                    log.warning("could not remove stale %s: %s", lock, e)
        else:
            self.profile_dir = tempfile.mkdtemp(prefix="browser-profile-")
            self.ephemeral = True

        port = await self.proxy.start()
        self.playwright = await async_playwright().start()
        options = dict(
            headless=True,
            viewport={"width": self.width, "height": self.height},
            screen={"width": self.width, "height": self.height},
            device_scale_factor=1,
            accept_downloads=True,
            args=CHROMIUM_ARGS,
            user_agent=default_user_agent(),
            service_workers="block",  # SW requests would bypass the route filter
            # every request goes through the egress proxy, loopback included ("<-loopback>" removes
            # Chromium's implicit bypass of localhost / 127.0.0.1 / [::1])
            proxy={"server": f"http://127.0.0.1:{port}", "bypass": "<-loopback>"},
            handle_sigint=False, handle_sigterm=False, handle_sighup=False,
        )
        if self.executable:
            self.context = await self.playwright.chromium.launch_persistent_context(
                self.profile_dir, executable_path=self.executable, **options)
        else:
            try:  # the new headless mode behaves like a real Chrome
                self.context = await self.playwright.chromium.launch_persistent_context(
                    self.profile_dir, channel="chromium", **options)
            except PlaywrightError as e:
                log.info("channel=chromium unavailable (%s); using the default headless build", short_error(e))
                self.context = await self.playwright.chromium.launch_persistent_context(
                    self.profile_dir, **options)
        self.context.set_default_timeout(15_000)
        self.context.on("page", self._on_context_page)
        self.context.on("close", lambda *_: self._on_context_closed())
        await self.context.route("**/*", self._route)
        try:
            await self.context.route_web_socket(re.compile(r".*"), self._route_ws)
        except Exception as e:  # noqa: BLE001 - older Playwright: websockets stay unfiltered here
            log.warning("websocket filtering unavailable: %s", short_error(e))
        await self._restore_session_cookies()
        for page in list(self.context.pages):
            self._register(page)
        if not self.tabs:
            self._register(await self.context.new_page())
        self.active_id = next(iter(self.tabs))
        self._activation = [self.active_id]

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.socket_path)), exist_ok=True)
        self.server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path,
                                                      limit=16 * 1024 * 1024)
        try:
            os.chmod(self.socket_path, 0o666)
        except OSError:
            pass
        self._spawn(self._idle_watch())
        log.info("browser server ready on %s (profile %s, idle %ss)", self.socket_path,
                 "ephemeral" if self.ephemeral else self.profile_dir, int(self.idle_seconds))

    async def wait(self) -> int:
        await self._done.wait()
        return self.exit_code

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def request_shutdown(self, reason: str, code: int = 0) -> None:
        if not self._stopping:
            self.exit_code = code
            self._spawn(self.shutdown(reason))

    async def shutdown(self, reason: str) -> None:
        if self._stopping:
            return
        self._stopping = True
        log.info("shutting down: %s", reason)
        if self.server is not None:
            self.server.close()
        await self._save_session_cookies()
        try:
            if self.context is not None:
                await asyncio.wait_for(self.context.close(), 20)
        except Exception as e:  # noqa: BLE001
            log.warning("closing the browser failed: %s", short_error(e))
        self.proxy.close()
        try:
            if self.playwright is not None:
                await asyncio.wait_for(self.playwright.stop(), 10)
        except Exception:  # noqa: BLE001
            pass
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        except OSError:
            pass
        if self.ephemeral and self.profile_dir:
            shutil.rmtree(self.profile_dir, ignore_errors=True)
        self._done.set()

    def _on_context_closed(self) -> None:
        if not self._stopping:
            log.error("the browser closed unexpectedly")
            self.request_shutdown("browser closed", code=2)

    async def _idle_watch(self) -> None:
        if self.idle_seconds <= 0:
            return
        interval = max(0.25, min(5.0, self.idle_seconds / 4))
        while not self._stopping:
            await asyncio.sleep(interval)
            if self._in_flight == 0 and time.monotonic() - self._last_activity >= self.idle_seconds:
                await self.shutdown(f"idle for {int(self.idle_seconds)} s")
                return

    def _touch(self) -> None:
        self._last_activity = time.monotonic()
        self.last_request = time.time()

    # -- session cookies (persistent profile only) -------------------------------------------

    def _session_cookies_path(self) -> str | None:
        if self.ephemeral or not self.profile_dir:
            return None
        return os.path.join(self.profile_dir, SESSION_COOKIES_FILE)

    async def _save_session_cookies(self) -> None:
        """Chromium writes only cookies with an expiry to the profile: keep the session cookies
        (expires -1, e.g. most logins) in a 0600 file next to them, written atomically. Values
        are never logged."""
        path = self._session_cookies_path()
        if path is None or self.context is None:
            return
        try:
            cookies = await asyncio.wait_for(self.context.cookies(), 5)
        except Exception as e:  # noqa: BLE001
            log.warning("could not read the session cookies: %s", short_error(e))
            return
        keep = [c for c in cookies if isinstance(c, dict) and c.get("expires", -1) in (-1, None)
                and isinstance(c.get("name"), str) and c.get("domain") and not c.get("partitionKey")]
        try:
            if not keep:
                if os.path.lexists(path):
                    os.unlink(path)
                return
            data = json.dumps(keep, ensure_ascii=True).encode("ascii")
            if len(data) > SESSION_COOKIES_MAX_BYTES:
                log.warning("session cookies too large to keep (%d bytes)", len(data))
                return
            tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            log.info("kept %d session cookie(s) for the next start", len(keep))
        except OSError as e:
            log.warning("could not keep the session cookies: %s", e)

    async def _restore_session_cookies(self) -> None:
        """Add the session cookies kept at the last shutdown, then delete the file (a crash later
        loses them, as it would in a real browser; a stale file never comes back)."""
        path = self._session_cookies_path()
        if path is None:
            return
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return
        except OSError as e:  # a link or worse: never followed, removed
            log.warning("ignoring the kept session cookies: %s", e)
            try:
                os.unlink(path)
            except OSError:
                pass
            return
        try:
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(SESSION_COOKIES_MAX_BYTES + 1)
            cookies = json.loads(raw) if len(raw) <= SESSION_COOKIES_MAX_BYTES else []
        except (OSError, ValueError):
            cookies = []
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        allowed = {"name", "value", "domain", "path", "httpOnly", "secure", "sameSite"}
        clean = []
        for cookie in cookies if isinstance(cookies, list) else []:
            if not isinstance(cookie, dict) or not isinstance(cookie.get("name"), str) \
                    or not isinstance(cookie.get("value"), str) or not isinstance(cookie.get("domain"), str):
                continue
            item = {k: v for k, v in cookie.items() if k in allowed}
            item.setdefault("path", "/")
            if item.get("sameSite") not in ("Strict", "Lax", "None"):
                item.pop("sameSite", None)
            clean.append(item)
        if not clean:
            return
        try:
            await self.context.add_cookies(clean)
            log.info("restored %d session cookie(s)", len(clean))
        except Exception as e:  # noqa: BLE001 - one bad cookie: try them one by one
            restored = 0
            for item in clean:
                try:
                    await self.context.add_cookies([item])
                    restored += 1
                except Exception:  # noqa: BLE001
                    pass
            log.info("restored %d of %d session cookie(s) (%s)", restored, len(clean), short_error(e))

    def _on_proxy_refused(self, refusal: EgressRefused) -> None:
        if self._in_call and len(self._proxy_refusals) < 50:
            self._proxy_refusals.append(refusal)

    # -- tabs and page events ---------------------------------------------------------------

    def _register(self, page) -> Tab:
        for tab in self.tabs.values():
            if tab.page is page:
                return tab
        self._tab_counter += 1
        tab = Tab(id=f"tab-{self._tab_counter}", page=page)
        self.tabs[tab.id] = tab
        page.on("close", lambda *_: self._forget(tab))
        page.on("download", self._on_download)
        page.on("dialog", lambda dialog: self._spawn(self._on_dialog(dialog)))
        page.on("request", lambda request: self._on_request(tab, request))
        page.on("requestfinished", lambda request: tab.pending_navs.pop(request, None))
        page.on("requestfailed", lambda request: tab.pending_navs.pop(request, None))
        return tab

    def _on_context_page(self, page) -> None:
        known = any(t.page is page for t in self.tabs.values())
        tab = self._register(page)
        if known:
            return
        if self._in_call:
            self._opened_in_call.append(tab.id)
        if len(self.tabs) > MAX_POPUP_TABS:
            self._notes.append("The page tried to open another tab; it was closed because too many tabs are open.")
            self._spawn(self._close_quietly(page))

    async def _close_quietly(self, page) -> None:
        try:
            await page.close()
        except Exception:  # noqa: BLE001
            pass

    def _forget(self, tab: Tab) -> None:
        if self.tabs.pop(tab.id, None) is None:
            return
        if tab.id in self._activation:
            self._activation.remove(tab.id)
        if tab.id in self._opened_in_call:
            self._opened_in_call.remove(tab.id)
        if self.active_id == tab.id:
            self.active_id = self._activation[-1] if self._activation else (next(reversed(self.tabs), None))

    async def _activate(self, tab: Tab) -> None:
        self.active_id = tab.id
        if tab.id in self._activation:
            self._activation.remove(tab.id)
        self._activation.append(tab.id)
        try:
            await tab.page.bring_to_front()
        except Exception:  # noqa: BLE001
            pass

    def _on_request(self, tab: Tab, request) -> None:
        try:
            if request.is_navigation_request() and request.frame == tab.page.main_frame:
                now = time.monotonic()
                tab.pending_navs[request] = now
                for stale in [r for r, t in tab.pending_navs.items() if now - t > 120]:
                    tab.pending_navs.pop(stale, None)
        except Exception:  # noqa: BLE001
            pass

    async def _on_dialog(self, dialog) -> None:
        kind = getattr(dialog, "type", "alert")
        message = clean_text(getattr(dialog, "message", ""), 200)
        note = None
        try:
            if kind == "beforeunload":
                await dialog.accept()
            elif kind == "alert":
                await dialog.accept()
                note = f"The page showed an alert: {quote(message)}"
            elif kind == "confirm":
                if matches_sensitive(message) and not self._call_allow_sensitive:
                    await dialog.dismiss()
                    self._call_dialog_refused = message
                elif matches_sensitive(message):
                    await dialog.accept()
                    self._call_dialog_sensitive = True
                    note = f"The page asked {quote(message)} and the browser answered OK."
                else:
                    await dialog.accept()
                    note = f"The page asked {quote(message)} and the browser answered OK."
            else:
                await dialog.dismiss()
                note = f"The page asked for input ({quote(message)}); such prompts are dismissed."
        except Exception as e:  # noqa: BLE001
            log.info("dialog handling failed: %s", short_error(e))
        if note:
            self._notes.append(note)

    def _on_download(self, download) -> None:
        did = str(uuid.uuid4())
        url = download_url(getattr(download, "url", ""))
        self._events.append({"type": "download_started", "download_id": did, "url": url})
        task = asyncio.ensure_future(self._finish_download(download, did, url))
        self._download_tasks.add(task)
        task.add_done_callback(self._download_tasks.discard)
        if self._in_call:
            self._call_downloads.add(task)

    async def _finish_download(self, download, did: str, url: str) -> None:
        name = None
        try:
            failure = await download.failure()
            if failure:
                self._events.append({"type": "download_failed", "download_id": did, "url": url,
                                     "error": clean_field(failure, 500)})
                return
            name = unique_name(self.downloads_dir, sanitize_filename(download.suggested_filename),
                               self._reserved_names)
            self._reserved_names.add(name)
            path = os.path.join(self.downloads_dir, name)
            await download.save_as(path)
            size = os.path.getsize(path)
            self._events.append({"type": "download_completed", "download_id": did, "url": url,
                                 "path": f"{DOWNLOADS_LABEL}/{name}", "size_bytes": int(size)})
        except Exception as e:  # noqa: BLE001
            self._events.append({"type": "download_failed", "download_id": did, "url": url,
                                 "error": short_error(e, 500)})
        finally:
            if name:
                self._reserved_names.discard(name)

    # -- network policy ---------------------------------------------------------------------

    async def _route(self, route, request) -> None:
        blocked = None
        try:
            if not self.allow_private:
                blocked = await self.policy.check_url(request.url)
        except Exception:  # noqa: BLE001
            blocked = None
        try:
            if blocked:
                try:
                    if request.is_navigation_request():
                        self._blocked_navs[id(request.frame.page)] = blocked
                except Exception:  # noqa: BLE001
                    pass
                log.info("blocked request to %s", blocked)
                await route.abort("blockedbyclient")
            elif self._is_resubmission(request):
                # back / forward / reload onto a page that was the answer to a form: Chromium sends
                # the form again (the route filter disables the HTTP cache)
                if self._call_allow_sensitive:
                    self._resubmitted = True
                    await route.fallback()
                else:
                    self._resubmit_refused = str(getattr(request, "method", "POST"))
                    await route.abort("aborted")
            else:
                await route.fallback()
        except Exception:  # noqa: BLE001 - the page went away meanwhile
            pass

    def _is_resubmission(self, request) -> bool:
        page = self._history_page
        if page is None:
            return False
        try:
            return (request.is_navigation_request() and request.frame == page.main_frame
                    and str(request.method).upper() not in ("GET", "HEAD"))
        except Exception:  # noqa: BLE001
            return False

    async def _route_ws(self, ws) -> None:
        try:
            blocked = None if self.allow_private else await self.policy.check_url(ws.url)
            if blocked:
                log.info("blocked websocket to %s", blocked)
                await ws.close(code=1008, reason="Blocked by the sandbox network policy")
            else:
                ws.connect_to_server()
        except Exception:  # noqa: BLE001
            pass

    # -- client protocol --------------------------------------------------------------------

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while not reader.at_eof():
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await self._write(writer, {"ok": False, "error": "Error: request too large."})
                    break
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("not an object")
                except ValueError:
                    await self._write(writer, {"ok": False, "error": "Error: invalid request (expected one JSON object per line)."})
                    continue
                response = await self.dispatch(request)
                if "id" in request:
                    response = {"id": request.get("id"), **response}
                await self._write(writer, response)
                if request.get("action") == "_shutdown":
                    self.request_shutdown("_shutdown requested")
                    break
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as e:  # noqa: BLE001
            log.exception("client handler failed: %s", e)
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, payload: dict) -> None:
        writer.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("ascii"))
        await writer.drain()

    async def dispatch(self, request: dict) -> dict:
        action = request.get("action")
        if not isinstance(action, str):
            return {"ok": False, "error": "Error: the request has no action.", "needs_confirmation": False}
        if self._stopping and action != "_ping":
            return {"ok": False, "error": "Error: the browser is shutting down.", "needs_confirmation": False}
        if action == "_ping":
            return {"ok": True}
        if action == "_touch":
            self._touch()
            return {"ok": True}
        if action == "_status":
            return await self._status()
        if action == "_shutdown":
            return {"ok": True}
        if action == "_peek":
            async with self.lock:
                return await self._peek()
        if action in DISABLED_MEMBERS:
            self._touch()
            return self._failure(f"Error: {action} is not enabled in this environment.")
        if action not in self.members:
            return self._failure(f"Error: Unknown browser action {quote(clean_text(action, 60))}.")
        inp = request.get("input")
        if inp is None:
            inp = {}
        if not isinstance(inp, dict):
            return self._failure("Error: input must be an object.")
        self._in_flight += 1
        self._touch()
        try:
            async with self.lock:
                return await self._call(action, inp, request.get("allow_sensitive") is True,
                                        force_state=request.get("force_state") is True)
        finally:
            self._in_flight -= 1
            self._touch()

    def _meta(self) -> dict:
        tab = self.tabs.get(self.active_id or "")
        if tab is None:
            return {"tab_id": None, "title": "", "url": ""}
        try:
            url = tab.page.url
        except Exception:  # noqa: BLE001
            url = ""
        return {"tab_id": tab.id, "title": clean_field(tab.title), "url": display_url(url)}

    def _failure(self, message: str, needs_confirmation: bool = False) -> dict:
        return {"ok": False, "error": message, "needs_confirmation": needs_confirmation, "meta": self._meta()}

    async def _call(self, member: str, inp: dict, allow_sensitive: bool, force_state: bool = False) -> dict:
        self._in_call = True
        self._call_started = time.monotonic()
        self._call_allow_sensitive = allow_sensitive
        self._call_dialog_sensitive = False
        self._call_dialog_refused = None
        self._proxy_refusals = []
        self._history_page = None
        self._resubmit_refused = None
        self._resubmitted = False
        self._opened_in_call = []
        self._notes = []
        self._call_downloads = set()
        self._blocked_navs = {}
        try:
            try:
                outcome = await asyncio.wait_for(self.members[member](inp), CALL_TIMEOUT_SECONDS)
            except ActionError as e:
                return self._failure(e.message, e.needs_confirmation)
            except asyncio.TimeoutError:
                return self._failure(f"Error: The browser action timed out after {int(CALL_TIMEOUT_SECONDS)} seconds.")
            except PlaywrightError as e:
                return self._failure(f"Error: {member} failed: {short_error(e)}")
            except Exception as e:  # noqa: BLE001
                log.exception("%s failed", member)
                return self._failure(f"Error: {member} failed: {type(e).__name__}: {short_error(e)}")
            if self._call_dialog_refused is not None:
                return self._failure(
                    f"Error: The page asked {quote(self._call_dialog_refused)} and the browser answered "
                    "Cancel, because this looks like a consequential action. Ask the user first with "
                    "browser_confirm, then repeat the action.", needs_confirmation=True)
            if member not in TAB_MEMBERS:
                await self._adopt_new_tabs()
            if self._call_downloads:
                await asyncio.wait(set(self._call_downloads), timeout=DOWNLOAD_WAIT_SECONDS)
            return await self._success(member, outcome, force_state)
        finally:
            self._in_call = False
            self._history_page = None

    async def _adopt_new_tabs(self) -> None:
        opened = [tid for tid in self._opened_in_call if tid in self.tabs]
        if opened:
            await self._activate(self.tabs[opened[-1]])

    async def _success(self, member: str, outcome: Outcome, force_state: bool = False) -> dict:
        entries = await self._state_entries()
        changes: list[dict] = []
        if member == "new_tab":
            if self.active_id:
                changes.append({"type": "tab_opened", "tab_id": self.active_id})
        elif member not in TAB_MEMBERS:
            for tid in dict.fromkeys(self._opened_in_call):
                if tid in self.tabs:
                    changes.append({"type": "tab_opened", "tab_id": tid})
        events, self._events = self._events, []
        changes.extend(merge_download_events(events))
        text = outcome.text
        if self._notes and member not in TAB_MEMBERS:
            text = ((text + "\n") if text else "") + "\n".join(self._notes)
        content, signature = assemble_content(member, entries=entries, text=text, image_b64=outcome.image,
                                              changes=changes, previous_signature=self._last_reported,
                                              force_state=force_state)
        problems = validate_content(member, content)
        if problems:  # never send something the API rejects: drop the state block on other members
            log.error("invalid %s content: %s", member, problems)
            self._events = events + self._events  # report the downloads on the next result
            if member in TAB_MEMBERS:
                return self._failure(f"Error: {member} could not report the tab state.")
            content = [b for b in content if b.get("type") != "browser_state"] or [text_block("Done.")]
        else:
            self._last_reported = signature
        sensitive = (outcome.sensitive or self._call_dialog_sensitive) and self._call_allow_sensitive
        return {"ok": True, "content": content, "meta": self._meta(), "sensitive_done": bool(sensitive)}

    async def _title(self, tab: Tab, timeout: float = 2.0) -> str:
        try:
            tab.title = await asyncio.wait_for(tab.page.title(), timeout)
        except Exception:  # noqa: BLE001 - mid-navigation: keep the last known title
            pass
        return tab.title

    async def _state_entries(self) -> list[dict]:
        raw = []
        for tab in list(self.tabs.values()):
            title = await self._title(tab)
            try:
                url = tab.page.url
            except Exception:  # noqa: BLE001
                url = ""
            raw.append((tab.id, title, display_url(url)))
        if self.tabs and self.active_id not in self.tabs:
            self.active_id = self._activation[-1] if self._activation else next(reversed(self.tabs))
        return make_tab_entries(raw, self.active_id)

    async def _status(self) -> dict:
        tabs = []
        for tab in list(self.tabs.values()):
            title = await self._title(tab, timeout=0.5)
            try:
                url = tab.page.url
            except Exception:  # noqa: BLE001
                url = ""
            tabs.append((tab.id, title, display_url(url)))
        return {"ok": True, "tabs": make_tab_entries(tabs, self.active_id), "started": self.started,
                "last_request": self.last_request}

    async def _peek(self) -> dict:
        tab = self.tabs.get(self.active_id or "")
        if tab is None:
            return self._failure("Error: No tab is open.")
        try:
            data = await tab.page.screenshot(type="jpeg", quality=PEEK_QUALITY, timeout=15_000)
        except Exception as e:  # noqa: BLE001
            return self._failure(f"Error: screenshot failed: {short_error(e)}")
        await self._title(tab, timeout=1.0)
        return {"ok": True, "image_b64": base64.b64encode(data).decode("ascii"),
                "media_type": "image/jpeg", "meta": self._meta()}

    # -- shared action helpers --------------------------------------------------------------

    async def _tab_for(self, inp: dict, create: bool = False) -> Tab:
        tab_id = inp.get("tab_id")
        if tab_id not in (None, ""):
            tab_id = check_tab_id(tab_id)
            tab = self.tabs.get(tab_id)
            if tab is None:
                raise ActionError(f"Error: No open tab has tab_id {quote(clean_text(tab_id, 80))}. "
                                  "Use list_tabs to see the open tabs.")
            if tab.id != self.active_id:
                await self._activate(tab)
            return tab
        tab = self.tabs.get(self.active_id or "")
        if tab is not None:
            return tab
        if create:
            page = await self.context.new_page()
            tab = self._register(page)
            if tab.id not in self._opened_in_call:
                self._opened_in_call.append(tab.id)
            await self._activate(tab)
            return tab
        raise ActionError("Error: No tab is open. Use new_tab or navigate first.")

    async def _agent(self, tab: Tab, args: dict):
        """Run the page script; once more after a navigation destroyed the page's context."""
        args = dict(args)
        for attempt in (1, 2):
            args["nextRef"] = self.next_ref
            try:
                result = await tab.page.evaluate(AGENT_JS, args)
                break
            except PlaywrightError as e:
                if attempt == 2 or not context_lost(e):
                    raise
                try:
                    await tab.page.wait_for_load_state("domcontentloaded", timeout=SETTLE_SECONDS * 1000)
                except Exception:  # noqa: BLE001
                    pass
        if isinstance(result, dict) and isinstance(result.get("nextRef"), int):
            self.next_ref = max(self.next_ref, result["nextRef"])
        return result

    async def _element(self, tab: Tab, ref: str):
        try:
            handle = await tab.page.evaluate_handle(AGENT_JS, {"op": "get", "ref": ref, "nextRef": self.next_ref})
        except PlaywrightError as e:
            raise ActionError(f"Error: Could not look up {ref}: {short_error(e)}") from e
        element = handle.as_element()
        if element is None:
            try:
                await handle.dispose()
            except Exception:  # noqa: BLE001
                pass
            raise ActionError(stale_ref_error(ref))
        return element

    async def _describe(self, tab: Tab, **where) -> dict | None:
        """describe() through the page script: a ref (``ref=``), or the fallbacks of
        ``_describe_at`` / ``_describe_focused`` when the DevTools protocol fails."""
        try:
            desc = await self._agent(tab, {"op": "describe", **where})
        except Exception as e:  # noqa: BLE001 - best effort: an unreadable target is not "sensitive"
            log.info("describe failed: %s", short_error(e))
            return None
        if isinstance(desc, dict) and desc.get("stale"):
            raise ActionError(stale_ref_error(where.get("ref", "the reference")))
        return desc if isinstance(desc, dict) else None

    # -- what an action would hit, through closed shadow roots and cross-origin frames ------------

    async def _page_cdp(self, tab: Tab):
        if tab.cdp is None:
            tab.cdp = await self.context.new_cdp_session(tab.page)
        return tab.cdp

    async def _frame_session(self, tab: Tab, frame_id: str):
        """(CDP session, Playwright Frame) of the out-of-process frame ``frame_id``, or None."""
        hit = tab.frame_sessions.get(frame_id)
        if hit is not None and not hit[1].is_detached():
            return hit
        tab.frame_sessions.pop(frame_id, None)
        if len(tab.frame_sessions) > 40 or len(tab.in_process) > 400:
            for session, _ in tab.frame_sessions.values():
                self._spawn(self._detach_quietly(session))
            tab.frame_sessions.clear()
            tab.in_process.clear()
        mapped = {id(frame) for _, frame in tab.frame_sessions.values()}
        for frame in list(tab.page.frames):
            if frame is tab.page.main_frame or frame.is_detached() or id(frame) in mapped:
                continue
            key = (id(frame), frame.url)
            if key in tab.in_process:
                continue
            try:
                session = await asyncio.wait_for(self.context.new_cdp_session(frame), 3)
            except Exception:  # noqa: BLE001 - a frame in its parent's process has no session of its own
                tab.in_process.add(key)
                continue
            try:
                tree = await asyncio.wait_for(session.send("Page.getFrameTree"), 3)
                fid = tree["frameTree"]["frame"]["id"]
            except Exception:  # noqa: BLE001
                self._spawn(self._detach_quietly(session))
                continue
            tab.frame_sessions[fid] = (session, frame)
            if fid == frame_id:
                return session, frame
        return None

    @staticmethod
    async def _detach_quietly(session) -> None:
        try:
            await session.detach()
        except Exception:  # noqa: BLE001
            pass

    async def _frame_info(self, frame, owner: dict) -> dict:
        title = ""
        try:
            title = await asyncio.wait_for(frame.title(), 1.0) if frame is not None else ""
        except Exception:  # noqa: BLE001
            pass
        return {"url": display_url(frame.url if frame is not None else owner.get("ownerSrc") or ""),
                "title": clean_text(title, 200), "ownerName": owner.get("name") or "",
                "ownerSrc": owner.get("ownerSrc") or ""}

    @staticmethod
    async def _call_on(session, object_id: str, declaration: str, args: dict | None = None,
                       by_value: bool = True):
        params = {"objectId": object_id, "functionDeclaration": declaration, "returnByValue": by_value,
                  "objectGroup": PROBE_GROUP}
        if args is not None:
            params["arguments"] = [{"value": args}]
        result = await session.send("Runtime.callFunctionOn", params)
        if result.get("exceptionDetails"):
            raise PlaywrightError("page script failed")
        return result.get("result") or {}

    @staticmethod
    async def _release(sessions) -> None:
        for session in sessions:
            try:
                await session.send("Runtime.releaseObjectGroup", {"objectGroup": PROBE_GROUP})
            except Exception:  # noqa: BLE001
                pass

    async def _describe_at(self, tab: Tab, x: float, y: float) -> dict | None:
        """describe() of the element a click at viewport (x, y) lands on: DOM.getNodeForLocation
        sees through closed shadow roots and same-process frames; an out-of-process (cross-site)
        frame is entered through its own DevTools session with the point moved into it. A frame
        that cannot be entered comes back as ``opaqueFrame`` (only payment-looking ones count)."""
        used = []
        try:
            session = await self._page_cdp(tab)
            used.append(session)
            vx, vy = float(x), float(y)
            frame_info = None
            for _ in range(MAX_FRAME_DEPTH):
                metrics = await session.send("Page.getLayoutMetrics")
                layout = metrics.get("cssLayoutViewport") or {}
                hit = await session.send("DOM.getNodeForLocation", {
                    "x": int(round(vx + float(layout.get("pageX") or 0))),
                    "y": int(round(vy + float(layout.get("pageY") or 0))),
                    "includeUserAgentShadowDOM": False, "ignorePointerEventsNone": False})
                backend = hit["backendNodeId"]
                resolved = await session.send("DOM.resolveNode", {"backendNodeId": backend, "objectGroup": PROBE_GROUP})
                desc = (await self._call_on(session, resolved["object"]["objectId"], DESCRIBE_NODE_FN, {})).get("value")
                if not isinstance(desc, dict):
                    return None
                if frame_info:
                    desc["frame"] = frame_info
                if not desc.get("frameOwner") or desc.get("frameOpen"):
                    return desc
                node = (await session.send("DOM.describeNode", {"backendNodeId": backend, "depth": 0, "pierce": True}))["node"]
                quad = (await session.send("DOM.getBoxModel", {"backendNodeId": backend}))["model"]["content"]
                left, top = min(quad[0::2]), min(quad[1::2])
                right, bottom = max(quad[0::2]), max(quad[1::2])
                if node.get("contentDocument") or not (left <= vx < right and top <= vy < bottom):
                    return desc  # the frame's border, or a frame this session already looked into
                found = await self._frame_session(tab, node.get("frameId") or "")
                if found is None:
                    desc["opaqueFrame"] = True
                    desc["frame"] = await self._frame_info(None, desc)
                    return desc
                session, frame = found
                used.append(session)
                frame_info = await self._frame_info(frame, desc)
                vx, vy = vx - left, vy - top
            return None
        except ActionError:
            raise
        except Exception as e:  # noqa: BLE001 - fall back to the page script (no closed roots, no frames)
            log.info("hit test through DevTools failed: %s", short_error(e))
            return await self._describe(tab, x=x, y=y)
        finally:
            await self._release(used)

    async def _describe_focused(self, tab: Tab) -> dict | None:
        """describe() of the element that has the keyboard focus, followed into closed shadow
        roots (DOM.describeNode lists them) and cross-origin frames (their own sessions)."""
        used = []
        try:
            session = await self._page_cdp(tab)
            used.append(session)
            start = await session.send("Runtime.evaluate", {"expression": "document", "objectGroup": PROBE_GROUP})
            obj = (start.get("result") or {}).get("objectId")
            frame_info = None
            for _ in range(MAX_FRAME_DEPTH * 4):
                if not obj:
                    return None
                walked = await self._call_on(session, obj, FOCUS_WALK_FN, by_value=False)
                obj = walked.get("objectId")
                if not obj or walked.get("subtype") == "null":
                    return None
                node = (await session.send("DOM.describeNode", {"objectId": obj, "depth": 0, "pierce": True}))["node"]
                inner = None
                for root in node.get("shadowRoots") or []:  # a closed root holding the focus
                    resolved = await session.send("DOM.resolveNode", {"backendNodeId": root["backendNodeId"],
                                                                      "objectGroup": PROBE_GROUP})
                    root_id = resolved["object"].get("objectId")
                    if root_id and (await self._call_on(session, root_id, "function() { return !!this.activeElement; }")).get("value"):
                        inner = root_id
                        break
                if inner:
                    obj = inner
                    continue
                if node.get("nodeName") in ("IFRAME", "FRAME") and node.get("frameId"):
                    document = node.get("contentDocument")
                    if document:  # same process, but the page script may not look in (cross-origin)
                        resolved = await session.send("DOM.resolveNode", {"backendNodeId": document["backendNodeId"],
                                                                          "objectGroup": PROBE_GROUP})
                        obj = resolved["object"].get("objectId")
                        continue
                    owner = (await self._call_on(session, obj, DESCRIBE_NODE_FN, {})).get("value") or {}
                    found = await self._frame_session(tab, node["frameId"])
                    if found is None:
                        owner["opaqueFrame"] = True
                        owner["frame"] = await self._frame_info(None, owner)
                        return owner
                    session, frame = found
                    used.append(session)
                    frame_info = await self._frame_info(frame, owner)
                    start = await session.send("Runtime.evaluate", {"expression": "document", "objectGroup": PROBE_GROUP})
                    obj = (start.get("result") or {}).get("objectId")
                    continue
                desc = (await self._call_on(session, obj, DESCRIBE_NODE_FN, {})).get("value")
                if isinstance(desc, dict) and frame_info:
                    desc["frame"] = frame_info
                return desc if isinstance(desc, dict) else None
            return None
        except ActionError:
            raise
        except Exception as e:  # noqa: BLE001
            log.info("focus lookup through DevTools failed: %s", short_error(e))
            return await self._describe(tab, focused=True)
        finally:
            await self._release(used)

    def _allow(self, label: str | None, performed: bool = False) -> bool:
        """Raise needs_confirmation for a consequential action that is not allowed; True when the
        action is consequential (and allowed). ``performed``: earlier steps of the same call (keys
        of a sequence, typed text) already happened."""
        if not label:
            return False
        if not self._call_allow_sensitive:
            message = confirmation_error(label)
            if performed:
                message += (" The steps before it in this action were already done (the focus may have "
                            "moved); after the user allowed it, repeat only the rest.")
            raise ActionError(message, needs_confirmation=True)
        return True

    def _guard(self, desc: dict | None, kind: str) -> bool:
        return self._allow(sensitive_label(desc, kind))

    def _viewport(self, tab: Tab) -> tuple[int, int]:
        """The tab's real viewport: a popup opened with its own size is smaller than the default."""
        try:
            size = tab.page.viewport_size
            if size and size.get("width") and size.get("height"):
                return int(size["width"]), int(size["height"])
        except Exception:  # noqa: BLE001
            pass
        return self.width, self.height

    async def _ref_point(self, tab: Tab, ref: str) -> tuple[float, float]:
        element = await self._element(tab, ref)
        try:
            try:
                await element.scroll_into_view_if_needed(timeout=5_000)
            except PlaywrightError:
                pass
            box = await element.bounding_box()
        finally:
            try:
                await element.dispose()
            except Exception:  # noqa: BLE001
                pass
        if not box or box["width"] <= 0 or box["height"] <= 0:
            raise ActionError(f"Error: {ref} is not visible on the page, so it cannot be used as a target. "
                              "Use read_page or find to choose another element.")
        width, height = self._viewport(tab)
        x0, y0 = max(box["x"], 0.0), max(box["y"], 0.0)
        x1, y1 = min(box["x"] + box["width"], float(width)), min(box["y"] + box["height"], float(height))
        if x1 <= x0 or y1 <= y0:
            raise ActionError(f"Error: {ref} is outside the visible area even after scrolling it into view.")
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def _coordinate(self, tab: Tab, target, label: str = "target") -> tuple[int, int]:
        width, height = self._viewport(tab)
        return parse_coordinate(target, width, height, label)

    async def _target_point(self, tab: Tab, target, label: str = "target"):
        """(x, y, how to name it in the result, describe() of the target, describe() of what is
        actually under the point: a ref's centre may land on a child, e.g. the "Delete" button in
        a cell or the "Pay now" button in the middle of a dialog)."""
        if isinstance(target, dict) and target.get("type") == "ref":
            ref = parse_ref(target, label)
            x, y = await self._ref_point(tab, ref)
            desc = await self._describe(tab, ref=ref)
            return x, y, ref, desc, await self._describe_at(tab, x, y)
        x, y = self._coordinate(tab, target, label)
        desc = await self._describe_at(tab, x, y)
        return float(x), float(y), f"({x}, {y})", desc, desc

    async def _settle(self, tab: Tab, timeout: float = SETTLE_SECONDS) -> None:
        """After an action that may navigate or open a tab: wait (bounded) for it to load."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        await asyncio.sleep(0.2)
        while loop.time() < deadline:
            pending = [r for r, started in tab.pending_navs.items() if started >= self._call_started - 0.5]
            if not pending:
                break
            await asyncio.sleep(0.05)
        pages = [tab.page] + [self.tabs[t].page for t in self._opened_in_call if t in self.tabs]
        for page in pages:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=remaining * 1000)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _where(how: str, desc: dict | None) -> str:
        if desc and (desc.get("name") or desc.get("role") not in (None, "generic")):
            return f"{how} ({describe_label(desc.get('role'), desc.get('name'))})"
        return how

    # -- members: navigation and tabs -------------------------------------------------------

    async def m_navigate(self, inp: dict) -> Outcome:
        url = inp.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ActionError("Error: navigate needs a url.")
        special = url.strip().lower()
        if special in ("back", "forward", "reload"):
            target = None
        else:
            target, error = normalize_navigation_url(url)
            if error:
                raise ActionError(error)
        tab = await self._tab_for(inp, create=True)
        page = tab.page
        if target is None:
            before = page.url
            self._history_page = page  # the route filter stops a form being sent again (POST)
            try:
                if special == "back":
                    response = await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                elif special == "forward":
                    response = await page.go_forward(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                else:
                    response = await page.reload(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeoutError as e:
                self._refuse_resubmission(special)
                raise ActionError(f"Error: Going {special} timed out after 30 seconds.") from e
            except PlaywrightError as e:
                self._refuse_resubmission(special)
                self._refuse_private(page, str(e))
                raise ActionError(f"Error: Could not go {special}: {short_error(e)}") from e
            finally:
                self._history_page = None
            self._refuse_resubmission(special)
            if response is None and special != "reload" and page.url == before:
                where = "earlier" if special == "back" else "later"
                raise ActionError(f"Error: There is no {where} page in this tab's history.")
            await self._network_idle(page)
            await self._refuse_private_landing(page)
            text = "Reloaded " + display_url(page.url) if special == "reload" else f"Navigated {special} to {display_url(page.url)}"
            if self._resubmitted:
                text += " (the page's form was sent again, as allowed)"
            return Outcome(text=text, sensitive=self._resubmitted)

        host = urlsplit(target).hostname or ""
        if await self.policy.is_blocked(host):
            raise ActionError(f"Error: Navigation refused. {clean_text(host, 200)} is a private or local address.")
        shown = clean_text(display_url(target), 300)
        response = None
        try:
            response = await page.goto(target, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError as e:
            raise ActionError(f"Error: Navigation to {shown} timed out after 30 seconds. "
                              "The page may be unavailable.") from e
        except PlaywrightError as e:
            if self._call_downloads or "Download is starting" in str(e):
                return Outcome(text=f"Opened {shown}: the server sent a file, which is being downloaded.")
            self._refuse_private(page, str(e))
            failed = self._proxy_failure(host)
            if failed is not None:
                raise ActionError(f"Error: Could not open {shown}: {failed}.") from e
            raise ActionError(f"Error: Could not open {shown}: {short_error(e)}") from e
        await self._network_idle(page)
        await self._refuse_private_landing(page)
        text = f"Navigated to {display_url(page.url)}"
        status = getattr(response, "status", None) if response is not None else None
        if status == 502:
            failed = self._proxy_failure(urlsplit(page.url).hostname or "")
            if failed is not None:
                raise ActionError(f"Error: Could not open {shown}: {failed}.")
        if isinstance(status, int) and status >= 400:
            text += f" (HTTP {status})"
        return Outcome(text=text)

    def _refuse_resubmission(self, special: str) -> None:
        if self._resubmit_refused:
            method = clean_text(self._resubmit_refused, 10).upper()
            raise ActionError(confirmation_error(
                f'navigate "{special}" would send the form of that page again ({method})'), needs_confirmation=True)

    def _refuse_private(self, page, error: str) -> None:
        """A navigation that failed because it was (or was redirected) to a private address: the
        route filter saw the navigation request, or the proxy refused a tunnel during this call."""
        blocked = self._blocked_navs.get(id(page))
        if not blocked and ("ERR_TUNNEL_CONNECTION_FAILED" in error or "ERR_PROXY" in error):
            blocked = next((r.host for r in self._proxy_refusals if r.reason == "private"), None)
        if blocked:
            raise ActionError(f"Error: Navigation refused. {clean_text(blocked, 200)} is a private or local address.")

    async def _refuse_private_landing(self, page) -> None:
        """A plain-http redirect to a private address ends on the proxy's refusal page: say so."""
        if self.allow_private:
            return
        host = urlsplit(page.url).hostname or ""
        if host and await self.policy.is_blocked(host):
            raise ActionError(f"Error: Navigation refused. The page redirected to {clean_text(host, 200)}, "
                              "a private or local address.")

    def _proxy_failure(self, host: str) -> str | None:
        name = (host or "").strip("[]").rstrip(".").lower()
        for refusal in self._proxy_refusals:
            if refusal.reason != "private" and refusal.host.strip("[]").rstrip(".").lower() == name:
                return f"{clean_text(refusal.host, 200)} {EgressRefused.REASONS.get(refusal.reason, refusal.reason)}"
        return None

    async def _network_idle(self, page) -> None:
        try:
            await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
        except Exception:  # noqa: BLE001 - busy pages never go idle
            pass

    async def m_new_tab(self, inp: dict) -> Outcome:
        if len(self.tabs) >= MAX_TABS:
            raise ActionError(f"Error: {len(self.tabs)} tabs are open, the maximum is {MAX_TABS}. "
                              "Close one with close_tab first.")
        page = await self.context.new_page()
        tab = self._register(page)
        await self._activate(tab)
        return Outcome()

    async def m_list_tabs(self, inp: dict) -> Outcome:
        return Outcome()

    async def m_switch_tab(self, inp: dict) -> Outcome:
        tab_id = check_tab_id(inp.get("tab_id"))
        tab = self.tabs.get(tab_id)
        if tab is None:
            raise ActionError(f"Error: No open tab has tab_id {quote(clean_text(tab_id, 80))}. "
                              "Use list_tabs to see the open tabs.")
        await self._activate(tab)
        return Outcome()

    async def m_close_tab(self, inp: dict) -> Outcome:
        tab_id = check_tab_id(inp.get("tab_id"))
        tab = self.tabs.get(tab_id)
        if tab is None:
            raise ActionError(f"Error: No open tab has tab_id {quote(clean_text(tab_id, 80))}. "
                              "Use list_tabs to see the open tabs.")
        try:
            await tab.page.close()
        except PlaywrightError as e:
            raise ActionError(f"Error: Could not close {tab_id}: {short_error(e)}") from e
        self._forget(tab)
        current = self.tabs.get(self.active_id or "")
        if current is not None:
            await self._activate(current)
        return Outcome()

    # -- members: seeing the page -----------------------------------------------------------

    async def m_screenshot(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        try:
            data = await tab.page.screenshot(type="jpeg", quality=SCREENSHOT_QUALITY, timeout=15_000)
        except PlaywrightError as e:
            raise ActionError(f"Error: The screenshot failed: {short_error(e)}") from e
        return Outcome(image=base64.b64encode(data).decode("ascii"))

    async def m_zoom(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        region = inp.get("region")
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ActionError("Error: region must be [x0, y0, x1, y1] in viewport pixels.")
        x0, y0, x1, y1 = (_as_int(v, "region") for v in region)
        vw, vh = self._viewport(tab)
        if not (0 <= x0 < x1 <= vw and 0 <= y0 < y1 <= vh):
            raise ActionError(f"Error: region [{x0}, {y0}, {x1}, {y1}] must satisfy 0 <= x0 < x1 <= {vw} "
                              f"and 0 <= y0 < y1 <= {vh}.")
        width, height = x1 - x0, y1 - y0
        scale = min(ZOOM_MAX_SIDE / max(width, height), ZOOM_MAX_SCALE)
        scale = max(scale, 0.1)
        cdp = await self._page_cdp(tab)
        offset = await tab.page.evaluate("() => [window.visualViewport ? visualViewport.pageLeft : scrollX,"
                                         " window.visualViewport ? visualViewport.pageTop : scrollY]")
        try:
            shot = await cdp.send("Page.captureScreenshot", {
                "format": "jpeg", "quality": SCREENSHOT_QUALITY,
                "clip": {"x": offset[0] + x0, "y": offset[1] + y0, "width": width, "height": height,
                         "scale": scale},
            })
        except PlaywrightError as e:
            raise ActionError(f"Error: The zoom failed: {short_error(e)}") from e
        return Outcome(image=shot["data"])

    async def m_read_page(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        flt = inp.get("filter")
        if flt not in (None, "", "all", "interactive"):
            raise ActionError("Error: filter must be \"interactive\", \"all\" or omitted.")
        depth = inp.get("depth")
        depth = 15 if depth in (None, "") else _as_int(depth, "depth")
        if depth < 1:
            raise ActionError("Error: depth must be at least 1.")
        ref = inp.get("ref")
        if ref not in (None, ""):
            ref = parse_ref({"type": "ref", "ref": ref}, "ref")
        else:
            ref = None
        result = await self._agent(tab, {"op": "tree", "filter": flt or None, "depth": min(depth, 200),
                                         "ref": ref, "cap": READ_PAGE_CAP})
        if not isinstance(result, dict):
            raise ActionError("Error: The page could not be read.")
        if result.get("stale"):
            raise ActionError(stale_ref_error(ref or "the reference"))
        nodes = result.get("nodes") or []
        scroll = result.get("scroll") or {}
        note = ""
        if not flt and not ref and scroll.get("h", 0) > scroll.get("vh", 0) + 4:
            note = (f"\n(Only elements in the viewport are listed; the page is scrolled to y={scroll.get('y', 0)} "
                    f"of {scroll.get('h', 0)} px. Use filter \"all\" for the whole page.)")
        # the whole answer, notices included, stays within READ_PAGE_CAP
        text = render_tree(nodes, READ_PAGE_CAP - len(note), bool(result.get("truncated")), notice_cap=READ_PAGE_CAP)
        if not text:
            text = "No elements found" + (" in the viewport." if not flt and not ref else ".")
        return Outcome(text=text + note)

    async def m_find(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        query = inp.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ActionError("Error: find needs a query such as \"search field\".")
        result = await self._agent(tab, {"op": "find", "limit": FIND_CANDIDATES})
        candidates = (result or {}).get("cands") or []
        picked = rank_find_candidates(query, candidates, FIND_LIMIT)
        shown_query = quote(clean_text(query, 100))
        if not picked:
            await self._agent(tab, {"op": "refs", "indices": []})
            return Outcome(text=f"No elements match {shown_query}. Try read_page or a screenshot.")
        refs = await self._agent(tab, {"op": "refs", "indices": [candidates[i]["i"] for i in picked]})
        lines = []
        for position, ref in zip(picked, (refs or {}).get("refs") or []):
            if not ref:
                continue
            node = dict(candidates[position])
            node["ref"] = ref
            node["level"] = 0
            lines.append(format_node_line(node, indent=False))
        if not lines:
            return Outcome(text=f"No elements match {shown_query}. Try read_page or a screenshot.")
        header = f"{len(lines)} match{'es' if len(lines) != 1 else ''} for {shown_query}:"
        return Outcome(text=header + "\n" + "\n".join(lines))

    async def m_get_page_text(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        try:
            html = await tab.page.content()
            body = await tab.page.evaluate("() => document.body ? document.body.innerText : ''")
        except PlaywrightError as e:
            raise ActionError(f"Error: The page text could not be read: {short_error(e)}") from e
        text = await asyncio.to_thread(_extract_text, html, body or "")
        if not text.strip():
            return Outcome(text="The page has no readable text.")
        return Outcome(text=cap_text(text, PAGE_TEXT_CAP))

    # -- members: pointer -------------------------------------------------------------------

    async def _click(self, inp: dict, button: str, count: int, guarded: bool) -> Outcome:
        tab = await self._tab_for(inp)
        modifiers = parse_modifiers(inp.get("modifiers"))
        x, y, how, desc, at_point = await self._target_point(tab, inp.get("target"))
        sensitive = self._allow(first_sensitive_label([at_point, desc], "click")) if guarded else False
        page = tab.page
        await page.mouse.move(x, y)
        for key in modifiers:
            await page.keyboard.down(key)
        try:
            if count == 2:
                await page.mouse.dblclick(x, y, button=button)
            else:
                await page.mouse.click(x, y, button=button, click_count=count)
        finally:
            for key in reversed(modifiers):
                try:
                    await page.keyboard.up(key)
                except Exception:  # noqa: BLE001
                    pass
        self._mouse_pressed = None
        await self._settle(tab)
        verb = {1: "Clicked", 2: "Double-clicked", 3: "Triple-clicked"}[count]
        if button != "left":
            verb = f"{button.capitalize()}-clicked"
        return Outcome(text=f"{verb} {self._where(how, desc)}.", sensitive=sensitive)

    async def m_left_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 1, guarded=True)

    async def m_right_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "right", 1, guarded=False)  # opens a context menu only

    async def m_middle_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "middle", 1, guarded=True)  # opens a link (e.g. /delete?id=7)

    async def m_double_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 2, guarded=True)

    async def m_triple_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 3, guarded=True)

    async def m_hover(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y, how, desc, _ = await self._target_point(tab, inp.get("target"))
        await tab.page.mouse.move(x, y)
        await asyncio.sleep(0.15)
        return Outcome(text=f"Hovered over {self._where(how, desc)}.")

    async def m_left_click_drag(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        start = inp.get("from", inp.get("from_"))
        x0, y0 = self._coordinate(tab, start, "from")
        x1, y1 = self._coordinate(tab, inp.get("target"), "target")
        # pressing on a button and letting go on it clicks it; dropping onto "Delete" acts too
        sensitive = self._allow(first_sensitive_label(
            [await self._describe_at(tab, x0, y0), await self._describe_at(tab, x1, y1)], "click"))
        mouse = tab.page.mouse
        await mouse.move(x0, y0)
        await mouse.down()
        try:
            await mouse.move(x1, y1, steps=12)
        finally:
            await mouse.up()
        self._mouse_pressed = None
        await self._settle(tab, timeout=1.5)
        return Outcome(text=f"Dragged from ({x0}, {y0}) to ({x1}, {y1}).", sensitive=sensitive)

    async def m_left_mouse_down(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = self._coordinate(tab, inp.get("target"))
        sensitive = self._allow(sensitive_label(await self._describe_at(tab, x, y), "click"))
        await tab.page.mouse.move(x, y)
        await tab.page.mouse.down()
        self._mouse_pressed = (tab.id, x, y)
        return Outcome(text=f"Pressed the left mouse button at ({x}, {y}).", sensitive=sensitive)

    async def m_left_mouse_up(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = self._coordinate(tab, inp.get("target"))
        sensitive = False
        if self._mouse_pressed != (tab.id, x, y):  # the press was not checked at this point
            sensitive = self._allow(sensitive_label(await self._describe_at(tab, x, y), "click"))
        await tab.page.mouse.move(x, y)
        await tab.page.mouse.up()
        self._mouse_pressed = None
        await self._settle(tab, timeout=1.5)
        return Outcome(text=f"Released the left mouse button at ({x}, {y}).", sensitive=sensitive)

    async def m_mouse_move(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = self._coordinate(tab, inp.get("target"))
        await tab.page.mouse.move(x, y)
        return Outcome(text=f"Moved the mouse to ({x}, {y}).")

    async def m_scroll(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        direction = inp.get("scroll_direction")
        if direction not in ("up", "down", "left", "right"):
            raise ActionError("Error: scroll_direction must be up, down, left or right.")
        amount = inp.get("scroll_amount")
        amount = 3 if amount in (None, "") else _as_int(amount, "scroll_amount")
        if not 1 <= amount <= 10:
            raise ActionError("Error: scroll_amount must be between 1 and 10.")
        x, y = self._coordinate(tab, inp.get("target"))
        delta = amount * SCROLL_NOTCH_PX
        dx, dy = {"up": (0, -delta), "down": (0, delta), "left": (-delta, 0), "right": (delta, 0)}[direction]
        await tab.page.mouse.move(x, y)
        await tab.page.mouse.wheel(dx, dy)
        await asyncio.sleep(0.35)
        text = f"Scrolled {direction} by {amount} at ({x}, {y})."
        try:
            pos = await self._agent(tab, {"op": "scroll"})
            if isinstance(pos, dict) and pos.get("h"):
                text += f" The page is now at y={pos['y']} of {pos['h']} px."
        except Exception:  # noqa: BLE001
            pass
        return Outcome(text=text)

    async def m_scroll_to(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        ref = parse_ref(inp.get("target"))
        element = await self._element(tab, ref)
        try:
            await element.scroll_into_view_if_needed(timeout=5_000)
        except PlaywrightError as e:
            raise ActionError(f"Error: {ref} could not be scrolled into view: {short_error(e)}") from e
        finally:
            try:
                await element.dispose()
            except Exception:  # noqa: BLE001
                pass
        return Outcome(text=f"Scrolled {ref} into view.")

    # -- members: keyboard ------------------------------------------------------------------

    async def _key_guard(self, tab: Tab, kind: str, state: dict, focus: dict | None = None) -> bool:
        """Before a key that may activate the focused element (Enter, Space): look at what has
        the focus NOW (earlier keys may have moved it; ``focus`` when just looked up). Raises
        needs_confirmation; returns False when the key may go, True when the call must stop here
        (a second consequential key: one confirmation covers one action)."""
        if focus is None:
            focus = await self._describe_focused(tab)
        label = sensitive_label(focus, kind)
        if not label:
            return False
        if state["sensitive"]:
            state["stopped"] = label
            return True
        state["sensitive"] = self._allow(label, performed=state["done"] > 0)
        return False

    @staticmethod
    def _stopped_note(state: dict) -> str:
        return (f" Stopped before the next consequential key ({state['stopped']}): one confirmation "
                "covers one action. Ask the user again with browser_confirm for the next one.")

    async def m_type(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        if not isinstance(text, str) or text == "":
            raise ActionError("Error: type needs the text to type.")
        if len(text) > MAX_TYPE_CHARS:
            raise ActionError(f"Error: type takes at most {MAX_TYPE_CHARS:,} characters at a time.")
        keyboard = tab.page.keyboard
        state = {"sensitive": False, "stopped": None, "done": 0}
        newline = "\n" in text or "\r" in text
        if not any(ch in "\r\n\t " for ch in text):
            await self._type_chunk(keyboard, text)
            return Outcome(text=f"Typed {len(text)} character{'s' if len(text) != 1 else ''}.")
        # A newline is an Enter key press; a space presses Space, which clicks a focused button or
        # link; a tab moves the focus. Check what has the focus before each Enter, and before a
        # space whenever the focus may have moved (at the start, after a tab or an Enter).
        buffer: list[str] = []
        focus: dict | None = None
        focus_known = False
        first = min((i for i in (text.find("\n"), text.find("\r")) if i >= 0), default=-1)
        if first >= 0 and "\t" not in text[:first]:
            # typing into a text field does not move the focus: when its first Enter would be
            # refused, refuse before typing anything (the model can repeat the whole text)
            focus = await self._describe_focused(tab)
            focus_known = True
            if focus and focus.get("textEntry") and not self._call_allow_sensitive:
                self._allow(sensitive_label(focus, "enter"))

        async def flush():
            if buffer:
                chunk = "".join(buffer)
                buffer.clear()
                await self._type_chunk(keyboard, chunk)
                state["done"] += len(chunk)

        for ch in text:
            if ch in "\r\n":
                await flush()
                if await self._key_guard(tab, "enter", state):
                    break
                await keyboard.type(ch)
                state["done"] += 1
                focus_known = False
            elif ch == " ":
                if not focus_known:
                    await flush()
                    focus = await self._describe_focused(tab)
                    focus_known = True
                if focus and focus.get("textEntry"):
                    buffer.append(ch)
                    continue
                await flush()
                if await self._key_guard(tab, "space", state, focus or {}):
                    break
                await keyboard.type(ch)
                state["done"] += 1
                # Space on a control (a click) may move the focus or change the page; on the page
                # itself (nothing focused) it only scrolls
                focus_known = not (focus and focus.get("actionable"))
            elif ch == "\t":
                buffer.append(ch)
                await flush()
                focus_known = False
            else:
                buffer.append(ch)
        if not state["stopped"]:
            await flush()
        if newline or state["sensitive"]:
            await self._settle(tab)
        done = state["done"]
        if state["stopped"]:
            result = f"Typed {done} of {len(text)} characters." + self._stopped_note(state)
        else:
            result = f"Typed {done} character{'s' if done != 1 else ''}."
        return Outcome(text=result, sensitive=state["sensitive"])

    @staticmethod
    async def _type_chunk(keyboard, chunk: str) -> None:
        if len(chunk) > TYPE_INSERT_THRESHOLD and "\t" not in chunk:
            # key events per character are slow for long text: insert the bulk, type the tail so
            # key listeners (autocomplete, validation) still see real key presses
            await keyboard.insert_text(chunk[:-TYPE_TAIL_CHARS])
            await keyboard.type(chunk[-TYPE_TAIL_CHARS:])
        else:
            await keyboard.type(chunk)

    async def m_key(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        chords = parse_key_sequence(text)
        repeat = inp.get("repeat")
        repeat = 1 if repeat in (None, "") else _as_int(repeat, "repeat")
        if not 1 <= repeat <= 100:
            raise ActionError("Error: repeat must be between 1 and 100.")
        activating = any(activation_kind(chord) for chord in chords)
        state = {"sensitive": False, "stopped": None, "done": 0}
        page = tab.page
        try:
            for _ in range(repeat):
                for chord in chords:
                    kind = activation_kind(chord)
                    if kind and await self._key_guard(tab, kind, state):
                        break
                    await page.keyboard.press(chord_to_playwright(chord))
                    state["done"] += 1
                if state["stopped"]:
                    break
        except PlaywrightError as e:
            message = short_error(e)
            if "Unknown key" in message:
                raise ActionError(f"Error: {message} in {quote(clean_text(text, 60))}.") from e
            raise
        if activating:
            await self._settle(tab)
        else:
            await asyncio.sleep(0.1)
        shown = clean_text(text, 60)
        times = f" {repeat} times" if repeat > 1 else ""
        result = f"Pressed {shown}{times}."
        if state["stopped"]:
            result = f"Pressed {state['done']} key(s) of {shown}{times}." + self._stopped_note(state)
        return Outcome(text=result, sensitive=state["sensitive"])

    async def m_hold_key(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        keys = [k for chord in parse_key_sequence(text) for k in chord]
        duration = _as_float(inp.get("duration"), "duration")
        if not 0 <= duration <= 30:
            raise ActionError("Error: duration must be between 0 and 30 seconds.")
        state = {"sensitive": False, "stopped": None, "done": 0}
        page = tab.page
        pressed = []
        try:
            for key in keys:
                kind = activation_kind(pressed + [key]) if key in ENTER_KEYS or key == "Space" else None
                if kind and await self._key_guard(tab, kind, state):
                    break
                await page.keyboard.down(key)
                pressed.append(key)
                state["done"] += 1
            await asyncio.sleep(duration)
        except PlaywrightError as e:
            raise ActionError(f"Error: {short_error(e)}") from e
        finally:
            for key in reversed(pressed):
                try:
                    await page.keyboard.up(key)
                except Exception:  # noqa: BLE001
                    pass
        result = f"Held {clean_text(text, 60)} for {duration:g} seconds."
        if state["stopped"]:
            result += self._stopped_note(state)
        return Outcome(text=result, sensitive=state["sensitive"])

    async def m_wait(self, inp: dict) -> Outcome:
        duration = _as_float(inp.get("duration"), "duration")
        if not 0 <= duration <= 30:
            raise ActionError("Error: duration must be between 0 and 30 seconds.")
        if inp.get("tab_id") not in (None, ""):
            await self._tab_for(inp)
        await asyncio.sleep(duration)
        return Outcome(text=f"Waited {duration:g} seconds.")

    # -- members: forms ---------------------------------------------------------------------

    async def m_form_input(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        ref = parse_ref(inp.get("target"))
        value = inp.get("value")
        if value is None or not isinstance(value, (str, int, float, bool)):
            raise ActionError("Error: value must be a string, a number or a boolean.")
        element = await self._element(tab, ref)
        try:
            desc = await self._describe(tab, ref=ref)
            sensitive = self._guard(desc, "form_input")
            info = await element.evaluate(
                """(e) => ({tag: e.tagName, type: (e.getAttribute('type') || '').toLowerCase(),
                           role: (e.getAttribute('role') || '').toLowerCase(), editable: e.isContentEditable,
                           checked: !!e.checked, aria: e.getAttribute('aria-checked'),
                           readonly: !!e.readOnly, disabled: !!e.disabled})""")
            label = describe_label((desc or {}).get("role"), (desc or {}).get("name"))
            if info["disabled"]:
                raise ActionError(f"Error: {ref} ({label}) is disabled.")
            tag, kind, role = info["tag"], info["type"], info["role"]
            shown = None
            if (tag == "INPUT" and kind in ("checkbox", "radio")) or role in ("checkbox", "switch", "radio", "menuitemcheckbox"):
                want = _as_bool(value)
                if tag == "INPUT":
                    try:
                        await element.set_checked(want, timeout=3_000)
                    except PlaywrightError:
                        await element.evaluate(
                            """(e, v) => { if (e.checked !== v) { e.checked = v;
                               e.dispatchEvent(new Event('input', {bubbles: true}));
                               e.dispatchEvent(new Event('change', {bubbles: true})); } }""", want)
                elif (info["aria"] == "true") != want:
                    await element.click(timeout=5_000)
                shown = "checked" if want else "unchecked"
            elif tag == "SELECT":
                if isinstance(value, bool):
                    raise ActionError(f"Error: {ref} is a select; give an option's value or visible text.")
                option = _value_text(value)
                try:
                    chosen = await element.select_option(option, timeout=5_000)
                except PlaywrightError as e:
                    raise ActionError(f"Error: {ref} has no option {quote(clean_text(option, 80))}.") from e
                if not chosen:
                    raise ActionError(f"Error: {ref} has no option {quote(clean_text(option, 80))}.")
                shown = quote(clean_text(option, 80))
            elif tag == "INPUT" and kind == "range":
                if isinstance(value, bool):
                    raise ActionError(f"Error: {ref} is a slider; give a number.")
                await element.evaluate(
                    """(e, v) => { e.value = v; e.dispatchEvent(new Event('input', {bubbles: true}));
                       e.dispatchEvent(new Event('change', {bubbles: true})); }""", _value_text(value))
                shown = quote(_value_text(value))
            elif tag in ("INPUT", "TEXTAREA") or info["editable"] or role in ("textbox", "searchbox", "spinbutton", "combobox"):
                if isinstance(value, bool):
                    raise ActionError(f"Error: {ref} is a text field; give a string or a number.")
                if info["readonly"]:
                    raise ActionError(f"Error: {ref} ({label}) is read-only.")
                text = _value_text(value)
                try:
                    await element.fill(text, timeout=5_000)
                except PlaywrightError as e:
                    raise ActionError(f"Error: Could not fill {ref}: {short_error(e)}") from e
                shown = '"\u2022\u2022\u2022\u2022"' if kind == "password" else quote(clean_text(text, 80))
            else:
                raise ActionError(f"Error: {ref} ({label}) is not a form field. Use left_click for buttons "
                                  "and links, or type into a focused field.")
        finally:
            try:
                await element.dispose()
            except Exception:  # noqa: BLE001
                pass
        await self._settle(tab, timeout=2.0)
        return Outcome(text=f"Set {ref} to {shown}.", sensitive=sensitive)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("true", "on", "yes", "checked", "1"):
        return True
    if text in ("false", "off", "no", "unchecked", "0", ""):
        return False
    raise ActionError("Error: a checkbox needs true or false.")


def _value_text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_TRAFILATURA = None


def _extract_text(html: str, body: str) -> str:
    """trafilatura on the rendered HTML (worker thread), else the body's innerText."""
    global _TRAFILATURA
    extracted = None
    try:
        if _TRAFILATURA is None:
            import trafilatura  # noqa: WPS433 - heavy import, only on first use

            _TRAFILATURA = trafilatura
        extracted = _TRAFILATURA.extract(html, include_tables=True, include_links=False,
                                         include_comments=False)
    except Exception:  # noqa: BLE001
        extracted = None
    return choose_page_text(extracted, body)


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Chromium for the browser toolset, on a unix socket.")
    parser.add_argument("--socket", required=True, help="unix socket path to listen on")
    parser.add_argument("--profile", default="", help='persistent profile directory; "" = ephemeral (deleted on exit)')
    parser.add_argument("--downloads", default="downloads", help="directory for downloaded files")
    parser.add_argument("--idle-seconds", type=float, default=960, help="exit after this long without requests (0 = never)")
    parser.add_argument("--viewport", default="1280x800", help="viewport WIDTHxHEIGHT (screenshot pixels = coordinates)")
    parser.add_argument("--allow-private", action="store_true", help="tests only: allow private and local addresses")
    parser.add_argument("--executable", default="", help="tests only: path of the Chromium binary")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    server = BrowserServer(args)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, server.request_shutdown, f"signal {sig.name}")
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await server.start()
    except Exception as e:  # noqa: BLE001
        log.error("startup failed: %s", short_error(e, 1000))
        await server.shutdown("startup failed")
        return 1
    return await server.wait()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s browser_server %(levelname)s %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        parse_viewport(args.viewport)
    except ValueError as e:
        log.error("%s", e)
        return 2
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
