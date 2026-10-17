#!/usr/bin/env python3
"""Persistent per-user browser for Anthropic's client-side browser toolset (browser_toolset_20260801).

Runs INSIDE the user's browser container (the sandbox image: Python 3.12, Playwright with Chromium,
trafilatura) as the non-root ``runner`` user. It drives ONE persistent Chromium profile and serves
the toolset's member calls over a unix socket, one JSON object per line each way::

    -> {"id": "7", "action": "left_click", "input": {"target": {"type": "ref", "ref": "ref_2"}},
        "allow_sensitive": false}
    <- {"id": "7", "ok": true, "content": [...API-ready blocks...], "meta": {...}, "sensitive_done": false}
    <- {"id": "7", "ok": false, "error": "Error: ...", "needs_confirmation": false, "meta": {...}}

``content`` is what the bot puts into the member's ``tool_result`` unchanged: only ``text``,
``image`` (base64 JPEG) and ``browser_state`` blocks, following the toolset rules (tab-management
members answer with exactly one ``browser_state`` block, ``new_tab`` carries exactly one
``tab_opened`` for the active tab, no ``browser_state`` on errors, never an empty
``state_changes``, sanitized tab fields). Internal actions: ``_ping`` (health check), ``_peek``
(viewport JPEG for the chat's live view), ``_status`` (tabs and timestamps), ``_touch`` (counts as
activity, so the idle exit restarts) and ``_shutdown``. Only member calls and ``_touch`` count as
activity; after ``--idle-seconds`` without any the server closes the browser and exits by itself,
so a crashed bot never leaves a browser running.

Element references: ``read_page`` and ``find`` tag elements ``ref_N``. An injected script keeps a
WeakMap element -> ref in the page and a ref -> WeakRef(element) map, so a ref is stable while the
document lives; the counter is kept per tab on this side and never restarts, so a ref handed out
before a navigation can never silently point at a different element afterwards (it is reported
stale). Ref targets are scrolled into view and clicked at the centre of their visible box with
the real mouse, like coordinate targets.

Safety (best effort, defence in depth next to the container's egress filter):
- every request (route ``**/*``) and websocket to a host that is, or resolves to, a private,
  loopback, link-local, reserved or multicast address is aborted (DNS answers cached 60 s);
  ``navigate`` refuses such hosts and every scheme but http(s); ``--allow-private`` (tests only)
  turns the address check off;
- clicks, ``form_input`` on non-text controls, Enter (``key``/``hold_key``/a newline in ``type``)
  on things that look like purchases, payments, sending, publishing, deleting or account/security
  changes are refused with ``needs_confirmation`` unless the request carries
  ``allow_sensitive: true``; then ``sensitive_done`` tells the bot the allowance was used. A page's
  own confirm() dialog that looks consequential is answered Cancel unless allowed.

Downloads land in ``--downloads`` under a sanitized unique name and are reported as
``download_started`` / ``download_completed`` (path ``downloads/<name>``) / ``download_failed``
state changes on the call during which they happened, or on the next successful one.

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
                 "--disable-blink-features=AutomationControlled"]
FALLBACK_CHROME_MAJOR = "141"

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
    U+2028/U+2029, at most ``limit`` characters."""
    text = "" if value is None else str(value)
    out = []
    for ch in text:
        if ch in "\u2028\u2029" or unicodedata.category(ch) in ("Cc", "Cs"):
            out.append(" ")
        else:
            out.append(ch)
    text = re.sub(r" {2,}", " ", "".join(out)).strip()
    return text[:limit]


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
    """Decides whether a host may be contacted; DNS answers are cached ``ttl`` seconds."""

    def __init__(self, allow_private: bool = False, ttl: float = DNS_CACHE_SECONDS, resolver=None):
        self.allow_private = allow_private
        self.ttl = ttl
        self._cache: dict[str, tuple[float, bool]] = {}
        self._resolver = resolver  # async (host) -> list[str]; tests inject one

    async def _resolve(self, host: str) -> list[str]:
        if self._resolver is not None:
            return await self._resolver(host)
        loop = asyncio.get_running_loop()
        infos = await asyncio.wait_for(loop.getaddrinfo(host, None, type=socket.SOCK_STREAM),
                                       DNS_TIMEOUT_SECONDS)
        return [info[4][0] for info in infos]

    async def is_blocked(self, host: str | None) -> bool:
        if self.allow_private or not host:
            return False
        literal = host_is_literal_private(host)
        if literal is not None:
            return literal
        name = host.strip().rstrip(".").lower()
        now = time.monotonic()
        cached = self._cache.get(name)
        if cached and cached[0] > now:
            return cached[1]
        try:
            addresses = await self._resolve(name)
            blocked = any(is_private_address(a) for a in addresses)
        except Exception:  # noqa: BLE001 - unresolvable here: the browser fails on its own
            blocked = False
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[name] = (now + self.ttl, blocked)
        return blocked

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
    r"\bsubscribe\b", r"\bdonate\b", r"\btransfer\b", r"\btop[\s-]?up\b",
    r"\bоплат\w*", r"\bкупить\b", r"\bзаказ\w*", r"\bоформить\b", r"\bперевести\b", r"\bподписаться\b",
    # sending and publishing
    r"\bsend\b", r"\bpost\b", r"\bpublish\b", r"\btweet\b", r"\breply\b", r"\bcomment\b",
    r"\bотправить\b", r"\bопубликовать\b", r"\bответить\b",
    # deleting data, account and security changes
    r"\bdelete\b", r"\bremove\s+(?:my\s+|the\s+|this\s+|your\s+)?account\b", r"\bdeactivate\b",
    r"\bclose\s+(?:my\s+|the\s+|this\s+|your\s+)?account\b",
    r"\bchange\s+(?:my\s+|the\s+|your\s+)?password\b", r"\bchange\s+(?:my\s+|the\s+|your\s+)?e-?mail\b",
    r"\bудалить\b", r"\bсменить\s+пароль\b",
]
_SENSITIVE_RE = re.compile("|".join(f"(?:{p})" for p in _SENSITIVE_PATTERNS), re.IGNORECASE)

# Phrases removed before matching: registration and login steps are never consequential.
_SAFE_PHRASES_RE = re.compile("|".join(f"(?:{p})" for p in [
    r"\b(?:re)?send\s+(?:me\s+)?(?:a\s+|the\s+|an?\s+|new\s+)*(?:verification\s+|confirmation\s+|security\s+|"
    r"login\s+|log[\s-]?in\s+|sign[\s-]?in\s+|magic\s+|one[\s-]time\s+|activation\s+|reset\s+)?"
    r"(?:code|link|otp|sms|pin|email)\b",
    r"\bотправить\s+(?:ещё\s+|еще\s+|повторно\s+)?(?:код|ссылк|смс|sms|письмо)\w*",
    r"\bмои\s+заказы\b", r"\bистори\w*\s+заказ\w*", r"\bстатус\w*\s+заказ\w*", r"\bотслежива\w*\s+заказ\w*",
]), re.IGNORECASE)

_COMPOSER_RE = re.compile(
    r"\bmessage|\breply|\bcomment|\bwrite\b|сообщени|комментар|напиш|\bответ", re.IGNORECASE)
_SEARCH_RE = re.compile(r"search|поиск|найти", re.IGNORECASE)


def matches_sensitive(name: str | None) -> bool:
    """Does an accessible name read like a purchase, payment, send/post or deletion?"""
    text = clean_text(name or "")
    if not text:
        return False
    text = _SAFE_PHRASES_RE.sub(" ", text)
    return bool(_SENSITIVE_RE.search(text))


def looks_like_composer(*texts: str | None) -> bool:
    joined = " ".join(clean_text(t or "") for t in texts)
    return bool(joined.strip()) and bool(_COMPOSER_RE.search(joined)) and not _SEARCH_RE.search(joined)


def describe_label(role: str | None, name: str | None) -> str:
    role = clean_text(role or "element", 30) or "element"
    name = clean_text(name or "", 80)
    return f"{role} {quote(name)}" if name else role


def sensitive_label(desc: dict | None, kind: str) -> str | None:
    """The guard's verdict: a label such as 'button "Place order"' when the action looks
    consequential, else None. ``kind`` is "click", "form_input" or "enter"."""
    if not desc:
        return None
    role = desc.get("role") or "element"
    name = desc.get("name") or ""
    if desc.get("actionable") is False and not desc.get("textEntry"):
        return None  # plain text, a heading, an empty area: clicking it does nothing consequential
    if kind == "click":
        return describe_label(role, name) if matches_sensitive(name) else None
    if kind == "form_input":
        if desc.get("textEntry"):
            return None  # filling a text field sends nothing by itself
        return describe_label(role, name) if matches_sensitive(name) else None
    # Enter: on a button or link it activates it; in a field it submits the form or a composer
    if not desc.get("textEntry"):
        return describe_label(role, name) if matches_sensitive(name) else None
    if not desc.get("multiline"):
        for button in desc.get("submit") or []:
            if matches_sensitive(button):
                return describe_label("button", button)
    if desc.get("inSearch"):
        return None
    if looks_like_composer(name, desc.get("placeholder"), desc.get("label")):
        shown = name or desc.get("placeholder") or desc.get("label") or ""
        return f"Enter in the text field {quote(clean_text(shown, 80))}"
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
                     previous_signature: tuple | None = None) -> tuple[list[dict], tuple | None]:
    """The success content for one member call and the signature of the state now reported.

    Tab-management members answer with exactly one browser_state block. Every other member
    answers with its text and/or image and adds browser_state only when the tab set, the active
    tab, a title or a URL changed since the last reported state, or there are state changes.
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
    include = bool(changes) or signature != previous_signature
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


def render_tree(nodes: list[dict], cap: int = READ_PAGE_CAP, truncated: bool = False) -> str:
    """read_page lines joined, cut at a line boundary under ``cap`` with a notice saying so."""
    lines: list[str] = []
    used = 0
    for node in nodes:
        line = format_node_line(node)
        if used + len(line) + 1 > cap:
            truncated = True
            break
        lines.append(line)
        used += len(line) + 1
    text = "\n".join(lines)
    if truncated:
        text += ("\n" if text else "") + truncation_notice(cap)
    return text


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

  function describe(el) {
    if (!el) return null;
    const ACTIONABLE = 'a[href],button,input,select,textarea,summary,label,[role=button],[role=link],' +
      '[role=menuitem],[role=menuitemcheckbox],[role=menuitemradio],[role=tab],[role=checkbox],' +
      '[role=switch],[role=radio],[role=option],[role=treeitem],[role=textbox],[role=searchbox],' +
      '[role=combobox],[onclick],[contenteditable=""],[contenteditable="true"]';
    let a = el.closest ? el.closest(ACTIONABLE) : null;
    if (a && a.tagName === 'LABEL') a = a.control || a;
    if (!a && el.isContentEditable) a = el;
    if (!a) {  // a div/span "button" without a role: the nearest cursor:pointer / onclick ancestor
      let cur = el;
      for (let i = 0; i < 8 && cur && cur.nodeType === 1 && cur !== cur.ownerDocument.body; i++, cur = cur.parentElement) {
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
    const form = target.form || (target.closest ? target.closest('form') : null);
    if (form) {
      const els = Array.from(form.elements || []).filter((b) =>
        (b.tagName === 'BUTTON' && (!b.getAttribute('type') || b.getAttribute('type').toLowerCase() === 'submit')) ||
        (b.tagName === 'INPUT' && ['submit', 'image'].includes((b.getAttribute('type') || '').toLowerCase())));
      submit = els.slice(0, 3).map((b) => accName(b, 'button')).filter((s) => s);
      if ((form.getAttribute('role') || '').toLowerCase() === 'search' || /search|поиск/i.test((form.getAttribute('action') || '') + ' ' + (form.getAttribute('aria-label') || '') + ' ' + (form.id || '') + ' ' + (form.className || ''))) inSearch = true;
    }
    if (target.closest && target.closest('[role=search], search')) inSearch = true;
    return {role, name, tag, type, textEntry, multiline, placeholder: cut(collapse(placeholder), 100), label: cut(collapse(label), 100), submit, inSearch, actionable: !!a};
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
    if (args.ref) { el = lookup(args.ref); if (!el) return {stale: true}; }
    else if (args.focused) { el = FOCUSED; if (el === document.body) el = null; }
    else el = deepPoint(document, args.x, args.y);
    return describe(el);
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
    next_ref: int = 1
    title: str = ""
    cdp: object = None
    pending_navs: dict = field(default_factory=dict)  # navigation request -> monotonic start


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

        self.playwright = None
        self.context = None
        self.server = None
        self.profile_dir = ""
        self.ephemeral = False

        self.tabs: dict[str, Tab] = {}
        self.active_id: str | None = None
        self._activation: list[str] = []
        self._tab_counter = 0

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
        try:
            if self.context is not None:
                await asyncio.wait_for(self.context.close(), 20)
        except Exception as e:  # noqa: BLE001
            log.warning("closing the browser failed: %s", short_error(e))
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
            else:
                await route.fallback()
        except Exception:  # noqa: BLE001 - the page went away meanwhile
            pass

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
                return await self._call(action, inp, bool(request.get("allow_sensitive")))
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

    async def _call(self, member: str, inp: dict, allow_sensitive: bool) -> dict:
        self._in_call = True
        self._call_started = time.monotonic()
        self._call_allow_sensitive = allow_sensitive
        self._call_dialog_sensitive = False
        self._call_dialog_refused = None
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
            return await self._success(member, outcome)
        finally:
            self._in_call = False

    async def _adopt_new_tabs(self) -> None:
        opened = [tid for tid in self._opened_in_call if tid in self.tabs]
        if opened:
            await self._activate(self.tabs[opened[-1]])

    async def _success(self, member: str, outcome: Outcome) -> dict:
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
                                              changes=changes, previous_signature=self._last_reported)
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
            args["nextRef"] = tab.next_ref
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
            tab.next_ref = max(tab.next_ref, result["nextRef"])
        return result

    async def _element(self, tab: Tab, ref: str):
        try:
            handle = await tab.page.evaluate_handle(AGENT_JS, {"op": "get", "ref": ref, "nextRef": tab.next_ref})
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
        try:
            desc = await self._agent(tab, {"op": "describe", **where})
        except Exception as e:  # noqa: BLE001 - best effort: an unreadable target is not "sensitive"
            log.info("describe failed: %s", short_error(e))
            return None
        if isinstance(desc, dict) and desc.get("stale"):
            raise ActionError(stale_ref_error(where.get("ref", "the reference")))
        return desc if isinstance(desc, dict) else None

    def _guard(self, desc: dict | None, kind: str) -> bool:
        """Raise needs_confirmation for a consequential action that is not allowed; True when the
        action is consequential (and allowed)."""
        label = sensitive_label(desc, kind)
        if not label:
            return False
        if not self._call_allow_sensitive:
            raise ActionError(confirmation_error(label), needs_confirmation=True)
        return True

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
        x0, y0 = max(box["x"], 0.0), max(box["y"], 0.0)
        x1, y1 = min(box["x"] + box["width"], float(self.width)), min(box["y"] + box["height"], float(self.height))
        if x1 <= x0 or y1 <= y0:
            raise ActionError(f"Error: {ref} is outside the visible area even after scrolling it into view.")
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    async def _target_point(self, tab: Tab, target, label: str = "target") -> tuple[float, float, str, dict | None]:
        """(x, y, how to name it in the result, describe() of the element there)."""
        if isinstance(target, dict) and target.get("type") == "ref":
            ref = parse_ref(target, label)
            x, y = await self._ref_point(tab, ref)
            desc = await self._describe(tab, ref=ref)
            return x, y, ref, desc
        x, y = parse_coordinate(target, self.width, self.height, label)
        desc = await self._describe(tab, x=x, y=y)
        return float(x), float(y), f"({x}, {y})", desc

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
            try:
                if special == "back":
                    response = await page.go_back(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                elif special == "forward":
                    response = await page.go_forward(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                else:
                    response = await page.reload(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            except PlaywrightTimeoutError as e:
                raise ActionError(f"Error: Going {special} timed out after 30 seconds.") from e
            except PlaywrightError as e:
                raise ActionError(f"Error: Could not go {special}: {short_error(e)}") from e
            if response is None and special != "reload" and page.url == before:
                where = "earlier" if special == "back" else "later"
                raise ActionError(f"Error: There is no {where} page in this tab's history.")
            await self._network_idle(page)
            text = "Reloaded " + display_url(page.url) if special == "reload" else f"Navigated {special} to {display_url(page.url)}"
            return Outcome(text=text)

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
            blocked = self._blocked_navs.get(id(page))
            if self._call_downloads or "Download is starting" in str(e):
                return Outcome(text=f"Opened {shown}: the server sent a file, which is being downloaded.")
            if blocked:
                raise ActionError(f"Error: Navigation refused. {clean_text(blocked, 200)} is a private or local address.") from e
            raise ActionError(f"Error: Could not open {shown}: {short_error(e)}") from e
        await self._network_idle(page)
        text = f"Navigated to {display_url(page.url)}"
        status = getattr(response, "status", None) if response is not None else None
        if isinstance(status, int) and status >= 400:
            text += f" (HTTP {status})"
        return Outcome(text=text)

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
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ActionError(f"Error: region [{x0}, {y0}, {x1}, {y1}] must satisfy 0 <= x0 < x1 <= {self.width} "
                              f"and 0 <= y0 < y1 <= {self.height}.")
        width, height = x1 - x0, y1 - y0
        scale = min(ZOOM_MAX_SIDE / max(width, height), ZOOM_MAX_SCALE)
        scale = max(scale, 0.1)
        if tab.cdp is None:
            tab.cdp = await self.context.new_cdp_session(tab.page)
        offset = await tab.page.evaluate("() => [window.visualViewport ? visualViewport.pageLeft : scrollX,"
                                         " window.visualViewport ? visualViewport.pageTop : scrollY]")
        try:
            shot = await tab.cdp.send("Page.captureScreenshot", {
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
        text = render_tree(nodes, READ_PAGE_CAP, bool(result.get("truncated")))
        scroll = result.get("scroll") or {}
        if not text:
            text = "No elements found" + (" in the viewport." if not flt and not ref else ".")
        if not flt and not ref and scroll.get("h", 0) > scroll.get("vh", 0) + 4:
            text += (f"\n(Only elements in the viewport are listed; the page is scrolled to y={scroll.get('y', 0)} "
                     f"of {scroll.get('h', 0)} px. Use filter \"all\" for the whole page.)")
        return Outcome(text=text)

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
        x, y, how, desc = await self._target_point(tab, inp.get("target"))
        sensitive = self._guard(desc, "click") if guarded else False
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
        await self._settle(tab)
        verb = {1: "Clicked", 2: "Double-clicked", 3: "Triple-clicked"}[count]
        if button != "left":
            verb = f"{button.capitalize()}-clicked"
        return Outcome(text=f"{verb} {self._where(how, desc)}.", sensitive=sensitive)

    async def m_left_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 1, guarded=True)

    async def m_right_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "right", 1, guarded=False)

    async def m_middle_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "middle", 1, guarded=False)

    async def m_double_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 2, guarded=True)

    async def m_triple_click(self, inp: dict) -> Outcome:
        return await self._click(inp, "left", 3, guarded=True)

    async def m_hover(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y, how, desc = await self._target_point(tab, inp.get("target"))
        await tab.page.mouse.move(x, y)
        await asyncio.sleep(0.15)
        return Outcome(text=f"Hovered over {self._where(how, desc)}.")

    async def m_left_click_drag(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        start = inp.get("from", inp.get("from_"))
        x0, y0 = parse_coordinate(start, self.width, self.height, "from")
        x1, y1 = parse_coordinate(inp.get("target"), self.width, self.height, "target")
        mouse = tab.page.mouse
        await mouse.move(x0, y0)
        await mouse.down()
        try:
            await mouse.move(x1, y1, steps=12)
        finally:
            await mouse.up()
        await self._settle(tab, timeout=1.5)
        return Outcome(text=f"Dragged from ({x0}, {y0}) to ({x1}, {y1}).")

    async def m_left_mouse_down(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = parse_coordinate(inp.get("target"), self.width, self.height)
        await tab.page.mouse.move(x, y)
        await tab.page.mouse.down()
        return Outcome(text=f"Pressed the left mouse button at ({x}, {y}).")

    async def m_left_mouse_up(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = parse_coordinate(inp.get("target"), self.width, self.height)
        await tab.page.mouse.move(x, y)
        await tab.page.mouse.up()
        await self._settle(tab, timeout=1.5)
        return Outcome(text=f"Released the left mouse button at ({x}, {y}).")

    async def m_mouse_move(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        x, y = parse_coordinate(inp.get("target"), self.width, self.height)
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
        x, y = parse_coordinate(inp.get("target"), self.width, self.height)
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

    async def m_type(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        if not isinstance(text, str) or text == "":
            raise ActionError("Error: type needs the text to type.")
        if len(text) > MAX_TYPE_CHARS:
            raise ActionError(f"Error: type takes at most {MAX_TYPE_CHARS:,} characters at a time.")
        sensitive = False
        newline = "\n" in text or "\r" in text
        if newline:  # a newline is an Enter key press
            sensitive = self._guard(await self._describe(tab, focused=True), "enter")
        keyboard = tab.page.keyboard
        if len(text) > TYPE_INSERT_THRESHOLD and not newline:
            # key events per character are slow for long text: insert the bulk, type the tail so
            # key listeners (autocomplete, validation) still see real key presses
            await keyboard.insert_text(text[:-TYPE_TAIL_CHARS])
            await keyboard.type(text[-TYPE_TAIL_CHARS:])
        else:
            await keyboard.type(text)
        if newline:
            await self._settle(tab)
        return Outcome(text=f"Typed {len(text)} character{'s' if len(text) != 1 else ''}.", sensitive=sensitive)

    async def m_key(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        chords = parse_key_sequence(text)
        repeat = inp.get("repeat")
        repeat = 1 if repeat in (None, "") else _as_int(repeat, "repeat")
        if not 1 <= repeat <= 100:
            raise ActionError("Error: repeat must be between 1 and 100.")
        has_enter = any(k in ENTER_KEYS for chord in chords for k in chord)
        sensitive = False
        if has_enter:
            sensitive = self._guard(await self._describe(tab, focused=True), "enter")
        page = tab.page
        try:
            for _ in range(repeat):
                for chord in chords:
                    await page.keyboard.press(chord_to_playwright(chord))
        except PlaywrightError as e:
            message = short_error(e)
            if "Unknown key" in message:
                raise ActionError(f"Error: {message} in {quote(clean_text(text, 60))}.") from e
            raise
        if has_enter:
            await self._settle(tab)
        else:
            await asyncio.sleep(0.1)
        shown = clean_text(text, 60)
        times = f" {repeat} times" if repeat > 1 else ""
        return Outcome(text=f"Pressed {shown}{times}.", sensitive=sensitive)

    async def m_hold_key(self, inp: dict) -> Outcome:
        tab = await self._tab_for(inp)
        text = inp.get("text")
        keys = [k for chord in parse_key_sequence(text) for k in chord]
        duration = _as_float(inp.get("duration"), "duration")
        if not 0 <= duration <= 30:
            raise ActionError("Error: duration must be between 0 and 30 seconds.")
        sensitive = False
        if any(k in ENTER_KEYS for k in keys):
            sensitive = self._guard(await self._describe(tab, focused=True), "enter")
        page = tab.page
        pressed = []
        try:
            for key in keys:
                await page.keyboard.down(key)
                pressed.append(key)
            await asyncio.sleep(duration)
        except PlaywrightError as e:
            raise ActionError(f"Error: {short_error(e)}") from e
        finally:
            for key in reversed(pressed):
                try:
                    await page.keyboard.up(key)
                except Exception:  # noqa: BLE001
                    pass
        return Outcome(text=f"Held {clean_text(text, 60)} for {duration:g} seconds.", sensitive=sensitive)

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
