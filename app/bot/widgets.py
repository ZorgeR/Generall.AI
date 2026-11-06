"""Telegram widgets (Bot API 10.4): mini-HTML the agent sends into the chat.

A widget is one self-contained HTML document (inline CSS and JS, ``data:`` images only) that every
Telegram client runs in a sandbox with no network and draws inside the message bubble
(TelegramMessenger/widget-sandbox: widget profile 1, SDK v1). The HTML never changes after it
is sent; what changes is its state, two texts of up to 1024 bytes that only the bot writes:
``chat_data`` (this message in this chat, never forwarded) and ``public_data`` (shown wherever the
widget is, forwards included, as of the forward). With ``allow_queries`` the widget may ask the
bot (``TelegramWidget.request(data)`` in a tap handler): the bot gets a ``widget_query`` update and
answers it within 15 s (``answerWidgetQuery``); ``bot/handlers/widget_queries.py`` turns such a tap
into an agent turn.

aiogram 3.31 speaks Bot API 10.3, so the three methods the bot uses are declared here
(``SendWidget``, ``EditMessageWidgetData``, ``AnswerWidgetQuery``) and called as ``await bot(m)``:
aiogram only serializes a method, the Bot API server is what must know it. A server without them
(the local ``telegram-bot-api`` before 10.4) answers 404. ``probe`` asks the server once at startup
(``sendWidget`` to chat 0: a server that knows the method refuses the chat with a 400, one that does not
answers 404) and ``mark_unsupported`` records a 404 from the probe or from any later call: from then on
the widget tools are not offered to the agent (``available``) and a call already under way fails with
``UNSUPPORTED_TEXT``, until a restart (a deploy that brings a newer server restarts the bot).

The registry of sent widgets (``data/_widgets/<uid>.json``: id, name, message, topic, the state
as last written) is bot-only state outside the workspace, like ``data/_mcp``: the sandbox can write
``data/<uid>``, and a record there could point an edit at another message. Read ``O_NOFOLLOW``,
written ``mkstemp`` + ``os.replace`` (0600, directory 0700), sync I/O under a per-user lock:
callers use ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import stat
import tempfile
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods.base import TelegramMethod
from aiogram.types import InlineKeyboardMarkup, Message, ReplyParameters

logger = logging.getLogger(__name__)

# ---- limits (docs/tgw-format.md of widget-sandbox; the server enforces them) -------------------
MAX_HTML_BYTES = 32_768  # UTF-8 bytes of HTML
MAX_PACKED_BYTES = 4_096  # after zlib level 9, as the server counts (PHP gzcompress($html, 9))
PACKED_MARGIN = 3_800  # what the widget-sandbox tools warn above
MAX_STATE_BYTES = 1_024  # chat_data / public_data, UTF-8 text without NUL
MAX_ANSWER_BYTES = 1_024  # answerWidgetQuery's data
MAX_SIZE_ANCHOR = 600  # width / height in CSS px; 0 = auto (clients clamp to 200-600 / 50-600)
MAX_CAPTION = 1_024
CLEARED_STATE = "null"  # an empty string cannot be sent (aiogram drops falsy fields): "null" stands for "no state"

MAX_WIDGETS = 50  # records kept per user (oldest dropped)
MAX_NAME_CHARS = 64
MAX_FILE_BYTES = 1024 * 1024  # a registry file larger than this reads as broken

PROBE_TIMEOUT = 10.0  # seconds the startup probe waits for the Bot API server
PROBE_CHAT_ID = 0  # no chat has this id: a server that knows sendWidget refuses it (400), nothing is sent
UNSUPPORTED_TEXT = (
    "Error: this bot's Telegram Bot API server does not support widgets yet (they need Bot API 10.4). "
    "The admin has to update it (docker compose pull telegram-bot-api). Show the content another way."
)


# ---- the Bot API methods (10.4) ----------------------------------------------------------------
class SendWidget(TelegramMethod[Message]):
    """``sendWidget``: a widget message. ``width`` / ``height`` 0 (or none) = auto."""

    __returning__ = Message
    __api_method__ = "sendWidget"

    chat_id: int | str
    html: str
    message_thread_id: int | None = None
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    chat_data: str | None = None
    public_data: str | None = None
    allow_queries: bool | None = None
    is_shared: bool | None = None
    disable_notification: bool | None = None
    reply_parameters: ReplyParameters | None = None
    reply_markup: InlineKeyboardMarkup | None = None


class EditMessageWidgetData(TelegramMethod[Message | bool]):
    """``editMessageWidgetData``: a silent edit of a widget message's state (a field not passed is unchanged)."""

    __returning__ = Message | bool
    __api_method__ = "editMessageWidgetData"

    chat_id: int | str | None = None
    message_id: int | None = None
    inline_message_id: str | None = None
    chat_data: str | None = None
    public_data: str | None = None


class AnswerWidgetQuery(TelegramMethod[bool]):
    """``answerWidgetQuery``: the answer to a ``widget_query`` (only the client that asked sees ``data``)."""

    __returning__ = bool
    __api_method__ = "answerWidgetQuery"

    widget_query_id: str
    data: str | None = None


# ---- a Bot API server without widgets ----------------------------------------------------------
_unsupported: str | None = None


def unsupported() -> str | None:
    """The reason widgets are off for this process, or None."""
    return _unsupported


def available() -> bool:
    """Whether the widget tools are offered to the agent: the server is not known to lack them
    (the probe said yes, or it could not tell and the first call will)."""
    return _unsupported is None


def mark_unsupported(reason: str) -> None:
    global _unsupported
    if _unsupported is None:
        logger.warning("Widgets are not supported by the Bot API server (%s): the widget tools are off until a restart "
                       "(they need Bot API 10.4)", reason)
    _unsupported = reason


def reset() -> None:
    """Tests: forget the unsupported flag."""
    global _unsupported
    _unsupported = None


def is_unsupported_error(e: BaseException) -> bool:
    """A server that does not know the method: 404 (the official one says "Not Found", the local
    one "Not Found: method not found"). Chats and messages that are not found are 400s."""
    text = str(e).lower()
    return isinstance(e, TelegramNotFound) or "method not found" in text or "unknown method" in text


async def probe(bot: Any, timeout: float = PROBE_TIMEOUT) -> bool | None:
    """Ask the Bot API server whether it knows ``sendWidget``, without sending anything: True (it
    refused chat 0 with a 400), False (404: marked unsupported) or None (no answer, another error:
    the tools stay listed and the first real call decides). Never raises."""
    try:
        await asyncio.wait_for(bot(SendWidget(chat_id=PROBE_CHAT_ID, html="<b>probe</b>")), timeout)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - the verdict is in the kind of error
        if is_unsupported_error(e):
            mark_unsupported(f"startup probe: {str(getattr(e, 'message', '') or e)}")
            return False
        if isinstance(e, TelegramBadRequest):
            logger.info("Widgets: the Bot API server knows sendWidget (Bot API 10.4); the widget tools are on")
            return True
        logger.warning("Could not tell whether the Bot API server supports widgets (%s: %s); the widget tools stay "
                       "listed and the first call decides", type(e).__name__, e)
        return None
    return True


def is_not_modified(e: BaseException) -> bool:
    return "not modified" in str(e).lower()


def is_message_gone(e: BaseException) -> bool:
    text = str(e).lower()
    return any(s in text for s in ("message_id_invalid", "message to edit not found", "message not found", "message can't be edited"))


# ---- size and content checks -------------------------------------------------------------------
def utf8(text: str) -> bytes | None:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return None


def measure(html: str) -> tuple[int, int]:
    """(HTML bytes, bytes after zlib level 9): the two numbers the server limits. Python's zlib at
    level 9 gives the same count as the server's PHP ``gzcompress($html, 9)``."""
    raw = html.encode("utf-8", "surrogatepass")
    return len(raw), len(zlib.compress(raw, 9))


def kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def check_html(html: str) -> str | None:
    """Why the server would refuse ``html``, or None."""
    if not isinstance(html, str) or not html.strip():
        return "Error: html is empty."
    raw = utf8(html)
    if raw is None:
        return "Error: html is not valid UTF-8 text."
    if b"\x00" in raw:
        return "Error: html contains NUL bytes; it must be plain text."
    size, packed = measure(html)
    if size > MAX_HTML_BYTES:
        return (f"Error: the widget is {size} bytes of HTML; the limit is {MAX_HTML_BYTES} bytes "
                f"(and {MAX_PACKED_BYTES} bytes compressed, it is {packed}). Make it smaller and send it again.")
    if packed > MAX_PACKED_BYTES:
        cut = 100 - (PACKED_MARGIN * 100 // packed)
        return (f"Error: the widget compresses to {packed} bytes; the limit is {MAX_PACKED_BYTES} (aim for "
                f"{PACKED_MARGIN}: cut about {cut}%). Remove comments, long texts, data: images and repeated "
                "markup (build repeated parts in JS), shorten class names, then send it again.")
    return None


def check_state(name: str, value: Any) -> str | None:
    """Why a ``chat_data`` / ``public_data`` value would be refused, or None (None = not given)."""
    if value is None:
        return None
    if not isinstance(value, str):
        return f"Error: {name} must be a string (JSON text, at most {MAX_STATE_BYTES} bytes)."
    raw = utf8(value)
    if raw is None or b"\x00" in raw:
        return f"Error: {name} must be UTF-8 text without NUL bytes."
    if len(raw) > MAX_STATE_BYTES:
        return f"Error: {name} is {len(raw)} bytes; the limit is {MAX_STATE_BYTES}. Send less (short keys, fewer points)."
    return None


def state_value(value: str | None) -> str | None:
    """What goes to the API: None stays None (not sent), an empty string becomes ``CLEARED_STATE``."""
    if value is None:
        return None
    return value if value != "" else CLEARED_STATE


_I = re.IGNORECASE
# Features every client removes or that throw: the widget would be broken, so it is not sent.
PROBLEMS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"<\s*(?:input|select|textarea|form|option)\b", _I),
     "form controls (<input>, <select>, <textarea>, <form>) are removed: a widget has no keyboard or text "
     "input; use buttons (steppers, chips, presets) and taps"),
    (re.compile(r"<\s*(?:iframe|video|audio|object|embed|dialog)\b", _I),
     "<iframe>, <video>, <audio>, <object>, <embed> and <dialog> are removed"),
    (re.compile(r"<\s*script\b[^>]*\bsrc\s*=", _I),
     "external scripts (<script src>) never load and no library is bundled: write plain JS (canvas or SVG for charts)"),
    (re.compile(r"<\s*link\b", _I), "<link> (stylesheets, fonts) never loads: put the CSS in <style>"),
    (re.compile(r"""\bsrc\s*=\s*["']?\s*(?:https?:)?//""", _I),
     "images load only from data: URLs (PNG, JPEG, GIF, WebP); draw with SVG or canvas instead"),
    (re.compile(r"""url\(\s*["']?\s*(?:https?:)?//""", _I), "CSS url() loads only data: images"),
    (re.compile(r"@import\b", _I), "@import never loads: put all CSS in <style>"),
    (re.compile(r"\bfetch\s*\(|\bXMLHttpRequest\b|\bnew\s+WebSocket\b|\bnew\s+EventSource\b|\bsendBeacon\s*\("),
     "a widget has no network (fetch, XMLHttpRequest, WebSocket throw): put the data into the HTML or the "
     "widget's state (chat_data / public_data) and update the state with update_widget"),
)
# Out of the profile or a likely mistake, but the widget still runs: reported with the result.
NOTES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\d\s*[dsl]?vh\b"), "vh units are removed: lay out with %, flex or grid"),
    (re.compile(r"overflow(?:-[xy])?\s*:\s*(?:auto|scroll)", _I),
     "nothing scrolls inside a widget (overflow auto/scroll becomes hidden); content past 600 px is clipped"),
    (re.compile(r"position\s*:\s*sticky", _I), "position: sticky becomes relative"),
    (re.compile(r"@font-face", _I), "@font-face is removed: system-ui, sans-serif, serif, monospace only"),
    (re.compile(r"(?<![\w.$])(?:alert|confirm|prompt)\s*\(\s*['\"`]"), "alert / confirm / prompt do not exist in a widget"),
    (re.compile(r"\b(?:keydown|keyup|keypress)\b"), "no key events reach a widget"),
    (re.compile(r"\btouch(?:start|move|end)\b"), "touch events are removed: use pointer events and click"),
    (re.compile(r"\bTelegram\.WebApp\b"), "Telegram.WebApp is the Mini Apps API: in a widget it is window.TelegramWidget"),
    (re.compile(r"\bTelegramWidget\.sendMessage\b"), "TelegramWidget.sendMessage does not exist: ask the bot with request(data)"),
)
REQUEST_CALL = re.compile(r"\.request\s*\(")


def lint(html: str, *, allow_queries: bool = False) -> tuple[list[str], list[str]]:
    """(problems, notes) found in ``html``: problems break the widget (it is not sent unless the
    caller insists), notes are worth knowing. Heuristics over the source, never a parser."""
    problems = [text for pattern, text in PROBLEMS if pattern.search(html)]
    notes = [text for pattern, text in NOTES if pattern.search(html)]
    if not allow_queries and "TelegramWidget" in html and REQUEST_CALL.search(html):
        notes.append("the HTML calls request() but allow_queries is off: every request fails with NOT_ALLOWED")
    return problems, notes


def raw_field(obj: Any, name: str) -> dict | None:
    """Field ``name`` of an aiogram object as the Bot API's JSON dict, or None. aiogram 3.31 (Bot API 10.3)
    keeps fields it does not know in ``model_extra``; a release with Bot API 10.4 will declare the widget
    fields as typed attributes instead: both are read, so upgrading aiogram does not silently lose them."""
    extra = getattr(obj, "model_extra", None)
    value = extra.get(name) if isinstance(extra, dict) else None
    if value is None and name in (getattr(type(obj), "model_fields", None) or {}):
        value = getattr(obj, name, None)
    if value is not None and not isinstance(value, dict) and hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)  # "from", not "from_user"
    return value if isinstance(value, dict) else None


def sent_widget_info(message: Any) -> dict:
    """The ``widget`` object of a sent message."""
    return raw_field(message, "widget") or {}


# ---- the registry (data/_widgets/<uid>.json) ---------------------------------------------------
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def store_dir() -> Path:
    return Path("./data") / "_widgets"


def _uid(user_id: str) -> str:
    uid = str(user_id)
    if not re.fullmatch(r"-?[0-9A-Za-z_]{1,64}", uid) or uid.startswith("_"):
        raise ValueError(f"bad user id {uid!r}")
    return uid


def user_path(user_id: str) -> Path:
    return store_dir() / f"{_uid(user_id)}.json"


def _lock(user_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(user_id), threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_name(name: Any) -> str:
    text = " ".join(str(name or "").split())[:MAX_NAME_CHARS]
    return text or "widget"


def _private_dir(directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _read(user_id: str) -> dict:
    path = user_path(user_id)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {"next": 1, "widgets": []}
    except OSError as e:
        logger.error("Could not read the widgets of %s: %s", user_id, e)
        return {"next": 1, "widgets": []}
    try:
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise OSError("not a regular file")
            raw = f.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise OSError("too large")
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - a broken file must not break every turn
        logger.error("Could not read the widgets of %s: %s", user_id, e)
        return {"next": 1, "widgets": []}
    widgets = [w for w in (data.get("widgets") if isinstance(data, dict) else None) or [] if _valid(w)]
    nxt = data.get("next") if isinstance(data, dict) else None
    if not isinstance(nxt, int) or nxt < 1:
        nxt = max([_number(w["id"]) for w in widgets] + [0]) + 1
    return {"next": nxt, "widgets": widgets}


def _valid(w: Any) -> bool:
    return (isinstance(w, dict) and isinstance(w.get("id"), str) and _number(w["id"]) > 0
            and isinstance(w.get("message_id"), int))


def _number(widget_id: str) -> int:
    m = re.fullmatch(r"[wW]?(\d{1,9})", str(widget_id).strip())
    return int(m.group(1)) if m else 0


def _write(user_id: str, data: dict) -> None:
    path = user_path(user_id)
    _private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def add(user_id: str, *, name: str, message_id: int, thread_id: int | None, allow_queries: bool,
        chat_data: str | None, public_data: str | None, caption: str | None, html_file: str | None,
        html_bytes: int, packed_bytes: int, telegram_id: str | None = None) -> dict:
    """Record a sent widget; returns the record with its id (``w<N>``, counting up per user)."""
    with _lock(user_id):
        data = _read(user_id)
        now = _now()
        record = {
            "id": f"w{data['next']}", "name": clean_name(name), "message_id": int(message_id),
            "thread_id": thread_id, "allow_queries": bool(allow_queries),
            "chat_data": chat_data, "public_data": public_data, "caption": caption or None,
            "html_file": html_file, "html_bytes": html_bytes, "packed_bytes": packed_bytes,
            "telegram_id": telegram_id, "created": now, "updated": now, "taps": 0, "gone": False,
        }
        data["next"] += 1
        data["widgets"] = (data["widgets"] + [record])[-MAX_WIDGETS:]
        _write(user_id, data)
        return dict(record)


def load(user_id: str) -> list[dict]:
    """The user's widgets, oldest first."""
    with _lock(user_id):
        return [dict(w) for w in _read(user_id)["widgets"]]


def find(user_id: str, ref: Any) -> dict | None:
    """A widget by its id (``w3``, ``W3``, ``3``), its message id, or its name (exact, case-insensitive,
    the newest of that name)."""
    text = str(ref if ref is not None else "").strip()
    if not text:
        return None
    widgets = load(user_id)
    if re.fullmatch(r"[wW]\d{1,9}", text) or re.fullmatch(r"\d{1,9}", text):
        number = _number(text)
        for w in widgets:
            if _number(w["id"]) == number:
                return w
        if text.isdigit():
            for w in reversed(widgets):
                if w["message_id"] == int(text):
                    return w
        return None
    lowered = text.casefold()
    for w in reversed(widgets):
        if str(w.get("name", "")).casefold() == lowered:
            return w
    return None


def by_message(user_id: str, message_id: int) -> dict | None:
    for w in reversed(load(user_id)):
        if w["message_id"] == message_id:
            return w
    return None


def update(user_id: str, widget_id: str, **fields: Any) -> dict | None:
    """Change fields of a record (``updated`` is set); None when it is gone from the registry."""
    with _lock(user_id):
        data = _read(user_id)
        for w in data["widgets"]:
            if w["id"] == widget_id:
                w.update(fields)
                w["updated"] = _now()
                _write(user_id, data)
                return dict(w)
    return None


def count_tap(user_id: str, widget_id: str) -> None:
    with _lock(user_id):
        data = _read(user_id)
        for w in data["widgets"]:
            if w["id"] == widget_id:
                w["taps"] = int(w.get("taps") or 0) + 1
                w["last_tap"] = _now()
                _write(user_id, data)
                return
