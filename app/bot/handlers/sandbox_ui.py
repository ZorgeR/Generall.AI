"""``/sandbox``, the ``sandbox_`` buttons, and the bot side of the sandbox session lifecycle.

Router on ``ui`` (auth, no quota). The sessions themselves (one persistent shell container and one
browser container per user) belong to ``secure_container.sessions.SessionManager``; it is imported
lazily (``session_manager()``), may be missing or never initialised (no Docker, tests): then
``/sandbox`` says the sandbox is unavailable and every hook here is a no-op. Every manager call
blocks (Docker, a unix socket), so each one runs in a worker thread.

``/sandbox`` is ONE legacy-Markdown message (``bot.ui.answer_md`` / ``edit_md``, plain fallback)
built from ``manager.status(uid)``: the sandbox (running for how long, closes at HH:MM if unused,
or not running and started on demand), the saved apt/pip packages that are reinstalled when it
starts, the browser (its tabs: title and host) or why it is not running, the stored logins (size)
and the idle timeout. Admins also see ``manager.server_status()`` (sessions and browsers against
their caps, free memory, the egress filter). Every page title, URL and package name is escaped.
Buttons: "⏳ Keep open N min" (``sandbox_keep`` → ``manager.extend``) and "⏹ Close sandbox &
browser" (``sandbox_close`` → ``manager.close``) while something runs, "🧹 Forget browser logins"
(``sandbox_forget`` → a confirmation: ``sandbox_forget_yes`` → ``manager.forget_browser_profile``,
``sandbox_back`` → the page again) while a profile is stored, and "🔄 Refresh" (``sandbox_refresh``).
Every button edits its message in place.

Idle warnings: ``reaper_loop`` (the ``sandbox-reaper`` background task of ``bot.app``) runs every
``REAP_INTERVAL`` seconds. It first closes idle sessions (``manager.reap_idle``) and edits the
user's remembered warning into "🔒 … closed after N minutes without use", then sends one plain-text
warning per ``manager.due_warnings()`` entry: what closes, in how many minutes and at what local time
(``bot.timezones``, the user's settings zone; UTC when unknown), with "👌 OK" (``sandbox_ok``: the
buttons go, the text stays) and "⏳ Keep open N min" (``sandbox_keep``: ``manager.extend`` → "✅ Kept
open until HH:MM."). The last warning per user is remembered in memory (``_warnings``); a button on
an older warning, or on one the user made obsolete by working again (a later closing time), is only
answered with the current closing time. The numbers come from ``sessions.SESSION_IDLE_SECONDS`` /
``SESSION_WARN_BEFORE_SECONDS`` (defaults when the module is absent). The loop never dies: a failed
manager call or a chat that cannot be messaged is logged and skipped.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import timezones
from bot.auth import auth
from bot.settings import UserSettings
from bot.ui import answer_md, edit_md, escape_markdown, is_not_modified

logger = logging.getLogger(__name__)

router = Router(name="sandbox")

PREFIX = "sandbox_"
CLOSE = "sandbox_close"
FORGET = "sandbox_forget"
FORGET_YES = "sandbox_forget_yes"
BACK = "sandbox_back"
REFRESH = "sandbox_refresh"
KEEP = "sandbox_keep"
OK = "sandbox_ok"

KINDS = ("shell", "browser")
REAP_INTERVAL = 60  # seconds between reaper ticks
CLEANUP_TIMEOUT = 120  # startup waits at most this long for the leftover containers to go
SHUTDOWN_TIMEOUT = 120
DEFAULT_IDLE_SECONDS = 1800  # used when secure_container.sessions is absent
DEFAULT_WARN_BEFORE_SECONDS = 300
NEWER_TOLERANCE = timedelta(seconds=5)  # a closing time this much later than the warning's = the user worked since
TABS_SHOWN = 6
TITLE_CHARS = 60
PACKAGES_SHOWN = 8

TITLE = "🧰 *Sandbox & browser*"
UNAVAILABLE = (
    f"{TITLE}\n\n"
    "The persistent sandbox is not available on this server right now. Commands still run, each in "
    "a fresh container, but nothing keeps running between them and the interactive browser is unavailable."
)
ERROR_TEXT = "❌ Could not read the sandbox state right now. Try again in a moment."
ACTION_FAILED = "❌ That did not work right now. Try again in a moment."
FORGET_CONFIRM = (
    "🧹 *Forget browser logins?*\n\n"
    "This closes the browser and deletes its cookies and saved logins. Sites will ask you to sign in "
    "again. It cannot be undone."
)
FORGOTTEN = "🧹 Browser logins forgotten: the browser was closed and its cookies deleted."
NOTHING_TO_FORGET = "🧹 No browser logins were stored."
CLOSED_BY_USER = "⏹ Closed. Your files and saved logins are kept; it starts again when needed."
NOTHING_RUNNING = "Nothing was running."
ALREADY_CLOSED = "It has already closed; it starts again when needed."
WARNING_PREFIX = "⏳ Your "  # how every warning starts (recognizes warnings sent before a restart)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f  ]+")


# ---------------------------------------------------------------------------
# The session manager (lazy, optional)
# ---------------------------------------------------------------------------
def session_manager() -> Any | None:
    """The process-wide ``SessionManager``, or None when sessions are off or unavailable."""
    try:
        from secure_container import sessions
    except Exception as e:  # noqa: BLE001 - no Docker SDK or no sessions module: sessions are off
        logger.debug("Sandbox sessions unavailable: %s", e)
        return None
    try:
        return sessions.get_manager()
    except Exception as e:  # noqa: BLE001
        logger.warning("Sandbox session manager lookup failed: %s", e)
        return None


def _sessions_seconds(name: str, default: int) -> int:
    try:
        from secure_container import sessions

        value = int(getattr(sessions, name, default))
    except Exception:  # noqa: BLE001 - module missing or a non-number: the documented default
        return default
    return value if value > 0 else default


def idle_seconds() -> int:
    """Seconds without use after which a session closes (``sessions.SESSION_IDLE_SECONDS``)."""
    return _sessions_seconds("SESSION_IDLE_SECONDS", DEFAULT_IDLE_SECONDS)


def warn_before_seconds() -> int:
    """How long before closing the user is warned (``sessions.SESSION_WARN_BEFORE_SECONDS``)."""
    return _sessions_seconds("SESSION_WARN_BEFORE_SECONDS", DEFAULT_WARN_BEFORE_SECONDS)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def duration_words(seconds: float, *, short: bool = False) -> str:
    """``30 minutes`` / ``1 minute`` / ``2 hours``; ``short``: ``30 min`` / ``2 h``."""
    seconds = max(0, int(seconds))
    if seconds >= 3600 and seconds % 3600 == 0:
        n = seconds // 3600
        unit = "h" if short else ("hour" if n == 1 else "hours")
    else:
        n = max(1, round(seconds / 60))
        unit = "min" if short else ("minute" if n == 1 else "minutes")
    return f"{n} {unit}"


def as_datetime(value: Any) -> datetime | None:
    """An aware UTC datetime from a datetime, an epoch number or an ISO string; None otherwise."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, timezone.utc) if value > 0 else None
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def clock(moment: datetime, zone: Any) -> str:
    """``14:35`` in the user's zone; ``14:35 UTC`` when the zone is unknown."""
    text = timezones.to_local(moment, zone).strftime("%H:%M")
    return text if zone is not None else f"{text} UTC"


def fmt_bytes(n: int) -> str:
    n = max(0, int(n or 0))
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.1f} GB"


def _clean(value: Any, limit: int) -> str:
    """One line of page-supplied text: control characters out, whitespace collapsed, clipped."""
    text = " ".join(_CONTROL_CHARS.sub(" ", str(value or "")).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _user_zone(user_id: str) -> Any:
    """The user's ZoneInfo (None = unknown, shown as UTC). Reads settings.json: call in a thread."""
    try:
        return timezones.zone_of(UserSettings(user_id))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def _kind(status: dict, kind: str) -> dict:
    value = status.get(kind) if isinstance(status, dict) else None
    return value if isinstance(value, dict) else {}


def running_kinds(status: dict) -> list[str]:
    return [kind for kind in KINDS if _kind(status, kind).get("running")]


def _kind_closes_at(info: dict, now: datetime) -> datetime | None:
    closes = as_datetime(info.get("closes_at"))
    if closes is None and info.get("idle_seconds_left") is not None:
        closes = now + timedelta(seconds=max(0, _int(info.get("idle_seconds_left"))))
    return closes


def closes_at(status: dict, now: datetime | None = None) -> datetime | None:
    """When the user's running sessions close if unused (the latest of them); None when none runs."""
    now = now or datetime.now(timezone.utc)
    times = [t for kind in running_kinds(status) if (t := _kind_closes_at(_kind(status, kind), now)) is not None]
    return max(times) if times else None


@dataclass
class SandboxView:
    status: dict
    zone: Any  # ZoneInfo | None
    browser_settings: dict
    server: dict | None = None  # admins only; {"error": ...} when server_status failed
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _gather(manager: Any, user_id: str, admin: bool) -> SandboxView:
    status = manager.status(user_id)
    settings = UserSettings(user_id)
    browser = settings.get("browser")
    server = None
    if admin:
        try:
            server = manager.server_status() or {}
        except Exception as e:  # noqa: BLE001 - the user part of the page still works
            logger.warning("Sandbox server status failed: %s", e)
            server = {"error": str(e)}
    return SandboxView(
        status=status if isinstance(status, dict) else {},
        zone=timezones.zone_of(settings),
        browser_settings=browser if isinstance(browser, dict) else {},
        server=server,
    )


async def collect(manager: Any, user_id: str) -> SandboxView:
    """Everything the page shows; the manager calls and the settings read run in a worker thread."""
    return await asyncio.to_thread(_gather, manager, str(user_id), auth.is_admin(str(user_id)))


def _age(info: dict, now: datetime) -> str:
    started = as_datetime(info.get("started"))
    if started is None:
        return ""
    return f" for {duration_words((now - started).total_seconds(), short=True)}"


def _closing(info: dict, view: SandboxView) -> str:
    closes = _kind_closes_at(info, view.now)
    return f" · closes at {clock(closes, view.zone)} if unused" if closes is not None else ""


def _packages_line(packages: Any) -> str:
    packages = packages if isinstance(packages, dict) else {}
    parts = []
    for manager in ("apt", "pip"):
        names = [n for n in (_clean(p, 40) for p in packages.get(manager) or []) if n]
        if not names:
            continue
        shown = ", ".join(escape_markdown(n) for n in names[:PACKAGES_SHOWN])
        more = f" +{len(names) - PACKAGES_SHOWN} more" if len(names) > PACKAGES_SHOWN else ""
        parts.append(f"{manager}: {shown}{more}")
    if not parts:
        return "📦 *Saved packages*: none"
    return "📦 *Saved packages* (reinstalled when the sandbox starts): " + " · ".join(parts)


def _tab_line(tab: Any) -> str:
    tab = tab if isinstance(tab, dict) else {}
    title = _clean(tab.get("title"), TITLE_CHARS) or "untitled"
    url = str(tab.get("url") or "")
    host = urlparse(url).hostname if "://" in url else None
    where = _clean(host or url, 48)
    marker = "▶️" if tab.get("active") else "•"
    line = f"  {marker} {escape_markdown(title)}"
    return f"{line} — {escape_markdown(where)}" if where else line


def _browser_lines(view: SandboxView) -> list[str]:
    info = _kind(view.status, "browser")
    if info.get("running"):
        tabs = [t for t in info.get("tabs") or [] if isinstance(t, dict)]
        count = f"{len(tabs)} tab{'' if len(tabs) == 1 else 's'}"
        lines = [f"🌐 *Browser*: running, {count}{_closing(info, view)}"]
        lines += [_tab_line(t) for t in tabs[:TABS_SHOWN]]
        if len(tabs) > TABS_SHOWN:
            lines.append(f"  … {len(tabs) - TABS_SHOWN} more")
        return lines
    if view.browser_settings.get("enabled", True) is False:
        return ["🌐 *Browser*: off (turn it on in /settings → 🌐 Browser)"]
    return ["🌐 *Browser*: not running (opens when I need a website)"]


def _logins_line(view: SandboxView) -> str:
    size = _int(view.status.get("profile_bytes"))
    if view.browser_settings.get("remember", True) is False:
        older = f"; an older profile of {fmt_bytes(size)} is stored" if size > 0 else ""
        return f"🔑 *Logins*: not kept (Remember logins is off in /settings){older}"
    return f"🔑 *Logins*: remembered ({fmt_bytes(size)})" if size > 0 else "🔑 *Logins*: none stored"


EGRESS = {"on": "on", "off": "off", "failed": "⚠️ failed, private networks are NOT blocked"}


def server_lines(server: dict) -> list[str]:
    """The admin block: sessions and browsers against their caps, memory, the egress filter."""
    if server.get("error"):
        return ["🛠 *Server* (admin): status unavailable, see the log"]

    def mb(value: Any) -> str:
        return f"{_int(value):,} MB" if value is not None else "unknown"

    egress = str(server.get("egress_filter") or "unknown")
    return [
        "🛠 *Server* (admin)",
        f"Sandboxes {_int(server.get('shells'))}/{_int(server.get('max_shells'))}"
        f" · browsers {_int(server.get('browsers'))}/{_int(server.get('max_browsers'))}",
        f"Memory: {mb(server.get('mem_available_mb'))} free of {mb(server.get('mem_total_mb'))},"
        f" {mb(server.get('reserve_mb'))} kept in reserve",
        f"Egress filter: {EGRESS.get(egress, escape_markdown(egress))}",
    ]


def render(view: SandboxView, note: str | None = None) -> str:
    """The /sandbox page (legacy Markdown; every page title, URL and package name escaped)."""
    status = view.status
    idle = _int(status.get("idle_timeout_seconds")) or idle_seconds()
    lines = [TITLE]
    if note:
        lines += ["", note]
    lines.append("")
    shell = _kind(status, "shell")
    if shell.get("running"):
        lines.append(f"🖥 *Sandbox*: running{_age(shell, view.now)}{_closing(shell, view)}")
    else:
        lines.append("🖥 *Sandbox*: not running (starts when I need to run something)")
    lines.append(_packages_line(status.get("packages")))
    lines += _browser_lines(view)
    lines.append(_logins_line(view))
    lines += [
        "",
        f"⏱ An unused sandbox or browser closes after {duration_words(idle)} (with a warning "
        f"{duration_words(warn_before_seconds())} before). Your workspace files always stay.",
    ]
    if view.server is not None:
        lines += ["", *server_lines(view.server)]
    return "\n".join(lines)


def _button(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def keep_label() -> str:
    return f"⏳ Keep open {duration_words(idle_seconds(), short=True)}"


def refresh_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_button("🔄 Refresh", REFRESH)]])


def keyboard(view: SandboxView) -> InlineKeyboardMarkup:
    """Keep open + Refresh and Close while something runs; Forget while a profile is stored."""
    rows: list[list[InlineKeyboardButton]] = []
    if running_kinds(view.status):
        rows.append([_button(keep_label(), KEEP), _button("🔄 Refresh", REFRESH)])
        rows.append([_button("⏹ Close sandbox & browser", CLOSE)])
    else:
        rows.append([_button("🔄 Refresh", REFRESH)])
    if _int(view.status.get("profile_bytes")) > 0:
        rows.append([_button("🧹 Forget browser logins", FORGET)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def forget_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        _button("✅ Yes, forget them", FORGET_YES),
        _button("↩️ Keep them", BACK),
    ]])


async def page(manager: Any, user_id: str, note: str | None = None) -> tuple[str, InlineKeyboardMarkup]:
    """Text and keyboard of the /sandbox page; never raises."""
    if manager is None:
        return UNAVAILABLE, refresh_keyboard()
    try:
        view = await collect(manager, user_id)
    except Exception:  # noqa: BLE001
        logger.exception("Sandbox status of %s failed", user_id)
        return (ERROR_TEXT if note is None else f"{note}\n\n{ERROR_TEXT}"), refresh_keyboard()
    return render(view, note), keyboard(view)


# ---------------------------------------------------------------------------
# Idle warnings
# ---------------------------------------------------------------------------
@dataclass
class ClosingWarning:
    message_id: int
    closes_at: datetime
    kinds: tuple[str, ...]


# The last warning sent to each user (chat id -> warning); in memory, so a restart forgets it.
_warnings: dict[str, ClosingWarning] = {}


def reset() -> None:
    """Forget every remembered warning (tests)."""
    _warnings.clear()


def _subject(kinds: Iterable[str]) -> tuple[str, bool]:
    """What closes, and whether it is plural."""
    kinds = set(kinds)
    if {"shell", "browser"} <= kinds:
        return "sandbox and browser", True
    if "browser" in kinds:
        return "browser", False
    return "sandbox", False


def warning_text(kinds: Iterable[str], closes: datetime, zone: Any) -> str:
    kinds = tuple(kinds)
    name, plural = _subject(kinds)
    if "shell" in kinds and "browser" in kinds:
        stops = "Running programs and open browser tabs will stop; your files and saved logins stay."
    elif "browser" in kinds:
        stops = "Open browser tabs will close; your saved logins stay."
    else:
        stops = "Running programs will stop; your files stay."
    return (
        f"⏳ Your {name} will close in {duration_words(warn_before_seconds())} (at {clock(closes, zone)}) "
        f"because {'they have' if plural else 'it has'} not been used.\n{stops}"
    )


def warning_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_button("👌 OK", OK), _button(keep_label(), KEEP)]])


def closed_text(kinds: Iterable[str]) -> str:
    name, plural = _subject(kinds)
    return (
        f"🔒 {name[0].upper()}{name[1:]} closed after {duration_words(idle_seconds())} without use. "
        f"Your files and saved logins are kept; {'they start' if plural else 'it starts'} again when needed."
    )


def kept_text(until: datetime, zone: Any) -> str:
    return f"✅ Kept open until {clock(until, zone)}."


def in_use_text(closes: datetime, zone: Any) -> str:
    return f"✅ In use again: it now closes at {clock(closes, zone)} if unused."


def _reaped(entry: Any) -> tuple[str | None, str]:
    if isinstance(entry, dict):
        user_id, kind = entry.get("user_id"), entry.get("kind")
    elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
        user_id, kind = entry[0], entry[1]
    else:
        return None, ""
    return (str(user_id) if user_id not in (None, "") else None), str(kind or "shell")


async def _edit_warning(bot: Bot, user_id: str, message_id: int, text: str) -> None:
    """Replace a warning's text; the buttons go with it. Best-effort."""
    try:
        await bot.edit_message_text(chat_id=int(user_id), message_id=message_id, text=text)
    except Exception as e:  # noqa: BLE001 - deleted, too old, chat gone
        if not is_not_modified(e):
            logger.warning("Could not update the sandbox warning of %s: %s", user_id, e)


async def _settle_warning(bot: Bot, user_id: str, text: str, *, skip_message_id: int | None = None) -> None:
    """Forget the user's remembered warning and turn it into ``text`` (unless it is ``skip_message_id``)."""
    record = _warnings.pop(str(user_id), None)
    if record is not None and record.message_id != skip_message_id:
        await _edit_warning(bot, user_id, record.message_id, text)


async def send_warning(bot: Bot, warning: dict) -> None:
    """Send one closing warning and remember it; a chat that cannot be messaged is logged and skipped."""
    user_id = str(warning.get("user_id") or "")
    kinds = tuple(k for k in warning.get("kinds") or () if k in KINDS) or ("shell",)
    closes = as_datetime(warning.get("closes_at")) or datetime.now(timezone.utc) + timedelta(seconds=warn_before_seconds())
    try:
        chat_id = int(user_id)
        if not auth.is_authorized(user_id):
            logger.info("Not warning %s about the sandbox closing: not authorized", user_id)
            return
        zone = await asyncio.to_thread(_user_zone, user_id)
        sent = await bot.send_message(chat_id=chat_id, text=warning_text(kinds, closes, zone), reply_markup=warning_keyboard())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not send the sandbox closing warning to %s: %s", user_id, e)
        return
    _warnings[user_id] = ClosingWarning(message_id=sent.message_id, closes_at=closes, kinds=kinds)


async def reaper_tick(bot: Bot, manager: Any) -> None:
    """Close idle sessions (editing their warnings), then send the warnings that are due. Never raises."""
    try:
        stopped = await asyncio.to_thread(manager.reap_idle)
    except Exception as e:  # noqa: BLE001
        logger.warning("Sandbox reaper: reap_idle failed: %s", e)
        stopped = []
    closed: dict[str, list[str]] = {}
    for entry in stopped or []:
        user_id, kind = _reaped(entry)
        if user_id is None:
            continue
        logger.info("Closed the idle %s session of %s", kind, user_id)
        closed.setdefault(user_id, []).append(kind)
    for user_id, kinds in closed.items():
        record = _warnings.pop(user_id, None)
        if record is not None:
            await _edit_warning(bot, user_id, record.message_id, closed_text(kinds))

    due_warnings = getattr(manager, "due_warnings", None)
    if due_warnings is None:
        return
    try:
        due = await asyncio.to_thread(due_warnings)
    except Exception as e:  # noqa: BLE001
        logger.warning("Sandbox reaper: due_warnings failed: %s", e)
        return
    for warning in due or []:
        if isinstance(warning, dict):
            await send_warning(bot, warning)


async def reaper_loop(bot: Bot, manager: Any, interval: float = REAP_INTERVAL) -> None:
    """The ``sandbox-reaper`` background task: one ``reaper_tick`` every ``interval`` seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            await reaper_tick(bot, manager)
        except Exception:  # noqa: BLE001 - the loop must outlive any single tick
            logger.exception("Sandbox reaper tick failed")


async def cleanup_stale(manager: Any) -> None:
    """Startup: remove containers a previous run left behind, before anyone can start a new one."""
    try:
        removed = await asyncio.wait_for(asyncio.to_thread(manager.cleanup_stale), CLEANUP_TIMEOUT)
    except Exception as e:  # noqa: BLE001 - incl. the timeout: startup goes on
        logger.warning("Sandbox cleanup at startup failed: %s", e)
        return
    if removed:
        logger.info("Removed %s leftover sandbox container(s)", removed)


async def shutdown_sessions(manager: Any | None = None) -> None:
    """Shutdown: stop every session (no-op without a manager). Never raises."""
    manager = manager if manager is not None else session_manager()
    if manager is None:
        return
    try:
        await asyncio.wait_for(asyncio.to_thread(manager.shutdown_all), SHUTDOWN_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        logger.warning("Sandbox shutdown failed: %s", e)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def _answer(callback: CallbackQuery, text: str | None = None) -> None:
    """Stop the button's spinner (with a toast when ``text``); best-effort, an old query raises."""
    try:
        await callback.answer(text)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not answer the sandbox button: %s", e)


@router.message(F.text, Command("sandbox"))
async def sandbox_command(message: Message, user_id: str) -> None:
    text, markup = await page(session_manager(), user_id)
    await answer_md(message, text, markup)


def _is_warning(user_id: str, message: Message) -> bool:
    record = _warnings.get(user_id)
    if record is not None and record.message_id == message.message_id:
        return True
    return (message.text or "").startswith(WARNING_PREFIX)


async def _remove_buttons(bot: Bot, message: Message) -> None:
    try:
        await bot.edit_message_reply_markup(chat_id=message.chat.id, message_id=message.message_id, reply_markup=None)
    except Exception as e:  # noqa: BLE001
        if not is_not_modified(e):
            logger.debug("Could not remove the warning buttons: %s", e)


async def _warning_button(callback: CallbackQuery, bot: Bot, message: Message, manager: Any, user_id: str) -> None:
    """👌 OK / ⏳ Keep open on a closing warning.

    Only the user's current warning extends the session, and only while its closing time still
    stands; on an older warning, or after the user worked again (a later closing time), the
    button just reports the current closing time.
    """
    keep = callback.data == KEEP
    record = _warnings.get(user_id)
    status: dict | None = None
    if manager is not None:
        try:
            status = await asyncio.to_thread(manager.status, user_id)
        except Exception as e:  # noqa: BLE001 - decide without it: the current warning still extends
            logger.warning("Sandbox status of %s failed: %s", user_id, e)
    zone = await asyncio.to_thread(_user_zone, user_id)
    gone = manager is None or (isinstance(status, dict) and not running_kinds(status))
    current = closes_at(status) if isinstance(status, dict) else None
    newer = record is None or (current is not None and current > record.closes_at + NEWER_TOLERANCE)

    if gone:
        if record is not None and record.message_id == message.message_id:
            _warnings.pop(user_id, None)
        await _answer(callback, ALREADY_CLOSED)
        if keep:
            await edit_md(message, ALREADY_CLOSED)
        else:
            await _remove_buttons(bot, message)
        return
    if newer:
        note = in_use_text(current, zone) if current is not None else None
        await _answer(callback, note)
        if keep and note:
            await edit_md(message, note)
        else:
            await _remove_buttons(bot, message)
        return
    if not keep:  # 👌 OK on the current warning: it stays as it is, without buttons
        if record.message_id == message.message_id:
            _warnings.pop(user_id, None)
        await _answer(callback)
        await _remove_buttons(bot, message)
        return
    try:
        until = await asyncio.to_thread(manager.extend, user_id)
    except Exception:  # noqa: BLE001
        logger.exception("Keeping the sandbox of %s open failed", user_id)
        await _answer(callback, ACTION_FAILED)
        return
    until = as_datetime(until)
    text = kept_text(until, zone) if until is not None else ALREADY_CLOSED
    await _answer(callback, text)
    await edit_md(message, text)
    await _settle_warning(bot, user_id, text, skip_message_id=message.message_id)


@router.callback_query(F.data.startswith(PREFIX))
async def sandbox_button(callback: CallbackQuery, bot: Bot, user_id: str) -> None:
    message = callback.message
    if not isinstance(message, Message):
        await _answer(callback)
        return
    data = callback.data or ""
    manager = session_manager()
    if data in (OK, KEEP) and _is_warning(user_id, message):
        await _warning_button(callback, bot, message, manager, user_id)
        return
    await _answer(callback)
    if data == FORGET:
        await edit_md(message, FORGET_CONFIRM, forget_keyboard())
        return
    if data not in (CLOSE, FORGET_YES, KEEP, REFRESH, BACK):
        logger.warning("Unknown sandbox callback from %s: %r", user_id, data)
        return
    note = None
    if manager is not None and data in (CLOSE, FORGET_YES, KEEP):
        try:
            note = await _apply(bot, manager, user_id, data)
        except Exception:  # noqa: BLE001
            logger.exception("Sandbox action %s of %s failed", data, user_id)
            note = ACTION_FAILED
    text, markup = await page(manager, user_id, note)
    await edit_md(message, text, markup)


async def _apply(bot: Bot, manager: Any, user_id: str, data: str) -> str:
    """Run one page action in a worker thread; the note shown above the refreshed page."""
    if data == CLOSE:
        if not await asyncio.to_thread(manager.close, user_id):
            return NOTHING_RUNNING
        await _settle_warning(bot, user_id, CLOSED_BY_USER)
        return CLOSED_BY_USER
    if data == FORGET_YES:
        return FORGOTTEN if await asyncio.to_thread(manager.forget_browser_profile, user_id) else NOTHING_TO_FORGET
    until = as_datetime(await asyncio.to_thread(manager.extend, user_id))
    if until is None:
        return ALREADY_CLOSED
    text = kept_text(until, await asyncio.to_thread(_user_zone, user_id))
    await _settle_warning(bot, user_id, text)
    return text
