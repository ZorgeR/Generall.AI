"""/reminders command and the ``reminder_*`` / ``reminders_*`` inline keyboard family.

Ported from the python-telegram-bot implementation and upgraded in the v3 wave
. Every callback string of the old UI still works (``reminders_page_<n>``,
``reminder_info_<id>``, ``reminder_delete_<id>``, ``reminder_toggle_<id>``, ``reminders_summary``,
``noop``) and the 5-per-page pagination stays. New:

* Times are shown in the user's zone (``bot.timezones.format_local``; UTC until ``/timezone``
  sets one), and the list is grouped into "⏰ Reminders" (type user) and "🤖 Tasks" (type agent).
* The per-reminder page has ``⏰ +1h`` / ``📅 +1d`` (``reminder_delay_<id>_<minutes>``),
  ``✏️ Edit text`` (``reminder_edit_<id>``), Delete / Disable and Back. Editing remembers
  ``user_id -> (reminder id, deadline)`` in ``_pending_edits`` for ``EDIT_TIMEOUT`` seconds; the
  text handler below, on this router (the ``ui`` router runs before the chat router), takes the
  user's next plain text as the new text, so it never reaches the agent; commands pass through
  and ``✖️ Cancel edit`` (``reminder_editcancel_<id>``) clears it.
* ``➕ New`` (``reminders_new``) explains that reminders are created by talking to the assistant.
* Every page is legacy Markdown, so zone names with underscores (``America/New_York``) are
  escaped wherever they land in a text (``_zone_line``, ``_format_time``, ``snooze_text``); a
  lone ``_`` would make Telegram reject the page or swallow it as italics. Button labels are
  plain text and stay unescaped.
* A fired user reminder carries snooze buttons (``bot.jobs.snooze_keyboard``):
  ``reminder_snooze_<id>_<minutes|tomorrow>``. A one-shot reminder is re-scheduled in place
  (status pending, new time); a periodic one keeps its series and gets a one-off copy
  (``snoozed_from``), so "every day at 9" does not drift to 9:10.

Other differences from the original, all deliberate:

* No authorization check in the handlers: ``AuthMiddleware`` already authorizes every invited user.
* ``noop`` spacer buttons get a dedicated handler that only answers the query.
* Reminder ids are opaque strings (uuid hex), never converted to ``int``.
* All reads and writes go through ``reminders_store`` (per-user lock), so they cannot race the
  scheduler or the agent tool.
* Status ``processing`` (an agent task currently running) is treated like ``pending`` for
  listing/counting and rendered as ``🏃 running``.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import timezones
from bot.settings import UserSettings
from bot.ui import answer_md, delete_quietly, edit_md, escape_markdown
from reminders_store import WEEKDAY_NAMES, new_reminder_id, parse_time, reminders_store

logger = logging.getLogger(__name__)

router = Router(name="reminders")

ITEMS_PER_PAGE = 5
ACTIVE_STATUSES = ("pending", "processing")
TEXT_PREVIEW_LEN = 35
EDIT_TIMEOUT = 300.0  # seconds the next text message counts as the new reminder text
SNOOZE_TOMORROW_HOUR = 9

NOOP = "noop"
BACK_TO_LIST = "reminders_page_1"
NEW = "reminders_new"
GROUP_TITLES = {"user": "⏰ Reminders", "agent": "🤖 Tasks"}
GROUP_ICONS = {"user": "⏰", "agent": "🤖"}

NEW_HINT = (
    "➕ *New reminder or task*\n\n"
    "Just tell me what you need, in your own words:\n"
    '• "remind me tomorrow at 9 to call Alex"\n'
    '• "every weekday at 8 check the news for me and message me only if there is something important"\n'
    '• "in 45 minutes remind me to take the pizza out"\n\n'
    "I schedule it in your local time and it shows up here."
)
EDIT_PROMPT = (
    "✏️ *Edit reminder text*\n\n"
    "Current text: {text}\n\n"
    "Send the new text as your next message (within 5 minutes), or cancel."
)

# user_id -> (reminder id, deadline as time.monotonic()); one pending edit per user
_pending_edits: dict[str, tuple[str, float]] = {}


# --------------------------------------------------------------------------- helpers


def _is_active(reminder: dict[str, Any]) -> bool:
    return reminder.get("status") in ACTIVE_STATUSES


def _is_completed(reminder: dict[str, Any]) -> bool:
    return reminder.get("status") == "completed"


def _is_enabled(reminder: dict[str, Any]) -> bool:
    return bool(reminder.get("enabled", True))


def _kind(reminder: dict[str, Any]) -> str:
    return "agent" if reminder.get("type") == "agent" else "user"


def _display_status(status: Any) -> str:
    return "🏃 running" if status == "processing" else str(status)


def _zone(user_id: str):
    """The user's saved zone or None (UTC); a settings problem never breaks the UI."""
    try:
        return timezones.zone_of(UserSettings(user_id))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read the time zone of %s: %s", user_id, e)
        return None


def _format_time(value: Any, zone, fmt: str | None = None) -> str:
    """An ISO timestamp in the user's zone: ``format_local`` escaped for legacy Markdown (the
    zone name may hold ``_``), or a short ``strftime`` pattern for button labels (plain text);
    the raw value if it does not parse."""
    if not value:
        return ""
    try:
        moment = parse_time(str(value))
    except (TypeError, ValueError):
        logger.warning("Unparseable reminder timestamp: %r", value)
        return str(value) if fmt is not None else escape_markdown(str(value))
    if fmt is None:
        return escape_markdown(timezones.format_local(moment, zone))
    return timezones.to_local(moment, zone).strftime(fmt)


def _sort_key(reminder: dict[str, Any]) -> str:
    return str(reminder.get("time") or "")


def _reminder_button_label(reminder: dict[str, Any], zone) -> str:
    formatted_time = _format_time(reminder.get("time"), zone, "%H:%M %d/%m")  # compact, local
    if reminder.get("is_periodic"):
        period_type = str(reminder.get("period_type") or "")
        formatted_time += f" ↻{reminder.get('period_interval', '')}{period_type[:1]}"
    text = str(reminder.get("text", ""))
    text_preview = text[:TEXT_PREVIEW_LEN] + ("..." if len(text) > TEXT_PREVIEW_LEN else "")
    return f"{GROUP_ICONS[_kind(reminder)]} {formatted_time} : {text_preview}"


def _zone_line(zone) -> str:
    if zone is None:
        return "Times in UTC. Set your zone with /timezone."
    return f"Times in {escape_markdown(timezones.zone_name(zone))}."


def _build_menu(all_reminders: list[dict[str, Any]], page: int, zone=None) -> tuple[str, InlineKeyboardMarkup]:
    """Build the status text and keyboard of the reminders list for ``page``.

    Active reminders come first (soonest first), then the agent tasks; a group title row opens
    each group where it starts on the page."""
    user_reminders = sorted((r for r in all_reminders if _is_active(r) and _kind(r) == "user"), key=_sort_key)
    agent_tasks = sorted((r for r in all_reminders if _is_active(r) and _kind(r) == "agent"), key=_sort_key)
    active_reminders = user_reminders + agent_tasks
    completed_reminders = [r for r in all_reminders if _is_completed(r)]

    total_pages = (len(active_reminders) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    # If the requested page is past the end (e.g. after a delete), show the last page.
    page = max(1, min(page, total_pages)) if total_pages else 1

    start_idx = (page - 1) * ITEMS_PER_PAGE
    current_reminders = active_reminders[start_idx : start_idx + ITEMS_PER_PAGE]

    keyboard: list[list[InlineKeyboardButton]] = []
    shown_group: str | None = None
    for reminder in current_reminders:
        kind = _kind(reminder)
        if kind != shown_group:
            keyboard.append([InlineKeyboardButton(text=GROUP_TITLES[kind], callback_data=NOOP)])
            shown_group = kind
        reminder_id = str(reminder.get("id", ""))
        keyboard.append(
            [InlineKeyboardButton(text=_reminder_button_label(reminder, zone), callback_data=f"reminder_info_{reminder_id}")]
        )
        keyboard.append(
            [
                InlineKeyboardButton(text=" ", callback_data=NOOP),
                InlineKeyboardButton(text="❌ Delete", callback_data=f"reminder_delete_{reminder_id}"),
                InlineKeyboardButton(
                    text="⏸️ Disable" if _is_enabled(reminder) else "▶️ Enable",
                    callback_data=f"reminder_toggle_{reminder_id}",
                ),
                InlineKeyboardButton(text=" ", callback_data=NOOP),
            ]
        )

    nav_buttons: list[InlineKeyboardButton] = []
    if page > 1:
        nav_buttons.append(InlineKeyboardButton(text="⬅️", callback_data=f"reminders_page_{page - 1}"))
    if total_pages > 1:
        nav_buttons.append(InlineKeyboardButton(text=f"{page}/{total_pages}", callback_data=NOOP))
    if page < total_pages:
        nav_buttons.append(InlineKeyboardButton(text="➡️", callback_data=f"reminders_page_{page + 1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([
        InlineKeyboardButton(text="➕ New", callback_data=NEW),
        InlineKeyboardButton(text="📊 Summary", callback_data="reminders_summary"),
    ])

    status_text = (
        "*Your Reminders*\n\n"
        f"⏰ Reminders: {len(user_reminders)} | 🤖 Tasks: {len(agent_tasks)} | "
        f"Completed: {len(completed_reminders)}\n"
        f"{_zone_line(zone)}\n\n"
        "Select a reminder to manage:"
    )
    return status_text, InlineKeyboardMarkup(inline_keyboard=keyboard)


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="⬅️ Back", callback_data=BACK_TO_LIST)]])


def _info_keyboard(reminder: dict[str, Any]) -> InlineKeyboardMarkup:
    reminder_id = str(reminder.get("id", ""))
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⏰ +1h", callback_data=f"reminder_delay_{reminder_id}_60"),
            InlineKeyboardButton(text="📅 +1d", callback_data=f"reminder_delay_{reminder_id}_1440"),
            InlineKeyboardButton(text="✏️ Edit text", callback_data=f"reminder_edit_{reminder_id}"),
        ],
        [
            InlineKeyboardButton(text="❌ Delete", callback_data=f"reminder_delete_{reminder_id}"),
            InlineKeyboardButton(
                text="⏸️ Disable" if _is_enabled(reminder) else "▶️ Enable",
                callback_data=f"reminder_toggle_{reminder_id}",
            ),
        ],
        [InlineKeyboardButton(text="⬅️ Back", callback_data=BACK_TO_LIST)],
    ])


def render_info(reminder: dict[str, Any], zone) -> str:
    """The detail view of one reminder (legacy Markdown), times in the user's zone."""
    kind = _kind(reminder)
    lines = [
        "*Task Details*" if kind == "agent" else "*Reminder Details*",
        "",
        f"Text: {escape_markdown(str(reminder.get('text', '')))}",
        f"Type: {escape_markdown(str(reminder.get('type', '')))}",
        f"Status: {_display_status(reminder.get('status'))}" + ("" if _is_enabled(reminder) else " (disabled)"),
        f"Created: {_format_time(reminder.get('created_at'), zone)}",
        f"Scheduled: {_format_time(reminder.get('time'), zone)}",
    ]
    if reminder.get("is_periodic"):
        lines.append(
            f"Repeats every: {reminder.get('period_interval', '')} {escape_markdown(str(reminder.get('period_type', '')))}"
        )
        days = reminder.get("weekdays")
        if days:
            names = [WEEKDAY_NAMES[d] for d in days if isinstance(d, int) and 0 <= d <= 6]
            lines.append("On: " + ", ".join(names))
        if reminder.get("next_trigger"):
            lines.append(f"Next Trigger: {_format_time(reminder['next_trigger'], zone)}")
    if reminder.get("last_triggered"):
        lines.append(f"Last Triggered: {_format_time(reminder['last_triggered'], zone)}")
    if kind == "agent" and reminder.get("last_result"):
        result = "quiet (nothing new)" if reminder.get("last_result") == "quiet" else "answered"
        quiet_runs = int(reminder.get("quiet_runs") or 0)
        lines.append(f"Last run: {result}" + (f", {quiet_runs} quiet run(s) so far" if quiet_runs else ""))
    if reminder.get("snoozed_from"):
        lines.append("Snoozed copy of a repeating reminder.")
    return "\n".join(lines)


def snooze_time(now: datetime, suffix: str, zone) -> datetime | None:
    """The UTC time a snooze button means: ``<minutes>`` from now, or ``tomorrow`` = 09:00 local."""
    if suffix == "tomorrow":
        local_tomorrow = timezones.to_local(now, zone) + timedelta(days=1)
        target = local_tomorrow.replace(hour=SNOOZE_TOMORROW_HOUR, minute=0, second=0, microsecond=0)
        return target.astimezone(timezone.utc)
    try:
        minutes = int(suffix)
    except ValueError:
        return None
    if minutes <= 0:
        return None
    return now + timedelta(minutes=minutes)


def snooze_text(text: str, when: datetime, zone) -> str:
    """The fired reminder message after a snooze (legacy Markdown, the new time in local time)."""
    local = escape_markdown(timezones.format_local(when, zone))
    return f"🔔 *Reminder*\n\n{escape_markdown(text)}\n\n⏰ Snoozed until {local}"


def _reschedule(reminder: dict[str, Any], when: datetime) -> None:
    """Make ``reminder`` fire at ``when`` (a one-shot that was completed becomes pending again)."""
    reminder["time"] = when.isoformat()
    if reminder.get("is_periodic"):
        reminder["next_trigger"] = when.isoformat()
    if reminder.get("status") in ("completed", "failed"):
        reminder["status"] = "pending"
    reminder.pop("completed_at", None)
    reminder["enabled"] = True


def pending_edit(user_id: str) -> str | None:
    """The reminder id the user's next text edits, or None (none pending or expired)."""
    entry = _pending_edits.get(str(user_id))
    if entry is None:
        return None
    reminder_id, deadline = entry
    if time.monotonic() > deadline:
        _pending_edits.pop(str(user_id), None)
        return None
    return reminder_id


def _edit_is_pending(message: Message) -> bool:
    """Filter: the user has a pending text edit (see ``reminder_edit_<id>``)."""
    return message.chat is not None and pending_edit(str(message.chat.id)) is not None


def reset() -> None:
    """Forget pending edits (tests)."""
    _pending_edits.clear()


# --------------------------------------------------------------------------- views


async def show_reminders_menu(message: Message, user_id: str, page: int = 1) -> None:
    """Send the main reminders menu (a fresh message) with the list of active reminders."""
    if not reminders_store.path(user_id).exists():
        await answer_md(message, "No reminders found.\n\n" + NEW_HINT)
        return

    try:
        all_reminders = await reminders_store.load(user_id)
    except Exception as e:  # noqa: BLE001
        logger.error("Error loading reminders for %s: %s", user_id, e)
        await message.answer("Error loading reminders.")
        return

    status_text, reply_markup = _build_menu(all_reminders, page, _zone(user_id))
    await answer_md(message, status_text, reply_markup)


async def _find(user_id: str, reminder_id: str) -> dict[str, Any] | None:
    reminders = await reminders_store.load(user_id)
    return next((r for r in reminders if str(r.get("id")) == reminder_id), None)


async def show_reminder_info(message: Message, user_id: str, reminder_id: str, *, fresh: bool = False) -> None:
    """Edit ``message`` into the detail view of one reminder (``fresh``: send a new message instead)."""
    try:
        reminder = await _find(user_id, reminder_id)
        if reminder is None:
            await edit_md(message, "Reminder not found.", _back_keyboard())
            return
        text = render_info(reminder, _zone(user_id))
        if fresh:
            await answer_md(message, text, _info_keyboard(reminder))
        else:
            await edit_md(message, text, _info_keyboard(reminder))
    except Exception as e:  # noqa: BLE001
        logger.error("Error showing reminder info: %s", e)
        await edit_md(message, "Error showing reminder information. Please try again.")


async def delete_reminder(message: Message, user_id: str, reminder_id: str) -> None:
    """Delete one reminder atomically, then replace the menu message with a fresh one."""

    def _delete(reminders: list[dict[str, Any]]) -> bool:
        before = len(reminders)
        reminders[:] = [r for r in reminders if str(r.get("id")) != reminder_id]
        return len(reminders) != before

    try:
        removed = await reminders_store.update(user_id, _delete)
        if not removed:
            logger.info("Delete requested for unknown reminder %s of user %s", reminder_id, user_id)
        await show_reminders_menu(message, user_id)
        await delete_quietly(message)
    except Exception as e:  # noqa: BLE001
        logger.error("Error deleting reminder: %s", e)
        await edit_md(message, "Error deleting reminder. Please try again.")


async def toggle_reminder(message: Message, user_id: str, reminder_id: str) -> None:
    """Flip one reminder's ``enabled`` flag atomically, then replace the menu message."""

    def _toggle(reminders: list[dict[str, Any]]) -> bool | None:
        for reminder in reminders:
            if str(reminder.get("id")) == reminder_id:
                reminder["enabled"] = not _is_enabled(reminder)
                return reminder["enabled"]
        return None

    try:
        new_state = await reminders_store.update(user_id, _toggle)
        if new_state is None:
            logger.info("Toggle requested for unknown reminder %s of user %s", reminder_id, user_id)
        await show_reminders_menu(message, user_id)
        await delete_quietly(message)
    except Exception as e:  # noqa: BLE001
        logger.error("Error toggling reminder: %s", e)
        await edit_md(message, "Error toggling reminder. Please try again.")


async def delay_reminder(message: Message, user_id: str, reminder_id: str, minutes: int) -> None:
    """``⏰ +1h`` / ``📅 +1d``: push the reminder ``minutes`` later (from its time, or from now when
    that is already past) and show the detail view again."""
    now = datetime.now(timezone.utc)

    def _delay(reminders: list[dict[str, Any]]) -> bool:
        for reminder in reminders:
            if str(reminder.get("id")) == reminder_id:
                try:
                    base = parse_time(str(reminder.get("time")))
                except (TypeError, ValueError):
                    base = now
                _reschedule(reminder, max(base, now) + timedelta(minutes=minutes))
                return True
        return False

    try:
        if not await reminders_store.update(user_id, _delay):
            await edit_md(message, "Reminder not found.", _back_keyboard())
            return
        await show_reminder_info(message, user_id, reminder_id)
    except Exception as e:  # noqa: BLE001
        logger.error("Error delaying reminder: %s", e)
        await edit_md(message, "Error changing the reminder. Please try again.")


async def snooze_reminder(user_id: str, reminder_id: str, suffix: str) -> datetime | None:
    """Re-schedule a fired reminder; returns the new UTC time, or None when it is unknown.

    A one-shot reminder is re-scheduled in place; a periodic one keeps its series and a one-off
    copy (``snoozed_from``) is added."""
    now = datetime.now(timezone.utc)
    when = snooze_time(now, suffix, _zone(user_id))
    if when is None:
        return None

    def _snooze(reminders: list[dict[str, Any]]) -> bool:
        for reminder in reminders:
            if str(reminder.get("id")) == reminder_id:
                if reminder.get("is_periodic"):
                    reminders.append({
                        "id": new_reminder_id(reminders),
                        "user_id": reminder.get("user_id", user_id),
                        "text": reminder.get("text", ""),
                        "time": when.isoformat(),
                        "timezone": reminder.get("timezone", ""),
                        "type": "user",
                        "status": "pending",
                        "created_at": now.isoformat(),
                        "is_periodic": False,
                        "period_type": None,
                        "period_interval": None,
                        "last_triggered": None,
                        "next_trigger": None,
                        "enabled": True,
                        "snoozed_from": reminder_id,
                    })
                else:
                    _reschedule(reminder, when)
                return True
        return False

    if not await reminders_store.update(user_id, _snooze):
        return None
    return when


async def start_edit(message: Message, user_id: str, reminder_id: str) -> None:
    """``✏️ Edit text``: remember the pending edit and ask for the new text."""
    reminder = await _find(user_id, reminder_id)
    if reminder is None:
        await edit_md(message, "Reminder not found.", _back_keyboard())
        return
    _pending_edits[str(user_id)] = (reminder_id, time.monotonic() + EDIT_TIMEOUT)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✖️ Cancel edit", callback_data=f"reminder_editcancel_{reminder_id}")],
    ])
    await edit_md(message, EDIT_PROMPT.format(text=escape_markdown(str(reminder.get("text", "")))), keyboard)


async def apply_edit(message: Message, user_id: str, reminder_id: str, new_text: str) -> None:
    """The user's next text becomes the reminder text; a fresh detail view confirms it."""

    def _set_text(reminders: list[dict[str, Any]]) -> bool:
        for reminder in reminders:
            if str(reminder.get("id")) == reminder_id:
                reminder["text"] = new_text
                return True
        return False

    try:
        if not await reminders_store.update(user_id, _set_text):
            await answer_md(message, "Reminder not found: it may have been deleted meanwhile.")
            return
        await answer_md(message, "✅ Reminder text updated.")
        await show_reminder_info(message, user_id, reminder_id, fresh=True)
    except Exception as e:  # noqa: BLE001
        logger.error("Error editing reminder text: %s", e)
        await answer_md(message, "Error changing the reminder. Please try again.")


async def show_reminders_summary(message: Message, user_id: str) -> None:
    """Edit ``message`` into the aggregate counts view."""
    try:
        reminders = await reminders_store.load(user_id)

        active_user = len([r for r in reminders if _is_active(r) and _kind(r) == "user" and _is_enabled(r)])
        active_agent = len([r for r in reminders if _is_active(r) and _kind(r) == "agent" and _is_enabled(r)])
        disabled = len([r for r in reminders if _is_active(r) and not _is_enabled(r)])
        completed = len([r for r in reminders if _is_completed(r)])
        periodic = len([r for r in reminders if r.get("is_periodic", False)])
        quiet_runs = sum(int(r.get("quiet_runs") or 0) for r in reminders)

        summary_text = (
            "*Reminders Summary*\n\n"
            f"Active User Reminders: {active_user}\n"
            f"Active Agent Tasks: {active_agent}\n"
            f"Disabled Reminders: {disabled}\n"
            f"Completed Reminders: {completed}\n"
            f"Periodic Reminders: {periodic}\n"
            f"Quiet task runs (nothing new): {quiet_runs}\n"
        )
        await edit_md(message, summary_text, _back_keyboard())
    except Exception as e:  # noqa: BLE001
        logger.error("Error showing reminders summary: %s", e)
        await edit_md(message, "Error showing reminders summary. Please try again.")


# --------------------------------------------------------------------------- handlers


@router.message(F.text, Command("reminders"))
async def reminders_command(message: Message) -> None:
    """Handle the /reminders command."""
    user_id = str(message.chat.id)
    _pending_edits.pop(user_id, None)  # a command ends a pending text edit
    await show_reminders_menu(message, user_id)


@router.message(F.text & ~F.text.startswith("/"), _edit_is_pending)
async def reminder_edit_text(message: Message) -> None:
    """The text following ``✏️ Edit text`` (within EDIT_TIMEOUT): the new reminder text."""
    user_id = str(message.chat.id)
    reminder_id = pending_edit(user_id)
    _pending_edits.pop(user_id, None)
    if reminder_id is None:  # expired between the filter and the handler
        return
    await apply_edit(message, user_id, reminder_id, (message.text or "").strip())


@router.callback_query(F.data == NOOP)
async def noop_button(callback: CallbackQuery) -> None:
    """Spacer / page-indicator / group-title buttons: just dismiss the spinner."""
    await callback.answer()


async def _answer(callback: CallbackQuery, text: str | None = None) -> None:
    try:
        await callback.answer(text)
    except Exception as e:  # noqa: BLE001 - the query may have expired
        logger.debug("Could not answer a reminders callback: %s", e)


@router.callback_query(F.data.startswith("reminder_") | F.data.startswith("reminders_"))
async def reminder_button(callback: CallbackQuery) -> None:
    """Handle reminder menu button presses."""
    await _answer(callback)
    if not isinstance(callback.message, Message):
        return

    message = callback.message
    user_id = str(message.chat.id)

    # "<prefix>_<action>_<payload>"; the payload (a reminder id, maybe "_<value>") is kept verbatim.
    parts = (callback.data or "").split("_", 2)
    action = parts[1] if len(parts) > 1 else None
    payload = parts[2] if len(parts) > 2 else ""

    if action == "page":
        try:
            page = int(payload)
        except ValueError:
            page = 1
        await show_reminders_menu(message, user_id, page)
        await delete_quietly(message)

    elif action == "info":
        await show_reminder_info(message, user_id, payload)

    elif action == "delete":
        await delete_reminder(message, user_id, payload)

    elif action == "toggle":
        await toggle_reminder(message, user_id, payload)

    elif action == "summary":
        await show_reminders_summary(message, user_id)

    elif action == "new":
        await edit_md(message, NEW_HINT, _back_keyboard())

    elif action == "delay":
        reminder_id, _, minutes = payload.rpartition("_")
        try:
            await delay_reminder(message, user_id, reminder_id, int(minutes))
        except ValueError:
            logger.warning("Bad delay callback from %s: %r", user_id, callback.data)

    elif action == "edit":
        await start_edit(message, user_id, payload)

    elif action == "editcancel":
        _pending_edits.pop(user_id, None)
        await show_reminder_info(message, user_id, payload)

    elif action == "snooze":
        reminder_id, _, suffix = payload.rpartition("_")
        when = await snooze_reminder(user_id, reminder_id, suffix)
        if when is None:
            await edit_md(message, (message.text or "🔔 Reminder") + "\n\n⚠️ Could not snooze: the reminder is gone.")
            return
        reminder = await _find(user_id, reminder_id)
        text = str((reminder or {}).get("text") or message.text or "")
        await edit_md(message, snooze_text(text, when, _zone(user_id)))

    else:
        logger.warning("Unknown reminders callback from %s: %r", user_id, callback.data)
