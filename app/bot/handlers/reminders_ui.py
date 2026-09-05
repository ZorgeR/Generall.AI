"""/reminders command and the ``reminder_*`` / ``reminders_*`` inline keyboard family.

Ported from the python-telegram-bot implementation. All texts, button labels,
callback_data strings and the 5-per-page pagination are preserved. Differences
from the original, all deliberate:

* No authorization check in the handlers: ``AuthMiddleware`` already authorizes
  every invited user, which also fixes the old bug where only ``TELEGRAM_CHAT_ID``
  members could press the buttons.
* ``noop`` spacer buttons get a dedicated handler that only answers the query,
  so they no longer show a loading spinner.
* Reminder ids are opaque strings (uuid hex), never converted to ``int``.
* All reads and writes go through ``reminders_store``; delete/toggle mutate under
  the store's per-user lock so they cannot race the scheduler or the agent tool.
* Status ``processing`` (an agent reminder currently running) is treated like
  ``pending`` for listing/counting and rendered as ``🏃 running``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.ui import answer_md, delete_quietly, edit_md, escape_markdown
from reminders_store import reminders_store

logger = logging.getLogger(__name__)

router = Router(name="reminders")

ITEMS_PER_PAGE = 5
ACTIVE_STATUSES = ("pending", "processing")
TEXT_PREVIEW_LEN = 35

NOOP = "noop"
BACK_TO_LIST = "reminders_page_1"


# --------------------------------------------------------------------------- helpers


def _is_active(reminder: dict[str, Any]) -> bool:
    return reminder.get("status") in ACTIVE_STATUSES


def _is_completed(reminder: dict[str, Any]) -> bool:
    return reminder.get("status") == "completed"


def _is_enabled(reminder: dict[str, Any]) -> bool:
    return bool(reminder.get("enabled", True))


def _display_status(status: Any) -> str:
    return "🏃 running" if status == "processing" else str(status)


def _format_time(value: Any, fmt: str) -> str:
    """Format an ISO timestamp; fall back to the raw value if it does not parse."""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).strftime(fmt)
    except (TypeError, ValueError):
        logger.warning("Unparseable reminder timestamp: %r", value)
        return str(value)


def _reminder_button_label(reminder: dict[str, Any]) -> str:
    formatted_time = _format_time(reminder.get("time"), "%H:%M %d/%m")  # More compact time format
    if reminder.get("is_periodic"):
        period_type = str(reminder.get("period_type") or "")
        formatted_time += f" ↻{reminder.get('period_interval', '')}{period_type[:1]}"
    text = str(reminder.get("text", ""))
    text_preview = text[:TEXT_PREVIEW_LEN] + ("..." if len(text) > TEXT_PREVIEW_LEN else "")
    return f"📅 {formatted_time} : {text_preview}"


def _build_menu(all_reminders: list[dict[str, Any]], page: int) -> tuple[str, InlineKeyboardMarkup]:
    """Build the status text and keyboard of the reminders list for ``page``."""
    active_reminders = [r for r in all_reminders if _is_active(r)]
    completed_reminders = [r for r in all_reminders if _is_completed(r)]

    total_pages = (len(active_reminders) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    # If the requested page is past the end (e.g. after a delete), show the last page.
    page = max(1, min(page, total_pages)) if total_pages else 1

    start_idx = (page - 1) * ITEMS_PER_PAGE
    current_reminders = active_reminders[start_idx : start_idx + ITEMS_PER_PAGE]

    keyboard: list[list[InlineKeyboardButton]] = []
    for reminder in current_reminders:
        reminder_id = str(reminder.get("id", ""))
        keyboard.append(
            [InlineKeyboardButton(text=_reminder_button_label(reminder), callback_data=f"reminder_info_{reminder_id}")]
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

    keyboard.append([InlineKeyboardButton(text="📊 Summary", callback_data="reminders_summary")])

    status_text = (
        "*Your Reminders*\n\n"
        f"Active: {len(active_reminders)} | "
        f"Completed: {len(completed_reminders)}\n\n"
        "Select a reminder to manage:"
    )
    return status_text, InlineKeyboardMarkup(inline_keyboard=keyboard)


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="⬅️ Back", callback_data=BACK_TO_LIST)]])


# --------------------------------------------------------------------------- views


async def show_reminders_menu(message: Message, user_id: str, page: int = 1) -> None:
    """Send the main reminders menu (a fresh message) with the list of active reminders."""
    if not reminders_store.path(user_id).exists():
        await message.answer("No reminders found.")
        return

    try:
        all_reminders = await reminders_store.load(user_id)
    except Exception as e:  # noqa: BLE001
        logger.error("Error loading reminders for %s: %s", user_id, e)
        await message.answer("Error loading reminders.")
        return

    status_text, reply_markup = _build_menu(all_reminders, page)
    await answer_md(message, status_text, reply_markup)


async def show_reminder_info(message: Message, user_id: str, reminder_id: str) -> None:
    """Edit ``message`` into the detail view of one reminder."""
    try:
        reminders = await reminders_store.load(user_id)
        reminder = next((r for r in reminders if str(r.get("id")) == reminder_id), None)
        if reminder is None:
            await edit_md(message, "Reminder not found.")
            return

        formatted_time = _format_time(reminder.get("time"), "%Y-%m-%d %H:%M:%S UTC")

        next_trigger = ""
        if reminder.get("is_periodic") and reminder.get("next_trigger"):
            next_trigger = f"\nNext Trigger: {_format_time(reminder['next_trigger'], '%Y-%m-%d %H:%M:%S UTC')}"

        last_triggered = ""
        if reminder.get("last_triggered"):
            last_triggered = f"\nLast Triggered: {_format_time(reminder['last_triggered'], '%Y-%m-%d %H:%M:%S UTC')}"

        periodic_info = ""
        if reminder.get("is_periodic"):
            periodic_info = (
                f"\nRepeats every: {reminder.get('period_interval', '')} "
                f"{escape_markdown(str(reminder.get('period_type', '')))}"
            )

        info_text = (
            "*Reminder Details*\n\n"
            f"Text: {escape_markdown(str(reminder.get('text', '')))}\n"
            f"Type: {escape_markdown(str(reminder.get('type', '')))}\n"
            f"Status: {_display_status(reminder.get('status'))}\n"
            f"Created: {_format_time(reminder.get('created_at'), '%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Scheduled: {formatted_time}"
            f"{periodic_info}"
            f"{next_trigger}"
            f"{last_triggered}"
        )
        await edit_md(message, info_text, _back_keyboard())
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


async def show_reminders_summary(message: Message, user_id: str) -> None:
    """Edit ``message`` into the aggregate counts view."""
    try:
        reminders = await reminders_store.load(user_id)

        active_user = len([r for r in reminders if _is_active(r) and r.get("type") == "user" and _is_enabled(r)])
        active_agent = len([r for r in reminders if _is_active(r) and r.get("type") == "agent" and _is_enabled(r)])
        disabled = len([r for r in reminders if _is_active(r) and not _is_enabled(r)])
        completed = len([r for r in reminders if _is_completed(r)])
        periodic = len([r for r in reminders if r.get("is_periodic", False)])

        summary_text = (
            "*Reminders Summary*\n\n"
            f"Active User Reminders: {active_user}\n"
            f"Active Agent Tasks: {active_agent}\n"
            f"Disabled Reminders: {disabled}\n"
            f"Completed Reminders: {completed}\n"
            f"Periodic Reminders: {periodic}\n"
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
    await show_reminders_menu(message, user_id)


@router.callback_query(F.data == NOOP)
async def noop_button(callback: CallbackQuery) -> None:
    """Spacer / page-indicator buttons: just dismiss the spinner."""
    await callback.answer()


@router.callback_query(F.data.startswith("reminder_") | F.data.startswith("reminders_"))
async def reminder_button(callback: CallbackQuery) -> None:
    """Handle reminder menu button presses."""
    await callback.answer()
    if not isinstance(callback.message, Message):
        return

    message = callback.message
    user_id = str(message.chat.id)

    # "<prefix>_<action>_<payload>"; the payload (a reminder id) is kept verbatim.
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

    else:
        logger.warning("Unknown reminders callback from %s: %r", user_id, callback.data)
