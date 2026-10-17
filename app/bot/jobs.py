"""Background scheduler: fires user reminders and queues agent reminders (scheduled tasks).

Runs every ``REMINDER_INTERVAL`` seconds. User reminders are sent directly, with snooze
buttons (``reminder_snooze_<id>_<minutes|tomorrow>``, handled in ``handlers/reminders_ui.py``).
Agent reminders are marked ``processing`` and submitted to the user's queue, so they never
run concurrently with that user's own messages.

Proactive tasks: an agent reminder runs ``run_turn`` with a framing
prompt (``task_prompt``: the task text, its id and the user's local time) and
``quiet_marker=QUIET_MARKER``. There is no separate announcement: the status message of the
turn is enough, and a run that answers exactly ``<no_update/>`` leaves nothing behind (the
status is deleted). The record keeps ``agent_response`` (the marker when quiet),
``last_result`` ("quiet" | "answered") and a ``quiet_runs`` counter. A task the user moved in
``/reminders`` while it ran keeps that new time (the run is recorded, the series not stepped).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot import timezones
from bot.agent_runner import QUIET_MARKER, run_turn
from bot.auth import auth
from bot.config import config
from bot.limits import check_user_limits
from bot.queue import Job, JobContext, QueueManager
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.ui import escape_markdown
from reminders_store import calculate_next_trigger, parse_time, reminders_store

logger = logging.getLogger(__name__)

REMINDER_INTERVAL = 10.0
TASK_HEADER = "🤖 *Scheduled task...*"
TASK_REASONING_CAPTION = "Scheduled task reasoning history."

# Snooze buttons under a fired user reminder: (label, callback suffix); minutes or "tomorrow"
# (= 09:00 local on the next day, see reminders_ui.snooze_time).
SNOOZE_OPTIONS: tuple[tuple[str, str], ...] = (("⏰ 10 min", "10"), ("1 h", "60"), ("tomorrow 09:00", "tomorrow"))


def _is_active(reminder: dict) -> bool:
    return reminder.get("enabled", True) is not False


def user_zone(user_id: str):
    """The user's saved zone (``timezone.name``) or None; a settings error means UTC."""
    try:
        return timezones.zone_of(UserSettings(user_id))
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read the time zone of %s: %s", user_id, e)
        return None


def task_prompt(reminder: dict, now: datetime, zone) -> str:
    """The framing prompt of a scheduled agent task (F11)."""
    return (
        f"[Scheduled task] This is a task you scheduled earlier (id {reminder.get('id')}; it is "
        f"{timezones.format_local(now, zone)}). Task: {reminder.get('text', '')}. Do it now. "
        f"If there is nothing new or worth telling the user, reply with exactly {QUIET_MARKER} and nothing else."
    )


def snooze_keyboard(reminder_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=label, callback_data=f"reminder_snooze_{reminder_id}_{suffix}")
        for label, suffix in SNOOZE_OPTIONS
    ]])


def _reset_stale_processing(reminder: dict, now: datetime) -> None:
    """A reminder left in 'processing' by a crash goes back to pending after the turn timeout."""
    since = reminder.get("processing_since")
    if reminder.get("status") == "processing" and since:
        try:
            if now - parse_time(since) > timedelta(seconds=config.turn_timeout_seconds * 2):
                reminder["status"] = "pending"
                reminder.pop("processing_since", None)
        except ValueError:
            reminder["status"] = "pending"


def _reschedule_or_complete(reminder: dict, now: datetime, response: str | None = None) -> None:
    if reminder.get("is_periodic"):
        try:
            next_trigger = calculate_next_trigger(
                reminder["time"], reminder.get("period_type"), reminder.get("period_interval"),
                weekdays=reminder.get("weekdays"), tz=reminder.get("timezone"),
            )
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Cannot reschedule reminder %s: %s", reminder.get("id"), e)
            reminder["status"] = "failed"
            reminder["enabled"] = False
        else:
            reminder["last_triggered"] = now.isoformat()
            reminder["time"] = next_trigger.isoformat()
            reminder["next_trigger"] = next_trigger.isoformat()
            reminder["status"] = "pending"
    else:
        reminder["status"] = "completed"
        reminder["completed_at"] = now.isoformat()
    _record_response(reminder, response)
    reminder.pop("processing_since", None)


def _record_response(reminder: dict, response: str | None) -> None:
    """The outcome of a scheduled task's run: agent_response, last_result, quiet_runs."""
    if response is None:
        return
    reminder["agent_response"] = response
    quiet = response.strip() == QUIET_MARKER
    reminder["last_result"] = "quiet" if quiet else "answered"
    if quiet:
        reminder["quiet_runs"] = int(reminder.get("quiet_runs") or 0) + 1


async def check_reminders(bot: Bot, queue: QueueManager) -> None:
    now = datetime.now(timezone.utc)
    for user_id in list(reminders_store.user_ids()):
        try:
            chat_id = int(user_id)
        except ValueError:
            continue  # not a chat directory
        if not auth.is_authorized(user_id):
            continue  # blocked or no longer authorized: their reminders stay pending
        due_user: list[dict] = []
        due_agent: list[dict] = []

        def _collect(reminders: list[dict]) -> None:
            for r in reminders:
                _reset_stale_processing(r, now)
                if r.get("status") != "pending" or not _is_active(r):
                    continue
                try:
                    if parse_time(r["time"]) > now:
                        continue
                except (KeyError, ValueError):
                    continue
                if r.get("type", "user") == "agent":
                    # Claim it atomically so the next tick does not queue it twice.
                    r["status"] = "processing"
                    r["processing_since"] = now.isoformat()
                    due_agent.append(dict(r))
                else:
                    due_user.append(dict(r))

        try:
            await reminders_store.update(user_id, _collect)
        except Exception as e:  # noqa: BLE001
            logger.error("Error scanning reminders for %s: %s", user_id, e)
            continue

        sender = ChatSender(bot, chat_id)
        delivered: set[str] = set()
        for r in due_user:
            try:
                await sender.send_text(
                    f"🔔 *Reminder*\n\n{escape_markdown(r['text'])}",
                    reply_markup=snooze_keyboard(str(r.get("id"))),
                )
                delivered.add(str(r.get("id")))
                logger.info("Sent reminder %s to %s", r.get("id"), user_id)
            except Exception as e:  # noqa: BLE001
                logger.error("Error sending reminder to %s (will retry next tick): %s", user_id, e)
        if delivered:
            # Only reminders that actually reached the user are rescheduled/completed.
            def _mark_delivered(reminders: list[dict]) -> None:
                for r in reminders:
                    if str(r.get("id")) in delivered and r.get("status") == "pending":
                        _reschedule_or_complete(r, now)

            try:
                await reminders_store.update(user_id, _mark_delivered)
            except Exception as e:  # noqa: BLE001
                logger.error("Error updating reminders for %s: %s", user_id, e)
        for r in due_agent:
            await queue.submit(Job(
                user_id=user_id,
                label="scheduled agent task",
                run=partial(run_agent_reminder, bot, user_id, r),
            ))


async def run_agent_reminder(bot: Bot, user_id: str, reminder: dict, ctx: JobContext) -> None:
    """Run one scheduled agent task as a turn (framing prompt, quiet marker) and record the outcome."""
    response: str | None = None
    try:
        prompt = task_prompt(reminder, datetime.now(timezone.utc), user_zone(user_id))
        _, _, limit = check_user_limits(user_id)
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=int(user_id), prompt=prompt, ctx=ctx, limit=limit,
            header=TASK_HEADER, reasoning_caption=TASK_REASONING_CAPTION, quiet_marker=QUIET_MARKER,
        )
        if result is None:
            response = None
        elif result.quiet:
            response = QUIET_MARKER
        else:
            response = result.response
    except asyncio.CancelledError:
        response = None
        raise
    finally:
        now = datetime.now(timezone.utc)

        def _finalize(reminders: list[dict]) -> None:
            for r in reminders:
                if str(r.get("id")) == str(reminder.get("id")):
                    if r.get("time") != reminder.get("time"):
                        # Moved (⏰ +1h / 📅 +1d in /reminders) while this run was going: the new
                        # time stands. Stepping from it would skip the run the user just asked for.
                        r["status"] = "pending"
                        r.pop("processing_since", None)
                        _record_response(r, response)
                    elif response is None and not r.get("is_periodic"):
                        r["status"] = "failed"
                        r.pop("processing_since", None)
                    else:
                        _reschedule_or_complete(r, now, response)
                    return

        try:
            await reminders_store.update(user_id, _finalize)
        except Exception as e:  # noqa: BLE001
            logger.error("Error finalizing reminder %s for %s: %s", reminder.get("id"), user_id, e)


async def reminders_loop(bot: Bot, queue: QueueManager, interval: float = REMINDER_INTERVAL) -> None:
    await asyncio.sleep(1)
    while True:
        try:
            await check_reminders(bot, queue)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Error in reminder checker: %s", e)
        await asyncio.sleep(interval)
