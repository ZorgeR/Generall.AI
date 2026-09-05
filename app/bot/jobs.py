"""Background scheduler: fires user reminders and queues agent reminders.

Runs every ``REMINDER_INTERVAL`` seconds. User reminders are sent directly.
Agent reminders are marked ``processing`` and submitted to the user's queue,
so they never run concurrently with that user's own messages.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial

from aiogram import Bot

from bot.agent_runner import run_turn
from bot.auth import auth
from bot.config import config
from bot.limits import check_user_limits
from bot.queue import Job, JobContext, QueueManager
from bot.sender import ChatSender
from bot.ui import escape_markdown
from reminders_store import calculate_next_trigger, parse_time, reminders_store

logger = logging.getLogger(__name__)

REMINDER_INTERVAL = 10.0


def _is_active(reminder: dict) -> bool:
    return reminder.get("enabled", True) is not False


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
            next_trigger = calculate_next_trigger(reminder["time"], reminder.get("period_type"), reminder.get("period_interval"))
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
    if response is not None:
        reminder["agent_response"] = response
    reminder.pop("processing_since", None)


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
                await sender.send_text(f"🔔 *Reminder*\n\n{escape_markdown(r['text'])}")
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
    sender = ChatSender(bot, int(user_id))
    text = reminder.get("text", "")
    response: str | None = None
    try:
        await sender.send_text("🤖 *Agent Reminder Task*\n\nProcessing scheduled task:\n" + escape_markdown(text))
        status = await sender.send_text("💭 *Processing Agent Task...*")
        _, _, limit = check_user_limits(user_id)
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=int(user_id), prompt=text, ctx=ctx, limit=limit, status=status,
            header="💭 *Processing Agent Task...*", reasoning_caption="Agent task reasoning history.",
        )
        response = result.response if result else None
    except asyncio.CancelledError:
        response = None
        raise
    finally:
        now = datetime.now(timezone.utc)

        def _finalize(reminders: list[dict]) -> None:
            for r in reminders:
                if str(r.get("id")) == str(reminder.get("id")):
                    if response is None and not r.get("is_periodic"):
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
