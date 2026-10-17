"""Buttons that act on a finished turn (``turn_*`` callbacks, see ``bot/turn_actions.py``).

``router`` (ui: auth only) handles actions that do not run the agent: 🔊 Listen
reads the stored answer aloud, 📄 Reasoning sends the stored reasoning file.
``turn_router`` (chat: auth + action limit, so its handlers get ``limit``) handles
the ones that start a new turn: 🔁 Retry runs the stored prompt again (a scheduled
task's with its quiet marker: "nothing new" ends as ``QUIET_NOTICE``), ▶️ Continue
asks the agent to finish what it listed, and a follow-up button sends its
suggestion. Those post a short echo of what was pressed and queue the turn on it
like a typed message (busy notice included), so the answer replies to the echo.

A press is answered right away; a turn whose record is gone (pruned, or from a
forged button) gets "This action has expired."
"""
from __future__ import annotations

import asyncio
import logging
from functools import partial

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, Message

from bot import media, turn_actions
from bot.agent_runner import run_turn
from bot.handlers import messages as chat_messages  # the chat queue and its busy notice
from bot.queue import Job, JobContext
from bot.sender import ChatSender

logger = logging.getLogger(__name__)

router = Router(name="turn-actions")
turn_router = Router(name="turn-actions-agent")

EXPIRED_TEXT = "This action has expired."
PREPARING_AUDIO_TEXT = "🔊 Preparing audio…"
AUDIO_FAILED_TEXT = "Could not generate audio."
CONTINUE_PROMPT = "Continue the task from where you stopped: do the remaining steps you listed."
RETRY_ECHO = "🔁 Retry"
CONTINUE_ECHO = "▶️ Continue"
FOLLOW_UP_ECHO = "💬 {}"
QUIET_NOTICE = "📭 Nothing new to report."  # a retried scheduled task answered its quiet marker

_P = turn_actions.CALLBACK_PREFIX
_listening: set[tuple[str, str]] = set()  # (user_id, turn_id) whose audio is being generated


def _stats():
    from stats import stats_tracker

    return stats_tracker


async def _answer(callback: CallbackQuery, text: str | None = None, *, alert: bool = False) -> None:
    try:
        await callback.answer(text, show_alert=alert)
    except Exception as e:  # noqa: BLE001 - the query may have expired
        logger.debug("Could not answer the turn button: %s", e)


async def _pressed(callback: CallbackQuery, user_id: str, action: str) -> tuple[Message, turn_actions.TurnRecord, int | None] | None:
    """(message, record, follow-up index) for a valid press of ``action``; None after the expired alert."""
    parsed = turn_actions.parse_callback(callback.data)
    message = callback.message
    record = None
    if parsed is not None and parsed[0] == action and isinstance(message, Message):
        record = await asyncio.to_thread(turn_actions.load, user_id, parsed[1])
    if record is None:
        await _answer(callback, EXPIRED_TEXT, alert=True)
        return None
    return message, record, parsed[2]


def _thread_id(message: Message, record: turn_actions.TurnRecord) -> int | None:
    return message.message_thread_id if message.is_topic_message else record.thread_id


# ---------------------------------------------------------------------------
# no agent run: listen, reasoning file
# ---------------------------------------------------------------------------
@router.callback_query(F.data.startswith(f"{_P}{turn_actions.LISTEN}_"))
async def on_listen(callback: CallbackQuery, bot: Bot, user_id: str) -> None:
    pressed = await _pressed(callback, user_id, turn_actions.LISTEN)
    if pressed is None:
        return
    message, record, _ = pressed
    key = (user_id, record.id)
    if key in _listening:
        await _answer(callback, PREPARING_AUDIO_TEXT)  # a second press while the first one works
        return
    if not turn_actions.tts_available() or not record.response.strip():
        await _answer(callback, AUDIO_FAILED_TEXT, alert=True)
        return
    _listening.add(key)
    sender = ChatSender(bot, message.chat.id, _thread_id(message, record))
    try:
        await _answer(callback, PREPARING_AUDIO_TEXT)
        from voice import VoiceManager

        audio = await media.synthesize_speech(record.response, VoiceManager().get_user_voice(user_id))
        if not audio:
            raise RuntimeError("no audio returned")
        await sender.send_voice(audio)
    except Exception as e:  # noqa: BLE001
        # The query is already answered, so the failure is a short message instead of an alert.
        logger.error("Listen failed for turn %s of %s: %s", record.id, user_id, e)
        try:
            await sender.send_text(f"❌ {AUDIO_FAILED_TEXT}", markdown=False)
        except Exception as e2:  # noqa: BLE001
            logger.debug("Could not report the failed audio: %s", e2)
    finally:
        _listening.discard(key)


@router.callback_query(F.data.startswith(f"{_P}{turn_actions.REASONING}_"))
async def on_reasoning(callback: CallbackQuery, bot: Bot, user_id: str) -> None:
    pressed = await _pressed(callback, user_id, turn_actions.REASONING)
    if pressed is None:
        return
    message, record, _ = pressed
    text = await asyncio.to_thread(turn_actions.load_reasoning, user_id, record.id)
    if not text:
        await _answer(callback, EXPIRED_TEXT, alert=True)
        return
    await _answer(callback)
    chat_id = message.chat.id
    try:
        await ChatSender(bot, chat_id, _thread_id(message, record)).send_document(
            text.encode("utf-8"),
            filename=f"reasoning_{chat_id}_{record.id}.txt",
            caption=record.reasoning_caption or turn_actions.DEFAULT_REASONING_CAPTION,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Could not send the reasoning file of turn %s to %s: %s", record.id, chat_id, e)


# ---------------------------------------------------------------------------
# a new agent turn: retry, continue, follow-up
# ---------------------------------------------------------------------------
async def _run_button_turn(
    bot: Bot, user_id: str, chat_id: int, thread_id: int | None, prompt: str | list, reply_to: int,
    limit: int | None, speak: bool, ctx: JobContext, *, quiet_marker: str | None = None,
) -> None:
    # A scheduled task's prompt asks for the quiet marker when there is nothing new: it must not
    # be shown as the answer, and the user who pressed the button still gets a short reply.
    await run_turn(
        bot=bot, user_id=user_id, chat_id=chat_id, prompt=prompt, thread_id=thread_id,
        reply_to_message_id=reply_to, ctx=ctx, limit=limit, speak=speak,
        quiet_marker=quiet_marker, quiet_notice=QUIET_NOTICE if quiet_marker else None,
    )


async def _start_turn(
    callback: CallbackQuery, bot: Bot, user_id: str, limit: int | None, message: Message,
    record: turn_actions.TurnRecord, *, prompt: str | list, echo: str, label: str, speak: bool = False,
    quiet_marker: str | None = None,
) -> None:
    """Echo the press as a plain message and queue the turn on it, like a typed message."""
    await _answer(callback)
    chat_id = message.chat.id
    thread_id = _thread_id(message, record)
    echo_message = await ChatSender(bot, chat_id, thread_id).send_text(echo, markdown=False)
    _stats().track_message_received(user_id, "button")
    run = partial(
        _run_button_turn, bot, user_id, chat_id, thread_id, prompt, echo_message.message_id, limit, speak,
        quiet_marker=quiet_marker,
    )
    await chat_messages.submit(echo_message, Job(user_id=user_id, label=label, run=run))


@turn_router.callback_query(F.data.startswith(f"{_P}{turn_actions.RETRY}_"))
async def on_retry(callback: CallbackQuery, bot: Bot, user_id: str, limit: int | None = None) -> None:
    pressed = await _pressed(callback, user_id, turn_actions.RETRY)
    if pressed is None:
        return
    message, record, _ = pressed
    await _start_turn(
        callback, bot, user_id, limit, message, record,
        prompt=record.prompt_blocks or record.prompt, echo=RETRY_ECHO, label="retry", speak=record.speak,
        quiet_marker=record.quiet_marker,
    )


@turn_router.callback_query(F.data.startswith(f"{_P}{turn_actions.CONTINUE}_"))
async def on_continue(callback: CallbackQuery, bot: Bot, user_id: str, limit: int | None = None) -> None:
    pressed = await _pressed(callback, user_id, turn_actions.CONTINUE)
    if pressed is None:
        return
    message, record, _ = pressed
    await _start_turn(
        callback, bot, user_id, limit, message, record,
        prompt=CONTINUE_PROMPT, echo=CONTINUE_ECHO, label="continue", speak=record.speak,
    )


@turn_router.callback_query(F.data.startswith(f"{_P}{turn_actions.FOLLOW_UP}_"))
async def on_follow_up(callback: CallbackQuery, bot: Bot, user_id: str, limit: int | None = None) -> None:
    pressed = await _pressed(callback, user_id, turn_actions.FOLLOW_UP)
    if pressed is None:
        return
    message, record, index = pressed
    if index is None or index >= len(record.follow_ups):
        await _answer(callback, EXPIRED_TEXT, alert=True)
        return
    suggestion = record.follow_ups[index]
    await _start_turn(
        callback, bot, user_id, limit, message, record,
        prompt=suggestion, echo=FOLLOW_UP_ECHO.format(suggestion), label="follow-up",
    )


@turn_router.callback_query(F.data.startswith(_P))
async def on_unknown(callback: CallbackQuery) -> None:
    """Any other ``turn_`` data (an older button format, or forged): never leave the spinner running."""
    await _answer(callback, EXPIRED_TEXT, alert=True)
