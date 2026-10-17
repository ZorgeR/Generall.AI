"""Answers to ``ask_user_choice`` questions (``ask_*`` callbacks).

A tap resolves the pending question in ``bot.questions``, which wakes the tool
call waiting for it, and turns the message into its final state: the question
and the chosen option, without buttons. A tap on a question that is gone gets
a short notice; when it expired (timed out, cancelled, or from before a restart)
any buttons still on it are taken off too.
"""
from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, Message

from bot import questions
from bot.sender import ChatSender

logger = logging.getLogger(__name__)

router = Router(name="questions")

EXPIRED_TEXT = "This question has expired."
ALREADY_ANSWERED_TEXT = "Already answered."


async def _notify(callback: CallbackQuery, text: str) -> None:
    try:
        await callback.answer(text)
    except Exception as e:  # noqa: BLE001 - a stale callback id must not break the handler
        logger.debug("Could not answer the callback: %s", e)


@router.callback_query(F.data.startswith(questions.CALLBACK_PREFIX))
async def on_question_answer(callback: CallbackQuery, bot: Bot) -> None:
    parsed = questions.parse_callback_data(callback.data)
    message = callback.message
    if parsed is None or not isinstance(message, Message):
        await _notify(callback, EXPIRED_TEXT)
        return
    qid, index = parsed
    # Read before resolving: the waiting tool call discards the question as soon as it wakes.
    pending = questions.get(qid)
    option = questions.resolve(qid, message.chat.id, index)
    if option is None or pending is None:
        status = questions.state(qid)
        if status == "answered":
            await _notify(callback, ALREADY_ANSWERED_TEXT)
            return
        await _notify(callback, EXPIRED_TEXT)
        if status != "pending":
            # Unknown (from before a restart) or closed: the tool call's "No answer" /
            # "Cancelled" edit is best-effort and skipped when the cancel hit mid-send, so
            # buttons that are still there get taken off (a no-op when none are left). A
            # still-pending question (tap from another chat, forged index) keeps its buttons.
            await questions.expire_message(bot, message)
        return
    await _notify(callback, questions.answered_line(option))
    await questions.show_outcome(ChatSender(bot, message.chat.id), pending, questions.answered_line(option), message)
