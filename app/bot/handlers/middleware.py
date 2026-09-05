"""Authorization and quota middleware.

Attach ``AuthMiddleware`` to a router's ``message`` and ``callback_query``
observers. It injects ``user_id`` (str chat id) and, when ``check_limits`` is
on, ``limit`` (int or None) into handler kwargs.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from bot.auth import auth
from bot.limits import check_user_limits

logger = logging.getLogger(__name__)

UNAUTHORIZED_TEXT = "Unauthorized. You need an invite to use this bot."
ADMIN_ONLY_TEXT = "Unauthorized. Only admin can use this command."


def event_chat_id(event: TelegramObject) -> int | None:
    if isinstance(event, Message):
        return event.chat.id
    if isinstance(event, CallbackQuery):
        if event.message is not None:
            return event.message.chat.id
        return event.from_user.id if event.from_user else None
    return None


async def _deny(event: TelegramObject, text: str) -> None:
    try:
        if isinstance(event, Message):
            await event.answer(text)
        elif isinstance(event, CallbackQuery):
            await event.answer(text, show_alert=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not deliver denial: %s", e)


class AuthMiddleware(BaseMiddleware):
    def __init__(self, *, require_admin: bool = False, check_limits: bool = False) -> None:
        self.require_admin = require_admin
        self.check_limits = check_limits

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat_id = event_chat_id(event)
        if chat_id is None:
            return None
        user_id = str(chat_id)
        if self.require_admin and not auth.is_admin(user_id):
            await _deny(event, ADMIN_ONLY_TEXT)
            return None
        if not auth.is_authorized(user_id):
            logger.warning("Unauthorized access attempt from %s", user_id)
            await _deny(event, UNAUTHORIZED_TEXT)
            return None
        data["user_id"] = user_id
        if self.check_limits:
            allowed, used, limit = check_user_limits(user_id)
            if not allowed:
                await _deny(event, f"⚠️ You've reached your action limit ({used}/{limit} for 30 days). Contact admin.")
                return None
            data["limit"] = limit
        return await handler(event, data)
