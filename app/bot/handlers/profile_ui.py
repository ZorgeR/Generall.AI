"""``/profile`` and the ``profile_`` buttons (docs/design-v3.md, F8). Router on ``ui`` (auth, no quota).

One legacy-Markdown message (``bot.ui.answer_md``, plain fallback) showing ``data/<uid>/profile.md``
as the agent keeps it (``bot.profile.ProfileStore``): sections in bold, entries as bullets, every
entry escaped because the agent wrote it from the user's words. Under it a hint that the profile is
changed by talking to the assistant ("remember that ...", "forget ...") and two buttons:
"🔄 Refresh" (``profile_refresh``) re-reads the file and edits the message in place; "🗑 Forget
everything" (``profile_clear``) first turns the message into a confirmation (``profile_clear_yes``
deletes the file, ``profile_cancel`` shows the profile again). File reads run in a worker thread.
Callbacks are answered best-effort (an old query raises) and ignored when the message is gone.
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.profile import Profile, profile_store
from bot.ui import answer_md, edit_md, escape_markdown

logger = logging.getLogger(__name__)

router = Router(name="profile")

REFRESH = "profile_refresh"
CLEAR = "profile_clear"
CLEAR_YES = "profile_clear_yes"
CANCEL = "profile_cancel"

HEADER = "👤 *Your profile*"
EMPTY = "Your profile is empty."
HINT = '💡 _Tell me "remember that ..." or "forget ..." to change it._'
CONFIRM = "🗑 *Forget everything?*\n\nThis deletes your whole profile: About, Preferences and Notes. It cannot be undone."
CLEARED = "🗑 Your profile was deleted."


def render(profile: Profile) -> str:
    """Legacy Markdown: header, the sections (bold title, bullets), the hint."""
    if profile.is_empty():
        return f"{HEADER}\n\n{EMPTY}\n\n{HINT}"
    parts = []
    for title, entries in profile.sections.items():
        if entries:
            parts.append(f"*{escape_markdown(title)}*\n" + "\n".join(f"• {escape_markdown(e)}" for e in entries))
    return "\n\n".join([HEADER, *parts, HINT])


def keyboard(profile: Profile) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text="🔄 Refresh", callback_data=REFRESH)]
    if not profile.is_empty():
        row.append(InlineKeyboardButton(text="🗑 Forget everything", callback_data=CLEAR))
    return InlineKeyboardMarkup(inline_keyboard=[row])


def confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Yes, forget everything", callback_data=CLEAR_YES),
        InlineKeyboardButton(text="↩️ Keep it", callback_data=CANCEL),
    ]])


async def load(user_id: str) -> Profile:
    return await asyncio.to_thread(profile_store.load, user_id)


# ---- handlers -----------------------------------------------------------------
@router.message(F.text, Command("profile"))
async def profile_command(message: Message, user_id: str) -> None:
    profile = await load(user_id)
    await answer_md(message, render(profile), keyboard(profile))


@router.callback_query(F.data.startswith("profile_"))
async def profile_button(callback: CallbackQuery, user_id: str) -> None:
    try:
        await callback.answer()
    except Exception as e:  # noqa: BLE001 - an old query; the action still happens
        logger.debug("Could not answer the profile button: %s", e)
    if not isinstance(callback.message, Message):
        return
    if callback.data == CLEAR:
        await edit_md(callback.message, CONFIRM, confirm_keyboard())
        return
    if callback.data == CLEAR_YES:
        await asyncio.to_thread(profile_store.clear, user_id)
        profile = await load(user_id)
        await edit_md(callback.message, f"{CLEARED}\n\n{HINT}", keyboard(profile))
        return
    if callback.data not in (REFRESH, CANCEL):
        logger.warning("Unknown profile callback from %s: %r", user_id, callback.data)
        return
    profile = await load(user_id)
    await edit_md(callback.message, render(profile), keyboard(profile))
