"""``/profile`` and the ``profile_`` buttons (docs/design-v3.md, F8). Router on ``ui`` (auth, no quota).

Legacy-Markdown messages (``bot.ui.answer_md``, plain fallback) showing ``data/<uid>/profile.md``
as the agent keeps it (``bot.profile.ProfileStore``): sections in bold, entries as bullets, every
entry escaped because the agent wrote it from the user's words. Under it a hint that the profile is
changed by talking to the assistant ("remember that ...", "forget ...") and two buttons:
"🔄 Refresh" (``profile_refresh``) re-reads the file and edits the message in place; "🗑 Forget
everything" (``profile_clear``) first turns the message into a confirmation (``profile_clear_yes``
deletes the file, ``profile_cancel`` shows the profile again). File reads run in a worker thread.

The store caps the file at 8 KB but Telegram caps a message at 4096 characters, so ``/profile``
sends the profile as pages (``render_pages``: sections packed whole, an entry never cut mid-line,
the buttons on the last page). The buttons edit one message in place and therefore show one page
(``render_one``); when more exists, a line says how many entries are left out and that ``/profile``
sends them all. Callbacks are answered best-effort (an old query raises) and ignored when the
message is gone.
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
MORE = "_… {n} more entries not shown here. Send /profile to see the whole profile._"
CONTINUED = " _(continued)_"
CONFIRM = "🗑 *Forget everything?*\n\nThis deletes your whole profile: About, Preferences and Notes. It cannot be undone."
CLEARED = "🗑 Your profile was deleted."

PAGE_CHARS = 4000  # Telegram allows 4096 UTF-16 units per message; the rest is room for the plain fallback
_FOOTER_RESERVE = 200  # the hint, or the MORE line plus the hint, fits under every page


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _section(title: str) -> str:
    return f"*{escape_markdown(title)}*"


def _bullet(entry: str) -> str:
    return f"• {escape_markdown(entry)}"


def render(profile: Profile) -> str:
    """Legacy Markdown of the whole profile: header, the sections (bold title, bullets), the hint.
    This is what ``render_pages`` gives when everything fits in one message."""
    if profile.is_empty():
        return f"{HEADER}\n\n{EMPTY}\n\n{HINT}"
    parts = []
    for title, entries in profile.sections.items():
        if entries:
            parts.append(_section(title) + "\n" + "\n".join(_bullet(e) for e in entries))
    return "\n\n".join([HEADER, *parts, HINT])


def render_pages(profile: Profile, limit: int = PAGE_CHARS) -> list[str]:
    """The profile as legacy-Markdown messages of at most ``limit`` UTF-16 units each.

    The header opens the first page and the hint closes the last. Sections are packed whole where
    they fit; one that runs over continues on the next page under its title again ("(continued)").
    An entry (one line) is never cut: the store keeps entries short, so each fits on a page.
    """
    if profile.is_empty():
        return [render(profile)]
    budget = limit - _FOOTER_RESERVE
    pages: list[str] = []
    page, open_section = HEADER, None
    for title, entries in profile.sections.items():
        for entry in entries:
            line = _bullet(entry)
            addition = f"\n{line}" if title == open_section else f"\n\n{_section(title)}\n{line}"
            if page != HEADER and _utf16_len(page) + _utf16_len(addition) > budget:
                pages.append(page)
                page = f"{_section(title)}{CONTINUED if title == open_section else ''}\n{line}"
            else:
                page += addition
            open_section = title
    pages.append(f"{page}\n\n{HINT}")
    return pages


def render_one(profile: Profile, limit: int = PAGE_CHARS) -> str:
    """What one message can show: the whole profile when it fits, else the first page and how many
    entries are left out. The buttons edit their message in place, so they get this."""
    pages = render_pages(profile, limit)
    if len(pages) == 1:
        return pages[0]
    shown = sum(1 for line in pages[0].splitlines() if line.startswith("• "))
    total = sum(len(entries) for entries in profile.sections.values())
    return f"{pages[0]}\n\n{MORE.format(n=total - shown)}\n\n{HINT}"


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
    pages = render_pages(profile)
    for page in pages[:-1]:
        await answer_md(message, page)
    await answer_md(message, pages[-1], keyboard(profile))


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
    await edit_md(callback.message, render_one(profile), keyboard(profile))
