"""/start, /invite, /listusers and /voice."""
from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.auth import auth
from bot.config import config
from bot.ui import answer_md, edit_md

logger = logging.getLogger(__name__)

public_router = Router(name="public")  # reachable without authorization
admin_router = Router(name="admin-commands")
voice_router = Router(name="voice")

WELCOME_AUTHORIZED = (
    "👋 Welcome to Generall.AI bot! Use me to get AI assistance.\n\n"
    "You can send me messages, voices, images, media groups, pdfs, and more to analyze. "
    "I have memory, access to the internet and a wide range of tools to help you."
)
WELCOME_UNAUTHORIZED = (
    "👋 Welcome! This bot requires an invitation to use.\n\n"
    "If you have an invite code, please use:\n<code>/invite YOUR_CODE_HERE</code>"
)


async def _invite_link(bot: Bot, code: str) -> str:
    me = await bot.me()
    return f"https://t.me/{me.username}?start=invite_{code}"


async def _redeem_invite(message: Message, bot: Bot, user_id: str, code: str) -> None:
    inviter = auth.find_invite(code)
    if not inviter:
        await message.answer("❌ Invalid or already used invite code.")
        return
    if inviter == user_id:
        await message.answer(
            "🔄 This is your own invite code! Share it with others instead.\n\n"
            "Forward this message to friends who want access to the bot:",
            parse_mode="HTML",
        )
        link = await _invite_link(bot, code)
        await message.answer(
            f"<b>Invite Link</b>\n\nUse this link to join: {link}\n\nOr use this command: <code>/invite {code}</code>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="📩 Share Invite", url=link)]]),
        )
        return
    auth.use_invite(code, user_id)
    await message.answer("✅ Invite accepted! You now have access to the bot.")
    if config.admin_id:
        try:
            await bot.send_message(
                chat_id=int(config.admin_id),
                text=(
                    f"🔔 New user joined!\nUser ID: <code>{user_id}</code>\n"
                    f"Invited by: <code>{inviter}</code>\nTotal users: {len(auth.authorized)}"
                ),
                parse_mode="HTML",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to notify admin: %s", e)


@public_router.message(CommandStart())
async def start_command(message: Message, bot: Bot, command: CommandObject) -> None:
    user_id = str(message.chat.id)
    args = (command.args or "").strip()
    if args.startswith("invite_"):
        await _redeem_invite(message, bot, user_id, args[len("invite_"):])
        return
    if auth.is_authorized(user_id):
        await message.answer(WELCOME_AUTHORIZED)
    else:
        await message.answer(WELCOME_UNAUTHORIZED, parse_mode="HTML")


@public_router.message(F.text, Command("invite"))
async def invite_command(message: Message, bot: Bot, command: CommandObject) -> None:
    user_id = str(message.chat.id)
    args = (command.args or "").strip()
    if args:
        await _redeem_invite(message, bot, user_id, args.split()[0])
        return
    if not auth.is_authorized(user_id):
        await message.answer("Unauthorized. You need an invite to use this bot.")
        return
    is_admin = auth.is_admin(user_id)
    count = auth.unused_invite_count(user_id)
    if not is_admin and count >= config.invite_limit:
        await message.answer(f"❌ You've reached your invite limit ({config.invite_limit}).")
        return
    code = auth.generate_invite(user_id)
    link = await _invite_link(bot, code)
    remaining = "unlimited" if is_admin else str(config.invite_limit - count - 1)
    await message.answer(
        f"🎟️ <b>New Invite Created</b>\n\nShare this link: {link}\n\n"
        f"Or use this command:\n<code>/invite {code}</code>\n\nInvites remaining: {remaining}/{config.invite_limit}",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="📩 Activate invite!", url=link)]]),
    )


@admin_router.message(F.text, Command("listusers"))
async def list_users_command(message: Message) -> None:
    users = sorted(auth.authorized)
    user_list = "\n".join(f"- <code>{u}</code>" for u in users)
    await message.answer(
        f"📊 <b>Bot Users Summary</b>\n\nTotal users: {len(users)}\nActive invites: {auth.total_unused_invites()}\n\n"
        f"<b>User IDs:</b>\n{user_list}",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /voice
# ---------------------------------------------------------------------------
def _voice_keyboard(current: str | None) -> InlineKeyboardMarkup:
    from voice import VoiceManager

    voices = VoiceManager().get_available_voices()
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for i, name in enumerate(voices.keys(), 1):
        label = f"✓ {name}" if name == current else name
        row.append(InlineKeyboardButton(text=label, callback_data=f"voice_{name}"))
        if i % 2 == 0 or i == len(voices):
            rows.append(row)
            row = []
    return InlineKeyboardMarkup(inline_keyboard=rows)


@voice_router.message(F.text, Command("voice"))
async def voice_command(message: Message, user_id: str) -> None:
    from voice import VoiceManager

    vm = VoiceManager()
    current = vm.get_voice_name(vm.get_user_voice(user_id))
    await answer_md(message, "*Choose a voice:*", reply_markup=_voice_keyboard(current))


@voice_router.callback_query(F.data.startswith("voice_"))
async def voice_button(callback: CallbackQuery, user_id: str) -> None:
    from voice import VoiceManager

    await callback.answer()
    if not isinstance(callback.message, Message):
        return
    name = callback.data.removeprefix("voice_")
    if VoiceManager().set_user_voice(user_id, name):
        await edit_md(callback.message, f"*Choose a voice:*\n\n_Voice set to:_ {name}", reply_markup=_voice_keyboard(name))
    else:
        await edit_md(callback.message, f"Error setting voice to: {name}\nPlease try again.")
