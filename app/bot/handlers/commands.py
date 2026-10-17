"""/start, /help, /invite, /listusers, /voice and /timezone, plus the command menu.

``COMMANDS`` is the one list of user commands: the menu Telegram shows behind the "/"
button (``bot_commands``, registered at startup by ``bot.app``) and the command section
of ``/help`` are both rendered from it. Admin-only commands are not in it.

``/timezone`` shows the saved zone and local time with a picker of
``timezones.COMMON_ZONES`` (``tz_set_<index>`` buttons) and the other ways to set it (share a
location, type a name); ``/timezone <name>`` validates with ``timezones.valid_zone`` (IANA names,
any capitalization, big-city aliases) and saves settings ``timezone.name``.
"""
from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import timezones
from bot.auth import auth
from bot.config import config
from bot.settings import UserSettings
from bot.ui import answer_md, edit_md, escape_markdown

logger = logging.getLogger(__name__)

public_router = Router(name="public")  # reachable without authorization
admin_router = Router(name="admin-commands")
voice_router = Router(name="voice")

# (command, menu description): descriptions are limited to 256 characters by Telegram.
COMMANDS: tuple[tuple[str, str], ...] = (
    ("start", "Welcome and a few things to try"),
    ("help", "What I can do, the commands and example prompts"),
    ("settings", "How I work: fast / balanced / deep, memory, replies, images, browser"),
    ("usage", "Your usage over the last 30 days"),
    ("reminders", "Your reminders and scheduled tasks"),
    ("profile", "What I remember about you"),
    ("timezone", "Set your time zone, e.g. /timezone Europe/Berlin"),
    ("voice", "Choose the voice of spoken replies"),
    ("mcp", "Connect MCP tool servers"),
    ("sandbox", "Your sandbox and browser: what runs, close them, forget logins"),
    ("cancel", "Stop the current task and clear the queue"),
)

WELCOME_AUTHORIZED = (
    "👋 Welcome to Generall.AI bot! Use me to get AI assistance.\n\n"
    "You can send me messages, voices, images, media groups, pdfs, and more to analyze. "
    "I have memory, access to the internet and a wide range of tools to help you."
)
WELCOME_UNAUTHORIZED = (
    "👋 Welcome! This bot requires an invitation to use.\n\n"
    "If you have an invite code, please use:\n<code>/invite YOUR_CODE_HERE</code>"
)
START_EXAMPLES = (
    "What happened in AI news this week? Give me the three most important items",
    "Send me a photo and ask: what is in it and where was it taken?",
    "Remind me tomorrow at 9 to call the dentist",
)
START_AUTHORIZED = (
    WELCOME_AUTHORIZED
    + "\n\n✍️ *Try:*\n"
    + "\n".join(f"• {example}" for example in START_EXAMPLES)
    + "\n\nSend /help for everything I can do and /settings to tune how I work."
)
HELP_UNAUTHORIZED = (
    "👋 This bot is invite-only.\n\n"
    "To get access, ask someone who already uses it for an invite: they create one with /invite "
    "and get a link to share with you. Open the link, or send the code it contains as\n"
    "`/invite YOUR_CODE`\n\n"
    "Once you are in, /help shows what I can do."
)
HELP_EXAMPLES = (
    "Research the latest news about solid-state batteries and summarize it with sources",
    "Reply to one of my answers with: translate this to Spanish",
    "Send a spreadsheet with the caption: make a chart of the monthly totals",
    "Write a Python script that renames the photos in my workspace by date, and run it",
    "Generate a logo for a bakery called Crumbs, minimal style, two variants",
    "Every weekday at 8, check the weather in Berlin and tell me if I need an umbrella",
)
HELP_ABOUT = (
    "🤖 *What I can do*\n"
    "• Chat by text or voice; when you speak, I answer with voice too\n"
    "• Look at photos, read documents (PDF, DOCX, XLSX, TXT, JSON) and watch videos\n"
    "• Research the web and read pages\n"
    "• Write and run code, work with the files in your workspace and send them back\n"
    "• Use websites in my own browser: click, type, fill forms and stay logged in; I ask you first "
    "before buying, paying, sending messages or deleting anything (/sandbox closes it or forgets logins)\n"
    "• Generate and edit images, make short videos\n"
    "• Set reminders and scheduled tasks that report back to you\n"
    "• Remember: conversation memory plus a profile of durable facts about you\n\n"
    "💡 *Tips*\n"
    "• Reply to a message, or quote a part of it, to tell me what you mean; forward messages for me to read\n"
    "• Press 🛑 Stop in the status message or send /cancel to stop a running task\n"
    "• Say \"remember that ...\" or \"forget ...\" to manage what I know about you"
)


def bot_commands() -> list[BotCommand]:
    """The command menu registered with ``set_my_commands`` at startup."""
    return [BotCommand(command=name, description=description) for name, description in COMMANDS]


def help_text(authorized: bool) -> str:
    """The ``/help`` message (legacy Markdown; ``answer_md`` falls back to plain text)."""
    if not authorized:
        return HELP_UNAUTHORIZED
    commands = "\n".join(f"/{name} — {description}" for name, description in COMMANDS if name != "start")
    examples = "\n".join(f"• {example}" for example in HELP_EXAMPLES)
    return f"{HELP_ABOUT}\n\n⌨️ *Commands*\n{commands}\n\n✍️ *Try*\n{examples}"


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
        await answer_md(message, START_AUTHORIZED)
    else:
        await message.answer(WELCOME_UNAUTHORIZED, parse_mode="HTML")


@public_router.message(F.text, Command("help"))
async def help_command(message: Message) -> None:
    """Public: someone without access learns how the invite flow works."""
    await answer_md(message, help_text(auth.is_authorized(str(message.chat.id))))


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


# ---------------------------------------------------------------------------
# /timezone
# ---------------------------------------------------------------------------
TZ_SET_PREFIX = "tz_set_"
TIMEZONE_HINT = (
    "Pick one below, send me your location, or type /timezone Europe/Berlin "
    "(any city name works too, e.g. /timezone Kyiv)."
)
TIMEZONE_UNKNOWN = (
    "❌ I don't know the time zone *{name}*.\n"
    "Use an IANA name such as Europe/Berlin or America/New\\_York, or a big city or country name."
)


def timezone_set_text(name: str) -> str:
    """Confirmation after the zone was saved (legacy Markdown), with the user's local time.

    The name sits inside a bold span, where legacy Markdown takes ``_`` literally (and forbids
    escaping), so ``America/New_York`` needs no escape here; ``timezone_overview`` shows it
    outside a span and escapes it."""
    zone = timezones.zone_named(name)
    local = timezones.format_local(timezones.now_local(None), zone, with_zone=False)
    return f"🕒 Time zone set to *{name}*.\nYour local time is {local}."


def timezone_overview(name: str) -> str:
    zone = timezones.zone_named(name)
    if zone is None:
        return f"🕒 *Your time zone* is not set: times are shown in UTC.\n\n{TIMEZONE_HINT}"
    local = timezones.format_local(timezones.now_local(None), zone, with_zone=False)
    return f"🕒 *Your time zone*: {escape_markdown(name)}\nLocal time: {local}\n\n{TIMEZONE_HINT}"


def timezone_keyboard(current: str | None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for i, name in enumerate(timezones.COMMON_ZONES):
        label = f"✓ {name}" if name == current else name
        button = InlineKeyboardButton(text=label, callback_data=f"{TZ_SET_PREFIX}{i}")
        if i % 2 == 0:
            rows.append([button])
        else:
            rows[-1].append(button)
    return InlineKeyboardMarkup(inline_keyboard=rows)


@voice_router.message(F.text, Command("timezone"))
async def timezone_command(message: Message, user_id: str, command: CommandObject) -> None:
    settings = UserSettings(user_id)
    args = (command.args or "").strip()
    if not args:
        current = str(settings.get("timezone", "name") or "")
        await answer_md(message, timezone_overview(current), reply_markup=timezone_keyboard(current or None))
        return
    name = timezones.valid_zone(args)
    if name is None:
        await answer_md(message, TIMEZONE_UNKNOWN.format(name=args.replace("*", "").replace("_", " ")))
        return
    settings.set("timezone", name, "name")
    await answer_md(message, timezone_set_text(name))


@voice_router.callback_query(F.data.startswith("tz_"))
async def timezone_button(callback: CallbackQuery, user_id: str) -> None:
    """``tz_set_<index>``: a picker button; the message becomes the confirmation."""
    try:
        await callback.answer()
    except Exception as e:  # noqa: BLE001 - the query may have expired
        logger.debug("Could not answer the timezone button: %s", e)
    if not isinstance(callback.message, Message):
        return
    data = callback.data or ""
    try:
        name = timezones.COMMON_ZONES[int(data[len(TZ_SET_PREFIX):])] if data.startswith(TZ_SET_PREFIX) else None
    except (ValueError, IndexError):
        name = None
    if name is None:
        logger.warning("Unknown timezone callback from %s: %r", user_id, data)
        return
    UserSettings(user_id).set("timezone", name, "name")
    await edit_md(callback.message, timezone_set_text(name), reply_markup=timezone_keyboard(name))
