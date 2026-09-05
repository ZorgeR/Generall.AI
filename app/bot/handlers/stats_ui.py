"""Admin ``/stats`` command and the ``stats_*`` inline keyboards.

Ported from the python-telegram-bot implementation in ``main_bot.py``. The
router is admin-only: attach ``AuthMiddleware(require_admin=True)`` to it, the
handlers themselves do no authorization checks.

Callback data (checked most-specific first, as in the original):

    stats_users_page_<n>      paginated user list (10 per page)
    stats_limit_<uid>_<n>     set <uid>'s 30-day action limit to <n> (0 = unlimited)
    stats_setlimit_<uid>      limit preset menu for <uid>
    stats_block_<uid>         block <uid>
    stats_unblock_<uid>       unblock <uid>
    stats_user_<uid>          per-user stats
    stats_back_main           aggregated stats
"""
from __future__ import annotations

import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.auth import auth
from bot.ui import answer_md, edit_md, escape_markdown
from stats import DEFAULT_ACTION_LIMIT, stats_tracker

logger = logging.getLogger(__name__)

router = Router(name="stats")

USERS_PER_PAGE = 10
LIMIT_PRESETS = [50, 100, 250, 500, 750, 1000]


async def get_telegram_user_display_name(bot: Bot, user_id: str) -> str:
    """Fetch user's display name from Telegram API"""
    try:
        chat = await bot.get_chat(int(user_id))
        if chat.username:
            return f"@{chat.username}"
        elif chat.first_name:
            name = chat.first_name
            if chat.last_name:
                name += f" {chat.last_name}"
            return name
    except Exception:  # noqa: BLE001
        pass
    return user_id  # Fallback to user_id if fetch fails


def format_stats_text(stats: dict, title: str = "") -> str:
    """Format stats dictionary into a readable text format"""
    text = ""
    if title:
        text += f"📊 *{escape_markdown(title)}*\n\n"

    # Messages Received
    msg_recv = stats.get("messages_received", {})
    msg_total = msg_recv.get("total", 0)
    text += f"├─ Messages Received: *{msg_total:,}* total\n"
    for msg_type in ["text", "voice", "video", "photo", "audio", "document"]:
        count = msg_recv.get(msg_type, 0)
        if count > 0:
            text += f"│  ├─ {msg_type}: {count:,}\n"

    # Messages Sent
    msg_sent = stats.get("messages_sent", 0)
    text += f"├─ Messages Sent: *{msg_sent:,}*\n"

    # Tools Used
    tools_total = stats.get("tools_total", 0)
    tools_used = stats.get("tools_used", {})
    text += f"├─ Tools Used: *{tools_total:,}* total\n"
    # Sort tools by count and show top ones
    sorted_tools = sorted(tools_used.items(), key=lambda x: x[1], reverse=True)[:10]
    for tool_name, count in sorted_tools:
        text += f"│  ├─ {escape_markdown(tool_name)}: {count:,}\n"

    # Describe Used
    describe_total = stats.get("describe_total", 0)
    describe_used = stats.get("describe_used", {})
    text += f"├─ Describe Used: *{describe_total:,}* total\n"
    sorted_describe = sorted(describe_used.items(), key=lambda x: x[1], reverse=True)
    for desc_type, count in sorted_describe:
        text += f"│  ├─ {escape_markdown(desc_type)}: {count:,}\n"

    # Media Groups
    media_groups = stats.get("media_groups_processed", 0)
    text += f"└─ Media Groups: *{media_groups:,}*\n"

    return text


def _fmt_tokens(n: int) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def format_usage_text(usage: dict) -> str:
    """Token accounting block (legacy Markdown) for the main and per-user stats views."""
    if not usage or not usage.get("api_calls"):
        return "🧮 Tokens: _no usage recorded yet_\n"
    prompt = usage["input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"]
    cached = int(round(100 * usage["cache_read_tokens"] / prompt)) if prompt else 0
    text = (
        f"🧮 Tokens: in *{_fmt_tokens(prompt)}* ({cached}% cached) · out *{_fmt_tokens(usage['output_tokens'])}*\n"
        f"│  ├─ API calls: {usage['api_calls']:,} · tool calls: {usage['tool_calls']:,}\n"
        f"│  └─ Estimated cost: *${usage['cost_usd']:.2f}*\n"
    )
    for model, u in list(usage.get("models", {}).items())[:4]:
        text += f"│     {escape_markdown(model)}: in {_fmt_tokens(u['input_tokens'] + u['cache_read_tokens'] + u['cache_write_tokens'])} · out {_fmt_tokens(u['output_tokens'])} · ${u['cost_usd']:.2f}\n"
    return text


def _build_main_stats() -> tuple[str, InlineKeyboardMarkup]:
    """Text and keyboard of the aggregated stats view (shared by /stats and stats_back_main)."""
    stats_30d = stats_tracker.get_aggregated_stats(days=30)
    stats_all = stats_tracker.get_aggregated_stats(days=None)

    text = "📊 *Usage Statistics (All Users)*\n\n"
    text += f"Total Users: *{stats_all.get('total_users', 0):,}*\n\n"

    text += "📅 *Last 30 Days:*\n"
    text += format_stats_text(stats_30d)
    text += format_usage_text(stats_tracker.get_usage(days=30))

    text += "\n📈 *All Time:*\n"
    text += format_stats_text(stats_all)
    text += format_usage_text(stats_tracker.get_usage(days=None))

    top = stats_tracker.get_users_ranked_by_cost(days=30, limit=5)
    if top:
        text += "\n💸 *Top spenders (30d):* " + ", ".join(f"`{uid}` ${cost:.2f}" for uid, cost in top) + "\n"

    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="👥 View Users", callback_data="stats_users_page_1")]]
    )
    return text, reply_markup


@router.message(F.text, Command("stats"))
async def stats_command(message: Message) -> None:
    """Handle the /stats command - admin only (guarded by the router's middleware)"""
    text, reply_markup = _build_main_stats()
    await answer_md(message, text, reply_markup)


@router.callback_query(F.data.startswith("stats_"))
async def stats_button(callback: CallbackQuery) -> None:
    """Handle stats menu button presses"""
    await callback.answer()

    message = callback.message
    if not isinstance(message, Message):
        return
    bot = callback.bot
    if bot is None:
        logger.warning("stats callback without a bound Bot instance; ignoring")
        return

    data = callback.data or ""

    if data.startswith("stats_users_page_"):
        page = int(data.replace("stats_users_page_", ""))
        await show_users_stats_page(message, bot, page)

    elif data.startswith("stats_limit_"):
        # stats_limit_{user_id}_{value} - set limit for user
        parts = data.replace("stats_limit_", "").rsplit("_", 1)
        target_user_id = parts[0]
        limit_value = int(parts[1])
        stats_tracker.set_user_limit(target_user_id, limit_value)
        logger.info("Admin set action limit for %s to %s", target_user_id, limit_value)
        await show_user_stats(message, bot, target_user_id)

    elif data.startswith("stats_setlimit_"):
        target_user_id = data.replace("stats_setlimit_", "")
        await show_set_limit(message, bot, target_user_id)

    elif data.startswith("stats_block_"):
        target_user_id = data.replace("stats_block_", "")
        auth.block(target_user_id)
        logger.info("Admin blocked user %s", target_user_id)
        await show_user_stats(message, bot, target_user_id)

    elif data.startswith("stats_unblock_"):
        target_user_id = data.replace("stats_unblock_", "")
        auth.unblock(target_user_id)
        logger.info("Admin unblocked user %s", target_user_id)
        await show_user_stats(message, bot, target_user_id)

    elif data.startswith("stats_user_"):
        target_user_id = data.replace("stats_user_", "")
        await show_user_stats(message, bot, target_user_id)

    elif data == "stats_back_main":
        await show_main_stats(message)


async def show_main_stats(message: Message) -> None:
    """Show main aggregated stats view"""
    text, reply_markup = _build_main_stats()
    await edit_md(message, text, reply_markup)


async def show_users_stats_page(message: Message, bot: Bot, page: int) -> None:
    """Show paginated list of users sorted by 30-day activity"""
    ranked_users = stats_tracker.get_users_ranked_by_activity(days=30)

    if not ranked_users:
        reply_markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="⬅️ Back to Stats", callback_data="stats_back_main")]]
        )
        await edit_md(message, "No user activity recorded yet.", reply_markup)
        return

    # Pagination
    total_pages = (len(ranked_users) + USERS_PER_PAGE - 1) // USERS_PER_PAGE
    page = max(1, min(page, total_pages))  # Clamp page number

    start_idx = (page - 1) * USERS_PER_PAGE
    end_idx = start_idx + USERS_PER_PAGE
    current_users = ranked_users[start_idx:end_idx]

    # Build the text with user names
    text = "👥 *Top Users (by 30-day activity)*\n\n"

    user_buttons: list[InlineKeyboardButton] = []
    for rank, (uid, activity) in enumerate(current_users, start=start_idx + 1):
        display_name = await get_telegram_user_display_name(bot, uid)
        text += f"{rank}. {escape_markdown(display_name)} - {activity:,} actions\n"

        button_name = display_name[:15] + "..." if len(display_name) > 18 else display_name
        user_buttons.append(InlineKeyboardButton(text=button_name, callback_data=f"stats_user_{uid}"))

    text += f"\nPage {page}/{total_pages}"

    # Create keyboard with user buttons (3 per row)
    keyboard: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(user_buttons), 3):
        keyboard.append(user_buttons[i:i + 3])

    # Add navigation buttons
    nav_row: list[InlineKeyboardButton] = []
    if page > 1:
        nav_row.append(InlineKeyboardButton(text="⬅️ Prev", callback_data=f"stats_users_page_{page - 1}"))
    nav_row.append(InlineKeyboardButton(text="📊 Back to Stats", callback_data="stats_back_main"))
    if page < total_pages:
        nav_row.append(InlineKeyboardButton(text="Next ➡️", callback_data=f"stats_users_page_{page + 1}"))
    keyboard.append(nav_row)

    await edit_md(message, text, InlineKeyboardMarkup(inline_keyboard=keyboard))


async def show_user_stats(message: Message, bot: Bot, target_user_id: str) -> None:
    """Show stats for a specific user"""
    display_name = await get_telegram_user_display_name(bot, target_user_id)

    stats_30d = stats_tracker.get_user_stats(target_user_id, days=30)
    stats_all = stats_tracker.get_user_stats(target_user_id, days=None)

    # Get block/limit status
    is_blocked = target_user_id in auth.blocked
    user_limit = stats_tracker.get_user_limit(target_user_id)
    user_action_count = stats_tracker.get_user_action_count(target_user_id, days=30)

    # Format the message
    text = f"📊 *User Stats: {escape_markdown(display_name)}*\n"
    text += f"ID: `{target_user_id}`\n"

    if is_blocked:
        text += "Status: 🚫 *BLOCKED*\n"
    else:
        text += "Status: ✅ Active\n"

    if user_limit is not None and user_limit > 0:
        text += f"Limit: *{user_action_count:,}/{user_limit:,}* actions (30d)\n"
    elif user_limit == 0:
        text += "Limit: ♾ *Unlimited*\n"
    else:
        text += f"Limit: *{user_action_count:,}/{DEFAULT_ACTION_LIMIT:,}* actions (30d, default)\n"

    text += "\n📅 *Last 30 Days:*\n"
    text += format_stats_text(stats_30d)
    text += format_usage_text(stats_tracker.get_usage(user_id=target_user_id, days=30))

    text += "\n📈 *All Time:*\n"
    text += format_stats_text(stats_all)
    text += format_usage_text(stats_tracker.get_usage(user_id=target_user_id, days=None))

    # Build keyboard with block/unblock and set limit buttons
    block_btn = InlineKeyboardButton(
        text="✅ Unblock User" if is_blocked else "🚫 Block User",
        callback_data=f"stats_unblock_{target_user_id}" if is_blocked else f"stats_block_{target_user_id}",
    )
    limit_btn = InlineKeyboardButton(text="⚙️ Set Limit", callback_data=f"stats_setlimit_{target_user_id}")

    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [block_btn, limit_btn],
            [InlineKeyboardButton(text="⬅️ Back to Users", callback_data="stats_users_page_1")],
        ]
    )
    await edit_md(message, text, reply_markup)


async def show_set_limit(message: Message, bot: Bot, target_user_id: str) -> None:
    """Show limit preset buttons for a user"""
    display_name = await get_telegram_user_display_name(bot, target_user_id)
    current_limit = stats_tracker.get_user_limit(target_user_id)
    used = stats_tracker.get_user_action_count(target_user_id, days=30)

    if current_limit is not None and current_limit > 0:
        limit_text = f"{current_limit:,}"
    elif current_limit == 0:
        limit_text = "Unlimited"
    else:
        limit_text = f"{DEFAULT_ACTION_LIMIT} (default)"

    text = f"⚙️ *Set Action Limit: {escape_markdown(display_name)}*\n\n"
    text += f"Current limit: *{limit_text}*\n"
    text += f"Used (30d): *{used:,}* actions\n\n"
    text += "Select new monthly limit:"

    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=str(p), callback_data=f"stats_limit_{target_user_id}_{p}")
                for p in LIMIT_PRESETS[:3]
            ],
            [
                InlineKeyboardButton(text=str(p), callback_data=f"stats_limit_{target_user_id}_{p}")
                for p in LIMIT_PRESETS[3:]
            ],
            [InlineKeyboardButton(text="♾ Unlimited", callback_data=f"stats_limit_{target_user_id}_0")],
            [InlineKeyboardButton(text="⬅️ Back", callback_data=f"stats_user_{target_user_id}")],
        ]
    )
    await edit_md(message, text, reply_markup)
