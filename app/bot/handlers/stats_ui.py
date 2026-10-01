"""Admin ``/stats`` command and the ``stats_*`` inline keyboards.

Ported from the python-telegram-bot implementation in ``main_bot.py``. The
router is admin-only: attach ``AuthMiddleware(require_admin=True)`` to it, the
handlers themselves do no authorization checks.

Each view is text (``bot.stats_report``: bars, a daily sparkline, deltas against the
previous 30 days) plus, on ``/stats`` and the 📈 button, a PNG dashboard
(``bot.stats_chart``). Both read one ``StatsReport``, collected in a worker thread.

Callback data (checked most-specific first, as in the original):

    stats_chart_all           dashboard image for all users
    stats_chart_<uid>         dashboard image for one user
    stats_users_page_<n>      paginated user list (10 per page)
    stats_limit_<uid>_<n>     set <uid>'s 30-day action limit to <n> (0 = unlimited)
    stats_setlimit_<uid>      limit preset menu for <uid>
    stats_block_<uid>         block <uid>
    stats_unblock_<uid>       unblock <uid>
    stats_user_<uid>          per-user stats
    stats_back_main           aggregated stats
"""
from __future__ import annotations

import asyncio
import logging
import time

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import stats_chart, stats_report
from bot.auth import auth
from bot.ui import answer_md, edit_md, escape_markdown
from stats import DEFAULT_ACTION_LIMIT, stats_tracker

logger = logging.getLogger(__name__)

router = Router(name="stats")

USERS_PER_PAGE = 10
LIMIT_PRESETS = [50, 100, 250, 500, 750, 1000]
NAME_TTL_SECONDS = 3600  # display names are cached: every view shows several

_names: dict[str, tuple[float, str]] = {}


async def get_telegram_user_display_name(bot: Bot, user_id: str) -> str:
    """Fetch user's display name from Telegram API (cached for an hour; the id when unknown)"""
    cached = _names.get(user_id)
    if cached and time.monotonic() - cached[0] < NAME_TTL_SECONDS:
        return cached[1]
    try:
        chat = await bot.get_chat(int(user_id))
    except Exception:  # noqa: BLE001
        return user_id  # not cached: a transient failure should not stick for an hour
    if chat.username:
        name = f"@{chat.username}"
    elif chat.first_name:
        name = f"{chat.first_name} {chat.last_name}" if chat.last_name else chat.first_name
    else:
        name = chat.title or user_id
    _names[user_id] = (time.monotonic(), name)
    return name


async def display_names(bot: Bot, user_ids: list[str]) -> dict[str, str]:
    names = await asyncio.gather(*(get_telegram_user_display_name(bot, uid) for uid in user_ids))
    return dict(zip(user_ids, names))


async def send_chart(bot: Bot, message: Message, report: stats_report.StatsReport, names: dict[str, str],
                     title: str) -> None:
    """Render the dashboard in a worker thread and send it as a photo; failures are logged, never raised."""
    try:
        await bot.send_chat_action(message.chat.id, "upload_photo", message_thread_id=message.message_thread_id)
    except Exception:  # noqa: BLE001 - cosmetic
        pass
    try:
        png = await asyncio.to_thread(stats_chart.render_dashboard, report, names, title=title)
    except Exception:  # noqa: BLE001
        logger.exception("Rendering the stats dashboard failed")
        return
    caption = f"📊 {title} · last {report.days} days · {report.generated_at:%Y-%m-%d %H:%M} UTC"
    try:
        await message.answer_photo(BufferedInputFile(png, filename="stats.png"), caption=caption)
    except TelegramAPIError as e:
        logger.warning("Could not send the stats dashboard: %s", e)


async def _main_view(bot: Bot) -> tuple[stats_report.StatsReport, dict[str, str], str, InlineKeyboardMarkup]:
    """Report, names, text and keyboard of the aggregated view (shared by /stats and stats_back_main)."""
    report = await asyncio.to_thread(stats_report.collect)
    names = await display_names(bot, report.user_ids())
    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="👥 View Users", callback_data="stats_users_page_1"),
            InlineKeyboardButton(text="📈 Chart", callback_data="stats_chart_all"),
        ]]
    )
    return report, names, stats_report.render_main(report, names), reply_markup


@router.message(F.text, Command("stats"))
async def stats_command(message: Message, bot: Bot) -> None:
    """Handle the /stats command - admin only (guarded by the router's middleware)"""
    report, names, text, reply_markup = await _main_view(bot)
    await send_chart(bot, message, report, names, "All users")
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

    if data.startswith("stats_chart_"):
        target = data.replace("stats_chart_", "")
        if target == "all":
            report, names, _, _ = await _main_view(bot)
            await send_chart(bot, message, report, names, "All users")
        else:
            report = await asyncio.to_thread(stats_report.collect, target)
            await send_chart(bot, message, report, {}, await get_telegram_user_display_name(bot, target))

    elif data.startswith("stats_users_page_"):
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
        await show_main_stats(message, bot)


async def show_main_stats(message: Message, bot: Bot) -> None:
    """Show main aggregated stats view"""
    _, _, text, reply_markup = await _main_view(bot)
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

    names = await display_names(bot, [uid for uid, _ in current_users])
    peak = ranked_users[0][1]
    rows = [
        [str(rank), stats_report.trim(stats_report.plain(names[uid]), 12), stats_report.hbar(activity, peak, 8),
         f"{activity:,}"]
        for rank, (uid, activity) in enumerate(current_users, start=start_idx + 1)
    ]
    text = "👥 *Users by 30-day activity* (actions)\n"
    text += stats_report.code_block(stats_report.table(rows, "rllr"))
    text += f"\nPage {page}/{total_pages}"

    user_buttons: list[InlineKeyboardButton] = []
    for uid, _ in current_users:
        display_name = names[uid]
        button_name = display_name[:15] + "..." if len(display_name) > 18 else display_name
        user_buttons.append(InlineKeyboardButton(text=button_name, callback_data=f"stats_user_{uid}"))

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
    report = await asyncio.to_thread(stats_report.collect, target_user_id)
    is_blocked = target_user_id in auth.blocked
    text = stats_report.render_user(
        report,
        display_name,
        blocked=is_blocked,
        limit=stats_tracker.get_user_limit(target_user_id),
        used=stats_tracker.get_user_action_count(target_user_id, days=30),
        default_limit=DEFAULT_ACTION_LIMIT,
    )

    # Build keyboard with block/unblock and set limit buttons
    block_btn = InlineKeyboardButton(
        text="✅ Unblock User" if is_blocked else "🚫 Block User",
        callback_data=f"stats_unblock_{target_user_id}" if is_blocked else f"stats_block_{target_user_id}",
    )
    limit_btn = InlineKeyboardButton(text="⚙️ Set Limit", callback_data=f"stats_setlimit_{target_user_id}")

    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [block_btn, limit_btn],
            [
                InlineKeyboardButton(text="📈 Chart", callback_data=f"stats_chart_{target_user_id}"),
                InlineKeyboardButton(text="⬅️ Back to Users", callback_data="stats_users_page_1"),
            ],
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
