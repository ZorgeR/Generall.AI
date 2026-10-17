"""``/usage``: the user's own usage (docs/design-v3.md, F4). Router on ``ui`` (auth, no quota).

One legacy-Markdown message (``bot.ui.answer_md``, plain fallback): a header, a monospace block
whose lines stay within ``stats_report.LINE_WIDTH`` so a phone does not wrap them, and a hint that
the Fast preset lowers cost. "🔄 Refresh" (``usage_refresh``) collects again and edits the message
in place. The admin sees their own numbers like everyone else.

The numbers: ``StatsTracker.get_user_usage_summary`` (requests, answers, tool calls, media
described, tokens, cost) and ``get_user_workspace_size`` (a directory walk), both blocking and run
in worker threads; ``bot.limits.check_user_limits`` for the rolling 30-day quota (SQLite too);
``reminders_store`` for the active reminders. ``stats`` is imported lazily, like everywhere in
``bot/``, so importing this module does not create ``data/stats.db``. Nothing user-written is
shown, so the only escaping needed is the code block's own (``stats_report.code_block``).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import stats_report as sr
from bot.config import config
from bot.limits import LIMIT_WINDOW_DAYS, check_user_limits
from bot.ui import answer_md, edit_md
from reminders_store import reminders_store

logger = logging.getLogger(__name__)

router = Router(name="usage")

DAYS = LIMIT_WINDOW_DAYS  # the quota window; the usage is shown for the same period
REFRESH = "usage_refresh"
ACTIVE_STATUSES = ("pending", "processing")  # as /reminders counts them
HINT = "💡 _/settings → Fast preset lowers cost_"
ERROR_TEXT = "❌ Could not load your usage right now. Try again in a moment."


@dataclass
class UsageView:
    summary: dict  # StatsTracker.get_user_usage_summary
    used: int  # actions in the quota window (check_user_limits)
    limit: int | None  # None = unlimited
    active_reminders: int
    workspace: dict  # {"files", "bytes"}
    generated_at: datetime


def _tracker():
    from stats import stats_tracker  # lazy: importing stats creates data/stats.db

    return stats_tracker


async def collect(user_id: str) -> UsageView:
    """Gather one view; the blocking parts run in worker threads."""
    user_id = str(user_id)
    tracker = _tracker()
    summary = await asyncio.to_thread(tracker.get_user_usage_summary, user_id, DAYS)
    _, used, limit = await asyncio.to_thread(check_user_limits, user_id)
    workspace = await asyncio.to_thread(tracker.get_user_workspace_size, user_id, config.data_dir)
    reminders = await reminders_store.load(user_id)
    active = sum(1 for r in reminders if r.get("status") in ACTIVE_STATUSES and r.get("enabled", True))
    return UsageView(summary, used, limit, active, workspace, datetime.now(timezone.utc))


# ---- formatting ---------------------------------------------------------------
def fmt_cost(x: float) -> str:
    """$12.34 from a dollar up, four decimals below it (a single turn is often fractions of a cent)."""
    x = float(x or 0)
    if x <= 0:
        return "$0.00"
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


def fmt_size(n_bytes: int) -> str:
    mb = int(n_bytes or 0) / (1024 * 1024)
    if mb >= 1000:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.1f} MB"


def quota_line(view: UsageView) -> str:
    if not view.limit:
        return f"Actions {sr.fmt_int(view.summary['actions'])} · unlimited"
    return f"Actions {sr.meter(view.used, view.limit)} {sr.fmt_int(view.used)}/{sr.fmt_int(view.limit)}"


def render(view: UsageView) -> str:
    """Legacy Markdown: header, one code block of ``LINE_WIDTH`` lines, the hint."""
    s = view.summary
    groups: list[list[list[str]]] = [[
        ["Requests", sr.fmt_int(s["requests"])],
        ["Answers", sr.fmt_int(s["answers"])],
        ["Tool calls", sr.fmt_int(s["tool_calls"])],
        ["Media described", sr.fmt_int(s["media_described"])],
    ]]
    if s["api_calls"]:
        prompt = s["input_tokens"] + s["cache_read_tokens"] + s["cache_write_tokens"]
        tokens = [
            ["Tokens in", sr.fmt_compact(prompt)],
            ["Cached", sr.pct(s["cache_read_tokens"], prompt) or "0%"],
            ["Tokens out", sr.fmt_compact(s["output_tokens"])],
            ["Est. cost", fmt_cost(s["cost_usd"])],
        ]
        daily = [d["cost_usd"] for d in s.get("daily", [])]
        if any(daily):
            tokens.append([f"Cost, {len(daily)} days", sr.sparkline(daily)])
        groups.append(tokens)
    else:
        groups.append([["Tokens", "none yet"]])
    groups.append([
        ["Reminders", sr.fmt_int(view.active_reminders)],
        ["Files", sr.fmt_int(view.workspace["files"])],
        ["Disk", fmt_size(view.workspace["bytes"])],
    ])
    # One table per group keeps the value column aligned across the whole block.
    rows = [row for group in groups for row in group + [["", ""]]][:-1]
    lines = [quota_line(view), ""] + sr.table(rows, "lr")
    lines = [sr.trim(line, sr.LINE_WIDTH) for line in lines]
    header = f"📊 *Your usage* · last {s['days']} days · {view.generated_at:%H:%M} UTC"
    return "\n".join([header, sr.code_block(lines), HINT])


def keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔄 Refresh", callback_data=REFRESH)]])


async def usage_text(user_id: str) -> str:
    try:
        return render(await collect(user_id))
    except Exception:  # noqa: BLE001 - the Refresh button stays, so this is not a dead end
        logger.exception("Collecting the usage of %s failed", user_id)
        return ERROR_TEXT


# ---- handlers -----------------------------------------------------------------
@router.message(F.text, Command("usage"))
async def usage_command(message: Message, user_id: str) -> None:
    await answer_md(message, await usage_text(user_id), keyboard())


@router.callback_query(F.data.startswith("usage_"))
async def usage_button(callback: CallbackQuery, user_id: str) -> None:
    """🔄 Refresh: the same view again, edited into the message the button is on."""
    try:
        await callback.answer()
    except Exception as e:  # noqa: BLE001 - an old query; the refresh still happens
        logger.debug("Could not answer the usage button: %s", e)
    if not isinstance(callback.message, Message):
        return
    if callback.data != REFRESH:
        logger.warning("Unknown usage callback from %s: %r", user_id, callback.data)
        return
    await edit_md(callback.message, await usage_text(user_id), keyboard())
