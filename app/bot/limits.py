"""Per-user action quota (rolling 30 days), backed by stats.db."""
from __future__ import annotations

from bot.auth import auth

LIMIT_WINDOW_DAYS = 30


def check_user_limits(user_id: str) -> tuple[bool, int, int | None]:
    """Return (allowed, used, limit). ``limit`` is None when unlimited (admin, 0 or unset)."""
    from stats import stats_tracker  # lazy: importing stats creates data/stats.db

    user_id = str(user_id)
    if auth.is_admin(user_id):
        return True, 0, None
    stats_tracker.ensure_user_limit(user_id)
    limit = stats_tracker.get_user_limit(user_id)
    used = stats_tracker.get_user_action_count(user_id, days=LIMIT_WINDOW_DAYS)
    if not limit:
        return True, used, None
    return used < limit, used, limit


def usage_line(user_id: str, limit: int | None) -> str:
    """Markdown v1 usage line for status messages, or '' when unlimited."""
    if not limit:
        return ""
    from stats import stats_tracker

    used = stats_tracker.get_user_action_count(str(user_id), days=LIMIT_WINDOW_DAYS)
    return f"📊 *Usage:* _{used}/{limit} actions (30d)_\n"
