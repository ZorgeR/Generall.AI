import asyncio
import importlib
import re
from datetime import datetime, timedelta, timezone

import pytest

from bot import stats_report as sr


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import stats as stats_module

    importlib.reload(stats_module)  # STATS_DB is relative: a fresh database in tmp_path
    return stats_module.stats_tracker


def _insert(tracker, user_id, event_type, subtype=None, *, days_ago=0.0, hour=None):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    if hour is not None:
        ts = ts.replace(hour=hour)
    with tracker._get_connection() as conn:
        conn.execute("INSERT INTO stats_events (user_id, event_type, event_subtype, timestamp) VALUES (?, ?, ?, ?)",
                     (user_id, event_type, subtype, ts.isoformat()))
        conn.commit()


def _usage(tracker, user_id, cost, *, days_ago=0.0, model="claude-sonnet-5-5"):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    with tracker._get_connection() as conn:
        conn.execute("INSERT INTO usage_events (user_id, timestamp, model, api_calls, input_tokens, output_tokens, "
                     "cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?, ?, ?, 1, 100, 10, 900, 0, ?)",
                     (user_id, ts.isoformat(), model, cost))
        conn.commit()


def _seed(tracker):
    for _ in range(3):
        _insert(tracker, "1", "message_received", "text")
    _insert(tracker, "1", "message_received", "voice", days_ago=2)
    _insert(tracker, "2", "message_received", "photo", days_ago=2)
    _insert(tracker, "2", "message_received", "document", days_ago=2)
    _insert(tracker, "1", "message_sent")
    _insert(tracker, "1", "tool_used", "read_file")
    _insert(tracker, "1", "tool_used", "a_rather_long_tool_name_for_the_table")
    _insert(tracker, "2", "message_received", "text", days_ago=40)  # previous period
    _insert(tracker, "3", "message_received", "text", days_ago=100)  # all time only
    _usage(tracker, "1", 2.5)
    _usage(tracker, "2", 0.5, days_ago=2, model="claude-haiku-4-5")
    _usage(tracker, "1", 1.0, days_ago=40)


# ---- helpers ----------------------------------------------------------------
def test_hbar_scales_in_eighths_and_never_hides_a_nonzero_value():
    assert sr.hbar(10, 10, 8) == "█" * 8
    assert sr.hbar(5, 10, 8) == "████"
    assert sr.hbar(1, 1000, 8) == "▏"
    assert sr.hbar(0, 10, 8) == "" and sr.hbar(3, 0, 8) == ""


def test_sparkline_keeps_idle_days_lowest():
    line = sr.sparkline([0, 1, 5, 10])
    assert line[0] == "▁" and line[1] != "▁" and line[-1] == "█" and len(line) == 4
    assert sr.sparkline([0, 0]) == "▁▁"


def test_delta_and_number_formats():
    assert sr.fmt_delta(112, 100) == "▲12%"
    assert sr.fmt_delta(97, 100) == "▼3%"
    assert sr.fmt_delta(100, 100) == "="
    assert sr.fmt_delta(5, 0) == "new" and sr.fmt_delta(0, 0) == ""
    assert sr.fmt_delta(1200, 100) == "▲12×"
    assert sr.fmt_compact(96_320_000) == "96.3M" and sr.fmt_compact(1_860_000) == "1.86M"
    assert sr.fmt_compact(261_200) == "261k" and sr.fmt_compact(4_400) == "4.4k" and sr.fmt_compact(999) == "999"
    assert sr.fmt_money(103.1) == "$103.10" and sr.fmt_money(0.004) == "<$0.01" and sr.fmt_money(12345) == "$12,345"
    assert sr.pct(1, 300) == "<1%" and sr.pct(150, 300) == "50%" and sr.pct(0, 300) == ""
    assert sr.meter(37, 50) == "███████░░░" and sr.meter(80, 50) == "██████████"


def test_table_aligns_and_code_block_neutralizes_backticks():
    assert sr.table([["a", "1"], ["bbb", "100"]], "lr") == ["a     1", "bbb 100"]
    assert sr.code_block(["x`y"]) == "```\nx'y\n```"
    assert sr.plain("Alice 🐱") == "Alice"


# ---- tracker queries --------------------------------------------------------
def test_window_totals_compare_periods(tracker):
    _seed(tracker)
    cur = tracker.get_window_totals(30)
    prev = tracker.get_window_totals(30, offset_days=30)
    everything = tracker.get_window_totals(None)
    assert (cur["received"], cur["sent"], cur["tools"], cur["active_users"]) == (6, 1, 2, 2)
    assert (prev["received"], prev["active_users"]) == (1, 1)
    assert everything["received"] == 8 and everything["active_users"] == 3
    assert abs(cur["cost_usd"] - 3.0) < 1e-9 and abs(prev["cost_usd"] - 1.0) < 1e-9
    assert tracker.get_window_totals(30, user_id="2")["received"] == 2
    assert tracker.get_window_totals(None, user_id="nobody")["last_seen"] is None


def test_daily_series_is_zero_filled_and_ends_today(tracker):
    _seed(tracker)
    days = tracker.get_daily_series(30)
    assert len(days) == 30
    assert days[-1]["date"] == datetime.now(timezone.utc).date().isoformat()
    assert days[-1]["received"] == {"text": 3} and days[-1]["received_total"] == 3
    assert days[-1]["sent"] == 1 and days[-1]["tools"] == 2 and days[-1]["active_users"] == 1
    assert abs(days[-1]["cost_usd"] - 2.5) < 1e-9 and days[-1]["prompt_tokens"] == 1000
    assert days[-3]["received"] == {"voice": 1, "photo": 1, "document": 1} and days[-3]["active_users"] == 2
    assert sum(d["received_total"] for d in days) == 6
    assert tracker.get_daily_series(30, user_id="2")[-3]["received_total"] == 2


def test_hourly_activity_buckets_by_utc_hour(tracker):
    _insert(tracker, "1", "message_received", "text", days_ago=1, hour=7)
    _insert(tracker, "1", "message_received", "text", days_ago=1, hour=7)
    _insert(tracker, "1", "message_sent", days_ago=1, hour=9)
    hours = tracker.get_hourly_activity(30)
    assert len(hours) == 24 and hours[7] == 2 and sum(hours) == 2


# ---- text views -------------------------------------------------------------
def _code_lines(text):
    return [line for block in re.findall(r"```\n(.*?)\n```", text, re.S) for line in block.split("\n")]


def test_render_main(tracker):
    _seed(tracker)
    report = sr.collect()
    assert report.user_ids() == ["1", "2"]
    text = sr.render_main(report, {"1": "@alice", "2": "Bob 🐱"})
    assert text.count("```") % 2 == 0 and len(text) < 4096
    assert all(len(line) <= sr.LINE_WIDTH for line in _code_lines(text)), _code_lines(text)
    assert "3 users · 2 active in 30 days · 1 today" in text
    assert re.search(r"Messages in +6 ▲500%", text)
    assert "a_rather_long_…" in text  # long tool names are trimmed
    assert "@alice" in text and "Bob " in text and "🐱" not in text  # emoji would break the columns
    assert "🗂 *All time* · 8 in" in text


def test_render_user_and_empty_database(tracker):
    _seed(tracker)
    text = sr.render_user(sr.collect("2"), "Bob_*", blocked=True, limit=None, used=37, default_limit=50)
    assert "👤 *Bob\\_\\**" in text and "🚫 *Blocked*" in text and "last seen 2d ago" in text
    assert "Quota `███████░░░` 37/50 (default)" in text
    assert "Active users" not in text and "Top spenders" not in text
    assert all(len(line) <= sr.LINE_WIDTH for line in _code_lines(text))
    assert "♾ unlimited" in sr.render_user(sr.collect("2"), "Bob", blocked=False, limit=0, used=3, default_limit=50)

    empty = sr.render_user(sr.collect("nobody"), "Nobody", blocked=False, limit=10, used=0, default_limit=50)
    assert "last seen never" in empty and "none in 30 days" in empty and "no usage recorded" in empty


# ---- chart --------------------------------------------------------------------
def test_dashboard_renders_png_with_and_without_data(tracker):
    pytest.importorskip("matplotlib")
    from bot import stats_chart

    empty = stats_chart.render_dashboard(sr.collect(), {}, title="All users")
    assert empty.startswith(b"\x89PNG")
    _seed(tracker)
    names = {"1": "$alice$ 🐱", "2": "Bob"}  # '$' must not switch matplotlib into mathtext
    assert stats_chart.render_dashboard(sr.collect(), names).startswith(b"\x89PNG")
    assert stats_chart.render_dashboard(sr.collect("1"), title="Алиса").startswith(b"\x89PNG")


# ---- handler ------------------------------------------------------------------
class _Chat:
    def __init__(self, username=None, first_name=None, last_name=None, title=None):
        self.username, self.first_name, self.last_name, self.title = username, first_name, last_name, title


class _Bot:
    def __init__(self):
        self.calls = 0

    async def get_chat(self, chat_id):
        self.calls += 1
        if chat_id == 1:
            return _Chat(username="alice")
        if chat_id == 2:
            return _Chat(first_name="Bob", last_name="B")
        raise RuntimeError("chat not found")


def test_display_names_are_cached_and_fall_back_to_the_id(tracker):
    from bot.handlers import stats_ui

    stats_ui._names.clear()
    bot = _Bot()
    names = asyncio.run(stats_ui.display_names(bot, ["1", "2", "3"]))
    assert names == {"1": "@alice", "2": "Bob B", "3": "3"}
    asyncio.run(stats_ui.display_names(bot, ["1", "2", "3"]))
    assert bot.calls == 4  # 1 and 2 cached; the failed lookup is retried
