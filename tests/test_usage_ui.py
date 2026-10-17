"""/usage: StatsTracker.get_user_usage_summary / get_user_workspace_size and bot/handlers/usage_ui.py."""
import importlib
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import Chat, Message

from bot import stats_report as sr
from bot.handlers import usage_ui
from reminders_store import RemindersStore

USER = "42"


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import stats as stats_module

    importlib.reload(stats_module)  # STATS_DB is relative: a fresh database in tmp_path
    return stats_module.stats_tracker


def _event(tracker, user_id, event_type, subtype=None, *, days_ago=0.0):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    with tracker._get_connection() as conn:
        conn.execute("INSERT INTO stats_events (user_id, event_type, event_subtype, timestamp) VALUES (?, ?, ?, ?)",
                     (user_id, event_type, subtype, ts.isoformat()))
        conn.commit()


def _usage(tracker, user_id, *, days_ago=0.0, model="claude-sonnet-5-5", api_calls=1, inp=100, out=10,
           cache_read=900, cache_write=0, cost=0.01):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    with tracker._get_connection() as conn:
        conn.execute("INSERT INTO usage_events (user_id, timestamp, model, api_calls, input_tokens, output_tokens, "
                     "cache_read_tokens, cache_write_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (user_id, ts.isoformat(), model, api_calls, inp, out, cache_read, cache_write, cost))
        conn.commit()


# ---- stats.py -----------------------------------------------------------------
def test_user_usage_summary_counts_events_tokens_models_and_days(tracker):
    for _ in range(3):
        _event(tracker, USER, "message_received", "text")
    _event(tracker, USER, "message_received", "photo", days_ago=29)
    _event(tracker, USER, "message_sent")
    _event(tracker, USER, "message_sent", days_ago=10)
    for tool in ("read_file", "search_web", "execute_python", "search_web"):
        _event(tracker, USER, "tool_used", tool)
    _event(tracker, USER, "describe_used", "image_anthropic")
    _event(tracker, USER, "media_group")
    _event(tracker, USER, "message_received", "text", days_ago=31)  # outside the window
    _event(tracker, "7", "message_received", "text")  # someone else
    _usage(tracker, USER, api_calls=3, inp=1000, out=200, cache_read=9000, cache_write=500, cost=0.05)
    _usage(tracker, USER, days_ago=2, model="claude-haiku-5-5", inp=300, out=50, cache_read=0, cost=0.001)
    _usage(tracker, USER, days_ago=40, cost=5.0)  # outside the window
    _usage(tracker, "7", cost=9.0)

    s = tracker.get_user_usage_summary(USER, days=30)
    assert s["days"] == 30 and s["user_id"] == USER
    assert s["requests"] == 4 and s["answers"] == 2 and s["tool_calls"] == 4
    assert s["media_described"] == 1 and s["albums"] == 1
    assert s["actions"] == 4 + 2 + 4 + 1 + 1  # every row counts against the limit
    assert s["api_calls"] == 4 and s["input_tokens"] == 1300 and s["output_tokens"] == 250
    assert s["cache_read_tokens"] == 9000 and s["cache_write_tokens"] == 500
    assert abs(s["cost_usd"] - 0.051) < 1e-9
    assert [m["model"] for m in s["models"]] == ["claude-sonnet-5-5", "claude-haiku-5-5"]  # costliest first
    sonnet = s["models"][0]
    assert sonnet["api_calls"] == 3 and sonnet["tokens"] == 1000 + 200 + 9000 + 500 and abs(sonnet["cost_usd"] - 0.05) < 1e-9
    assert len(s["daily"]) == 7 and s["daily"][-1]["date"] == datetime.now(timezone.utc).date().isoformat()
    assert abs(s["daily"][-1]["cost_usd"] - 0.05) < 1e-9 and s["daily"][-1]["prompt_tokens"] == 10500
    assert abs(sum(d["cost_usd"] for d in s["daily"]) - 0.051) < 1e-9
    assert all(d["cost_usd"] == 0 for d in s["daily"][:-3])


def test_user_usage_summary_of_an_unknown_user_is_all_zeros(tracker):
    s = tracker.get_user_usage_summary("nobody")
    assert s["actions"] == 0 and s["requests"] == 0 and s["api_calls"] == 0 and s["cost_usd"] == 0.0
    assert s["models"] == [] and len(s["daily"]) == 7 and all(d["cost_usd"] == 0 for d in s["daily"])


def test_workspace_size_counts_every_file_and_never_follows_links(tracker, tmp_path):
    root = tmp_path / "data" / USER
    (root / "images").mkdir(parents=True)
    (root / "images" / "a.jpg").write_bytes(b"x" * 1000)
    (root / ".hidden").write_bytes(b"y" * 24)  # hidden and temporary files count too
    (root / "temp_code_ab12.py").write_text("print(1)")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 10_000)
    os.symlink(outside, root / "link.bin")  # counted by its own size, the target is not read
    os.symlink(tmp_path, root / "loop")  # a directory link out of the workspace is not followed
    size = tracker.get_user_workspace_size(USER, base_dir=str(tmp_path / "data"))
    assert size["files"] == 4
    assert 1000 + 24 + 8 <= size["bytes"] < 1000 + 24 + 8 + 10_000
    assert tracker.get_user_workspace_size("nobody", base_dir=str(tmp_path / "data")) == {"files": 0, "bytes": 0}


# ---- rendering ----------------------------------------------------------------
def _summary(**over):
    base = {
        "user_id": USER, "days": 30, "actions": 37, "requests": 12, "answers": 12, "tool_calls": 10,
        "media_described": 3, "albums": 0, "api_calls": 40, "input_tokens": 120_000, "output_tokens": 45_000,
        "cache_read_tokens": 880_000, "cache_write_tokens": 0, "cost_usd": 1.2345,
        "models": [], "daily": [{"date": "2026-10-1%d" % i, "cost_usd": c, "prompt_tokens": 0, "output_tokens": 0}
                                for i, c in enumerate([0, 0.1, 0.3, 0, 0.5, 0.2, 0.1])],
    }
    base.update(over)
    return base


def _view(summary=None, *, used=37, limit=50, reminders=2, files=14, size=3_355_443):
    return usage_ui.UsageView(summary or _summary(), used, limit, reminders, {"files": files, "bytes": size},
                              datetime(2026, 10, 17, 14, 5, tzinfo=timezone.utc))


def _block_lines(text):
    assert text.count("```") == 2
    return text.split("```")[1].strip("\n").split("\n")


def test_render_shows_every_number_in_lines_that_fit_a_phone():
    text = usage_ui.render(_view())
    assert text.startswith("📊 *Your usage* · last 30 days · 14:05 UTC")
    assert text.endswith(usage_ui.HINT)
    lines = _block_lines(text)
    assert lines[0] == f"Actions {sr.meter(37, 50)} 37/50"
    block = "\n".join(lines)
    for expected in ("Requests", "12", "Tool calls", "10", "Media described", "3", "Tokens in", "1M", "Cached", "88%",
                     "Tokens out", "45k", "Est. cost", "$1.23", "Cost, 7 days", "Reminders", "2", "Files", "14",
                     "Disk", "3.2 MB"):
        assert expected in block, expected
    assert all(len(line) <= sr.LINE_WIDTH for line in lines), lines
    assert "▁" in block and "█" in block  # the 7-day cost sparkline


def test_render_edge_values_stay_within_the_line_width():
    huge = _summary(actions=1_234_567, requests=999_999, answers=999_999, tool_calls=9_999_999, media_described=12_345,
                    api_calls=5_000_000, input_tokens=9_876_543_210, output_tokens=123_456_789,
                    cache_read_tokens=98_765_432_100, cache_write_tokens=12_345_678_900, cost_usd=123_456.789)
    for view in (
        _view(huge, used=9_999, limit=10_000, reminders=1_234, files=1_234_567, size=1_234_567_890_123),
        _view(huge, used=0, limit=None),
        _view(_summary(api_calls=0, cost_usd=0, input_tokens=0, output_tokens=0, cache_read_tokens=0, actions=0),
              used=0, limit=None, reminders=0, files=0, size=0),
    ):
        lines = _block_lines(usage_ui.render(view))
        assert all(len(line) <= sr.LINE_WIDTH for line in lines), lines


def test_render_unlimited_and_empty_usage():
    text = usage_ui.render(_view(_summary(api_calls=0, actions=5), used=0, limit=None))
    lines = _block_lines(text)
    assert lines[0] == "Actions 5 · unlimited"
    assert any(line.startswith("Tokens") and line.endswith("none yet") for line in lines)
    assert "Cost, 7 days" not in text and "Cached" not in text


def test_cost_and_size_formats():
    assert usage_ui.fmt_cost(0) == "$0.00" and usage_ui.fmt_cost(0.00421) == "$0.0042"
    assert usage_ui.fmt_cost(0.9999) == "$0.9999" and usage_ui.fmt_cost(12.345) == "$12.35" and usage_ui.fmt_cost(1234.5) == "$1,234.50"
    assert usage_ui.fmt_size(0) == "0.0 MB" and usage_ui.fmt_size(3_355_443) == "3.2 MB"
    assert usage_ui.fmt_size(2 * 1024 ** 3) == "2.0 GB"


# ---- handlers -----------------------------------------------------------------
class FakeTracker:
    def __init__(self, summary):
        self.summary = summary
        self.calls = 0

    def get_user_usage_summary(self, user_id, days):
        assert (user_id, days) == (USER, 30)
        self.calls += 1
        return dict(self.summary)

    def get_user_workspace_size(self, user_id, base_dir):
        assert user_id == USER and base_dir == "data"
        return {"files": 14, "bytes": 3_355_443}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Fake stats, a reminders store under tmp_path, a 37/50 quota; returns the fake tracker."""
    fake = FakeTracker(_summary())
    monkeypatch.setattr(usage_ui, "_tracker", lambda: fake)
    monkeypatch.setattr(usage_ui, "check_user_limits", lambda uid: (True, 37, 50))
    store = RemindersStore(base_dir=tmp_path)
    monkeypatch.setattr(usage_ui, "reminders_store", store)
    return fake, store


def make_message():
    """A real aiogram Message (the callback handler checks the type) whose Telegram calls are mocks."""
    msg = Message(message_id=7, date=datetime.now(timezone.utc), chat=Chat(id=int(USER), type="private"))
    object.__setattr__(msg, "answer", AsyncMock(return_value=msg))  # the model is frozen: bypass pydantic
    object.__setattr__(msg, "edit_text", AsyncMock(return_value=msg))
    return msg


def make_callback(data, message=None, *, answer=None):
    return SimpleNamespace(data=data, message=message if message is not None else make_message(),
                           answer=answer or AsyncMock())


def _buttons(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


async def test_usage_command_answers_with_the_report_and_a_refresh_button(wired):
    fake, store = wired
    await store.add(USER, {"text": "a", "type": "user", "status": "pending"})
    await store.add(USER, {"text": "b", "type": "agent", "status": "processing"})
    await store.add(USER, {"text": "c", "type": "user", "status": "pending", "enabled": False})
    await store.add(USER, {"text": "d", "type": "user", "status": "completed"})
    message = make_message()
    await usage_ui.usage_command(message, user_id=USER)
    message.answer.assert_awaited_once()
    text, kw = message.answer.await_args.args[0], message.answer.await_args.kwargs
    assert kw["parse_mode"] == "Markdown" and _buttons(kw["reply_markup"]) == ["usage_refresh"]
    lines = _block_lines(text)
    assert lines[0] == f"Actions {sr.meter(37, 50)} 37/50"
    assert any(line.startswith("Reminders") and line.endswith(" 2") for line in lines)  # enabled, pending/processing
    assert any(line.startswith("Files") and line.endswith("14") for line in lines)
    assert any(line.startswith("Disk") and line.endswith("3.2 MB") for line in lines)
    assert fake.calls == 1


async def test_unlimited_users_see_the_word_and_their_action_count(wired, monkeypatch):
    monkeypatch.setattr(usage_ui, "check_user_limits", lambda uid: (True, 0, None))  # admin / limit 0
    message = make_message()
    await usage_ui.usage_command(message, user_id=USER)
    assert _block_lines(message.answer.await_args.args[0])[0] == "Actions 37 · unlimited"


async def test_refresh_edits_the_message_in_place(wired):
    fake, _ = wired
    message = make_message()
    callback = make_callback("usage_refresh", message)
    await usage_ui.usage_button(callback, user_id=USER)
    callback.answer.assert_awaited_once()
    message.edit_text.assert_awaited_once()
    text, kw = message.edit_text.await_args.args[0], message.edit_text.await_args.kwargs
    assert "📊 *Your usage*" in text and kw["parse_mode"] == "Markdown"
    assert _buttons(kw["reply_markup"]) == ["usage_refresh"]
    assert message.answer.await_count == 0  # edited, not re-sent
    assert fake.calls == 1


async def test_refresh_still_happens_when_the_query_is_too_old(wired):
    too_old = TelegramBadRequest(method=AnswerCallbackQuery(callback_query_id="1"), message="Bad Request: query is too old")
    message = make_message()
    await usage_ui.usage_button(make_callback("usage_refresh", message, answer=AsyncMock(side_effect=too_old)), user_id=USER)
    message.edit_text.assert_awaited_once()


async def test_refresh_falls_back_to_plain_text_when_markdown_is_refused(wired):
    message = make_message()
    message.edit_text.side_effect = [TelegramBadRequest(method=AnswerCallbackQuery(callback_query_id="1"),
                                                        message="Bad Request: can't parse entities"), message]
    await usage_ui.usage_button(make_callback("usage_refresh", message), user_id=USER)
    assert message.edit_text.await_count == 2
    second = message.edit_text.await_args_list[1]
    assert "parse_mode" not in second.kwargs and "*" not in second.args[0]


async def test_unknown_usage_callbacks_and_inaccessible_messages_are_ignored(wired):
    message = make_message()
    await usage_ui.usage_button(make_callback("usage_something", message), user_id=USER)
    await usage_ui.usage_button(make_callback("usage_refresh", SimpleNamespace(chat=None)), user_id=USER)
    assert message.edit_text.await_count == 0


async def test_a_failing_collection_is_not_a_dead_end(wired, monkeypatch):
    def boom(uid):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(usage_ui, "check_user_limits", boom)
    message = make_message()
    await usage_ui.usage_command(message, user_id=USER)
    text, kw = message.answer.await_args.args[0], message.answer.await_args.kwargs
    assert text == usage_ui.ERROR_TEXT and _buttons(kw["reply_markup"]) == ["usage_refresh"]


def test_importing_the_module_does_not_import_stats():
    """bot/ never imports stats at module level: that would create data/stats.db on import."""
    import ast

    tree = ast.parse(open(usage_ui.__file__, encoding="utf-8").read())
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = [alias.name for node in top_level if isinstance(node, ast.Import) for alias in node.names]
    modules += [node.module for node in top_level if isinstance(node, ast.ImportFrom)]
    assert "stats" not in modules, modules
