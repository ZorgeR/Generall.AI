"""/sandbox, the sandbox_ buttons, the idle warnings of the sandbox-reaper and the app hooks (bot/handlers/sandbox_ui.py).

The SessionManager (secure_container/sessions.py) is replaced by ``FakeManager``, installed as
``secure_container.sessions`` so ``session_manager()`` is exercised for real; every manager call
records whether it ran off the event loop's thread.
"""
import asyncio
import logging
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import secure_container
from aiogram.types import Chat, Message

from bot import app, runtime
from bot.handlers import sandbox_ui as sb
from bot.settings import UserSettings
from bot.ui import escape_markdown

USER = "42"
NOW = datetime.now(timezone.utc).replace(microsecond=0)
CLOSES = datetime(2026, 10, 17, 12, 35, tzinfo=timezone.utc)  # 14:35 in Berlin (UTC+2 until Oct 25)


def not_running(**extra):
    return {
        "shell": {"running": False},
        "browser": {"running": False, "tabs": []},
        "profile_bytes": 0,
        "packages": {"apt": [], "pip": []},
        "idle_timeout_seconds": 1800,
        "warned": False,
        **extra,
    }


def running(**extra):
    return {
        "shell": {"running": True, "started": NOW - timedelta(minutes=7), "last_used": NOW,
                  "idle_seconds_left": 1500, "closes_at": CLOSES, "mem_mb": 120},
        "browser": {"running": True, "started": NOW.timestamp() - 60, "last_used": NOW.isoformat(),
                    "idle_seconds_left": 1700, "closes_at": CLOSES.isoformat(), "tabs": [
                        {"tab_id": "tab-1", "title": "Shop *deals* [new]\n_sale_", "url": "https://my_shop.example.com/a_b?x=1", "active": False},
                        {"tab_id": "tab-2", "title": "", "url": "about:blank", "active": True},
                    ]},
        "profile_bytes": int(2.5 * 1024 * 1024),
        "packages": {"apt": ["jq", "lib_foo*"], "pip": ["requests"]},
        "idle_timeout_seconds": 1800,
        "warned": False,
        **extra,
    }


SERVER = {"shells": 1, "max_shells": 4, "browsers": 0, "max_browsers": 2, "mem_available_mb": 2345,
          "mem_total_mb": 7812, "reserve_mb": 1536, "egress_filter": "on"}


class FakeManager:
    def __init__(self, status=None):
        self.state = status if status is not None else not_running()
        self.server = dict(SERVER)
        self.calls: list[tuple] = []
        self.on_loop: list[str] = []  # manager methods that were called on the event loop's thread
        self.extend_result = CLOSES + timedelta(minutes=30)
        self.close_result = True
        self.forget_result = True
        self.reaped: list = []
        self.due: list = []
        self.errors: dict[str, Exception] = {}

    def _call(self, name, *args):
        self.calls.append((name, *args))
        if threading.current_thread() is threading.main_thread():
            self.on_loop.append(name)
        if name in self.errors:
            raise self.errors[name]

    def names(self):
        return [c[0] for c in self.calls]

    def status(self, user_id):
        self._call("status", user_id)
        return self.state

    def server_status(self):
        self._call("server_status")
        return self.server

    def close(self, user_id, kind=None):
        self._call("close", user_id)
        return self.close_result

    def forget_browser_profile(self, user_id):
        self._call("forget_browser_profile", user_id)
        return self.forget_result

    def extend(self, user_id):
        self._call("extend", user_id)
        return self.extend_result

    def reap_idle(self):
        self._call("reap_idle")
        reaped, self.reaped = self.reaped, []
        return reaped

    def due_warnings(self):
        self._call("due_warnings")
        due, self.due = self.due, []
        return due

    def cleanup_stale(self):
        self._call("cleanup_stale")
        return 2

    def shutdown_all(self):
        self._call("shutdown_all")


class FakeAuth:
    def __init__(self, admins=(), blocked=()):
        self.admins, self.blocked = set(admins), set(blocked)

    def is_admin(self, user_id):
        return str(user_id) in self.admins

    def is_authorized(self, user_id):
        return str(user_id) not in self.blocked


def install_sessions(monkeypatch, manager, **constants):
    """A fake ``secure_container.sessions`` module whose get_manager() returns ``manager``."""
    module = ModuleType("secure_container.sessions")
    module.get_manager = lambda: manager
    module.SESSION_IDLE_SECONDS = constants.get("idle", 1800)
    module.SESSION_WARN_BEFORE_SECONDS = constants.get("warn", 300)
    monkeypatch.setitem(sys.modules, "secure_container.sessions", module)
    monkeypatch.setattr(secure_container, "sessions", module, raising=False)
    return module


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # UserSettings under tmp_path
    monkeypatch.setattr(sb, "auth", FakeAuth())
    sb.reset()
    yield
    sb.reset()


@pytest.fixture
def manager(monkeypatch):
    m = FakeManager()
    install_sessions(monkeypatch, m)
    yield m
    assert m.on_loop == [], f"blocking manager calls on the event loop: {m.on_loop}"


@pytest.fixture
def ui(monkeypatch):
    """Record answer_md / edit_md instead of talking to Telegram."""
    calls = []

    async def answer_md(message, text, reply_markup=None):
        calls.append(("answer_md", text, reply_markup))

    async def edit_md(message, text, reply_markup=None):
        calls.append(("edit_md", text, reply_markup, message.message_id))

    monkeypatch.setattr(sb, "answer_md", answer_md)
    monkeypatch.setattr(sb, "edit_md", edit_md)
    return calls


def berlin():
    UserSettings(USER).set("timezone", "Europe/Berlin", "name")


def make_message(message_id=7, text="🧰 Sandbox & browser"):
    return Message(message_id=message_id, date=datetime.now(timezone.utc), chat=Chat(id=int(USER), type="private"), text=text)


def make_callback(data, message=None):
    return SimpleNamespace(data=data, message=message if message is not None else make_message(), answer=AsyncMock())


def make_bot(*, send_error=None):
    bot = SimpleNamespace(sent=0)

    async def send_message(**kw):
        if send_error is not None and send_error(kw):
            raise RuntimeError("Forbidden: bot was blocked by the user")
        bot.sent += 1
        return SimpleNamespace(message_id=500 + bot.sent)

    bot.send_message = AsyncMock(side_effect=send_message)
    bot.edit_message_text = AsyncMock()
    bot.edit_message_reply_markup = AsyncMock()
    return bot


def assert_markdown_balanced(text):
    """Legacy Markdown parses: once the escaped characters are gone, every entity mark is paired."""
    plain = re.sub(r"\\[_*`\[]", "", text)
    for mark in "*_`":
        assert plain.count(mark) % 2 == 0, (mark, text)
    assert "[" not in plain, text


def datas(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


def labels(markup):
    return [b.text for row in markup.inline_keyboard for b in row]


# ---- the manager lookup and its constants ---------------------------------------------------
def test_no_sessions_module_means_no_manager_and_default_numbers(monkeypatch):
    monkeypatch.delattr(secure_container, "sessions", raising=False)
    monkeypatch.setitem(sys.modules, "secure_container.sessions", None)  # import fails
    assert sb.session_manager() is None
    assert (sb.idle_seconds(), sb.warn_before_seconds()) == (1800, 300)


def test_a_manager_that_was_never_initialised_is_none(monkeypatch):
    install_sessions(monkeypatch, None)
    assert sb.session_manager() is None


def test_numbers_come_from_the_sessions_module(monkeypatch):
    install_sessions(monkeypatch, None, idle=3600, warn=600)
    assert sb.warning_text(["shell"], CLOSES, None).startswith("⏳ Your sandbox will close in 10 minutes (at 12:35 UTC)")
    assert sb.closed_text(["shell"]).startswith("🔒 Sandbox closed after 1 hour without use.")
    assert sb.keep_label() == "⏳ Keep open 1 h"
    install_sessions(monkeypatch, None, idle="garbage", warn=0)
    assert (sb.idle_seconds(), sb.warn_before_seconds()) == (1800, 300)


def test_duration_words_and_times():
    assert sb.duration_words(1800) == "30 minutes" and sb.duration_words(60) == "1 minute"
    assert sb.duration_words(7200) == "2 hours" and sb.duration_words(3600, short=True) == "1 h"
    assert sb.duration_words(20, short=True) == "1 min" and sb.duration_words(5400, short=True) == "90 min"
    assert sb.as_datetime(CLOSES.isoformat()) == CLOSES == sb.as_datetime(CLOSES.timestamp())
    assert sb.as_datetime("2026-10-17T12:35:00Z") == CLOSES == sb.as_datetime(CLOSES.replace(tzinfo=None))
    assert sb.as_datetime("soon") is None and sb.as_datetime(None) is None and sb.as_datetime(0) is None
    assert sb.fmt_bytes(512) == "512 B" and sb.fmt_bytes(2048) == "2 KB" and sb.fmt_bytes(int(2.5 * 1024 ** 2)) == "2.5 MB"


# ---- /sandbox --------------------------------------------------------------------------------
async def test_sandbox_unavailable_without_a_manager(monkeypatch, ui):
    install_sessions(monkeypatch, None)
    await sb.sandbox_command(make_message(), USER)
    (kind, text, markup), = ui
    assert kind == "answer_md" and text == sb.UNAVAILABLE and "not available" in text
    assert datas(markup) == [sb.REFRESH]


async def test_sandbox_page_when_nothing_runs(manager, ui):
    await sb.sandbox_command(make_message(), USER)
    (_, text, markup), = ui
    assert text.startswith(sb.TITLE)
    assert "🖥 *Sandbox*: not running (starts when I need to run something)" in text
    assert "📦 *Saved packages*: none" in text
    assert "🌐 *Browser*: not running (opens when I need a website)" in text
    assert "🔑 *Logins*: none stored" in text
    assert "closes after 30 minutes (with a warning 5 minutes before)" in text
    assert_markdown_balanced(text)
    assert "Server" not in text and "server_status" not in manager.names()
    assert datas(markup) == [sb.REFRESH]  # nothing to keep open, close or forget
    assert manager.calls == [("status", USER)]


async def test_sandbox_page_when_both_run(manager, ui):
    manager.state = running()
    berlin()
    await sb.sandbox_command(make_message(), USER)
    (_, text, markup), = ui
    assert "🖥 *Sandbox*: running for 7 min · closes at 14:35 if unused" in text
    assert "🌐 *Browser*: running, 2 tabs · closes at 14:35 if unused" in text
    # page titles, hosts and package names are escaped; control characters and newlines are gone
    assert f"• {escape_markdown('Shop *deals* [new] _sale_')} — {escape_markdown('my_shop.example.com')}" in text
    assert "▶️ untitled — about:blank" in text
    assert f"apt: jq, {escape_markdown('lib_foo*')} · pip: requests" in text and "reinstalled when the sandbox starts" in text
    assert "🔑 *Logins*: remembered (2.5 MB)" in text
    assert_markdown_balanced(text)
    assert "\n_sale_" not in text and "a_b?x=1" not in text  # one line per tab, host only
    assert labels(markup) == ["⏳ Keep open 30 min", "🔄 Refresh", "⏹ Close sandbox & browser", "🧹 Forget browser logins"]
    assert datas(markup) == [sb.KEEP, sb.REFRESH, sb.CLOSE, sb.FORGET]


async def test_sandbox_page_times_are_utc_without_a_zone_and_survive_odd_values(manager, ui):
    state = running()
    state["shell"] = {"running": True, "idle_seconds_left": 600}  # no started / closes_at
    state["browser"] = {"running": True, "tabs": [{"title": "x" * 500, "url": "https://example.org/"}] * 9}
    state["packages"] = {"apt": [f"pkg{i}" for i in range(12)], "pip": None}
    manager.state = state
    await sb.sandbox_command(make_message(), USER)
    text = ui[0][1]
    shell_line = next(line for line in text.splitlines() if line.startswith("🖥"))
    assert shell_line.startswith("🖥 *Sandbox*: running · closes at ") and shell_line.endswith(" UTC if unused")
    assert "running, 9 tabs" in text and "  … 3 more" in text and "x" * 60 not in text
    assert "pkg7 +4 more" in text and "pkg8" not in text


async def test_sandbox_page_browser_off_and_logins_not_kept(manager, ui):
    UserSettings(USER).set("browser", {"enabled": False, "stream": True, "remember": False})
    manager.state = not_running(profile_bytes=4096)
    await sb.sandbox_command(make_message(), USER)
    text, markup = ui[0][1], ui[0][2]
    assert "🌐 *Browser*: off (turn it on in /settings → 🌐 Browser)" in text
    assert "🔑 *Logins*: not kept (Remember logins is off in /settings); an older profile of 4 KB is stored" in text
    assert datas(markup) == [sb.REFRESH, sb.FORGET]  # the old profile can still be forgotten


async def test_admin_sees_the_server(manager, ui, monkeypatch):
    monkeypatch.setattr(sb, "auth", FakeAuth(admins={USER}))
    await sb.sandbox_command(make_message(), USER)
    text = ui[-1][1]
    assert "🛠 *Server* (admin)\nSandboxes 1/4 · browsers 0/2" in text
    assert "Memory: 2,345 MB free of 7,812 MB, 1,536 MB kept in reserve" in text and "Egress filter: on" in text
    manager.server = {**SERVER, "egress_filter": "failed", "mem_total_mb": None}
    await sb.sandbox_command(make_message(), USER)
    assert "Egress filter: ⚠️ failed, private networks are NOT blocked" in ui[-1][1] and "free of unknown" in ui[-1][1]
    assert_markdown_balanced(ui[-1][1])
    manager.errors["server_status"] = RuntimeError("docker down")
    await sb.sandbox_command(make_message(), USER)
    assert "🛠 *Server* (admin): status unavailable" in ui[-1][1] and "🖥 *Sandbox*" in ui[-1][1]


async def test_a_failing_status_shows_an_error_with_refresh(manager, ui):
    manager.errors["status"] = RuntimeError("boom")
    await sb.sandbox_command(make_message(), USER)
    (_, text, markup), = ui
    assert text == sb.ERROR_TEXT and datas(markup) == [sb.REFRESH]


# ---- page buttons ----------------------------------------------------------------------------
async def test_refresh_and_back_redraw_the_page_in_place(manager, ui):
    for data in (sb.REFRESH, sb.BACK):
        cb = make_callback(data, make_message(55))
        await sb.sandbox_button(cb, make_bot(), USER)
        cb.answer.assert_awaited_once()
        kind, text, markup, message_id = ui[-1]
        assert (kind, message_id) == ("edit_md", 55) and text.startswith(sb.TITLE)
    assert manager.names() == ["status", "status"]


async def test_close(manager, ui):
    manager.state = running()
    bot = make_bot()
    await sb.sandbox_button(make_callback(sb.CLOSE), bot, USER)
    assert manager.calls[0] == ("close", USER)
    assert ui[-1][1].startswith(f"{sb.TITLE}\n\n{sb.CLOSED_BY_USER}\n\n")
    manager.close_result = False
    await sb.sandbox_button(make_callback(sb.CLOSE), bot, USER)
    assert f"\n\n{sb.NOTHING_RUNNING}\n\n" in ui[-1][1]
    manager.errors["close"] = RuntimeError("docker")
    await sb.sandbox_button(make_callback(sb.CLOSE), bot, USER)
    assert f"\n\n{sb.ACTION_FAILED}\n\n" in ui[-1][1]


async def test_forget_asks_first(manager, ui):
    await sb.sandbox_button(make_callback(sb.FORGET), make_bot(), USER)
    kind, text, markup, _ = ui[-1]
    assert text == sb.FORGET_CONFIRM and datas(markup) == [sb.FORGET_YES, sb.BACK]
    assert manager.calls == []  # nothing happens before the confirmation
    await sb.sandbox_button(make_callback(sb.FORGET_YES), make_bot(), USER)
    assert manager.calls[0] == ("forget_browser_profile", USER)
    assert f"\n\n{sb.FORGOTTEN}\n\n" in ui[-1][1]
    manager.forget_result = False
    await sb.sandbox_button(make_callback(sb.FORGET_YES), make_bot(), USER)
    assert f"\n\n{sb.NOTHING_TO_FORGET}\n\n" in ui[-1][1]


async def test_keep_open_from_the_page(manager, ui):
    berlin()
    manager.state = running()
    await sb.sandbox_button(make_callback(sb.KEEP), make_bot(), USER)
    assert manager.calls[0] == ("extend", USER)
    assert "\n\n✅ Kept open until 15:05.\n\n" in ui[-1][1]
    manager.extend_result = None
    await sb.sandbox_button(make_callback(sb.KEEP), make_bot(), USER)
    assert f"\n\n{sb.ALREADY_CLOSED}\n\n" in ui[-1][1]


async def test_buttons_without_a_manager_say_unavailable(monkeypatch, ui):
    install_sessions(monkeypatch, None)
    await sb.sandbox_button(make_callback(sb.CLOSE), make_bot(), USER)
    assert ui[-1][1] == sb.UNAVAILABLE


async def test_unknown_and_inaccessible_buttons(manager, ui):
    await sb.sandbox_button(make_callback("sandbox_explode"), make_bot(), USER)
    cb = SimpleNamespace(data=sb.CLOSE, message=None, answer=AsyncMock())
    await sb.sandbox_button(cb, make_bot(), USER)
    cb.answer.assert_awaited_once()
    assert ui == [] and manager.calls == []


async def test_an_expired_query_does_not_lose_the_tap(manager, ui):
    cb = make_callback(sb.CLOSE)
    cb.answer = AsyncMock(side_effect=RuntimeError("query is too old"))
    await sb.sandbox_button(cb, make_bot(), USER)
    assert manager.calls[0] == ("close", USER) and ui[-1][0] == "edit_md"


# ---- idle warnings ---------------------------------------------------------------------------
async def test_reaper_sends_one_warning_with_buttons_in_local_time(manager):
    berlin()
    manager.due = [{"user_id": USER, "kinds": ["shell", "browser"], "closes_at": CLOSES}]
    bot = make_bot()
    await sb.reaper_tick(bot, manager)
    assert manager.names() == ["reap_idle", "due_warnings"]
    kw = bot.send_message.await_args.kwargs
    assert kw["chat_id"] == int(USER)
    assert kw["text"] == (
        "⏳ Your sandbox and browser will close in 5 minutes (at 14:35) because they have not been used.\n"
        "Running programs and open browser tabs will stop; your files and saved logins stay."
    )
    assert labels(kw["reply_markup"]) == ["👌 OK", "⏳ Keep open 30 min"]
    assert datas(kw["reply_markup"]) == [sb.OK, sb.KEEP]
    assert sb._warnings[USER] == sb.ClosingWarning(message_id=501, closes_at=CLOSES, kinds=("shell", "browser"))
    await sb.reaper_tick(bot, manager)  # nothing due any more: nothing sent again
    assert bot.send_message.await_count == 1


async def test_warning_without_a_zone_is_in_utc_and_names_what_closes(manager):
    bot = make_bot()
    manager.due = [{"user_id": USER, "kinds": ["shell"], "closes_at": CLOSES}]
    await sb.reaper_tick(bot, manager)
    assert bot.send_message.await_args.kwargs["text"] == (
        "⏳ Your sandbox will close in 5 minutes (at 12:35 UTC) because it has not been used.\n"
        "Running programs will stop; your files stay."
    )
    manager.due = [{"user_id": USER, "kinds": ["browser"], "closes_at": CLOSES.isoformat()}]
    await sb.reaper_tick(bot, manager)
    assert bot.send_message.await_args.kwargs["text"].startswith("⏳ Your browser will close in 5 minutes (at 12:35 UTC) because it has")
    assert sb._warnings[USER].message_id == 502  # the newest warning is the one remembered


def warn(manager, bot):
    manager.due = [{"user_id": USER, "kinds": ["shell", "browser"], "closes_at": CLOSES}]
    return sb.reaper_tick(bot, manager)


async def test_ok_removes_the_buttons_and_keeps_the_text(manager, ui):
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)
    cb = make_callback(sb.OK, make_message(501, "⏳ Your sandbox and browser will close ..."))
    await sb.sandbox_button(cb, bot, USER)
    assert bot.edit_message_reply_markup.await_args.kwargs == {"chat_id": int(USER), "message_id": 501, "reply_markup": None}
    assert ui == [] and "extend" not in manager.names()
    assert USER not in sb._warnings
    cb.answer.assert_awaited_once()


async def test_keep_extends_and_edits_the_warning(manager, ui):
    berlin()
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)
    cb = make_callback(sb.KEEP, make_message(501, "⏳ Your sandbox and browser will close ..."))
    await sb.sandbox_button(cb, bot, USER)
    assert ("extend", USER) in manager.calls
    assert ui == [("edit_md", "✅ Kept open until 15:05.", None, 501)]  # no buttons
    assert USER not in sb._warnings
    bot.edit_message_text.assert_not_awaited()  # nothing else to settle


async def test_keep_after_the_sandbox_closed(manager, ui):
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)
    manager.state = not_running()  # closed in the meantime (e.g. from /sandbox)
    await sb.sandbox_button(make_callback(sb.KEEP, make_message(501, "⏳ Your sandbox ...")), bot, USER)
    assert "extend" not in manager.names()
    assert ui == [("edit_md", sb.ALREADY_CLOSED, None, 501)] and USER not in sb._warnings
    # extend racing the reaper: nothing ran any more by the time it was called
    await warn(manager, bot)
    manager.state, manager.extend_result = running(), None
    await sb.sandbox_button(make_callback(sb.KEEP, make_message(502, "⏳ Your sandbox ...")), bot, USER)
    assert ui[-1] == ("edit_md", sb.ALREADY_CLOSED, None, 502)


async def test_keep_on_a_warning_the_user_made_obsolete_reports_the_current_time(manager, ui):
    berlin()
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)
    later = CLOSES + timedelta(minutes=20)  # the user worked again: a later closing time
    manager.state["shell"]["closes_at"] = later
    manager.state["browser"]["closes_at"] = later
    cb = make_callback(sb.KEEP, make_message(501, "⏳ Your sandbox ..."))
    await sb.sandbox_button(cb, bot, USER)
    assert "extend" not in manager.names()
    assert ui == [("edit_md", "✅ In use again: it now closes at 14:55 if unused.", None, 501)]
    assert cb.answer.await_args.args == ("✅ In use again: it now closes at 14:55 if unused.",)
    cb = make_callback(sb.OK, make_message(501, "⏳ Your sandbox ..."))
    await sb.sandbox_button(cb, bot, USER)
    assert cb.answer.await_args.args == ("✅ In use again: it now closes at 14:55 if unused.",)
    assert bot.edit_message_reply_markup.await_count == 1


async def test_keep_on_an_older_warning_extends_and_settles_the_current_one(manager, ui):
    berlin()
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)  # 501, an earlier idle period
    await warn(manager, bot)  # 502, the current warning
    await sb.sandbox_button(make_callback(sb.KEEP, make_message(501, "⏳ Your sandbox ...")), bot, USER)
    assert ("extend", USER) in manager.calls
    assert ui == [("edit_md", "✅ Kept open until 15:05.", None, 501)]
    assert bot.edit_message_text.await_args.kwargs == {"chat_id": int(USER), "message_id": 502, "text": "✅ Kept open until 15:05."}
    assert USER not in sb._warnings


async def test_a_warning_from_before_a_restart(manager, ui):
    manager.state = not_running()  # a restart closed everything
    await sb.sandbox_button(make_callback(sb.KEEP, make_message(9, "⏳ Your sandbox will close ...")), make_bot(), USER)
    assert ui == [("edit_md", sb.ALREADY_CLOSED, None, 9)] and "extend" not in manager.names()


async def test_page_keep_also_settles_a_pending_warning(manager, ui):
    berlin()
    manager.state = running()
    bot = make_bot()
    await warn(manager, bot)
    await sb.sandbox_button(make_callback(sb.KEEP, make_message(77)), bot, USER)  # the /sandbox page's button
    assert "\n\n✅ Kept open until 15:05.\n\n" in ui[-1][1] and ui[-1][3] == 77
    assert bot.edit_message_text.await_args.kwargs == {"chat_id": int(USER), "message_id": 501, "text": "✅ Kept open until 15:05."}
    assert USER not in sb._warnings


async def test_reaper_turns_the_warning_into_closed(manager):
    bot = make_bot()
    await warn(manager, bot)
    manager.reaped = [(USER, "shell"), (USER, "browser"), ("7", "shell")]  # 7 was never warned: no edit
    await sb.reaper_tick(bot, manager)
    bot.edit_message_text.assert_awaited_once()
    assert bot.edit_message_text.await_args.kwargs == {
        "chat_id": int(USER), "message_id": 501,
        "text": "🔒 Sandbox and browser closed after 30 minutes without use. Your files and saved logins are kept; "
                "they start again when needed.",
    }
    assert USER not in sb._warnings
    assert sb.closed_text(["shell"]) == (
        "🔒 Sandbox closed after 30 minutes without use. Your files and saved logins are kept; it starts again when needed.")


async def test_reaper_survives_failures(manager, caplog):
    bot = make_bot(send_error=lambda kw: kw["chat_id"] == 7)
    manager.due = [
        {"user_id": "7", "kinds": ["shell"], "closes_at": CLOSES},  # blocked the bot: logged, skipped
        {"user_id": "not-a-chat", "kinds": ["shell"], "closes_at": CLOSES},
        {"user_id": USER, "kinds": ["shell"], "closes_at": CLOSES},
    ]
    with caplog.at_level(logging.WARNING, logger="bot.handlers.sandbox_ui"):
        await sb.reaper_tick(bot, manager)
    assert [c.kwargs["chat_id"] for c in bot.send_message.await_args_list] == [7, int(USER)]
    assert set(sb._warnings) == {USER}
    assert any("Could not send the sandbox closing warning to 7" in r.getMessage() for r in caplog.records)
    # a failing edit of a warning is logged too
    bot.edit_message_text.side_effect = RuntimeError("message to edit not found")
    manager.reaped = [(USER, "shell")]
    await sb.reaper_tick(bot, manager)
    assert USER not in sb._warnings
    # manager failures do not stop the tick or the loop
    manager.errors["reap_idle"] = RuntimeError("docker down")
    manager.due = [{"user_id": USER, "kinds": ["shell"], "closes_at": CLOSES}]
    await sb.reaper_tick(bot, manager)
    assert bot.send_message.await_count == 3


async def test_unauthorized_users_are_not_warned(manager, monkeypatch):
    monkeypatch.setattr(sb, "auth", FakeAuth(blocked={USER}))
    bot = make_bot()
    await warn(manager, bot)
    bot.send_message.assert_not_awaited()
    assert sb._warnings == {}


async def test_reaper_loop_keeps_ticking_after_errors(manager, monkeypatch):
    ticks = []

    async def tick(bot, m):
        ticks.append(m)
        if len(ticks) == 1:
            raise RuntimeError("unexpected")

    monkeypatch.setattr(sb, "reaper_tick", tick)
    task = asyncio.create_task(sb.reaper_loop(make_bot(), manager, interval=0.001))
    for _ in range(200):
        if len(ticks) >= 3:
            break
        await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert len(ticks) >= 3 and ticks[0] is manager


async def test_a_manager_without_warnings_still_reaps(monkeypatch):
    class OldManager:
        def __init__(self):
            self.reaps = 0

        def reap_idle(self):
            self.reaps += 1
            return []

    m = OldManager()
    await sb.reaper_tick(make_bot(), m)
    assert m.reaps == 1


# ---- the app hooks ---------------------------------------------------------------------------
@pytest.fixture
async def tasks():
    yield runtime.background_tasks
    for task in runtime.background_tasks:
        task.cancel()
    await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
    runtime.background_tasks.clear()


async def test_startup_cleans_up_and_starts_the_reaper(manager, tasks):
    await app.start_sandbox_sessions(make_bot())
    assert manager.names() == ["cleanup_stale"]
    assert [t.get_name() for t in tasks] == ["sandbox-reaper"]


async def test_startup_goes_on_when_the_cleanup_fails(manager, tasks):
    manager.errors["cleanup_stale"] = RuntimeError("docker down")
    await app.start_sandbox_sessions(make_bot())
    assert [t.get_name() for t in tasks] == ["sandbox-reaper"]


async def test_startup_and_shutdown_without_a_manager(monkeypatch, tasks):
    install_sessions(monkeypatch, None)
    await app.start_sandbox_sessions(make_bot())
    assert tasks == []
    monkeypatch.setattr(runtime, "queue", SimpleNamespace(shutdown=AsyncMock()))
    await app.on_shutdown(make_bot())
    runtime.queue.shutdown.assert_awaited_once()


async def test_on_startup_runs_the_sandbox_hooks(manager, tasks, monkeypatch):
    monkeypatch.setattr(app.auth, "load", lambda: None)
    monkeypatch.setattr(app, "configure_ffmpeg", lambda: None)

    async def idle(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(app, "reminders_loop", idle)
    monkeypatch.setattr(app, "refresh_admin_servers", idle)
    bot = SimpleNamespace(set_my_commands=AsyncMock(), me=AsyncMock(return_value=SimpleNamespace(username="b")))
    await app.on_startup(bot)
    assert "sandbox-reaper" in [t.get_name() for t in tasks]
    assert manager.names() == ["cleanup_stale"]


async def test_shutdown_stops_every_session_after_the_jobs(manager, monkeypatch, tasks):
    order = []

    async def queue_shutdown():
        order.append("queue")

    monkeypatch.setattr(runtime, "queue", SimpleNamespace(shutdown=queue_shutdown))
    original = manager.shutdown_all
    manager.shutdown_all = lambda: (order.append("sessions"), original())
    await app.on_shutdown(make_bot())
    assert order == ["queue", "sessions"]
    manager.errors["shutdown_all"] = RuntimeError("docker down")
    await sb.shutdown_sessions()  # logged, never raised
