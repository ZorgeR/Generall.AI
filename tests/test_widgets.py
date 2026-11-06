"""Telegram widgets (Bot API 10.4): bot/widgets.py, agents/widget_tools.py, bot/handlers/widget_queries.py.

The agent sends mini-HTML widgets (send_widget: the server's size limits and the out-of-profile
check first), changes their state (update_widget: editMessageWidgetData, silent), lists them and
reads the authoring guide. Taps that ask the bot arrive as ``widget_query`` updates aiogram does not
know: an outer middleware answers them at once and turns the request into an agent turn (joining a
running turn of the topic, or a queued quiet-marker turn; requests of one widget coalesce while it
waits). The registry of sent widgets is bot-only state outside the workspace.
"""
import json
import os
import random
import re
import stat
import string
import sys
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import Message, Update

from agents import widget_tools
from agents.widget_tools import WIDGET_GUIDE, WidgetTools
from bot import inbox, widgets
from bot.handlers import messages
from bot.handlers import widget_queries as wq

USER = "42"
CHAT = {"id": 42, "type": "private"}

GOOD_HTML = (
    "<style>body{margin:0;padding:12px}button{width:100%}</style>"
    "<div id='v'></div><button id='b'>Refresh</button>"
    "<script>const TW=window.TelegramWidget,$=id=>document.getElementById(id);"
    "function draw(){try{$('v').textContent=JSON.parse(TW.chatData).n}catch(e){}}"
    "$('b').onclick=()=>{TW&&TW.request('refresh')};if(TW)TW.onEvent('chatDataChanged',draw);draw();</script>"
)


def noise(n: int, seed: int = 1) -> str:
    rnd = random.Random(seed)
    return "".join(rnd.choice(string.ascii_letters + string.digits) for _ in range(n))


class FakeBot:
    """``await bot(method)`` records the Bot API method objects; ``fail`` maps a method name to an exception."""

    def __init__(self):
        self.calls = []
        self.fail: dict[str, Exception] = {}
        self._id = 900

    async def __call__(self, method, request_timeout=None):
        self.calls.append(method)
        error = self.fail.get(method.__api_method__)
        if error is not None:
            raise error
        if method.__api_method__ == "sendWidget":
            self._id += 1
            return Message.model_validate({
                "message_id": self._id, "date": 1762400000, "chat": CHAT,
                "widget": {"width": 0, "height": 0, **({"id": "777"} if method.is_shared else {})},
            })
        return True

    def of(self, api_method):
        return [m for m in self.calls if m.__api_method__ == api_method]


def bad_request(text: str) -> TelegramBadRequest:
    return TelegramBadRequest(method=SendMessage(chat_id=1, text="x"), message=text)


def sender(bot, thread_id=None):
    return SimpleNamespace(bot=bot, chat_id=42, thread_id=thread_id)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / USER).mkdir(parents=True)
    widgets.reset()
    wq.reset()
    yield tmp_path
    widgets.reset()
    wq.reset()


# ---- limits and checks -------------------------------------------------------------------------
def test_measure_counts_like_the_server():
    html = GOOD_HTML * 3
    size, packed = widgets.measure(html)
    assert size == len(html.encode())
    assert packed == len(zlib.compress(html.encode(), 9))


def test_check_html_limits():
    assert widgets.check_html(GOOD_HTML) is None
    assert "empty" in widgets.check_html("  ")
    assert "NUL" in widgets.check_html("<b>\x00</b>")
    assert "32768" in widgets.check_html("a" * 40_000)
    error = widgets.check_html(f"<p>{noise(9000)}</p>")  # small but incompressible
    assert error and "compresses to" in error and "4096" in error
    assert "UTF-8" in widgets.check_html("<b>\ud800</b>")


def test_check_state_and_the_cleared_value():
    assert widgets.check_state("chat_data", None) is None
    assert widgets.check_state("chat_data", '{"a":1}') is None
    assert "1024" in widgets.check_state("chat_data", "x" * 1025)
    assert "1024" in widgets.check_state("public_data", "é" * 600)  # counted in bytes
    assert "string" in widgets.check_state("chat_data", {"a": 1})
    assert "NUL" in widgets.check_state("chat_data", "a\x00")
    assert widgets.state_value("") == widgets.CLEARED_STATE == "null"
    assert widgets.state_value(None) is None and widgets.state_value("x") == "x"


@pytest.mark.parametrize("html, needle", [
    ("<input type=text>", "form controls"),
    ("<TEXTAREA></TEXTAREA>", "form controls"),
    ("<script src='https://cdn.x/chart.js'></script>", "external scripts"),
    ("<img src=\"https://x.com/a.png\">", "data: URLs"),
    ("<div style=\"background:url('//x.com/a.png')\"></div>", "url()"),
    ("<script>fetch('/api')</script>", "no network"),
    ("<script>new WebSocket('wss://x')</script>", "no network"),
    ("<link rel=stylesheet href=a.css>", "<link>"),
    ("<iframe src=x></iframe>", "<iframe>"),
])
def test_lint_problems(html, needle):
    problems, _ = widgets.lint(html)
    assert any(needle in p for p in problems), problems


def test_lint_notes_and_clean_widgets():
    _, notes = widgets.lint("<div style='height:100vh;overflow:auto'></div>")
    assert any("vh" in n for n in notes) and any("scrolls" in n for n in notes)
    _, notes = widgets.lint(GOOD_HTML, allow_queries=False)
    assert any("allow_queries" in n for n in notes)
    assert widgets.lint(GOOD_HTML, allow_queries=True) == ([], [])
    # an https link is fine (a tap opens it), so are words like "prompt" in text
    assert widgets.lint("<a href='https://example.com'>prompt: fetch the data</a>") == ([], [])


def test_the_guide_example_is_a_valid_widget():
    """The example the agent copies patterns from must pass our own checks and fit the limits."""
    example = WIDGET_GUIDE.split("Refresh tap\n", 1)[1].split("\nwith chat_data", 1)[0]
    assert example.startswith("<style>") and example.rstrip().endswith("</script>")
    assert widgets.check_html(example) is None
    assert widgets.lint(example, allow_queries=True) == ([], [])
    state = re.search(r"with chat_data '(.*?)' and", WIDGET_GUIDE, re.S).group(1).replace("\n", "")
    json.loads(state)


def test_methods_serialize_without_unset_fields():
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession

    bot = Bot(token="42:abc")
    method = widgets.SendWidget(chat_id=1, html="<b>x</b>", allow_queries=True, chat_data="null", width=None)
    fields = [f[0]["name"] for f in AiohttpSession().build_form_data(bot, method)._fields]
    assert fields == ["chat_id", "html", "chat_data", "allow_queries"]
    assert method.__api_method__ == "sendWidget"
    assert widgets.EditMessageWidgetData(chat_id=1, message_id=2, public_data="x").__api_method__ == "editMessageWidgetData"
    assert widgets.AnswerWidgetQuery(widget_query_id="q", data="{}").__api_method__ == "answerWidgetQuery"


# ---- the registry --------------------------------------------------------------------------------
def add(name="Dash", message_id=901, **kw):
    fields = dict(name=name, message_id=message_id, thread_id=None, allow_queries=True, chat_data='{"n":1}',
                  public_data=None, caption=None, html_file=None, html_bytes=10, packed_bytes=8)
    fields.update(kw)
    return widgets.add(USER, **fields)


def test_registry_ids_lookup_and_updates(env):
    first, second = add("BTC dashboard", 901), add("Quiz", 902, thread_id=5)
    assert (first["id"], second["id"]) == ("w1", "w2")
    assert widgets.find(USER, "w2")["name"] == "Quiz"
    assert widgets.find(USER, "W1")["message_id"] == 901
    assert widgets.find(USER, "2")["id"] == "w2"  # a bare number is an id first
    assert widgets.find(USER, "902")["id"] == "w2"  # then a message id
    assert widgets.find(USER, "btc DASHBOARD")["id"] == "w1"
    assert widgets.find(USER, "nope") is None and widgets.find(USER, "") is None
    assert widgets.by_message(USER, 902)["id"] == "w2"
    widgets.update(USER, "w1", chat_data='{"n":2}')
    assert widgets.find(USER, "w1")["chat_data"] == '{"n":2}'
    widgets.count_tap(USER, "w1")
    assert widgets.find(USER, "w1")["taps"] == 1


def test_registry_keeps_the_newest_and_never_reuses_ids(env, monkeypatch):
    monkeypatch.setattr(widgets, "MAX_WIDGETS", 3)
    for i in range(5):
        add(f"n{i}", 900 + i)
    ids = [w["id"] for w in widgets.load(USER)]
    assert ids == ["w3", "w4", "w5"]
    assert add("again", 999)["id"] == "w6"


def test_registry_is_bot_only_state_outside_the_workspace(env):
    add()
    path = env / "data" / "_widgets" / f"{USER}.json"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not any((env / "data" / USER).rglob("*.json"))


def test_registry_never_follows_a_link(env):
    target = env / "elsewhere.json"
    target.write_text(json.dumps({"next": 9, "widgets": [{"id": "w8", "message_id": 1, "name": "planted"}]}))
    directory = env / "data" / "_widgets"
    directory.mkdir(parents=True)
    os.symlink(target, directory / f"{USER}.json")
    assert widgets.load(USER) == []
    assert widgets.find(USER, "planted") is None


def test_registry_refuses_odd_user_ids():
    with pytest.raises(ValueError):
        widgets.user_path("../x")
    with pytest.raises(ValueError):
        widgets.user_path("_sandbox")


# ---- send_widget ---------------------------------------------------------------------------------
async def test_send_widget_sends_records_and_saves_the_html(env):
    bot = FakeBot()
    tools = WidgetTools(USER, sender(bot, thread_id=5))
    result = await tools.execute_tool("send_widget", {
        "html": GOOD_HTML, "name": "Counter", "chat_data": '{"n":1}', "allow_queries": True, "width": 0, "height": 320,
        "caption": "A counter",
    })
    assert result.startswith('Widget w1 "Counter" sent (message 901;'), result
    (method,) = bot.of("sendWidget")
    assert (method.chat_id, method.message_thread_id, method.html) == (42, 5, GOOD_HTML)
    assert (method.allow_queries, method.chat_data, method.public_data) == (True, '{"n":1}', None)
    assert (method.width, method.height, method.caption) == (None, 320, "A counter")
    record = widgets.find(USER, "w1")
    assert record["message_id"] == 901 and record["thread_id"] == 5 and record["allow_queries"]
    saved = env / "data" / USER / record["html_file"]
    assert record["html_file"].startswith("widgets/counter-") and saved.read_text() == GOOD_HTML
    assert record["html_file"] in result


async def test_send_widget_refuses_what_the_server_or_the_apps_would(env):
    bot = FakeBot()
    tools = WidgetTools(USER, sender(bot))
    too_big = await tools.execute_tool("send_widget", {"html": f"<p>{noise(9000)}</p>", "name": "x"})
    assert too_big.startswith("Error:") and "compresses" in too_big
    broken = await tools.execute_tool("send_widget", {"html": "<input id=a><script>fetch('x')</script>", "name": "x"})
    assert broken.startswith("Error: the widget was not sent") and "form controls" in broken and "no network" in broken
    state = await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "x", "public_data": "x" * 2000})
    assert "public_data" in state and "1024" in state
    both = await tools.execute_tool("send_widget", {"html": GOOD_HTML, "html_file": "a.html", "name": "x"})
    assert "not both" in both
    assert bot.calls == []
    forced = await tools.execute_tool("send_widget", {"html": "<input>" + GOOD_HTML, "name": "x", "ignore_checks": True})
    assert "sent" in forced and "Sent despite" in forced and len(bot.calls) == 1


async def test_send_widget_reports_notes_with_the_result(env):
    tools = WidgetTools(USER, sender(FakeBot()))
    result = await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "x"})
    assert "allow_queries" in result and "Notes" in result


async def test_send_widget_from_a_workspace_file(env):
    bot = FakeBot()
    workspace = env / "data" / USER
    (workspace / "build").mkdir()
    (workspace / "build" / "w.html").write_text(GOOD_HTML)
    tools = WidgetTools(USER, sender(bot))
    result = await tools.execute_tool("send_widget", {"html_file": "build/w.html", "name": "Built", "allow_queries": True})
    assert "sent" in result and bot.of("sendWidget")[0].html == GOOD_HTML
    assert widgets.find(USER, "Built")["html_file"] == "build/w.html"
    assert not (workspace / "widgets").exists()  # no second copy

    (env / "secret.html").write_text("<b>secret</b>")
    os.symlink(env / "secret.html", workspace / "link.html")
    assert "not found" in await tools.execute_tool("send_widget", {"html_file": "link.html", "name": "x"})
    assert "not found" in await tools.execute_tool("send_widget", {"html_file": "../../secret.html", "name": "x"})
    assert len(bot.calls) == 1


async def test_saved_html_never_follows_a_planted_link(env):
    outside = env / "outside"
    outside.mkdir()
    os.symlink(outside, env / "data" / USER / "widgets")
    tools = WidgetTools(USER, sender(FakeBot()))
    result = await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "x", "allow_queries": True})
    assert "sent" in result
    assert list(outside.iterdir()) == []
    folder = env / "data" / USER / "widgets"
    assert folder.is_dir() and not folder.is_symlink() and len(list(folder.iterdir())) == 1


async def test_a_server_without_widgets_turns_the_tools_off(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = TelegramNotFound(method=SendMessage(chat_id=1, text="x"), message="Not Found")
    tools = WidgetTools(USER, sender(bot))
    assert await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "x"}) == widgets.UNSUPPORTED_TEXT
    assert widgets.unsupported()
    assert await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "x"}) == widgets.UNSUPPORTED_TEXT
    assert await tools.execute_tool("update_widget", {"widget": "w1", "chat_data": "{}"}) == widgets.UNSUPPORTED_TEXT
    assert len(bot.calls) == 1


async def test_telegram_refusals_reach_the_agent_with_a_hint(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = bad_request("Bad Request: INPUT_MEDIA_WIDGET_NOT_ALLOWED")
    result = await WidgetTools(USER, sender(bot)).execute_tool("send_widget", {"html": GOOD_HTML, "name": "x"})
    assert result.startswith("Error: Telegram refused the widget") and "private chats and groups" in result
    assert widgets.load(USER) == [] and widgets.unsupported() is None


# ---- the startup probe ---------------------------------------------------------------------------
async def test_probe_finds_a_server_without_widgets(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = TelegramNotFound(method=SendMessage(chat_id=1, text="x"), message="Not Found: method not found")
    assert await widgets.probe(bot) is False
    (method,) = bot.calls
    assert method.chat_id == widgets.PROBE_CHAT_ID == 0  # no chat has that id: nothing can be sent
    assert not widgets.available() and "startup probe" in widgets.unsupported()
    assert widget_tools.widgets_available() is False


async def test_probe_finds_a_server_with_widgets(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = bad_request("Bad Request: chat not found")
    assert await widgets.probe(bot) is True
    assert widgets.available() and widget_tools.widgets_available()


async def test_probe_without_an_answer_keeps_the_tools(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = ConnectionError("server not up yet")
    assert await widgets.probe(bot) is None
    assert widgets.available()

    async def hangs(method, request_timeout=None):
        import asyncio

        await asyncio.sleep(60)

    assert await widgets.probe(hangs, timeout=0.05) is None
    assert widgets.available()


# ---- update_widget / list_widgets / widget_guide -------------------------------------------------
async def test_update_widget_edits_the_state_silently(env):
    bot = FakeBot()
    tools = WidgetTools(USER, sender(bot))
    await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "Dash", "chat_data": '{"n":1}', "allow_queries": True})
    result = await tools.execute_tool("update_widget", {"widget": "dash", "chat_data": '{"n":2}', "public_data": ""})
    assert "updated (chat_data and public_data)" in result
    (edit,) = bot.of("editMessageWidgetData")
    assert (edit.chat_id, edit.message_id, edit.chat_data, edit.public_data) == (42, 901, '{"n":2}', "null")
    record = widgets.find(USER, "w1")
    assert (record["chat_data"], record["public_data"]) == ('{"n":2}', "null")

    bot.fail["editMessageWidgetData"] = bad_request("Bad Request: message is not modified")
    assert "nothing changed" in await tools.execute_tool("update_widget", {"widget": "w1", "chat_data": '{"n":2}'})

    bot.fail["editMessageWidgetData"] = bad_request("Bad Request: MESSAGE_ID_INVALID")
    gone = await tools.execute_tool("update_widget", {"widget": "w1", "chat_data": '{"n":3}'})
    assert "gone" in gone and widgets.find(USER, "w1")["gone"]
    assert "gone" in await tools.execute_tool("update_widget", {"widget": "w1", "chat_data": "{}"})


async def test_update_widget_argument_errors(env):
    tools = WidgetTools(USER, sender(FakeBot()))
    assert "give chat_data" in await tools.execute_tool("update_widget", {"widget": "w1"})
    assert "no widget" in await tools.execute_tool("update_widget", {"widget": "w9", "chat_data": "{}"})
    assert "1024" in await tools.execute_tool("update_widget", {"widget": "w1", "chat_data": "x" * 2000})


async def test_list_widgets_and_the_guide(env):
    tools = WidgetTools(USER, sender(FakeBot(), thread_id=7))
    assert await tools.execute_tool("list_widgets", {}) == "No widgets sent in this chat yet."
    await tools.execute_tool("send_widget", {"html": GOOD_HTML, "name": "Dash", "chat_data": '{"n":1}', "allow_queries": True})
    listing = await tools.execute_tool("list_widgets", {})
    assert listing.startswith('w1 · "Dash" · message 901 · topic 7 · taps ask you') and 'chat_data: {"n":1}' in listing
    assert await tools.execute_tool("widget_guide", {}) == WIDGET_GUIDE
    assert "widget_guide" in WidgetTools.prompt_notes and "<no_update/>" in WidgetTools.prompt_notes
    assert {t["name"] for t in tools.tools_schema} == {"send_widget", "update_widget", "list_widgets", "widget_guide"}


def test_the_patcher_never_names_the_widget_tools():
    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    for name in ("WidgetTools", "widget_tools", "send_widget", "update_widget", "_send", "_save_html"):
        assert name not in source, f"tool_integrator.py names {name}"


# ---- widget queries ------------------------------------------------------------------------------
class FakeQueue:
    def __init__(self):
        self.jobs = []
        self.pending = False

    def submit_nowait(self, job):
        self.jobs.append(job)
        return SimpleNamespace(was_busy=False)

    def has_pending_user_jobs(self, user_id, thread_id=None):
        return self.pending


class _Stats:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


@pytest.fixture
def chat(monkeypatch):
    """The chat side faked: queue, auth, quota, stats and run_turn (which records its arguments)."""
    import bot.agent_runner as agent_runner
    import bot.limits as limits

    queue = FakeQueue()
    monkeypatch.setattr(messages, "queue", queue)
    monkeypatch.setattr(wq, "auth", SimpleNamespace(is_authorized=lambda uid: uid != "666"))
    quota = {"allowed": True}
    monkeypatch.setattr(limits, "check_user_limits", lambda uid: (quota["allowed"], 50, 50))
    stats = ModuleType("stats")
    stats.stats_tracker = _Stats()
    monkeypatch.setitem(sys.modules, "stats", stats)
    turns = []

    async def run_turn(**kw):
        turns.append(kw)

    monkeypatch.setattr(agent_runner, "run_turn", run_turn)
    inbox.reset()
    yield SimpleNamespace(queue=queue, quota=quota, stats=stats.stats_tracker, turns=turns)
    inbox.reset()


def query(data="refresh", message_id=901, chat=CHAT, qid="q1", **extra):
    return {"id": qid, "from": {"id": 42, "first_name": "Alice", "username": "alice"}, "chat": chat,
            "message_id": message_id, "chat_instance": "-1", "data": data, **extra}


def answers(bot):
    return [json.loads(m.data) for m in bot.of("answerWidgetQuery")]


async def test_a_tap_is_answered_at_once_and_queued_as_a_quiet_turn(env, chat):
    add("Dash", 901, thread_id=5, chat_data='{"n":1}')
    bot = FakeBot()
    assert await wq.handle_query(bot, query("refresh")) == wq.ACK
    assert answers(bot) == [{"ok": True}] and bot.of("answerWidgetQuery")[0].widget_query_id == "q1"
    (job,) = chat.queue.jobs
    assert (job.label, job.thread_id, job.from_user) == (wq.WIDGET_LABEL, 5, True)
    assert ("track_message_received", (USER, "widget")) in chat.stats.calls
    assert widgets.find(USER, "w1")["taps"] == 1

    await wq.handle_query(bot, query("tab:2", qid="q2"))  # joins the queued turn
    await wq.handle_query(bot, query("tab:2", qid="q3"))  # the same again at once: acknowledged, dropped
    assert len(chat.queue.jobs) == 1 and answers(bot) == [{"ok": True}] * 3

    await job.run(SimpleNamespace())
    (turn,) = chat.turns
    assert turn["quiet_marker"] == "<no_update/>" and turn["header"] == wq.WIDGET_HEADER
    assert (turn["reply_to_message_id"], turn["thread_id"], turn["chat_id"], turn["user_id"]) == (901, 5, 42, USER)
    prompt = turn["prompt"]
    assert prompt.startswith('[Widget request] Widget w1 "Dash" (message 901) sent 2 requests:')
    assert '"refresh"' in prompt and '"tab:2"' in prompt and "Alice (@alice), user id 42" in prompt
    assert 'chat_data: {"n":1}' in prompt and "update_widget" in prompt and "<no_update/>" in prompt

    await wq.handle_query(bot, query("refresh", qid="q4"))  # the turn has started: a new one is queued
    assert len(chat.queue.jobs) == 2


async def test_a_cancelled_turn_lets_the_next_tap_start_a_new_one(env, chat):
    add("Dash", 901)
    bot = FakeBot()
    await wq.handle_query(bot, query("a"))
    chat.queue.jobs[0].on_drop()
    await wq.handle_query(bot, query("b", qid="q2"))
    assert len(chat.queue.jobs) == 2


@pytest.mark.parametrize("case, expected", [
    ("unknown message", "this widget does not take requests"),
    ("no requests", "this widget does not take requests"),
    ("unauthorized", "not authorized"),
    ("inline", "not available here"),
    ("empty", "empty request"),
    ("quota", "action limit reached"),
])
async def test_requests_that_do_not_reach_the_agent_get_an_error_answer(env, chat, case, expected):
    add("Dash", 901)
    add("Static", 902, allow_queries=False)
    q = {
        "unknown message": query(message_id=555),
        "no requests": query(message_id=902),
        "unauthorized": query(chat={"id": 666, "type": "private"}),
        "inline": {"id": "q1", "from": {"id": 42}, "inline_message_id": "AQ", "data": "x"},
        "empty": query(data=""),
        "quota": query(),
    }[case]
    if case == "quota":
        chat.quota["allowed"] = False
    bot = FakeBot()
    outcome = await wq.handle_query(bot, q)
    assert outcome["ok"] is False and expected in outcome["error"]
    assert answers(bot) == [outcome]
    assert chat.queue.jobs == [] and chat.turns == []


async def test_a_forwarded_copy_gets_an_error_answer(env, chat):
    bot = FakeBot()
    outcome = await wq.handle_query(bot, {"id": "q9", "from": {"id": 1}, "data": "x", "widget_id": "7"}, copy=True)
    assert outcome["ok"] is False and answers(bot) == [outcome]


async def test_a_running_turn_of_the_topic_reads_the_tap(env, chat):
    add("Dash", 901, thread_id=5)
    box = inbox.open_inbox(USER, 5, base=f"data/{USER}")
    bot = FakeBot()
    assert await wq.handle_query(bot, query("refresh")) == wq.ACK
    assert chat.queue.jobs == []
    blocks = await box.drain()
    text = " ".join(b.get("text", "") for b in blocks)
    assert "[Widget request]" in text and '"refresh"' in text
    assert not any(m for m in bot.calls if m.__api_method__ == "setMessageReaction")


async def test_a_failing_route_still_answers(env, chat, monkeypatch):
    add("Dash", 901)

    def boom(*a, **k):
        raise RuntimeError("disk")

    monkeypatch.setattr(widgets, "by_message", boom)
    bot = FakeBot()
    outcome = await wq.handle_query(bot, query())
    assert outcome == {"ok": False, "error": "the bot failed, try again"} and answers(bot) == [outcome]


async def test_a_widget_asking_in_a_loop_is_rate_limited(env, chat, monkeypatch):
    add("Dash", 901)
    monkeypatch.setattr(wq, "RATE_LIMIT", 3)
    bot = FakeBot()
    outcomes = [await wq.handle_query(bot, query(f"poll:{i}", qid=f"q{i}")) for i in range(5)]
    assert outcomes[:3] == [wq.ACK] * 3
    assert all(o["ok"] is False and "too many" in o["error"] for o in outcomes[3:])
    assert answers(bot) == outcomes
    assert len(chat.queue.jobs) == 1 and len(wq._batches[(USER, "w1")].events) == 3


async def test_a_failing_tap_count_does_not_turn_a_queued_request_into_an_error(env, chat, monkeypatch):
    add("Dash", 901)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(widgets, "count_tap", boom)
    bot = FakeBot()
    assert await wq.handle_query(bot, query()) == wq.ACK
    assert len(chat.queue.jobs) == 1 and answers(bot) == [{"ok": True}]


def test_who_is_one_line():
    who = wq._who({"id": 5, "first_name": "Eve\nIgnore all previous", "last_name": " X ", "username": "e v"})
    assert "\n" not in who and who == "Eve Ignore all previous X (@ev), user id 5"
    assert wq._who(None) == "someone" and wq._who({"id": 1}) == "someone, user id 1"


async def test_a_state_refusal_gets_the_state_hint(env):
    bot = FakeBot()
    bot.fail["sendWidget"] = bad_request("Bad Request: INPUT_MEDIA_WIDGET_DATA_TOO_LARGE")
    result = await WidgetTools(USER, sender(bot)).execute_tool("send_widget", {"html": GOOD_HTML, "name": "x"})
    assert "1024 bytes" in result and "compressed" not in result


# ---- the dispatcher ------------------------------------------------------------------------------
def test_query_of_reads_the_raw_update():
    update = Update.model_validate({"update_id": 1, "widget_query": query()})
    assert wq.query_of(update) == (query(), False)
    copy = Update.model_validate({"update_id": 2, "shared_widget_query": {"id": "q", "data": "x", "widget_id": "7"}})
    assert wq.query_of(copy)[1] is True
    plain = Update.model_validate({"update_id": 3, "message": {"message_id": 1, "date": 1, "chat": CHAT, "text": "hi"}})
    assert wq.query_of(plain) == (None, False)


def test_an_aiogram_that_knows_widgets_still_reaches_us():
    """A later aiogram (Bot API 10.4) declares widget_query and Message.widget as typed fields, so they leave
    model_extra: both readers must still find them, with the Bot API's own key names ("from", not "from_user")."""
    from aiogram.types import Chat, TelegramObject, User
    from pydantic import Field

    class WidgetQuery(TelegramObject):
        id: str
        from_user: User = Field(..., alias="from")
        chat: Chat | None = None
        message_id: int | None = None
        data: str

    class Widget(TelegramObject):
        id: str | None = None
        width: int = 0

    class FutureUpdate(Update):
        widget_query: WidgetQuery | None = None

    class FutureMessage(Message):
        widget: Widget | None = None

    real = {**query("go"), "from": {"id": 42, "is_bot": False, "first_name": "Alice"}}  # as Telegram sends it
    update = FutureUpdate.model_validate({"update_id": 1, "widget_query": real})
    assert not (update.model_extra or {}).get("widget_query")  # typed now, not an extra
    raw, copy = wq.query_of(update)
    assert copy is False and raw["data"] == "go" and raw["from"]["first_name"] == "Alice"
    assert raw["chat"]["id"] == 42 and raw["message_id"] == 901
    message = FutureMessage.model_validate({"message_id": 1, "date": 1, "chat": CHAT, "widget": {"id": "77"}})
    assert widgets.sent_widget_info(message)["id"] == "77"
    assert wq.query_of(Update.model_validate({"update_id": 2})) == (None, False)


async def test_the_dispatcher_routes_widget_queries_and_polls_for_them(monkeypatch):
    import warnings

    from aiogram import Bot

    import bot.app as app

    handled = []

    async def handle(bot, raw, copy=False):
        handled.append((raw["data"], copy))
        return wq.ACK

    monkeypatch.setattr(wq, "handle_query", handle)
    dp = app.create_dispatcher()
    assert "widget_query" in app.allowed_updates(dp)
    assert {"message", "callback_query"} <= set(app.allowed_updates(dp))
    bot = Bot(token="42:abc")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no "unknown update type" warning
            await dp.feed_update(bot, Update.model_validate({"update_id": 5, "widget_query": query("go")}))
    finally:
        await bot.session.close()
    assert handled == [("go", False)]


def test_widget_tools_module_never_imports_the_sandbox():
    assert "secure_container" not in Path(widget_tools.__file__).read_text()
