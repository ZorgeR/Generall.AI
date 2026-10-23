"""agents/browser_tools.py: the browser provider and the sandbox status, with a fake session manager.

Covers member execution (success, errors, the confirmation refusal, a full server, a crash),
``browser_confirm`` granting and consuming the one-action allowance, ``browser_show_screenshot``
with and without a crop, the throttled live view (one photo per turn, edited in place),
``sandbox_status`` in the user's time zone, the static prompt notes, the registration gating in
``ChainOfThoughtAgent`` and the content/transcript helpers that meet browser results
(browser_state blocks, the image-count compaction boundary). ``ChatSender.send_photo(silent)``
and ``edit_photo`` are covered at the end.
"""
import asyncio
import base64
import copy
import importlib
import io
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from PIL import Image

from agents import browser_tools as B
from agents import content as C
from agents import transcript as T

STATE = {"type": "browser_state", "tabs": [{"tab_id": "tab-1", "title": "Shop", "url": "https://shop.example/", "active": True}]}


def jpeg(size=(1280, 800), color=(200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="JPEG", quality=80)
    return out.getvalue()


class CapacityError(RuntimeError):
    """Stands in for secure_container.sessions.CapacityError (matched by class name)."""


class Manager:
    def __init__(self, responses=None, peek=None, status=None):
        self.responses = list(responses or [])
        self.requests: list[dict] = []
        self.kwargs: list[dict] = []
        self.peek = peek
        self.status_value = status or {}
        self.peeks = 0

    def browser_call(self, uid, request, *, remember=True, timeout=75):
        self.requests.append(request)
        self.kwargs.append({"uid": uid, "remember": remember, "timeout": timeout})
        item = self.responses.pop(0) if self.responses else {"ok": True, "content": [{"type": "text", "text": "ok"}]}
        if isinstance(item, BaseException):
            raise item
        return item

    def browser_peek(self, uid):
        self.peeks += 1
        return self.peek

    def status(self, uid):
        return self.status_value


def peek(data=None, title="Shop", url="https://shop.example/cart"):
    return {"ok": True, "image_b64": base64.b64encode(data or jpeg()).decode(), "media_type": "image/jpeg",
            "meta": {"tab_id": "tab-1", "title": title, "url": url}}


class Sender:
    def __init__(self):
        self.bot = object()
        self.chat_id = 5
        self.photos: list[dict] = []
        self.edits: list[dict] = []
        self.next_id = 100

    async def send_photo(self, source, *, filename=None, caption=None, silent=False):
        self.next_id += 1
        self.photos.append({"data": source, "filename": filename, "caption": caption, "silent": silent, "id": self.next_id})
        return SimpleNamespace(message_id=self.next_id)

    async def edit_photo(self, message, source, *, filename=None, caption=None):
        self.edits.append({"message_id": message.message_id, "data": source, "caption": caption})
        return message


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def tools(manager=None, sender=None, **browser):
    s = {"browser": {"enabled": True, "stream": True, "remember": True, **browser}, "timezone": {"name": ""}}
    t = B.BrowserTools("7", sender, s, manager if manager is not None else Manager())
    t._clock = Clock()
    return t


# ---- execute_member ---------------------------------------------------------------------------
async def test_member_success_passes_content_through():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    manager = Manager([{"ok": True, "content": [{"type": "text", "text": "Clicked"}, {"type": "text", "text": ""}, image, STATE,
                                                 {"type": "document", "x": 1}], "meta": {"title": "Shop", "url": "u"}}])
    t = tools(manager, remember=False)
    content, is_error = await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_2"}})
    assert is_error is False
    assert content == [{"type": "text", "text": "Clicked"}, image, STATE]  # empty text and foreign blocks dropped
    request = manager.requests[0]
    assert request["action"] == "left_click" and request["input"] == {"target": {"type": "ref", "ref": "ref_2"}}
    assert request["allow_sensitive"] is False and request["id"]
    assert manager.kwargs[0] == {"uid": "7", "remember": False, "timeout": B.BROWSER_CALL_TIMEOUT}
    assert t._last_action == "left_click ref_2" and t._last_meta == {"title": "Shop", "url": "u"}


async def test_the_first_member_call_of_a_turn_asks_for_the_tab_state():
    # review: browser_state was left out whenever unchanged since the last report to ANY
    # conversation (forum topics share one browser; a cancelled turn's result is thrown away), so a
    # turn's first read_page could carry no tab list. Until one call of THIS turn succeeded, the
    # request asks the server for the full state.
    manager = Manager([{"ok": False, "error": "Error: No tab is open. Use new_tab or navigate first."},
                       {"ok": True, "content": [{"type": "text", "text": "page"}, STATE]},
                       {"ok": True, "content": [{"type": "text", "text": "page"}]}])
    t = tools(manager)
    await t.execute_member("read_page", {})
    await t.execute_member("read_page", {})
    await t.execute_member("screenshot", {})
    assert [r["force_state"] for r in manager.requests] == [True, True, False]
    fresh = tools(manager)  # the next turn: a new instance asks again
    await fresh.execute_member("find", {"query": "search"})
    assert manager.requests[-1]["force_state"] is True


async def test_member_with_empty_content_says_done():
    t = tools(Manager([{"ok": True, "content": []}]))
    assert await t.execute_member("hover", {"target": {"type": "coordinate", "x": 1, "y": 2}}) == ("Done.", False)


async def test_member_errors():
    manager = Manager([
        {"ok": False, "error": "Error: Navigation refused. 10.0.0.1 is a private or local address.", "needs_confirmation": False},
        {"ok": False, "error": "timeout"},
        {"ok": False, "needs_confirmation": True,
         "error": "Error: This looks like a consequential action (button \"Place order\"). Ask the user first with browser_confirm, then repeat the action."},
        CapacityError("The server is short on memory (612 MB free); no new browser can start right now. Try again in a few minutes."),
        OSError("socket gone"),
        "not a dict",
    ])
    t = tools(manager)
    assert await t.execute_member("navigate", {"url": "http://10.0.0.1"}) == ("Error: Navigation refused. 10.0.0.1 is a private or local address.", True)
    assert await t.execute_member("wait", {"duration": 1}) == ("Error: timeout", True)
    text, is_error = await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_9"}})
    assert is_error and "browser_confirm" in text
    text, is_error = await t.execute_member("navigate", {"url": "a.com"})
    assert is_error and text.startswith("The server is short on memory")  # the readable reason as is
    text, is_error = await t.execute_member("screenshot", {})
    assert is_error and "socket gone" in text
    assert await t.execute_member("screenshot", {}) == ("Error: the browser gave no answer.", True)


async def test_disabled_members_and_no_manager(monkeypatch):
    manager = Manager()
    t = tools(manager)
    assert await t.execute_member("javascript_exec", {"text": "1"}) == ("Error: javascript_exec is not enabled in this environment.", True)
    assert manager.requests == []
    monkeypatch.setattr(B, "session_manager", lambda: None)
    orphan = B.BrowserTools("7", None, {}, None)
    text, is_error = await orphan.execute_member("navigate", {"url": "a.com"})
    assert is_error and "not available" in text


def test_the_toolset_entry_disables_the_risky_members():
    entry = tools().toolset
    assert entry["type"] == "browser_toolset_20260801" and "name" not in entry
    assert entry["configs"] == {m: {"enabled": False} for m in ("javascript_exec", "file_upload", "read_console", "read_network")}
    entry["configs"].clear()
    assert tools().toolset["configs"]  # a fresh copy each time
    assert {t["name"] for t in tools().tools_schema} == {"browser_confirm", "browser_show_screenshot"}
    assert B.BrowserTools.toolset_name == "browser"


# ---- browser_confirm --------------------------------------------------------------------------
@pytest.fixture
def answers(monkeypatch):
    """Scripted answers to bot.questions.ask, with what was asked: a plain string is a tap on that
    option (what ask() returns for a tap), anything else (None, a typed Answer) is returned as it is."""
    from bot import questions

    asked: list[dict] = []
    script: list = []

    async def ask(sender, question, options, timeout):
        asked.append({"question": question, "options": options, "timeout": timeout})
        answer = script.pop(0)
        return questions.Answer(answer, tapped=True) if type(answer) is str else answer

    monkeypatch.setattr(questions, "ask", ask)
    return SimpleNamespace(asked=asked, script=script)


async def test_confirm_allow_grants_one_action(answers):
    manager = Manager([{"ok": True, "content": [{"type": "text", "text": "Order placed"}], "sensitive_done": True},
                       {"ok": True, "content": [{"type": "text", "text": "Clicked"}]}], peek=peek())
    sender = Sender()
    t = tools(manager, sender)
    answers.script.append(B.ALLOW_OPTION)
    result = await t.confirm("Place the order for 2 tickets (48 EUR) on shop.example")
    assert result.startswith("The user allowed: Place the order for 2 tickets (48 EUR) on shop.example")
    assert answers.asked[0]["options"] == ["✅ Allow", "❌ Cancel"] and answers.asked[0]["timeout"] == 300
    assert "Place the order" in answers.asked[0]["question"]
    assert len(sender.photos) == 1 and sender.photos[0]["caption"] == "🛡 Place the order for 2 tickets (48 EUR) on shop.example"
    assert sender.photos[0]["data"][:3] == b"\xff\xd8\xff"

    assert t.allowance_active()
    await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_5"}})
    assert manager.requests[0]["allow_sensitive"] is True
    assert not t.allowance_active()  # consumed by the response that did the sensitive action
    await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_6"}})
    assert manager.requests[1]["allow_sensitive"] is False


async def test_the_allowance_survives_harmless_actions_and_expires(answers):
    manager = Manager([{"ok": True, "content": []}] * 3)
    t = tools(manager, Sender())
    answers.script.append(B.ALLOW_OPTION)
    await t.confirm("Send the message to Anna")
    await t.execute_member("read_page", {})  # nothing sensitive happened: still allowed
    assert manager.requests[0]["allow_sensitive"] is True and t.allowance_active()
    t._clock.now += B.ALLOWANCE_SECONDS + 1
    assert not t.allowance_active()
    await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_1"}})
    assert manager.requests[1]["allow_sensitive"] is False


async def test_confirm_decline_and_timeout(answers):
    t = tools(Manager(), Sender())  # no browser running: the question goes out without a picture
    answers.script.extend([B.CANCEL_OPTION, None])
    assert (await t.confirm("Delete the account")).startswith("The user declined: Delete the account")
    assert not t.allowance_active()
    result = await t.confirm("Delete the account")
    assert result.startswith("No answer within 5 minutes") and "Delete the account" in result
    assert not t.allowance_active() and t.sender.photos == []


async def test_only_a_tap_on_allow_grants_the_allowance(answers):
    """A typed "yes" (or even the button's own text, typed) goes back to the agent as text and allows nothing."""
    from bot import questions

    manager = Manager([{"ok": True, "content": []}])
    t = tools(manager, Sender())
    answers.script.extend([questions.Answer("yes, go ahead", tapped=False), questions.Answer(B.ALLOW_OPTION, tapped=False)])
    for typed in ("yes, go ahead", B.ALLOW_OPTION):
        result = await t.confirm("Pay 10 EUR")
        assert result.startswith(f'User answered in text instead of tapping a button: "{typed}"')
        assert "not a permission" in result and not t.allowance_active()
    await t.execute_member("left_click", {"target": {"type": "ref", "ref": "ref_1"}})
    assert manager.requests[0]["allow_sensitive"] is False

    # the typed text reached the turn as a message of its own: the result points at it, still no permission
    answers.script.append(questions.Answer("yes, go ahead", tapped=False, delivered=True))
    result = await t.confirm("Pay 10 EUR")
    assert result.startswith(questions.TYPED_DELIVERED + ". That is not a permission") and "go ahead" not in result
    assert not t.allowance_active()


async def test_a_plain_string_from_a_stand_in_is_no_tap(monkeypatch):
    """The allowance needs an ``Answer`` that says it was tapped: anything else allows nothing."""
    from bot import questions

    async def ask(*a, **k):
        return B.ALLOW_OPTION  # a str, not an Answer

    monkeypatch.setattr(questions, "ask", ask)
    t = tools(Manager(), Sender())
    assert not (await t.confirm("Pay 10 EUR")).startswith("The user allowed") and not t.allowance_active()


async def test_confirm_needs_a_chat_and_a_summary(answers):
    assert (await tools(Manager(), None).confirm("x")).startswith("Error: there is no Telegram chat")
    assert (await tools(Manager(), Sender()).confirm("   ")).startswith("Error: 'summary'")
    assert answers.asked == []


async def test_confirm_propagates_cancellation(monkeypatch):
    from bot import questions

    async def ask(*a, **k):
        raise asyncio.CancelledError

    monkeypatch.setattr(questions, "ask", ask)
    with pytest.raises(asyncio.CancelledError):
        await tools(Manager(), Sender()).confirm("Pay 10 EUR")


# ---- browser_show_screenshot ------------------------------------------------------------------
async def test_show_screenshot_whole_view_and_region():
    data = jpeg()
    sender = Sender()
    t = tools(Manager(peek=peek(data)), sender)
    assert await t.execute_tool("browser_show_screenshot", {"caption": "Which seat do you want?"}) == "Sent."
    assert sender.photos[0]["data"] == data and sender.photos[0]["caption"] == "Which seat do you want?"
    assert sender.photos[0]["silent"] is False

    assert await t.execute_tool("browser_show_screenshot", {"caption": "Captcha", "region": [100, 200, 300, 300]}) == "Sent."
    with Image.open(io.BytesIO(sender.photos[1]["data"])) as cropped:
        assert cropped.size == (600, 300)  # 200x100 scaled up for legibility, at most 3x


async def test_show_screenshot_errors():
    sender = Sender()
    assert (await tools(Manager(peek=peek()), sender).show_screenshot("x", [1, 2, 3])).startswith("Error: region")
    assert (await tools(Manager(peek=peek()), sender).show_screenshot("x", [5000, 5000, 6000, 6000])).startswith("Error: could not crop")
    assert await tools(Manager(peek=None), sender).show_screenshot("x") == "Error: the browser is not running; open a page first."
    assert (await tools(Manager(peek=peek()), None).show_screenshot("x")).startswith("Error: there is no Telegram chat")
    assert sender.photos == []


# ---- the live view ------------------------------------------------------------------------------
async def test_live_view_sends_once_then_edits_with_throttle():
    manager = Manager([{"ok": True, "content": []}] * 3, peek=peek(jpeg(color=(1, 2, 3))))
    sender = Sender()
    t = tools(manager, sender)
    await t.execute_member("navigate", {"url": "shop.example"})
    await t.after_batch()
    assert len(sender.photos) == 1 and sender.edits == []
    first = sender.photos[0]
    assert first["silent"] is True
    assert first["caption"] == "🌐 Shop\nhttps://shop.example/cart\n▶️ navigate shop.example"

    await t.execute_member("type", {"text": "hunter2-secret"})
    t._clock.now += 1.0
    manager.peek = peek(jpeg(color=(9, 9, 9)))
    await t.after_batch()  # within 2.5 s of the last update: skipped
    assert sender.edits == [] and manager.peeks == 1

    t._clock.now += 2.0
    await t.after_batch()
    assert len(sender.photos) == 1 and len(sender.edits) == 1
    assert sender.edits[0]["message_id"] == first["id"]
    assert sender.edits[0]["caption"].endswith("▶️ type 14 characters") and "hunter2" not in sender.edits[0]["caption"]

    t._clock.now += 3.0
    await t.after_batch()  # same picture, same caption: nothing to edit
    assert len(sender.edits) == 1


async def test_live_view_off_or_without_browser():
    sender = Sender()
    off = tools(Manager(peek=peek()), sender, stream=False)
    await off.after_batch()
    idle = tools(Manager(peek=None), sender)
    await idle.after_batch()
    assert sender.photos == [] and sender.edits == []


async def test_live_view_failures_are_not_fatal():
    class Broken(Sender):
        async def send_photo(self, *a, **k):
            raise RuntimeError("Telegram is down")

    t = tools(Manager(peek=peek()), Broken())
    await t.after_batch()  # logged, not raised
    assert t._live_message is None


# ---- sandbox_status -----------------------------------------------------------------------------
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
STATUS = {
    "shell": {"running": True, "started": NOW, "last_used": NOW, "idle_seconds_left": 1500, "mem_mb": 120,
              "mem_limit_mb": 768, "closes_at": NOW + timedelta(minutes=25)},
    "browser": {"running": True, "idle_seconds_left": 600, "closes_at": (NOW + timedelta(minutes=10)).isoformat(),
                "tabs": [{"tab_id": "tab-1", "title": "Shop", "url": "https://shop.example/", "active": True},
                         {"tab_id": "tab-2", "title": "", "url": "about:blank", "active": False}]},
    "profile_bytes": 3 * 1024 * 1024,
    "packages": {"apt": ["ffmpeg"], "pip": ["pandas", "httpx"]},
    "idle_timeout_seconds": 1800,
}


def test_status_text_in_the_users_zone():
    text = B.format_status(STATUS, {"timezone": {"name": "Europe/Berlin"}}, now=NOW)
    assert "Shell session: running, closes at 2026-10-17 14:25 (Europe/Berlin, UTC+2) if idle (in 25 min), memory 120 MB used of 768 MB." in text
    assert "Browser: running, closes at 2026-10-17 14:10 (Europe/Berlin, UTC+2) if idle (in 10 min); 2 tab(s):" in text
    assert "  - tab-1 (active): Shop | https://shop.example/" in text and "  - tab-2: (untitled) | about:blank" in text
    assert "Browser logins: remembered (3.0 MB profile)." in text
    assert "apt: ffmpeg; pip: pandas, httpx" in text
    assert "Idle timeout: 30 min (the user is warned 5 minutes before closing)." in text
    assert "Times in UTC" not in text and "warned" not in text.split("Idle timeout")[0]
    assert "The user was warned" in B.format_status({**STATUS, "warned": True}, {}, now=NOW)


def test_status_text_in_utc_and_when_nothing_runs():
    text = B.format_status(STATUS, {}, now=NOW)
    assert "closes at 2026-10-17 12:25 (UTC)" in text and "Times in UTC" in text
    idle = B.format_status({"shell": {"running": False}, "browser": {"running": False}, "profile_bytes": 0,
                            "packages": {"apt": [], "pip": []}}, {"browser": {"remember": False}}, now=NOW)
    assert "Shell session: not running (it starts with the next command)." in idle
    assert "Browser: not running" in idle and "Browser logins: not remembered" in idle
    assert "Packages reinstalled when a session starts: none." in idle
    # closes_at missing: computed from idle_seconds_left
    computed = B.format_status({"shell": {"running": True, "idle_seconds_left": 120}}, {}, now=NOW)
    assert "closes at 2026-10-17 12:02 (UTC) if idle (in 2 min)" in computed


async def test_sandbox_status_tool():
    t = B.SandboxTools("7", {"timezone": {"name": "Europe/Berlin"}}, Manager(status=STATUS))
    assert [s["name"] for s in t.tools_schema] == ["sandbox_status"]
    text = await t.execute_tool("sandbox_status", {})
    assert text.startswith("Shell session: running") and "Europe/Berlin" in text

    class Failing(Manager):
        def status(self, uid):
            raise RuntimeError("docker unreachable")

    assert "docker unreachable" in await B.SandboxTools("7", {}, Failing()).execute_tool("sandbox_status", {})


# ---- prompt notes and settings ------------------------------------------------------------------------
@pytest.fixture
def sessions(monkeypatch):
    """A stand-in secure_container.sessions module in sys.modules."""
    module = types.ModuleType(B.SESSIONS_MODULE)
    module.SESSION_IDLE_SECONDS = 1800
    module.SESSION_MEM_MB = 768
    module.manager = None
    module.get_manager = lambda: module.manager
    monkeypatch.setitem(sys.modules, B.SESSIONS_MODULE, module)
    return module


def test_prompt_notes(sessions):
    sandbox = B.SandboxTools("7").prompt_notes
    for needle in ("/home/runner/workspace", "30 minutes", "warned 5 minutes before it closes", "must be in the workspace",
                   "nohup", "installed_packages.txt", "installed_python_packages.txt", "private and local networks are blocked",
                   "sandbox_status", "fresh bash shell", "768 MB"):
        assert needle in sandbox, needle
    sessions.SESSION_IDLE_SECONDS = 900
    assert "15 minutes" in B.SandboxTools("7").prompt_notes

    browser = tools().prompt_notes
    for needle in ("browser_confirm", "browser_show_screenshot", "captcha", "end your turn", "strong password",
                   "browse_webpage", "read_page", "ref", "remembered", "15 minutes", "Accepting terms", "registering"):
        assert needle in browser, needle


def test_prompt_notes_fall_back_without_the_sessions_module(monkeypatch):
    monkeypatch.delitem(sys.modules, B.SESSIONS_MODULE, raising=False)
    assert B.session_manager() is None
    assert "30 minutes" in B.SandboxTools("7").prompt_notes and "768 MB" in B.SandboxTools("7").prompt_notes


def test_browser_settings_defaults():
    assert B.browser_settings({}) == {"enabled": True, "stream": True, "remember": True}
    assert B.browser_settings(None)["enabled"] is True
    assert B.browser_settings({"browser": {"enabled": False, "stream": False}}) == {"enabled": False, "stream": False, "remember": True}


# ---- registration ------------------------------------------------------------------------------
@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    return importlib.import_module("agents.main")


def chain(agents_main, **browser):
    from bot.settings import DEFAULT_SETTINGS

    s = copy.deepcopy(DEFAULT_SETTINGS)
    s.setdefault("browser", {}).update(browser)
    return agents_main.ChainOfThoughtAgent(user_id="42", user_settings=s)


def provider_types(agent):
    return [type(p).__name__ for p, _ in agent.agent.extra_providers]


def test_registration_needs_a_manager_and_the_setting(agents_main, sessions):
    sessions.SESSIONS_ENABLED = True
    none = chain(agents_main)  # get_manager() is None
    assert none.browser_tools is None and none.sandbox_tools is None
    assert "BrowserTools" not in provider_types(none) and "SandboxTools" not in provider_types(none)

    sessions.manager = Manager()
    both = chain(agents_main)
    assert provider_types(both)[-2:] == ["SandboxTools", "BrowserTools"]
    assert both.browser_tools._manager is sessions.manager
    tools_sent = both.agent.get_tools_schema()
    assert any(t.get("type") == "browser_toolset_20260801" for t in tools_sent)
    assert {"sandbox_status", "browser_confirm", "browser_show_screenshot"} <= {t.get("name") for t in tools_sent}
    notes = both._provider_notes()
    assert "persistent Linux sandbox session" in notes and "browser_confirm" in notes

    off = chain(agents_main, enabled=False)
    assert off.browser_tools is None and off.sandbox_tools is not None
    assert not any(t.get("type") == "browser_toolset_20260801" for t in off.agent.get_tools_schema())
    assert "sandbox_status" in {t.get("name") for t in off.agent.get_tools_schema()}
    assert "browser_confirm" not in off._provider_notes()

    sessions.SESSIONS_ENABLED = False  # SANDBOX_SESSIONS=false: commands run in one-off containers
    one_off = chain(agents_main)
    assert one_off.sandbox_tools is None and one_off.browser_tools is not None
    assert "persistent Linux sandbox session" not in one_off._provider_notes()


def test_a_failing_manager_lookup_means_no_sandbox_tools(agents_main, sessions):
    def broken():
        raise RuntimeError("init failed")

    sessions.get_manager = broken
    agent = chain(agents_main)
    assert agent.browser_tools is None and agent.sandbox_tools is None


# ---- content helpers and the transcript -------------------------------------------------------------
def test_content_helpers_handle_browser_state():
    state = {**STATE, "state_changes": [{"type": "download_completed", "download_id": "d1", "url": "https://x/f.pdf",
                                         "path": "downloads/f.pdf", "size_bytes": 10}]}
    preview = C.result_preview([{"type": "text", "text": "Clicked"}, state])
    assert preview == "Clicked\n[browser: 1 tab(s); active tab-1: Shop https://shop.example/; download completed downloads/f.pdf]"
    assert C.cap_blocks([{"type": "text", "text": "y" * 50}, state], 10)[1] is state
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "A" * 4000}}
    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1", "toolset_name": "browser",
                                              "content": [{"type": "text", "text": "ok"}, image, state]}]}]
    stripped = C.strip_media(messages)
    block = stripped[0]["content"][0]
    assert block["toolset_name"] == "browser" and block["content"] == [{"type": "text", "text": "ok"}, {"type": "text", "text": "[image]"}, state]
    assert 0 < C.estimate_block_tokens(state) < 100


def _turn(i, images, browser=True):
    """One user turn whose tool result holds ``images`` screenshots."""
    pic = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    result = {"type": "tool_result", "tool_use_id": f"b{i}", "content": [{"type": "text", "text": "shot"}] + [dict(pic) for _ in range(images)]}
    call = {"type": "tool_use", "id": f"b{i}", "name": "screenshot", "input": {}}
    if browser:
        result["toolset_name"] = call["toolset_name"] = "browser"
    return [
        {"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": f"s{i}"}, call]},
        {"role": "user", "content": [result]},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {i}"}]},
    ]


async def test_too_many_images_trigger_a_compaction():
    messages = [m for i in range(5) for m in _turn(i, 9)]  # 45 screenshots, far below the token budget
    assert T.count_images(messages) == 45 > T.MAX_TRANSCRIPT_IMAGES
    assert T.estimate_tokens(messages) < 120000

    async def summarize(old):
        raise AssertionError("pictures alone never summarize")

    stats = await T.prune(messages, max_context_tokens=120000, keep_tool_results_turns=3, summarize=summarize)
    # the two oldest turns lost theirs (27 left), then the oldest 7 of the next turn: down to the target
    assert T.count_images(messages) == T.IMAGE_PRUNE_TARGET == 20
    assert stats["cleared"] == 2 and stats["cleared_media"] == 7
    assert stats["summarized"] == 0 and stats["thinking_stripped"] == 5
    third = messages[10]["content"][0]["content"]
    assert third[1:8] == [{"type": "text", "text": T.SCREENSHOT_CLEARED_MARKER}] * 7 and third[8]["type"] == "image"
    assert not any(b.get("type") == "thinking" for m in messages for b in m["content"])
    assert messages[2]["content"][0]["content"] == T.CLEARED_MARKER  # old tool results cleared as usual
    assert messages[2]["content"][0]["toolset_name"] == "browser"


async def test_images_in_the_kept_turns_are_cleared_down_to_the_newest_turn():
    messages = [m for i in range(3) for m in _turn(i, 25)]  # 75 screenshots, all in the last 3 turns
    stats = await T.prune(messages, max_context_tokens=120000, keep_tool_results_turns=3)
    assert T.count_images(messages) == 25  # 50 after the oldest turn still is too many: only the newest keeps them
    assert stats["cleared_media"] == 50 and stats["cleared"] == 0
    middle = messages[6]["content"][0]["content"]
    assert middle[0] == {"type": "text", "text": "shot"}
    assert middle[1] == {"type": "text", "text": T.SCREENSHOT_CLEARED_MARKER}  # a browser picture has no file to reopen


async def test_few_images_change_nothing():
    messages = [m for i in range(4) for m in _turn(i, 10, browser=False)]
    before = copy.deepcopy(messages)
    stats = await T.prune(messages, max_context_tokens=120000, keep_tool_results_turns=3)
    assert messages == before and stats == {"cleared": 0, "cleared_media": 0, "summarized": 0, "thinking_stripped": 0}


def browser_server():
    """The sandbox's browser server, loaded by path (a standalone script): its ``validate_content``
    states what the API accepts in a member's tool_result."""
    import importlib.util
    from pathlib import Path

    name = "browser_server_for_tools_tests"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "app" / "secure_container" / "scripts" / "browser_server.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look the module up while the class is created
        spec.loader.exec_module(module)
    return sys.modules[name]


def assert_valid_member_results(messages):
    """Every successful browser member result is still one the API accepts (errors carry no state)."""
    names = {b["id"]: b["name"] for m in messages if isinstance(m.get("content"), list)
             for b in m["content"] if b.get("type") == "tool_use"}
    for m in messages:
        for b in m["content"] if isinstance(m.get("content"), list) else []:
            if b.get("type") == "tool_result" and b.get("toolset_name") and not b.get("is_error"):
                content = b["content"] if isinstance(b["content"], list) else [{"type": "text", "text": b["content"]}]
                assert browser_server().validate_content(names[b["tool_use_id"]], content) == [], b


def _browsing_turn(i, shots):
    """A browsing turn as the loop stores it: one screenshot per assistant message, each with its state."""
    pic = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    turn = [{"role": "user", "content": [{"type": "text", "text": "<context>\nnow\n</context>"}, {"type": "text", "text": f"q{i}"}]}]
    for j in range(shots):
        tid = f"t{i}_{j}"
        turn.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x", "signature": "s"},
            {"type": "tool_use", "id": tid, "name": "screenshot", "input": {}, "toolset_name": "browser"}]})
        turn.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "toolset_name": "browser",
                                                  "content": [dict(pic), copy.deepcopy(STATE)]}]})
    turn.append({"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}]})
    return turn


@pytest.mark.parametrize("shots, most_compactions", [(14, 5), (8, 3)])
async def test_an_image_compaction_leaves_room_for_the_next_turns(shots, most_compactions):
    """Clearing to just under the cap made the next browsing turn cross it again: a full cache miss
    and every thinking block stripped on (nearly) every turn. A compaction leaves IMAGE_PRUNE_TARGET
    pictures (the newest turn keeps all of its own), so the next one is turns away."""
    messages: list[dict] = []
    compacted: list[int] = []
    for turn in range(12):
        newest = len(messages)
        messages.extend(_browsing_turn(turn, shots))
        before = copy.deepcopy(messages)
        stats = await T.prune(messages, max_context_tokens=120000, keep_tool_results_turns=3)
        if messages == before:
            assert T.count_images(messages) <= T.MAX_TRANSCRIPT_IMAGES
            continue
        compacted.append(turn)
        assert stats["thinking_stripped"] > 0 and stats["summarized"] == 0
        assert T.count_images(messages) <= max(T.IMAGE_PRUNE_TARGET, shots)
        assert T.count_images(messages[newest:]) == shots  # the newest turn keeps its pictures
        assert_valid_member_results(messages)
    assert compacted, "the cap was never reached"
    assert all(b - a >= 2 for a, b in zip(compacted, compacted[1:])), f"compactions on consecutive turns: {compacted}"
    assert len(compacted) <= most_compactions


OPENED = {"type": "browser_state", "tabs": [
    {"tab_id": "tab-1", "title": "Shop", "url": "https://shop.example/", "active": False},
    {"tab_id": "tab-2", "title": "New Tab", "url": "about:blank", "active": True},
], "state_changes": [{"type": "tab_opened", "tab_id": "tab-2"}]}


def _member_turn(i):
    """One turn with a batch of browser members (and a normal tool) as the loop stores it."""
    pic = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
    calls = [  # member, its result content, is_error
        ("navigate", [{"type": "text", "text": "Navigated to https://shop.example/"}, STATE], False),
        ("screenshot", [{"type": "text", "text": "1280x800"}, dict(pic), STATE], False),
        ("list_tabs", [copy.deepcopy(STATE)], False),
        ("new_tab", [copy.deepcopy(OPENED)], False),
        ("scroll", [{"type": "text", "text": "Scrolled down."}], False),
        ("switch_tab", "Error: there is no tab tab-9.", True),
    ]
    uses = [{"type": "tool_use", "id": f"{name}{i}", "name": name, "input": {}, "toolset_name": "browser"} for name, _, _ in calls]
    results = [{"type": "tool_result", "tool_use_id": f"{name}{i}", "content": copy.deepcopy(content), "toolset_name": "browser",
                **({"is_error": True} if error else {})} for name, content, error in calls]
    uses.append({"type": "tool_use", "id": f"web{i}", "name": "search_web", "input": {"query": "x"}})
    results.append({"type": "tool_result", "tool_use_id": f"web{i}", "content": "Error: search failed", "is_error": True})
    return [
        {"role": "user", "content": [{"type": "text", "text": f"turn {i}"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": f"s{i}"}, *uses]},
        {"role": "user", "content": results},
        {"role": "assistant", "content": [{"type": "text", "text": f"answer {i}"}]},
    ]


async def test_cleared_browser_results_keep_their_state():
    """A cleared member result must stay a valid toolset result: a tab member answers with exactly one
    browser_state block (the plain marker string was refused), the others keep theirs behind the marker."""
    messages = _member_turn(0) + _member_turn(1)
    assert_valid_member_results(messages)

    media_only = copy.deepcopy(messages)  # clear_old_media alone: the picture goes, the state stays
    assert T.clear_old_media(media_only, keep_turns=1) == 1
    assert media_only[2]["content"][1]["content"] == [
        {"type": "text", "text": "1280x800"}, {"type": "text", "text": T.SCREENSHOT_CLEARED_MARKER}, STATE]
    assert_valid_member_results(media_only)

    stats = await T.prune(messages, max_context_tokens=10, keep_tool_results_turns=1)
    assert stats["cleared"] == 5 and stats["thinking_stripped"] == 2  # the tab members had nothing to clear
    navigate, screenshot, list_tabs, new_tab, scroll, switch_tab, web = messages[2]["content"]
    marker = {"type": "text", "text": T.CLEARED_MARKER}
    assert navigate["content"] == [marker, STATE] and screenshot["content"] == [marker, STATE]
    assert list_tabs["content"] == [STATE] and new_tab["content"] == [OPENED]
    assert scroll["content"] == T.CLEARED_MARKER and "is_error" not in scroll
    assert switch_tab["content"] == T.CLEARED_MARKER and switch_tab["is_error"] is True  # still an error: no state needed
    assert all(r["toolset_name"] == "browser" for r in (navigate, screenshot, list_tabs, new_tab, scroll, switch_tab))
    assert web == {"type": "tool_result", "tool_use_id": "web0", "content": T.CLEARED_MARKER}  # a normal tool as before
    assert messages[6]["content"] == _member_turn(1)[2]["content"]  # the kept turn is untouched
    assert_valid_member_results(messages)
    assert T.clear_old_tool_results(messages, keep_turns=1) == 0  # idempotent


# ---- ChatSender photo helpers --------------------------------------------------------------------
class FakeBot:
    def __init__(self, edit_error=None):
        self.calls: list[tuple[str, dict]] = []
        self.edit_error = edit_error

    async def send_photo(self, **kw):
        self.calls.append(("send_photo", kw))
        return SimpleNamespace(message_id=1)

    async def edit_message_media(self, **kw):
        self.calls.append(("edit_message_media", kw))
        if self.edit_error is not None:
            raise self.edit_error
        return True


async def test_sender_photo_helpers():
    from aiogram.exceptions import TelegramBadRequest
    from aiogram.methods import EditMessageMedia
    from aiogram.types import BufferedInputFile, InputMediaPhoto

    from bot.sender import ChatSender

    bot = FakeBot()
    sender = ChatSender(bot, 5, thread_id=9)
    await sender.send_photo(b"\xff\xd8\xffdata", filename="browser.jpg", caption="hi", silent=True)
    name, kw = bot.calls[0]
    assert kw["disable_notification"] is True and kw["message_thread_id"] == 9
    assert isinstance(kw["photo"], BufferedInputFile) and kw["photo"].filename == "browser.jpg"
    await sender.send_photo(b"x", caption="loud")
    assert "disable_notification" not in bot.calls[1][1]

    message = SimpleNamespace(message_id=77)
    assert await sender.edit_photo(message, b"\xff\xd8\xffnew", filename="browser.jpg", caption="c") is message
    _, kw = bot.calls[2]
    assert kw["chat_id"] == 5 and kw["message_id"] == 77
    assert isinstance(kw["media"], InputMediaPhoto) and kw["media"].caption == "c"

    method = EditMessageMedia(media=InputMediaPhoto(media="x"), chat_id=5, message_id=77)
    not_modified = TelegramBadRequest(method=method, message="Bad Request: message is not modified")
    assert await ChatSender(FakeBot(not_modified), 5).edit_photo(message, b"x") is None
    assert await ChatSender(FakeBot(RuntimeError("down")), 5).edit_photo(message, b"x") is None


def test_the_patcher_never_touches_the_browser_providers():
    """Like the other registered providers, these run in the bot process; the browser itself is in
    its own container (SessionManager). A patcher rebinding one of their methods would impose its
    own signature on it (that is how image editing once broke)."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    for name in ("BrowserTools", "SandboxTools", "browser_tools", "execute_member", "browser_confirm",
                 "browser_show_screenshot", "sandbox_status", "after_batch"):
        assert name not in source, name


def test_the_real_sessions_module_has_what_the_agent_reads():
    """Contract with secure_container.sessions (the agent only looks it up in sys.modules)."""
    sessions = pytest.importorskip("secure_container.sessions")
    for name in ("SESSION_IDLE_SECONDS", "SESSION_WARN_BEFORE_SECONDS", "SESSION_MEM_MB", "SESSIONS_ENABLED"):
        assert hasattr(sessions, name), name
    assert B.is_capacity_error(sessions.CapacityError("The server is short on memory"))
    assert callable(sessions.get_manager)
    for method in ("browser_call", "browser_peek", "status"):
        assert callable(getattr(sessions.SessionManager, method)), method
