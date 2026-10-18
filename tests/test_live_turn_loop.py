"""Live turns in the agent core: the inbox and background tool calls.

``AgentAnthropic.generate_response`` drains the turn's inbox (``TurnInbox``, bot/inbox.py) after
each tool batch and sends what it holds BEHIND the tool results in the same user message, stored
exactly as sent (transcript and legacy mode); never before the first call, never after the last.
``run_tool_batch`` hands a normal call to the ``BackgroundHost`` (bot/background.py) once it ran
``TOOL_BACKGROUND_AFTER_SECONDS``, or ``BACKGROUND_ON_MESSAGE_AFTER`` while a user message waits:
the result becomes a placeholder, the host's task still yields the tool's raw result, and a Stop
cancels only the calls that were not handed over. ``NEVER_BACKGROUND`` tools and toolset members
always run in the foreground. Without inbox and host (tests, subagents, other callers) nothing
changes. The bot side is faked here; only the protocols in agents.task_tools are used.
"""
import asyncio
import copy
import importlib
import json
import os
import re
import threading
import time
from types import SimpleNamespace

import anthropic
import pytest

from agents import task_tools
from agents import transcript as T
from agents.trace import ToolTrace, TurnBudget

USER_TAG = '<user_message sent_during_turn="true" time="2026-10-18 12:01 UTC">'
PIXEL = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    module = importlib.import_module("agents.main")
    monkeypatch.setattr(module, "_thinking_binding_refused", False)
    monkeypatch.setattr(module.stats_tracker, "track_tool_used", lambda uid, name: None)
    return module


# ---- fakes of the bot side (the protocols in agents.task_tools) ---------------------------------
def user_block(text):
    return {"type": "text", "text": f"{USER_TAG}\n{text}\n</user_message>"}


def notification_block(task_id, result="ok"):
    return {"type": "text", "text": f'<task_notification id="{task_id}" tool="run_command" status="completed" elapsed="95s">\n<result>\n{result}\n</result>\n</task_notification>'}


class FakeInbox:
    """Items in arrival order; ``wait`` wakes on a delivery; ``drain`` hands everything over."""

    def __init__(self):
        self.items: list[tuple[str, list[dict], str]] = []  # (kind, blocks, text view)
        self.drains = 0
        self.waits = 0
        self.texts: list[str] = []
        self.event = asyncio.Event()

    def deliver_user(self, text, *media):
        self.items.append(("message", [user_block(text), *media], text))
        self.event.set()

    def deliver_task(self, task_id, result="ok"):
        self.items.append(("task", [notification_block(task_id, result)], ""))
        self.event.set()

    def reset(self):
        self.items.clear()
        self.event.clear()

    def has_user_messages(self):
        return any(kind == "message" for kind, _, _ in self.items)

    def has_pending(self):
        return bool(self.items)

    async def drain(self):
        self.drains += 1
        blocks = [b for _, item, _ in self.items for b in item]
        self.texts += [text for kind, _, text in self.items if kind == "message"]
        self.items.clear()
        self.event.clear()
        return blocks

    async def wait(self, timeout):
        self.waits += 1
        if not self.items:
            try:
                await asyncio.wait_for(self.event.wait(), timeout)
            except asyncio.TimeoutError:
                return "timeout"
        return "message" if self.has_user_messages() else "task"

    @property
    def drained_user_texts(self):
        return list(self.texts)


class FakeHost:
    def __init__(self, cap=4):
        self.cap = cap
        self.adopted: list[dict] = []
        self.t0 = time.monotonic()

    def can_adopt(self):
        return sum(1 for a in self.adopted if not a["task"].done()) < self.cap

    def adopt(self, task, **kw):
        task_id = f"t{len(self.adopted) + 1}"
        self.adopted.append({"id": task_id, "task": task, "at": time.monotonic() - self.t0, **kw})
        return task_id

    def list(self):
        return [{"id": a["id"], "tool": a["tool_name"], "args": a["display_args"],
                 "status": "running" if not a["task"].done() else "done", "elapsed_s": 1} for a in self.adopted]

    def cancel(self, task_id):
        for a in self.adopted:
            if a["id"] == task_id and not a["task"].done():
                a["task"].cancel()
                return True
        return False


class Tools:
    """An awaited provider: ``slow`` sleeps ``delay``, ``boom`` raises after it, ``pic`` returns
    blocks; the NEVER_BACKGROUND names behave like ``slow``. Records cancellations."""

    NAMES = ["slow", "slow_b", "boom", "pic", "ask_user_choice", "wait_for_tasks", "send_user_telegram_message"]

    def __init__(self):
        self.tools_schema = [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}} for n in self.NAMES]
        self.cancelled: list[str] = []
        self.finished: list[str] = []
        self.on_start = None

    async def execute_tool(self, name, args):
        if self.on_start is not None:
            self.on_start(name, args)
        try:
            await asyncio.sleep(args.get("delay", 0.01))
        except asyncio.CancelledError:
            self.cancelled.append(name)
            raise
        if name == "boom":
            raise RuntimeError("tool exploded")
        self.finished.append(name)
        if name == "pic":
            return [{"type": "text", "text": "a picture"}, PIXEL]
        return f"{name} done"


class SlowOnly(Tools):
    NAMES = ["slow"]


class BlockingTools:
    """A provider registered with blocking=True: execute_tool runs in asyncio.to_thread."""

    def __init__(self):
        self.tools_schema = [{"name": "crunch", "description": "c", "input_schema": {"type": "object", "properties": {}}}]
        self.threads: list[str] = []

    def execute_tool(self, name, args):
        self.threads.append(threading.current_thread().name)
        time.sleep(args.get("delay", 0.01))
        return f"crunched {args.get('n')}"


class FakeBrowser:
    toolset_name = "browser"
    toolset = {"type": "browser_toolset_20260801"}
    tools_schema = []

    async def execute_member(self, name, tool_input):
        await asyncio.sleep(tool_input.get("delay", 0.01))
        return [{"type": "text", "text": f"{name} ok"}], False


def call(tool_id, name, **args):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=args)


def text(t):
    return SimpleNamespace(type="text", text=t)


def thinking(sig):
    return SimpleNamespace(type="thinking", thinking=f"reasoning {sig}", signature=sig)


def reply(*blocks, stop="end_turn"):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop, input_transformations=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeStream:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.outcome


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(stream=self._stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return FakeStream(self.script.pop(0))


def settings(changes=None):
    """The default settings with ``{(category, key): value}`` changes."""
    from bot.settings import DEFAULT_SETTINGS

    s = copy.deepcopy(DEFAULT_SETTINGS)
    for (category, key), value in (changes or {}).items():
        s[category][key] = value
    return s


def make_agent(agents_main, script=(), inbox=None, host=None):
    agent = agents_main.AgentAnthropic(user_id="7")
    agent.client = FakeClient(script)
    agent.user_interactions = Tools()  # an awaited slot
    agent.inbox = inbox
    agent.background = host
    return agent


QUESTION = [{"role": "user", "content": [{"type": "text", "text": "do it"}]}]


async def run_loop(agent, user_settings=None, update_status=None):
    return await agent.generate_response(
        messages=copy.deepcopy(QUESTION), system_role="sys", question="do it",
        user_settings=user_settings or settings(), update_status=update_status, trace=ToolTrace(),
    )


# ---- the inbox in the loop ------------------------------------------------------------------------
async def test_inbox_blocks_go_behind_the_tool_results_in_the_same_message(agents_main):
    inbox = FakeInbox()
    agent = make_agent(agents_main, [reply(text("checking"), call("c1", "slow"), stop="tool_use"), reply(text("done"))], inbox)
    agent.user_interactions.on_start = lambda name, args: inbox.deliver_user("also check B", PIXEL)
    response, messages = await run_loop(agent)

    assert response == "done"
    first, second = agent.client.requests
    assert first["messages"] == QUESTION  # nothing before the first call
    sent = second["messages"][-1]
    assert sent == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "slow done"},
        user_block("also check B"), PIXEL,
    ]}
    assert messages[2] == sent and messages[: len(second["messages"])] == second["messages"]  # what is returned is what was sent
    assert inbox.drains == 1 and inbox.drained_user_texts == ["also check B"]


async def test_nothing_is_injected_before_the_first_call_after_end_turn_or_into_judge_notices(agents_main):
    inbox = FakeInbox()
    inbox.deliver_user("early")  # arrived while the model was thinking: waits for a tool boundary
    agent = make_agent(agents_main, [reply(text("draft")), reply(text("final"))], inbox)
    verdicts = iter([False, True])

    async def judge(question, answer):
        return next(verdicts)

    agent.judge_response = judge
    response, messages = await run_loop(agent, settings({("judge", "enabled"): True}))
    assert response == "final"
    assert inbox.drains == 0 and inbox.has_user_messages()  # left for the bot side to requeue
    assert all(USER_TAG not in json.dumps(r["messages"]) for r in agent.client.requests)


async def test_the_judge_and_the_critique_see_the_messages_read_mid_turn(agents_main):
    """The answer must cover what the user added during the turn, so both check it against that too."""
    seen: list[tuple[str, str]] = []

    async def judge(question, answer):
        seen.append(("judge", question))
        return True

    async def critique(question, answer, dialog_history=[], trace=None):
        seen.append(("critique", question))
        return None

    both = settings({("judge", "enabled"): True, ("critique", "enabled"): True})
    inbox = FakeInbox()
    agent = make_agent(agents_main, [reply(call("c1", "slow"), stop="tool_use"), reply(text("A and B done"))], inbox)
    agent.user_interactions.on_start = lambda name, args: inbox.deliver_user("also B")
    agent.judge_response, agent.critique_response = judge, critique
    await run_loop(agent, both)
    expected = "do it\n\n[Sent while I was working]\nalso B"
    assert seen == [("critique", expected), ("judge", expected)]

    seen.clear()  # nothing read: the bare question, as before
    plain = make_agent(agents_main, [reply(text("done"))], FakeInbox())
    plain.judge_response, plain.critique_response = judge, critique
    await run_loop(plain, both)
    assert seen == [("critique", "do it"), ("judge", "do it")]


async def test_the_forced_final_call_injects_nothing(agents_main):
    inbox = FakeInbox()
    agent = make_agent(agents_main, [reply(call("c1", "slow"), stop="tool_use"), reply(text("summary"))], inbox)
    agent.user_interactions.on_start = lambda name, args: inbox.deliver_task("t9")
    response, messages = await run_loop(agent, settings({("tools", "max_iteration"): 1}))
    assert response == "summary" and agent.limit_reached
    assert inbox.drains == 1  # after the one batch, never for the SYSTEM NOTICE
    notice = agent.client.requests[1]["messages"][-1]
    assert notice["content"][0]["text"].startswith("SYSTEM NOTICE") and len(notice["content"]) == 1
    assert agent.client.requests[1]["messages"][-3]["content"][1] == notification_block("t9")


async def test_status_reads_your_message_once_per_injection(agents_main):
    inbox = FakeInbox()
    agent = make_agent(agents_main, [
        reply(call("c1", "slow"), stop="tool_use"), reply(call("c2", "slow"), stop="tool_use"), reply(text("done")),
    ], inbox)
    deliveries = iter([lambda: (inbox.deliver_user("one"), inbox.deliver_user("two"), inbox.deliver_task("t1")),
                       lambda: inbox.deliver_task("t2")])
    agent.user_interactions.on_start = lambda name, args: next(deliveries)()
    updates = []

    async def update_status(step, details, iteration, critique):
        updates.append((step, details))

    await run_loop(agent, update_status=update_status)
    reading = [u for u in updates if u[0] == "reading your message"]
    assert reading == [("reading your message", "Reading your 2 new messages")]  # once; a notification alone says nothing
    i = updates.index(reading[0])
    assert updates[i + 1][0] == "executing-tools" and updates[i + 1][1] == "Executing tools"  # the next call's own status was skipped
    second = agent.client.requests[2]["messages"][-1]["content"]
    assert [b.get("type") for b in second] == ["tool_result", "text"] and second[1] == notification_block("t2")


async def test_an_inbox_without_user_tags_still_counts_drained_messages(agents_main):
    """The status falls back on the inbox's own bookkeeping when the blocks carry another format."""
    inbox = FakeInbox()
    agent = make_agent(agents_main, [reply(call("c1", "slow"), stop="tool_use"), reply(text("done"))], inbox)

    def plain_message(name, args):
        inbox.items.append(("message", [{"type": "text", "text": "hello there"}, {"type": "text", "text": "  "}], "hello there"))

    agent.user_interactions.on_start = plain_message
    updates = []

    async def update_status(step, details, iteration, critique):
        updates.append((step, details))

    await run_loop(agent, update_status=update_status)
    assert ("reading your message", "Reading your new message") in updates
    assert agent.client.requests[1]["messages"][-1]["content"][1:] == [{"type": "text", "text": "hello there"}]  # empty text dropped


async def test_a_failing_inbox_never_ends_the_turn(agents_main):
    class Broken(FakeInbox):
        async def drain(self):
            raise RuntimeError("inbox closed")

        def has_user_messages(self):
            raise RuntimeError("inbox closed")

    agent = make_agent(agents_main, [reply(call("c1", "slow"), stop="tool_use"), reply(text("done"))], Broken(), FakeHost())
    response, messages = await run_loop(agent)
    assert response == "done" and messages[2]["content"] == [{"type": "tool_result", "tool_use_id": "c1", "content": "slow done"}]


# ---- background calls -----------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(
    r"^Still running in the background as task (t\d+) \(started \d\d:\d\d:\d\d UTC\)\. Its result will arrive as a "
    r"<task_notification> after a later tool call, or as a new message if this turn has ended\. Meanwhile you can "
    r"answer the user, use other tools, or call wait_for_tasks\.$"
)


async def test_a_long_call_moves_to_the_background(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.15)
    host = FakeHost()
    agent = make_agent(agents_main, host=host)
    agent.max_tool_result_chars = 5000
    trace = ToolTrace()
    refreshed = []

    async def refresh(details):
        refreshed.append(details)

    t0, wall = time.monotonic(), time.time()
    results = await agent.run_tool_batch([call("c1", "slow", delay=1.0, command="sleep 1"), call("c2", "slow_b", delay=0.02)], trace=trace, refresh=refresh)
    assert time.monotonic() - t0 < 0.7, "the turn waited for the slow call"

    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert PLACEHOLDER_RE.match(results[0]["content"]).group(1) == "t1" and "is_error" not in results[0]
    assert results[1] == {"type": "tool_result", "tool_use_id": "c2", "content": "slow_b done"}
    assert [(c.name, c.ok, c.preview) for c in trace.calls] == [("slow", True, "moved to the background as t1"), ("slow_b", True, "slow_b done")]
    assert trace.running == 0 and "slow moved to the background as t1" in refreshed

    adopted = host.adopted[0]
    assert adopted["tool_name"] == "slow" and adopted["tool_use_id"] == "c1" and adopted["display_args"] == "sleep 1"
    assert adopted["max_chars"] == 5000 and abs(adopted["started"] - wall) < 0.5  # wall clock (time.time()) at the call's start
    assert 0.14 <= adopted["at"] < 0.6
    assert await adopted["task"] == "slow done"  # the host gets the tool's raw result
    assert agent.user_interactions.cancelled == []


def test_the_placeholder_names_the_start_in_utc(agents_main):
    from datetime import datetime, timezone

    started = datetime(2026, 10, 18, 12, 1, 5, tzinfo=timezone.utc).timestamp()
    assert agents_main.background_placeholder("t3", started) == (
        "Still running in the background as task t3 (started 12:01:05 UTC). Its result will arrive as a <task_notification> "
        "after a later tool call, or as a new message if this turn has ended. Meanwhile you can answer the user, use other "
        "tools, or call wait_for_tasks."
    )


async def test_the_adopted_task_yields_the_raw_result_or_the_exception(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.1)
    host = FakeHost()
    agent = make_agent(agents_main, host=host)
    agent.max_tool_result_chars = 3  # the host caps; the task itself is untouched
    results = await agent.run_tool_batch([call("c1", "boom", delay=0.4), call("c2", "pic", delay=0.4)])
    assert all(PLACEHOLDER_RE.match(r["content"]) for r in results)
    with pytest.raises(RuntimeError, match="tool exploded"):
        await host.adopted[0]["task"]
    assert await host.adopted[1]["task"] == [{"type": "text", "text": "a picture"}, PIXEL]


async def test_a_quick_call_is_the_same_with_or_without_a_host(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 5)
    blocks = [call("c1", "slow"), call("c2", "boom"), call("c3", "pic")]
    plain_trace, live_trace = ToolTrace(), ToolTrace()
    plain = await make_agent(agents_main).run_tool_batch(blocks, trace=plain_trace)
    host = FakeHost()
    live = await make_agent(agents_main, inbox=FakeInbox(), host=host).run_tool_batch(blocks, trace=live_trace)
    assert live == plain and host.adopted == []
    assert [(c.name, c.ok, c.preview) for c in live_trace.calls] == [(c.name, c.ok, c.preview) for c in plain_trace.calls]


async def test_a_waiting_user_message_moves_a_call_after_a_few_seconds(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)  # the clock alone never moves it
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.2)
    monkeypatch.setattr(agents_main, "BACKGROUND_POLL_SECONDS", 5.0)  # only the inbox can wake the watcher in time
    inbox, host = FakeInbox(), FakeHost()
    agent = make_agent(agents_main, inbox=inbox, host=host)

    async def write_at(seconds, words):
        await asyncio.sleep(seconds)
        inbox.deliver_user(words)

    # a message after the minimum: the call goes at once
    writer = asyncio.get_running_loop().create_task(write_at(0.35, "how is it going?"))
    t0 = time.monotonic()
    results = await agent.run_tool_batch([call("c1", "slow", delay=3.0)])
    await writer
    assert PLACEHOLDER_RE.match(results[0]["content"]) and 0.3 <= time.monotonic() - t0 < 1.0
    assert inbox.waits >= 1  # it listened instead of polling

    # a message before the minimum: the call still gets its few seconds
    inbox.reset()
    host.t0 = time.monotonic()
    writer = asyncio.get_running_loop().create_task(write_at(0.02, "and another thing"))
    results = await agent.run_tool_batch([call("c2", "slow", delay=3.0)])
    await writer
    assert PLACEHOLDER_RE.match(results[0]["content"]) and 0.19 <= host.adopted[1]["at"] < 1.0

    # nobody writes and the clock trigger is off: the call runs to its end
    inbox.reset()
    results = await agent.run_tool_batch([call("c3", "slow", delay=0.3)])
    assert results[0]["content"] == "slow done" and len(host.adopted) == 2


async def test_a_message_still_being_prepared_moves_no_call_until_it_is_ready(agents_main, monkeypatch):
    """With the real inbox: a message still being prepared (a video being described can take minutes)
    is no reason to move calls to the background, since the agent cannot read it yet; once it is
    ready, the watcher sees it at its next look and the call goes."""
    from bot import inbox as real_inbox

    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.2)
    monkeypatch.setattr(agents_main, "BACKGROUND_POLL_SECONDS", 0.1)
    monkeypatch.setattr(real_inbox, "DRAIN_WAIT", 0.05)
    real_inbox.reset()
    box = real_inbox.open_inbox("7", None)
    preparing = asyncio.get_running_loop().create_future()
    box.deliver(real_inbox.InboxItem(kind=real_inbox.USER, future=preparing))
    host = FakeHost()
    agent = make_agent(agents_main, inbox=box, host=host)
    for i in range(2):
        results = await agent.run_tool_batch([call(f"c{i}", "slow", delay=0.5)])
        assert results[0]["content"] == "slow done" and await box.drain() == []
    assert host.adopted == []

    asyncio.get_running_loop().call_later(0.3, preparing.set_result, real_inbox.Prepared(prompt="also B"))
    t0 = time.monotonic()
    results = await agent.run_tool_batch([call("c9", "slow", delay=3.0)])
    assert PLACEHOLDER_RE.match(results[0]["content"]) and time.monotonic() - t0 < 1.0
    assert "also B" in (await box.drain())[0]["text"]
    host.adopted[0]["task"].cancel()
    real_inbox.finish(box, discard=True)


async def test_a_pending_notification_does_not_spin_the_watcher(agents_main, monkeypatch):
    """inbox.wait returns at once while a notification is pending: the watcher polls instead."""
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.1)
    monkeypatch.setattr(agents_main, "BACKGROUND_POLL_SECONDS", 0.05)
    inbox, host = FakeInbox(), FakeHost()
    inbox.deliver_task("t7")
    agent = make_agent(agents_main, inbox=inbox, host=host)

    async def write_later():
        await asyncio.sleep(0.25)
        inbox.deliver_user("hi")

    writer = asyncio.get_running_loop().create_task(write_later())
    results = await agent.run_tool_batch([call("c1", "slow", delay=2.0)])
    await writer
    assert PLACEHOLDER_RE.match(results[0]["content"]) and 0.24 <= host.adopted[0]["at"] < 1.0
    assert inbox.waits == 0  # never awaited while something was pending


async def test_an_inbox_that_returns_at_once_does_not_spin_the_watcher(agents_main, monkeypatch):
    """A closed inbox (or any wait that wakes with nothing pending) makes the watcher poll."""
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)
    monkeypatch.setattr(agents_main, "BACKGROUND_POLL_SECONDS", 0.1)

    class Closed(FakeInbox):
        async def wait(self, timeout):
            self.waits += 1
            return "timeout"

    inbox = Closed()
    agent = make_agent(agents_main, inbox=inbox, host=FakeHost())
    results = await agent.run_tool_batch([call("c1", "slow", delay=0.5)])
    assert results[0]["content"] == "slow done" and inbox.waits == 1


async def test_a_full_host_keeps_the_call_until_it_has_room(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.1)
    monkeypatch.setattr(agents_main, "BACKGROUND_POLL_SECONDS", 0.05)
    host = FakeHost(cap=0)
    agent = make_agent(agents_main, host=host)
    results = await agent.run_tool_batch([call("c1", "slow", delay=0.3)])
    assert results[0]["content"] == "slow done" and host.adopted == []

    async def make_room():
        await asyncio.sleep(0.3)
        host.cap = 4

    host.t0 = time.monotonic()
    room = asyncio.get_running_loop().create_task(make_room())
    results = await agent.run_tool_batch([call("c2", "slow", delay=2.0)])
    await room
    assert PLACEHOLDER_RE.match(results[0]["content"]) and 0.29 <= host.adopted[0]["at"] < 1.0


async def test_a_host_that_refuses_leaves_the_call_in_the_foreground(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.05)
    host = FakeHost()
    attempts = []

    def refuse(task, **kw):
        attempts.append(kw["tool_name"])
        raise RuntimeError("registry closed")

    host.adopt = refuse
    agent = make_agent(agents_main, host=host)
    results = await agent.run_tool_batch([call("c1", "slow", delay=0.3)])
    assert results[0]["content"] == "slow done" and attempts == ["slow"]  # asked once, then waited


def test_never_background_names(agents_main):
    assert agents_main.NEVER_BACKGROUND >= {
        "ask_user_choice", "browser_confirm", "browser_show_screenshot", "wait_for_tasks", "list_tasks", "cancel_task",
        "send_user_telegram_message", "set_message_reaction", "send_voice_message",
    }
    assert agents_main.TOOL_BACKGROUND_AFTER_SECONDS == 30.0 or os.getenv("TOOL_BACKGROUND_AFTER_SECONDS")
    assert agents_main.BACKGROUND_ON_MESSAGE_AFTER == 3.0


@pytest.mark.parametrize("raw, value", [(None, 30.0), ("", 30.0), ("45", 45.0), ("0", 0.0), ("-5", 0.0), ("soon", 30.0)])
def test_the_background_delay_comes_from_the_environment(agents_main, monkeypatch, raw, value):
    if raw is None:
        monkeypatch.delenv("TOOL_BACKGROUND_AFTER_SECONDS", raising=False)
    else:
        monkeypatch.setenv("TOOL_BACKGROUND_AFTER_SECONDS", raw)
    assert agents_main._env_seconds("TOOL_BACKGROUND_AFTER_SECONDS", 30.0) == value


async def test_never_background_tools_and_toolset_members_stay_in_the_foreground(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.05)
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.05)
    inbox, host = FakeInbox(), FakeHost()
    inbox.deliver_user("hurry up")
    agent = make_agent(agents_main, inbox=inbox, host=host)
    agent.register_provider(FakeBrowser())
    member = anthropic.types.ToolUseBlock(id="b1", name="navigate", input={"delay": 0.3}, type="tool_use", toolset_name="browser")
    blocks = [call("c1", "ask_user_choice", delay=0.3), call("c2", "wait_for_tasks", delay=0.3),
              call("c3", "send_user_telegram_message", delay=0.3), member]
    results = await agent.run_tool_batch(blocks)
    assert host.adopted == []
    assert [r["content"] for r in results[:3]] == ["ask_user_choice done", "wait_for_tasks done", "send_user_telegram_message done"]
    assert results[3]["toolset_name"] == "browser" and results[3]["content"] == [{"type": "text", "text": "navigate ok"}]


async def test_stop_cancels_running_calls_but_never_adopted_ones(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.1)
    host = FakeHost()
    agent = make_agent(agents_main, host=host)
    tools = agent.user_interactions
    batch = asyncio.get_running_loop().create_task(agent.run_tool_batch([
        call("c1", "slow", delay=0.6),             # handed over at 0.1 s
        call("c2", "ask_user_choice", delay=5.0),  # never handed over: the batch waits for it
    ]))
    await asyncio.sleep(0.3)
    assert len(host.adopted) == 1 and not batch.done()
    batch.cancel()  # the user's Stop cancels the turn's task
    with pytest.raises(asyncio.CancelledError):
        await batch
    await asyncio.sleep(0.05)
    assert tools.cancelled == ["ask_user_choice"]
    adopted = host.adopted[0]["task"]
    assert not adopted.cancelled() and await adopted == "slow done"  # it ran on to its end
    assert tools.finished == ["slow"]


async def test_stop_cancels_a_watched_call_that_was_not_handed_over(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 10)
    inbox, host = FakeInbox(), FakeHost()
    agent = make_agent(agents_main, inbox=inbox, host=host)
    batch = asyncio.get_running_loop().create_task(agent.run_tool_batch([call("c1", "slow", delay=5.0)]))
    await asyncio.sleep(0.15)
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch
    await asyncio.sleep(0.05)
    assert agent.user_interactions.cancelled == ["slow"] and host.adopted == []


async def test_a_call_in_a_worker_thread_is_handed_over_and_survives_stop(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.1)
    host = FakeHost()
    agent = make_agent(agents_main, host=host)
    worker = BlockingTools()
    agent.register_provider(worker, blocking=True)
    batch = asyncio.get_running_loop().create_task(agent.run_tool_batch([call("c1", "crunch", delay=0.5, n=7)]))
    results = await batch
    assert PLACEHOLDER_RE.match(results[0]["content"])
    assert worker.threads and worker.threads[0] != threading.current_thread().name  # still off the event loop
    assert await host.adopted[0]["task"] == "crunched 7"


async def test_the_loop_answers_with_the_placeholder_and_reads_the_message(agents_main, monkeypatch):
    """End to end: the user writes during a slow call, the call is moved, the message is read
    behind the placeholder, and the turn goes on without waiting for the call."""
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.1)
    inbox, host = FakeInbox(), FakeHost()
    agent = make_agent(agents_main, [
        reply(thinking("s1"), call("c1", "slow", delay=1.5), stop="tool_use"),
        reply(thinking("s2"), call("c2", "send_user_telegram_message", text="still running, B is next"), stop="tool_use"),
        reply(text("A is running in the background; here is B.")),
    ], inbox, host)

    async def write():
        await asyncio.sleep(0.2)
        inbox.deliver_user("also do B")

    writer = asyncio.get_running_loop().create_task(write())
    t0 = time.monotonic()
    response, messages = await run_loop(agent)
    await writer
    assert time.monotonic() - t0 < 1.2 and response.startswith("A is running")
    second = agent.client.requests[1]["messages"][-1]["content"]
    assert second[0]["tool_use_id"] == "c1" and PLACEHOLDER_RE.match(second[0]["content"])
    assert second[1:] == [user_block("also do B")]
    assert agent.client.requests[1]["messages"][-2]["content"][0] == {"type": "thinking", "thinking": "reasoning s1", "signature": "s1"}
    assert messages[: len(agent.client.requests[2]["messages"])] == agent.client.requests[2]["messages"]
    assert await host.adopted[0]["task"] == "slow done"


# ---- defaults, subagents --------------------------------------------------------------------------
def make_chain(agents_main, monkeypatch, script, inbox=None, background=None, user_settings=None):
    s = user_settings or settings()
    kwargs = {}
    if inbox is not None:
        kwargs["inbox"] = inbox
    if background is not None:
        kwargs["background"] = background
    chain = agents_main.ChainOfThoughtAgent(user_id="42", user_settings=s, **kwargs)
    client = FakeClient(script)
    chain.agent.client = client
    saved = []

    async def complex_(question, history=None):
        return "complex"

    async def save(question, response, messages):
        saved.append(question)
        return ""

    async def context(question, update_status=None, messages=None):
        return "<context>now</context>", "<context>now</context>"

    monkeypatch.setattr(chain, "_classify_complexity", complex_)
    monkeypatch.setattr(chain, "_save_conversation", save)
    monkeypatch.setattr(chain, "_request_context", context)
    return chain, client, saved


def tool_names(chain):
    return [t.get("name") for t in chain.agent.get_tools_schema()]


def test_defaults_are_unchanged_without_inbox_or_host(agents_main, monkeypatch):
    bare = agents_main.AgentAnthropic(user_id="7")
    assert bare.inbox is None and bare.background is None
    plain, _, _ = make_chain(agents_main, monkeypatch, [])
    assert plain.agent.inbox is None and plain.agent.background is None and plain.task_tools is None
    assert not set(tool_names(plain)) & task_tools.TASK_TOOL_NAMES
    assert "<live_turn>" not in plain._provider_notes()

    inbox, host = FakeInbox(), FakeHost()
    live, _, _ = make_chain(agents_main, monkeypatch, [], inbox, host)
    assert live.agent.inbox is inbox and live.agent.background is host and live.inbox is inbox
    assert tool_names(live) == tool_names(plain) + ["wait_for_tasks", "list_tasks", "cancel_task"]
    notes = live._provider_notes()
    assert notes.count("<live_turn>") == 1 and notes == make_chain(agents_main, monkeypatch, [], FakeInbox(), FakeHost())[0]._provider_notes()


async def test_without_a_host_no_call_ever_goes_to_the_background(agents_main, monkeypatch):
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.01)
    inbox = FakeInbox()
    inbox.deliver_user("hello")
    agent = make_agent(agents_main, inbox=inbox)  # an inbox alone
    results = await agent.run_tool_batch([call("c1", "slow", delay=0.2)])
    assert results == [{"type": "tool_result", "tool_use_id": "c1", "content": "slow done"}]


async def test_task_tools_in_the_loop(agents_main, monkeypatch):
    """wait_for_tasks is a normal tool call: the notification it waited for follows its result."""
    inbox, host = FakeInbox(), FakeHost()
    chain, client, _ = make_chain(agents_main, monkeypatch, [
        reply(call("w1", "wait_for_tasks", timeout_seconds=30), stop="tool_use"), reply(text("t1 finished: ok")),
    ], inbox, host)
    host.adopted.append({"id": "t1", "task": asyncio.get_running_loop().create_future(), "tool_name": "run_command", "display_args": "make"})

    async def finish_later():
        await asyncio.sleep(0.1)
        host.adopted[0]["task"].set_result("ok")
        inbox.deliver_task("t1", "build ok")

    finisher = asyncio.get_running_loop().create_task(finish_later())
    response, _ = await chain.generate_response("wait for the build")
    await finisher
    assert response == "t1 finished: ok"
    sent = client.requests[1]["messages"][-1]["content"]
    assert sent == [
        {"type": "tool_result", "tool_use_id": "w1", "content": "A background task finished; its notification follows."},
        notification_block("t1", "build ok"),
    ]


async def test_subagents_get_no_inbox_host_or_task_tools(agents_main, monkeypatch):
    from agents.subagent import SubagentTools

    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.05)
    inbox, host = FakeInbox(), FakeHost()
    parent = make_agent(agents_main, inbox=inbox, host=host)
    parent.user_interactions = SlowOnly()
    parent.register_provider(task_tools.TaskTools(host, inbox))
    other = SimpleNamespace(tools_schema=[{"name": "view_image"}])
    parent.register_provider(other)
    parent.budget = TurnBudget(limit=20)
    sub = SubagentTools(parent, settings())
    child = sub._child(agents_main.ANTHROPIC_MODEL, None, TurnBudget(limit=5))
    assert child.inbox is None and child.background is None
    assert [p for p, _ in child.extra_providers] == [other]
    assert not sub._available_tool_names() & task_tools.TASK_TOOL_NAMES
    assert not set(t["name"] for t in child.get_tools_schema()) & task_tools.TASK_TOOL_NAMES
    inbox.deliver_user("for the main agent")
    results = await child.run_tool_batch([call("c1", "slow", delay=0.2)])
    assert results[0]["content"] == "slow done" and host.adopted == [] and inbox.has_user_messages()


def test_a_subagent_never_gets_a_tool_that_talks_to_or_waits_for_the_user(agents_main, monkeypatch):
    """run_subagent may be moved to the background at any moment (even while the child waits for a
    tap): an open question after the turn ended would take the user's next text as its answer. So
    the child never has the NEVER_BACKGROUND tools, whatever subset was asked for."""
    chain, _, _ = make_chain(agents_main, monkeypatch, [], FakeInbox(), FakeHost())
    sub = chain.subagents
    user_facing = {"ask_user_choice", "send_user_telegram_message", "send_voice_message", "set_message_reaction"}
    assert user_facing <= set(tool_names(chain))  # the main agent keeps them
    child = sub._child(agents_main.ANTHROPIC_MODEL, None, TurnBudget(limit=5))
    names = {t["name"] for t in child.get_tools_schema()}
    assert not names & agents_main.NEVER_BACKGROUND
    assert {"schedule_reminder", "search_web", "view_image"} <= names  # everything else stays
    assert not sub._available_tool_names() & agents_main.NEVER_BACKGROUND
    subset = sub._child(agents_main.ANTHROPIC_MODEL, {"ask_user_choice", "search_web"}, TurnBudget(limit=5))
    assert subset.allowed_tools == {"search_web"}


async def test_a_backgrounded_subagents_tokens_are_recorded_as_they_arrive(agents_main, monkeypatch):
    """run_subagent moved to the background: the child's model calls reach the turn's trace after
    bot.agent_runner recorded the turn's usage; each one is recorded when it arrives, once."""
    from agents.subagent import SubagentTools
    from bot import agent_runner

    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0.1)

    async def slow_stream(self, kwargs, on_event=None):
        await asyncio.sleep(0.4)  # the child's model call
        return reply(text("REPORT"))

    monkeypatch.setattr(agents_main.AgentAnthropic, "_stream_message", slow_stream)
    host = FakeHost()
    parent = agents_main.AgentAnthropic(user_id="7")
    parent.background, parent.inbox = host, FakeInbox()
    parent.budget = TurnBudget(limit=20)
    parent.subagents = SubagentTools(parent, settings())
    trace = ToolTrace()
    parent.subagents.trace = trace
    results = await parent.run_tool_batch([call("c1", "run_subagent", task="research")], trace=trace)
    assert PLACEHOLDER_RE.match(results[0]["content"])

    rows: list[dict] = []
    tracker = SimpleNamespace(track_usage=lambda uid, **k: rows.append(k))
    agent_runner.record_usage(tracker, "7", trace)  # the turn ends now: nothing of the child's yet
    agent_runner.record_late_usage(tracker, "7", trace)
    assert rows == []
    assert await host.adopted[0]["task"] == "REPORT"
    assert [(r["model"], r["api_calls"], r["input_tokens"], r["output_tokens"]) for r in rows] == [(agents_main.ANTHROPIC_MODEL, 1, 10, 5)]


async def test_a_backgrounded_subagent_stops_editing_the_status(agents_main, monkeypatch):
    from agents.subagent import SubagentTools

    parent = agents_main.AgentAnthropic(user_id="7")
    parent.budget = TurnBudget(limit=20)
    sub = SubagentTools(parent, settings())
    seen = []

    async def parent_status(step, details, iteration, critique):
        seen.append(details)

    sub.update_status = parent_status
    moved = asyncio.Event()

    async def child_loop(self, **kw):
        await kw["update_status"](step="executing-tools", details="before", iteration=1, critique=0)
        await moved.wait()
        await kw["update_status"](step="executing-tools", details="after", iteration=2, critique=0)
        return "REPORT", []

    monkeypatch.setattr(agents_main.AgentAnthropic, "generate_response", child_loop)
    import contextvars

    state = task_tools.CallState("run_subagent")
    context = contextvars.copy_context()
    context.run(task_tools.bind_call, state)
    task = asyncio.get_running_loop().create_task(sub.run({"task": "research"}), context=context)
    await asyncio.sleep(0.02)
    state.background_id = "t1"  # what run_tool_batch does when it hands the call over
    moved.set()
    assert await task == "REPORT"
    assert seen == ["before"]

    # in the foreground it reports as before
    moved.set()
    assert await sub.run({"task": "research"}) == "REPORT" and seen == ["before", "before", "after"]


# ---- what is stored -----------------------------------------------------------------------------
def stored():
    return T.transcript_store.load("42", None)


async def test_transcript_stores_the_injected_message_exactly_as_sent(agents_main, monkeypatch):
    inbox, host = FakeInbox(), FakeHost()
    chain, client, saved = make_chain(agents_main, monkeypatch, [
        reply(thinking("s1"), text("checking"), call("c1", "get_time_in_timezone", timezone="Europe/Lisbon"), stop="tool_use"),
        reply(thinking("s2"), text("It is 12:00, and B is done too.")),
    ], inbox, host)

    async def run_tool(name, args):
        inbox.deliver_user("also B please", PIXEL)
        return "12:00"

    monkeypatch.setattr(chain.agent, "execute_tool", run_tool)
    response, _ = await chain.generate_response("what time is it in Lisbon?")
    assert response == "It is 12:00, and B is done too."

    sent = client.requests[-1]["messages"]
    assert sent[-1]["content"] == [{"type": "tool_result", "tool_use_id": "c1", "content": "12:00"}, user_block("also B please"), PIXEL]
    messages = stored().messages
    assert messages[: len(sent)] == sent  # stored == sent: no edit, no strip
    assert [b["signature"] for m in messages for b in m["content"] if b.get("type") == "thinking"] == ["s1"]
    assert T.message_text(messages[-1]) == response

    memory = "\n\n[Sent while I was working]\nalso B please"
    assert saved == [saved[0]] and saved[0].endswith("what time is it in Lisbon?" + memory)
    with open("data/42/short_term_memory/dialog_history.json", encoding="utf-8") as f:
        history = json.load(f)
    assert history[-2]["content"] == saved[0] and history[-1]["content"] == response

    # the next turn starts from exactly that history (append-only)
    again, client2, _ = make_chain(agents_main, monkeypatch, [reply(text("hi again"))], FakeInbox(), FakeHost())
    await again.generate_response("thanks")
    assert client2.requests[0]["messages"][: len(sent)] == sent


async def test_legacy_mode_injects_through_the_same_loop(agents_main, monkeypatch):
    inbox, host = FakeInbox(), FakeHost()
    s = settings({("transcript", "enabled"): False, ("semantic_search", "enabled"): False})
    chain, client, saved = make_chain(agents_main, monkeypatch, [
        reply(call("c1", "get_time_in_timezone", timezone="UTC"), stop="tool_use"), reply(text("12:00, and B.")),
    ], inbox, host, s)

    async def run_tool(name, args):
        inbox.deliver_user("and B")
        return "12:00"

    monkeypatch.setattr(chain.agent, "execute_tool", run_tool)
    response, thread_messages = await chain.generate_response("time?")
    assert response == "12:00, and B."
    assert client.requests[1]["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "c1", "content": "12:00"}, user_block("and B")]
    assert not T.transcript_store.exists("42", None)
    assert saved[0].endswith("time?\n\n[Sent while I was working]\nand B")
    with open("data/42/short_term_memory/dialog_history.json", encoding="utf-8") as f:
        assert json.load(f)[-2]["content"] == saved[0]


async def test_memory_keeps_the_bare_question_when_nothing_was_read(agents_main, monkeypatch):
    chain, _, saved = make_chain(agents_main, monkeypatch, [reply(text("hi"))], FakeInbox(), FakeHost())
    await chain.generate_response("hello")
    assert saved[0].endswith("hello") and "[Sent while I was working]" not in saved[0]
    plain, _, saved_plain = make_chain(agents_main, monkeypatch, [reply(text("hi"))])
    await plain.generate_response("hello")
    assert saved_plain[0].endswith("hello")
