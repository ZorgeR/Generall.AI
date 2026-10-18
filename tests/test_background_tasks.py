"""Tool calls moved to the background (bot/background.py): ids and the cap, the <task_notification>
format, delivery to the running turn of the task's topic or a "background task finished" turn
through the queue, notifications joining a queued one, cancel_task, /cancel (cancels them),
the Stop button (keeps them), shutdown, and run_turn handing the agent its inbox and host."""
import asyncio
import sys
import time
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest

import bot.agent_runner as agent_runner
import bot.limits as limits
import bot.streaming as streaming
from bot import background, inbox, rich, runtime
from bot.handlers import messages
from bot.queue import Job, QueueManager

USER = "42"
REAL_RUN_TURN = agent_runner.run_turn  # the ``turns`` fixture replaces the module's for notification turns
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


class FakeBot:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return SimpleNamespace(message_id=kw.get("message_id"))

    async def delete_message(self, **kw):
        return True


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None, track_usage=lambda *a, **k: None,
        get_user_action_count=lambda *a, **k: 0, track_message_received=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    monkeypatch.setattr(limits, "check_user_limits", lambda uid: (True, 0, 50))
    rich.reset()
    inbox.reset()
    background.reset()
    yield
    inbox.reset()
    background.reset()


@pytest.fixture
async def qm(monkeypatch):
    """The process queue (notification turns) and the handlers' queue, as one fresh QueueManager."""
    queue = QueueManager(max_concurrent_turns=4, turn_timeout=30)
    monkeypatch.setattr(runtime, "queue", queue)
    monkeypatch.setattr(messages, "queue", queue)
    yield queue
    await queue.shutdown()


@pytest.fixture
def turns(monkeypatch) -> list[dict]:
    """``run_turn`` as the notification job imports it: records its arguments."""
    seen: list[dict] = []

    async def fake(**kw):
        seen.append(kw)

    monkeypatch.setattr(agent_runner, "run_turn", fake)
    return seen


async def until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def host(thread_id=None) -> background.BackgroundHost:
    return background.host_for(USER, bot=FakeBot(), chat_id=42, thread_id=thread_id)


async def result_after(value, delay=0.0):
    await asyncio.sleep(delay)
    if isinstance(value, BaseException):
        raise value
    return value


def adopt(h, coro, tool="run_command", max_chars=None, started=None) -> str:
    return h.adopt(
        asyncio.create_task(coro), tool_name=tool, tool_use_id="toolu_1", display_args='{"command": "make"}',
        started=time.monotonic() if started is None else started, max_chars=max_chars,
    )


def notification_text(blocks) -> str:
    return blocks[0]["text"]


# ---- the host ------------------------------------------------------------------------------------
async def test_ids_increase_per_user_and_the_cap_is_four_running():
    h = host()
    gate = asyncio.Event()
    ids = [adopt(h, gate.wait()) for _ in range(4)]
    assert ids == ["t1", "t2", "t3", "t4"] and not h.can_adopt()
    entry = h.list()[0]
    assert entry["id"] == "t1" and entry["tool"] == "run_command" and entry["args"] == '{"command": "make"}'
    assert entry["status"] == "running" and entry["started"].tzinfo is not None and entry["elapsed_s"] >= 0
    gate.set()
    await until(lambda: h.can_adopt() and all(e["status"] == "done" for e in h.list()))
    assert adopt(h, result_after("ok")) == "t5"  # ids never repeat
    assert background.host_for(USER) is h  # one host per user, across turns


async def test_started_may_be_either_clock():
    h = host()
    a = adopt(h, asyncio.sleep(1), started=time.monotonic() - 95)
    b = adopt(h, asyncio.sleep(1), started=time.time() - 95)
    entries = {e["id"]: e for e in h.list()}
    for task_id in (a, b):
        assert 94 <= entries[task_id]["elapsed_s"] <= 96
        assert abs((datetime.now(timezone.utc) - entries[task_id]["started"]).total_seconds() - 95) < 2
    h.cancel_all()


# ---- notifications ------------------------------------------------------------------------------------
async def test_a_finished_task_reaches_the_running_turn_of_its_topic():
    box = inbox.open_inbox(USER, 5)
    h = host(thread_id=5)
    task_id = adopt(h, result_after("built in 95 s", 0.01))
    assert await box.wait(2) == "task"
    blocks = await box.drain()
    text = notification_text(blocks)
    assert text.startswith(f'<task_notification id="{task_id}" tool="run_command" status="completed" elapsed="')
    assert text.endswith("<result>\nbuilt in 95 s\n</result>\n</task_notification>")
    assert h.list()[0]["status"] == "done"


async def test_results_are_capped_and_images_kept_and_errors_are_failures():
    box = inbox.open_inbox(USER, None)
    h = host()
    adopt(h, result_after([{"type": "text", "text": "x" * 50}, IMAGE]), max_chars=10)
    adopt(h, result_after("Error (exit code 2):\nno such file"))
    adopt(h, result_after(ValueError("boom")))
    await until(lambda: len(box._items) == 3)
    blocks = await box.drain()
    assert blocks[0]["text"].count("x") == 10 and "[truncated 40 characters]" in blocks[0]["text"]
    assert blocks[1] == IMAGE  # images follow their notification's text block
    assert 'status="failed"' in blocks[2]["text"] and "Error (exit code 2)" in blocks[2]["text"]
    assert 'status="failed"' in blocks[3]["text"] and "Error: ValueError: boom" in blocks[3]["text"]
    assert [e["status"] for e in h.list()] == ["done", "failed", "failed"]


async def test_an_idle_user_gets_a_background_task_finished_turn(qm, turns):
    h = host(thread_id=7)
    adopt(h, result_after("report.pdf is ready"))
    await until(lambda: turns)
    (turn,) = turns
    assert turn["prompt"][0]["text"] == background.NOTIFICATION_PROMPT
    assert "report.pdf is ready" in turn["prompt"][1]["text"] and turn["prompt"][1]["text"].startswith("<task_notification")
    assert turn["header"] == background.NOTIFICATION_HEADER and turn["quiet_marker"] == agent_runner.QUIET_MARKER
    assert turn["thread_id"] == 7 and turn["chat_id"] == 42 and turn["limit"] == 50 and turn["ctx"] is not None


async def test_notifications_join_a_notification_turn_that_has_not_started(qm, turns):
    gate = asyncio.Event()

    async def busy(ctx):
        await gate.wait()

    await qm.submit(Job(user_id=USER, label="something else", run=busy))
    h = host()
    adopt(h, result_after("first"))
    adopt(h, result_after("second"))
    await until(lambda: all(e["status"] == "done" for e in h.list()))
    assert qm.status(USER).pending == 1  # one turn for both
    gate.set()
    await until(lambda: turns)
    texts = [b["text"] for b in turns[0]["prompt"]]
    assert len(texts) == 3 and "first" in texts[1] and "second" in texts[2]


async def test_a_running_turn_of_another_topic_does_not_get_it(qm, turns):
    box = inbox.open_inbox(USER, 5)
    h = host(thread_id=None)
    adopt(h, result_after("done"))
    await until(lambda: turns)
    assert not box.has_pending() and turns[0]["thread_id"] is None


async def test_an_unread_notification_becomes_a_turn_when_the_turn_ends(qm, turns):
    box = inbox.open_inbox(USER, None)
    h = host()
    adopt(h, result_after("late result"))
    await until(lambda: box.has_pending())
    inbox.finish(box)
    await until(lambda: turns)
    assert "late result" in turns[0]["prompt"][1]["text"]


async def test_cancel_task_reports_to_a_running_turn_only(qm, turns):
    h = host()
    task_id = adopt(h, asyncio.sleep(30))
    assert h.cancel(task_id) is True and h.cancel(task_id) is False and h.cancel("t99") is False
    await until(lambda: h.list()[0]["status"] == "cancelled")
    await asyncio.sleep(0.05)
    assert turns == []  # nobody to tell: no turn of its own for a cancellation
    box = inbox.open_inbox(USER, None)
    task_id = adopt(h, asyncio.sleep(30))
    h.cancel(task_id)
    assert await box.wait(2) == "task"
    assert 'status="cancelled"' in (await box.drain())[0]["text"]


# ---- /cancel, Stop, shutdown -------------------------------------------------------------------------------
def use_adopting_agent(monkeypatch, started: asyncio.Event) -> dict:
    """A ChainOfThoughtAgent stand-in that moves one long call to the background, then keeps working."""
    seen: dict = {}

    class Agent:
        def __init__(self, **kw):
            seen.update(kw)
            self.agent = SimpleNamespace(limit_reached=False)

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            task = asyncio.create_task(asyncio.sleep(30))
            seen["task"] = task
            seen["id"] = seen["background"].adopt(task, tool_name="run_command", tool_use_id="toolu_9", display_args="{}",
                                                  started=time.monotonic(), max_chars=1000)
            started.set()
            await asyncio.sleep(30)

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    return seen


def turn_job(bot, thread_id=None) -> Job:
    async def run(ctx):
        return await REAL_RUN_TURN(bot=bot, user_id=USER, chat_id=42, prompt="build it", thread_id=thread_id, ctx=ctx)

    return Job(user_id=USER, label="text message", run=run, from_user=True, thread_id=thread_id)


async def test_run_turn_hands_the_agent_its_inbox_and_host(monkeypatch, qm):
    started = asyncio.Event()
    seen = use_adopting_agent(monkeypatch, started)
    job = turn_job(FakeBot(), thread_id=3)
    await qm.submit(job)
    await started.wait()
    assert isinstance(seen["inbox"], inbox.TurnInbox) and seen["inbox"].thread_id == 3 and inbox.current(USER) is seen["inbox"]
    assert isinstance(seen["background"], background.BackgroundHost) and seen["background"].thread_id == 3
    await qm.cancel_current(USER, job.id)
    await until(lambda: not qm.is_busy(USER))
    assert seen["inbox"].closed and inbox.current(USER) is None
    background.cancel_all(USER)


async def test_stop_keeps_background_tasks_and_slash_cancel_ends_them(monkeypatch, qm, turns):
    started = asyncio.Event()
    seen = use_adopting_agent(monkeypatch, started)
    job = turn_job(FakeBot())
    await qm.submit(job)
    await started.wait()
    assert await qm.cancel_current(USER, job.id) is True  # the Stop button
    await until(lambda: not qm.is_busy(USER))
    assert not seen["task"].done() and seen["background"].list()[0]["status"] == "running"

    text = await messages._cancel_for(USER)  # /cancel
    assert text == "🛑 Cancelled 1 background task(s)."
    await until(lambda: seen["task"].done())
    assert seen["task"].cancelled() and seen["background"].list()[0]["status"] == "cancelled"
    await asyncio.sleep(0.05)
    assert turns == [] and qm.status(USER).pending == 0  # nothing announced


async def test_slash_cancel_forgets_a_queued_notification_turn(qm, turns):
    gate = asyncio.Event()

    async def busy(ctx):
        await gate.wait()

    await qm.submit(Job(user_id=USER, label="something", run=busy))
    h = host()
    adopt(h, result_after("first"))
    await until(lambda: qm.status(USER).pending == 1)
    await messages._cancel_for(USER)
    adopt(h, result_after("after the cancel"))  # must not join the dropped turn
    await until(lambda: turns)
    assert len(turns) == 1 and "after the cancel" in turns[0]["prompt"][1]["text"]


async def test_shutdown_cancels_every_task_silently(qm, turns):
    h = host()
    other = background.host_for("77", bot=FakeBot(), chat_id=77)
    tasks = [asyncio.create_task(asyncio.sleep(30)) for _ in range(2)]
    h.adopt(tasks[0], tool_name="a", tool_use_id="1", display_args="", started=time.monotonic(), max_chars=None)
    other.adopt(tasks[1], tool_name="b", tool_use_id="2", display_args="", started=time.monotonic(), max_chars=None)
    await background.shutdown()
    assert all(t.cancelled() for t in tasks)
    await asyncio.sleep(0.05)
    assert turns == []


async def test_app_shutdown_cancels_the_background_tasks_before_the_queue(monkeypatch):
    from bot import app

    order: list[str] = []

    async def bg_shutdown():
        order.append("background")

    async def queue_shutdown():
        order.append("queue")

    async def sessions():
        order.append("sessions")

    monkeypatch.setattr(background, "shutdown", bg_shutdown)
    monkeypatch.setattr(runtime, "queue", SimpleNamespace(shutdown=queue_shutdown))
    monkeypatch.setattr(app.sandbox_ui, "shutdown_sessions", sessions)
    await app.on_shutdown(FakeBot())
    assert order == ["background", "queue", "sessions"]


# ---- the agent core's loop with the real inbox and host ---------------------------------------------------
class _Reply(SimpleNamespace):
    pass


def _reply(*blocks, stop="end_turn"):
    return _Reply(content=list(blocks), stop_reason=stop, input_transformations=[],
                  usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0))


class _Stream:
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


class _Client:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.messages = SimpleNamespace(stream=self._stream)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        import copy

        self.requests.append(copy.deepcopy(kwargs))
        return _Stream(self.script.pop(0))


class _SlowTool:
    tools_schema = [{"name": "slow", "description": "slow", "input_schema": {"type": "object", "properties": {}}}]

    def __init__(self, on_start):
        self.on_start = on_start

    async def execute_tool(self, name, args):
        self.on_start(args)
        await asyncio.sleep(args["delay"])
        return f"slow {args['delay']} done"


async def test_the_agent_loop_reads_the_real_inbox_and_hands_calls_to_the_real_host(monkeypatch, tmp_path):
    """End to end over the protocols: a user message makes the loop move a long call to the host and
    is injected behind the placeholder result; the call's result comes back as a notification at the
    next tool boundary; the turn record side sees the drained text."""
    import copy
    import importlib
    import os

    from bot.settings import DEFAULT_SETTINGS

    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    monkeypatch.delitem(sys.modules, "stats", raising=False)  # agents.main keeps the real tracker
    agents_main = importlib.import_module("agents.main")
    monkeypatch.setattr(agents_main, "_thinking_binding_refused", False)
    monkeypatch.setattr(agents_main.stats_tracker, "track_tool_used", lambda uid, name: None)
    monkeypatch.setattr(agents_main, "TOOL_BACKGROUND_AFTER_SECONDS", 0)
    monkeypatch.setattr(agents_main, "BACKGROUND_ON_MESSAGE_AFTER", 0.1)

    box = inbox.open_inbox(USER, None)
    h = host()

    def on_start(args):
        if args["delay"] == 0.6:  # the first call: the user writes while it runs
            box.deliver(inbox.InboxItem(kind=inbox.USER, future=inbox.ready_future(inbox.Prepared(prompt="how is it going?"))))

    from types import SimpleNamespace as NS

    agent = agents_main.AgentAnthropic(user_id=USER)
    agent.client = _Client([
        _reply(NS(type="tool_use", id="c1", name="slow", input={"delay": 0.6}), stop="tool_use"),
        _reply(NS(type="tool_use", id="c2", name="slow", input={"delay": 0.9}), stop="tool_use"),
        _reply(NS(type="text", text="all done")),
    ])
    agent.user_interactions = _SlowTool(on_start)
    agent.inbox, agent.background = box, h
    from agents.trace import ToolTrace

    response, messages = await agent.generate_response(
        messages=[{"role": "user", "content": [{"type": "text", "text": "do it"}]}], system_role="sys", question="do it",
        user_settings=copy.deepcopy(DEFAULT_SETTINGS), update_status=None, trace=ToolTrace(),
    )
    assert response == "all done"
    second = agent.client.requests[1]["messages"][-1]["content"]
    assert second[0]["tool_use_id"] == "c1" and "Still running in the background as task t1" in str(second[0]["content"])
    assert second[1]["text"].startswith('<user_message sent_during_turn="true"') and "how is it going?" in second[1]["text"]
    third = agent.client.requests[2]["messages"][-1]["content"]
    assert third[0] == {"type": "tool_result", "tool_use_id": "c2", "content": "slow 0.9 done"}
    assert third[1]["text"].startswith('<task_notification id="t1" tool="slow" status="completed"')
    assert "slow 0.6 done" in third[1]["text"]
    assert box.drained_user_texts == ["how is it going?"] and h.list()[0]["status"] == "done"
