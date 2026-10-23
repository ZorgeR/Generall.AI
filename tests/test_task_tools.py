"""agents.task_tools: the background-task tools of a live turn and the per-call context.

``TaskTools(background, inbox)`` offers ``wait_for_tasks`` (waits on ``inbox.wait`` within
5..600 s and returns the results that finished, or says what woke it), ``get_task_result`` (a
finished task's output as the tool's own result: the system notice of a finished task carries no
output), ``list_tasks`` and ``cancel_task`` over the bot side's ``BackgroundHost``; its
``prompt_notes`` are static text for the cached system prompt; a subagent never inherits it. ``CallState`` / ``in_background`` tell code inside a tool call that the call
was moved to the background.
"""
import asyncio
import contextvars
import re
from datetime import datetime, timezone

import pytest

from agents import task_tools
from agents.task_tools import TaskTools


class FakeHost:
    def __init__(self, tasks=None, cancellable=(), results=None):
        self.tasks = list(tasks or [])
        self.cancellable = set(cancellable)
        self.cancelled: list[str] = []
        self.results = dict(results or {})  # finished tasks: id -> output
        self.read_ids: list[str] = []

    def unread(self):
        return [t for t in self.results if t not in self.read_ids]

    def read(self, task_id):
        if task_id not in self.results:
            return None
        self.read_ids.append(task_id)
        return [{"type": "text", "text": f"Result of {task_id}: {self.results[task_id]}"}]

    def can_adopt(self):
        return True

    def adopt(self, task, **kw):
        raise AssertionError("TaskTools never adopts")

    def list(self):
        return [dict(t) for t in self.tasks]

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        return task_id in self.cancellable


class FakeInbox:
    def __init__(self, outcome="timeout", pending=False):
        self.outcome = outcome
        self.pending = pending
        self.waits: list[float] = []

    def has_user_messages(self):
        return False

    def has_pending(self):
        return self.pending

    async def drain(self):
        return []

    async def wait(self, timeout):
        self.waits.append(timeout)
        return self.outcome

    @property
    def drained_user_texts(self):
        return []


STARTED = datetime(2026, 10, 18, 12, 1, 5, tzinfo=timezone.utc)


def running(tid="t1", tool="run_command", elapsed=125, args="sleep 300"):
    return {"id": tid, "tool": tool, "args": args, "status": "running", "started": STARTED, "elapsed_s": elapsed}


# ---- schema and notes ------------------------------------------------------------------------
def test_schema_names_and_bounds():
    tools = TaskTools(FakeHost(), FakeInbox())
    names = [t["name"] for t in tools.tools_schema]
    assert names == ["wait_for_tasks", "get_task_result", "list_tasks", "cancel_task"]
    assert set(names) == task_tools.TASK_TOOL_NAMES
    wait = tools.tools_schema[0]["input_schema"]["properties"]["timeout_seconds"]
    assert (wait["minimum"], wait["maximum"]) == (5, 600)
    assert tools.tools_schema[1]["input_schema"]["required"] == ["task_id"]
    assert tools.tools_schema[3]["input_schema"]["required"] == ["task_id"]
    for t in tools.tools_schema:
        assert t["description"] and t["input_schema"]["type"] == "object"


def test_prompt_notes_are_static_and_cover_the_live_turn():
    one, two = TaskTools(FakeHost(), FakeInbox()), TaskTools(FakeHost([running()]), None)
    assert one.prompt_notes == two.prompt_notes == TaskTools.prompt_notes  # a class constant: no per-turn text
    notes = one.prompt_notes
    for needle in ('<user_message sent_during_turn="true"', "send_user_telegram_message", "never ignore it",
                   "ask_user_choice", "Still running in the background as task", "system message", "get_task_result",
                   "wait_for_tasks", "list_tasks", "cancel_task", "result will follow"):
        assert needle in notes, needle
    assert "<task_notification" not in notes  # the old format: output next to the user's words
    assert "{" not in notes  # nothing left to format
    assert not re.search(r"\d\d:\d\d|20\d\d-", notes), "no times or dates in the static prompt"


def test_task_tools_are_never_inherited_by_subagents():
    assert TaskTools.inherit_to_subagents is False


# ---- wait_for_tasks ----------------------------------------------------------------------------
@pytest.mark.parametrize("outcome, text", [
    ("task", "A background task ended with nothing to read (cancelled, or its result was read already); list_tasks shows it."),
    ("message", "A new user message arrived; it follows."),
])
async def test_wait_says_what_woke_it(outcome, text):
    inbox = FakeInbox(outcome)
    tools = TaskTools(FakeHost([running()]), inbox)
    assert await tools.execute_tool("wait_for_tasks", {"timeout_seconds": 30}) == text
    assert inbox.waits == [30.0]


async def test_wait_timeout_names_the_running_tasks():
    tools = TaskTools(FakeHost([running(), running("t2", "search_web", 40, "q"), {**running("t3"), "status": "done"}]), FakeInbox())
    assert await tools.execute_tool("wait_for_tasks", {}) == (
        "Nothing finished within 120 s; tasks still running: t1 (run_command, 2 min), t2 (search_web, 40 s)."
    )


async def test_wait_with_nothing_running_returns_at_once():
    inbox = FakeInbox("task")
    tools = TaskTools(FakeHost([{**running(), "status": "done"}]), inbox)
    assert await tools.execute_tool("wait_for_tasks", {"timeout_seconds": 60}) == "No background tasks are running."
    assert inbox.waits == []
    # ... unless a notification (or a message) is already pending: then the wait returns what it is
    inbox.pending = True
    assert await tools.execute_tool("wait_for_tasks", {}) == (
        "A background task ended with nothing to read (cancelled, or its result was read already); list_tasks shows it.")


async def test_wait_returns_the_results_that_finished():
    """The output comes back as this tool's own result, never next to the user's words."""
    host = FakeHost([running(), {**running("t2"), "status": "done"}])
    inbox = FakeInbox("task")

    async def wait(timeout):
        inbox.waits.append(timeout)
        host.results["t1"] = "built"
        host.tasks[0]["status"] = "done"
        return "task"

    inbox.wait = wait
    tools = TaskTools(host, inbox)
    assert await tools.execute_tool("wait_for_tasks", {"timeout_seconds": 30}) == [{"type": "text", "text": "Result of t1: built"}]
    assert host.read_ids == ["t1"] and inbox.waits == [30]


async def test_wait_returns_unread_results_at_once():
    """A task that finished before the call (its notice may not have been read yet) is not waited for."""
    host = FakeHost([running(), {**running("t2"), "status": "done"}, {**running("t3"), "status": "failed"}],
                    results={"t2": "ok", "t3": "Error: boom"})
    inbox = FakeInbox("timeout")
    blocks = await TaskTools(host, inbox).execute_tool("wait_for_tasks", {})
    assert blocks == [{"type": "text", "text": "Result of t2: ok"}, {"type": "text", "text": "Result of t3: Error: boom"}]
    assert inbox.waits == [] and host.unread() == []


async def test_wait_says_so_when_a_result_is_gone():
    host = FakeHost([{**running(), "status": "done"}], results={"t1": "x"})
    host.read = lambda task_id: None  # pruned between unread() and read()
    assert await TaskTools(host, FakeInbox()).execute_tool("wait_for_tasks", {}) == (
        "A background task finished, but its result is no longer available (list_tasks shows it).")


# ---- get_task_result ---------------------------------------------------------------------------
async def test_get_task_result():
    host = FakeHost([running(), {**running("t2", "search_web"), "status": "done"}, {**running("t3"), "status": "cancelled"}],
                    results={"t2": "found it"})
    tools = TaskTools(host, FakeInbox())
    assert await tools.execute_tool("get_task_result", {"task_id": "T2"}) == [{"type": "text", "text": "Result of t2: found it"}]
    assert host.read_ids == ["t2"]
    assert await tools.execute_tool("get_task_result", {"task_id": "t1"}) == (
        "Task t1 (run_command) is still running (2 min). A system message will say when it ends; wait_for_tasks waits for it.")
    assert await tools.execute_tool("get_task_result", {"task_id": "t3"}) == "Error: the result of task t3 is no longer available."
    assert await tools.execute_tool("get_task_result", {"task_id": "t9"}) == (
        "Error: no background task t9 in this conversation (list_tasks shows the tasks).")
    assert (await tools.execute_tool("get_task_result", {})).startswith("Error: 'task_id' is required")
    assert host.read_ids == ["t2"]  # only a finished result is read


@pytest.mark.parametrize("given, used", [(1, 5), (5, 5), (999, 600), ("90", 90), ("x", 120), (None, 120), (12.7, 12)])
async def test_wait_timeout_is_clamped(given, used):
    inbox = FakeInbox()
    tools = TaskTools(FakeHost([running()]), inbox)
    await tools.execute_tool("wait_for_tasks", {"timeout_seconds": given})
    assert inbox.waits == [float(used)]


async def test_wait_after_the_last_task_ended_without_a_notification():
    host = FakeHost([running()])
    inbox = FakeInbox("timeout")

    async def wait(timeout):
        host.tasks = [{**running(), "status": "cancelled"}]
        return "timeout"

    inbox.wait = wait
    assert await TaskTools(host, inbox).execute_tool("wait_for_tasks", {"timeout_seconds": 5}) == (
        "Nothing finished within 5 s. No background tasks are running."
    )


async def test_wait_without_an_inbox_polls_the_host(monkeypatch):
    monkeypatch.setattr(task_tools, "POLL_SECONDS", 0.01)
    host = FakeHost([running()])
    tools = TaskTools(host, None)

    async def finish_later():
        await asyncio.sleep(0.05)
        host.tasks = [{**running(), "status": "done"}]
        host.results["t1"] = "ok"

    asyncio.get_running_loop().create_task(finish_later())
    assert await tools.wait_for_tasks(5) == [{"type": "text", "text": "Result of t1: ok"}]


# ---- list_tasks / cancel_task --------------------------------------------------------------------
async def test_list_tasks():
    host = FakeHost([running(), {"id": "t2", "tool": "search_web", "args": "", "status": "failed", "started": STARTED, "elapsed_s": 3725,
                                 "unread": True}])
    text = await TaskTools(host, FakeInbox()).execute_tool("list_tasks", {})
    assert text.splitlines() == [
        "Background tasks:",
        "- t1 · run_command · running · 2 min · started 12:01:05 UTC · args: sleep 300",
        "- t2 · search_web · failed · 1 h 2 min · result not read yet · started 12:01:05 UTC",
    ]
    assert await TaskTools(FakeHost(), FakeInbox()).execute_tool("list_tasks", {}) == "No background tasks in this conversation."


async def test_cancel_task():
    host = FakeHost([running()], cancellable={"t1"})
    tools = TaskTools(host, FakeInbox())
    assert await tools.execute_tool("cancel_task", {"task_id": " T1 "}) == "Task t1 was cancelled."
    assert await tools.execute_tool("cancel_task", {"task_id": "3"}) == "Error: no running background task t3 (list_tasks shows the tasks)."
    assert (await tools.execute_tool("cancel_task", {})).startswith("Error: 'task_id' is required")
    assert host.cancelled == ["t1", "t3"]


async def test_a_failing_host_is_an_error_text_not_a_crash():
    class Broken(FakeHost):
        def list(self):
            raise RuntimeError("registry gone")

        def cancel(self, task_id):
            raise RuntimeError("registry gone")

    tools = TaskTools(Broken(), FakeInbox())
    assert await tools.execute_tool("list_tasks", {}) == "No background tasks in this conversation."
    assert (await tools.execute_tool("cancel_task", {"task_id": "t1"})).startswith("Error: could not cancel t1")
    assert await tools.execute_tool("nope", {}) == "Unknown tool: nope"


def test_helpers():
    assert [task_tools.format_duration(s) for s in (0, 59, 60, 3599, 3600, 3660, "x", None)] == [
        "0 s", "59 s", "1 min", "59 min", "1 h", "1 h 1 min", "?", "?"]
    assert [task_tools.normalize_task_id(v) for v in ("t3", " T3 ", "3", None, "")] == ["t3", "t3", "t3", "", ""]


# ---- the per-call context ------------------------------------------------------------------------
async def test_in_background_follows_the_call_state():
    assert task_tools.current_call() is None and task_tools.in_background() is False
    state = task_tools.CallState("run_command")
    seen = []

    async def inside():
        seen.append((task_tools.current_call() is state, task_tools.in_background()))
        await asyncio.sleep(0.02)
        # tasks the call starts (a subagent's own tool batch) share the same state
        child = asyncio.get_running_loop().create_task(asyncio.sleep(0, result=task_tools.in_background()))
        seen.append((task_tools.current_call() is state, await child))

    context = contextvars.copy_context()
    context.run(task_tools.bind_call, state)
    task = asyncio.get_running_loop().create_task(inside(), context=context)
    await asyncio.sleep(0.01)
    state.background_id = "t1"  # the watcher handed it over meanwhile
    await task
    assert seen == [(True, False), (True, True)]
    assert task_tools.current_call() is None  # the caller's own context is untouched
