"""Background tasks of a live turn: ``wait_for_tasks``, ``list_tasks``, ``cancel_task``.

A tool call that is still running after ``agents.main.TOOL_BACKGROUND_AFTER_SECONDS`` (or
while the user wrote and the call has run for a few seconds) is moved to the background by
``AgentAnthropic.run_tool_batch``: its tool_result becomes a placeholder, the turn goes on, and
the bot side (``bot/background.py``, the ``BackgroundHost``) owns the asyncio task from then on.
When it ends, a ``<task_notification>`` reaches the agent through the turn's inbox
(``bot/inbox.py``, the ``TurnInbox``) after a later tool call, or as a new turn when none runs.

This provider is registered by ``ChainOfThoughtAgent`` only when the caller gives it a
background host; subagents never get it (``inherit_to_subagents``). It only talks to the two
protocols below, never to Telegram.

It also owns the per-call context (``CallState``): every call that may go to the background
runs in an asyncio task whose context carries one, so code running inside the call (a
subagent's status updates) can tell that the turn no longer waits for it (``in_background``).
"""
from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

WAIT_DEFAULT_SECONDS = 120
WAIT_MIN_SECONDS = 5
WAIT_MAX_SECONDS = 600
POLL_SECONDS = 1.0  # waiting without an inbox: how often the host's task list is looked at

TASK_TOOL_NAMES = frozenset({"wait_for_tasks", "list_tasks", "cancel_task"})


class TurnInbox(Protocol):
    """What the agent core needs from the bot side's per-turn inbox (``bot/inbox.py``)."""

    def has_user_messages(self) -> bool: ...
    def has_pending(self) -> bool: ...
    async def drain(self) -> list[dict]: ...
    async def wait(self, timeout: float) -> str: ...  # "message" | "task" | "timeout"

    @property
    def drained_user_texts(self) -> list[str]: ...


class BackgroundHost(Protocol):
    """What the agent core needs from the bot side's background task host (``bot/background.py``).

    ``list`` and ``cancel`` cover the running turn's conversation (its forum topic) only; ``adopt``
    may refuse (a falsy id: the call stays in the foreground)."""

    def can_adopt(self) -> bool: ...
    def adopt(self, task: asyncio.Task, *, tool_name: str, tool_use_id: str, display_args: str,
              started: float, max_chars: int | None) -> str | None: ...
    def list(self) -> list[dict]: ...
    def cancel(self, task_id: str) -> bool: ...


# ---- the call a piece of code runs in -------------------------------------------------------
@dataclass
class CallState:
    """One tool call that may be moved to the background; ``background_id`` is set when it was."""

    tool_name: str
    background_id: str | None = None

    @property
    def detached(self) -> bool:
        return self.background_id is not None


_CALL_STATE: ContextVar[CallState | None] = ContextVar("agent_tool_call", default=None)


def bind_call(state: CallState) -> None:
    """Make ``state`` the current call (run inside the context the call's task is created with)."""
    _CALL_STATE.set(state)


def current_call() -> CallState | None:
    return _CALL_STATE.get()


def in_background() -> bool:
    """True inside a tool call that was moved to the background (its turn may be over)."""
    state = _CALL_STATE.get()
    return state is not None and state.detached


# ---- helpers ------------------------------------------------------------------------------
def format_duration(seconds: Any) -> str:
    """``45 s``, ``2 min``, ``1 h 5 min``."""
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    if total < 60:
        return f"{total} s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def normalize_task_id(value: Any) -> str:
    """``"t3"`` from ``"t3"``, ``" T3 "`` or ``"3"``."""
    text = str(value or "").strip().lower()
    return f"t{text}" if text.isdigit() else text


def _started_text(value: Any) -> str:
    strftime = getattr(value, "strftime", None)
    return f"started {strftime('%H:%M:%S')} UTC" if strftime else ""


def _running(tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if str(t.get("status")) == "running"]


def running_summary(tasks: list[dict]) -> str:
    """``t1 (run_command, 2 min), t2 (search_web, 40 s)``."""
    return ", ".join(f"{t.get('id')} ({t.get('tool')}, {format_duration(t.get('elapsed_s'))})" for t in tasks)


class TaskTools:
    """Tool provider: wait for, list and cancel the background tasks of this conversation."""

    # registered by the main agent only: a subagent has no inbox and no background host
    inherit_to_subagents = False

    prompt_notes = """<live_turn>
The user can keep writing while you work. A message they sent during this turn reaches you after a tool call, behind the tool results, as <user_message sent_during_turn="true" time="..."> ... </user_message> (pictures and documents they attached follow it). Read it right away and never ignore it: it may change the task, add details, ask a question or ask you to stop. Adapt your plan to it. If it asks something quick while your work will take longer, answer it briefly with send_user_telegram_message and carry on. Your final answer must cover it too.

A tool call that runs long, or is still running when the user writes, is moved to the background: its tool result then says "Still running in the background as task t<N>" instead of the output. It keeps running; do not start it again. Its outcome arrives later as <task_notification id="t<N>" tool="..." status="completed|failed|cancelled" elapsed="..."> with the <result> inside: after one of your later tool calls while this turn runs, or as a new message to you after the turn has ended. Meanwhile you can answer the user, use other tools, or call wait_for_tasks to wait for the next notification or user message. list_tasks shows the background tasks, cancel_task stops one.

If you give your final answer while a background task still runs, tell the user its result will follow in a separate message.
</live_turn>"""

    def __init__(self, background: BackgroundHost, inbox: TurnInbox | None = None) -> None:
        self.background = background
        self.inbox = inbox
        self.tools_schema = [
            {
                "name": "wait_for_tasks",
                "description": (
                    "Wait until a background task finishes or the user sends a new message, whichever comes first, "
                    "at most timeout_seconds. The finished task's <task_notification> (or the user's message) follows "
                    "this tool's result. Use it when you have nothing else to do until a background task is done."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timeout_seconds": {
                            "type": "integer", "minimum": WAIT_MIN_SECONDS, "maximum": WAIT_MAX_SECONDS,
                            "description": f"How long to wait at most, {WAIT_MIN_SECONDS}-{WAIT_MAX_SECONDS} s (default {WAIT_DEFAULT_SECONDS}).",
                        },
                    },
                },
            },
            {
                "name": "list_tasks",
                "description": "List this conversation's background tasks: id, tool, arguments, status (running, done, failed, cancelled) and how long they ran.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "cancel_task",
                "description": "Cancel a running background task by its id (t1, t2, ...), e.g. when the user no longer needs its result.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string", "description": "The task id, e.g. t3."}},
                    "required": ["task_id"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        args = tool_args if isinstance(tool_args, dict) else {}
        if tool_name == "wait_for_tasks":
            return await self.wait_for_tasks(args.get("timeout_seconds"))
        if tool_name == "list_tasks":
            return self.list_tasks()
        if tool_name == "cancel_task":
            return self.cancel_task(args.get("task_id"))
        return f"Unknown tool: {tool_name}"

    @staticmethod
    def _timeout(value: Any) -> int:
        try:
            seconds = int(float(value)) if value is not None and str(value).strip() else WAIT_DEFAULT_SECONDS
        except (TypeError, ValueError):
            seconds = WAIT_DEFAULT_SECONDS
        return max(WAIT_MIN_SECONDS, min(WAIT_MAX_SECONDS, seconds))

    def _tasks(self) -> list[dict]:
        try:
            return [t for t in (self.background.list() or []) if isinstance(t, dict)]
        except Exception as e:  # noqa: BLE001 - the host is the bot's; a failure is not the turn's
            print(f"WARNING: could not list background tasks: {e}")
            return []

    def _inbox_pending(self) -> bool:
        try:
            return bool(self.inbox is not None and self.inbox.has_pending())
        except Exception:  # noqa: BLE001
            return False

    async def wait_for_tasks(self, timeout_seconds: Any = None) -> str:
        timeout = self._timeout(timeout_seconds)
        if not _running(self._tasks()) and not self._inbox_pending():
            return "No background tasks are running."
        if self.inbox is not None:
            outcome = await self.inbox.wait(float(timeout))
        else:
            outcome = await self._poll_tasks(timeout)
        if outcome == "task":
            return "A background task finished; its notification follows."
        if outcome == "message":
            return "A new user message arrived; it follows."
        still = _running(self._tasks())
        if still:
            return f"Nothing finished within {timeout} s; tasks still running: {running_summary(still)}."
        return f"Nothing finished within {timeout} s. No background tasks are running."

    async def _poll_tasks(self, timeout: int) -> str:
        """Without an inbox: wait until one of the running tasks is no longer running."""
        watched = {t.get("id") for t in _running(self._tasks())}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            running = {t.get("id") for t in _running(self._tasks())}
            if watched - running:
                return "task"
        return "timeout"

    def list_tasks(self) -> str:
        tasks = self._tasks()
        if not tasks:
            return "No background tasks in this conversation."
        lines = ["Background tasks:"]
        for t in tasks:
            parts = [str(t.get("id")), str(t.get("tool")), str(t.get("status")), format_duration(t.get("elapsed_s"))]
            started = _started_text(t.get("started"))
            if started:
                parts.append(started)
            line = "- " + " · ".join(parts)
            args = str(t.get("args") or "").strip()
            if args:
                line += f" · args: {args}"
            lines.append(line)
        return "\n".join(lines)

    def cancel_task(self, task_id: Any) -> str:
        tid = normalize_task_id(task_id)
        if not tid:
            return "Error: 'task_id' is required (e.g. t1; list_tasks shows the ids)."
        try:
            cancelled = bool(self.background.cancel(tid))
        except Exception as e:  # noqa: BLE001
            return f"Error: could not cancel {tid}: {e}"
        if cancelled:
            return f"Task {tid} was cancelled."
        return f"Error: no running background task {tid} (list_tasks shows the tasks)."
