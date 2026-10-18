"""Tool calls that outlive their turn step: the per-user background task host.

The agent loop (``agents/main.py``) moves a tool call that runs too long, or that a user
message interrupts, to the background: it answers the call with a placeholder and hands
the still-running asyncio task to ``BackgroundHost.adopt`` (the ``BackgroundHost``
protocol of the agent core). From then on the host owns it, across turns:

* when it ends, its result (the tool's raw result: a string, a list of content blocks, or
  an exception) becomes one ``<task_notification>`` item (``notification_blocks``: text
  capped at the call's ``max_chars``, images kept);
* the notification goes to the inbox of the user's running turn in the topic the task was
  started from (``bot.inbox.deliver_task``), where the agent reads it at its next tool
  boundary; with no such turn it becomes a "background task finished" turn through the
  user's queue, like a scheduled task (``NOTIFICATION_PROMPT``, header
  ``NOTIFICATION_HEADER``, the quiet marker allowed). Notifications that finish while such
  a turn is still queued join it instead of queueing another;
* ``/cancel`` cancels the user's tasks (``cancel_all``, nothing is announced), the Stop
  button does not; shutdown cancels everyone's (``shutdown``).

At most ``MAX_BACKGROUND_TASKS`` run per user (``can_adopt``). Task ids are "t1", "t2", …
per user, increasing for the life of the process.

No import-time side effects: ``agents``, ``bot.runtime`` and ``bot.agent_runner`` are imported
where they are used.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import Any

from bot import inbox as turn_inbox
from bot.queue import Job, JobContext

logger = logging.getLogger(__name__)

MAX_BACKGROUND_TASKS = 4  # running at once, per user
MAX_LISTED = 20  # finished tasks kept for list()
SHUTDOWN_GRACE = 3.0
NOTIFICATION_PROMPT = (
    "[Background task finished] A task you started earlier finished while no turn was running. "
    "Tell the user what matters from it. If there is nothing worth telling, reply with exactly <no_update/>."
)
NOTIFICATION_HEADER = "🔔 *Background task finished*"
NOTIFICATION_LABEL = "background task result"
NOTIFICATION_CAPTION = "Background task reasoning history."

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"
_NOTIFY_STATUS = {DONE: "completed", FAILED: "failed", CANCELLED: "cancelled"}


def _wall(started: float) -> float:
    """``started`` as a ``time.time()`` value: the agent core may pass either clock."""
    if started > 1e9:  # already epoch seconds
        return started
    return time.time() - (time.monotonic() - started)


@dataclass(eq=False)
class BackgroundTask:
    id: str
    tool: str
    tool_use_id: str
    args: str
    task: asyncio.Task
    started_wall: float
    max_chars: int | None
    thread_id: int | None
    status: str = RUNNING
    finished_wall: float | None = None
    silent: bool = False  # cancelled by /cancel or shutdown: nothing is announced

    def elapsed(self) -> int:
        end = self.finished_wall if self.finished_wall is not None else time.time()
        return max(0, int(round(end - self.started_wall)))

    def entry(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "started": datetime.fromtimestamp(self.started_wall, timezone.utc),
            "elapsed_s": self.elapsed(),
        }


# ---- the notification ---------------------------------------------------------------------------
def _attr(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _outcome(task: asyncio.Task) -> tuple[str, Any]:
    """(status, result) of a finished task: the raw result, or the error as text."""
    if task.cancelled():
        return CANCELLED, "The task was cancelled before it finished."
    error = task.exception()
    if error is not None:
        detail = " ".join(str(error).split())
        return FAILED, f"Error: {type(error).__name__}{': ' + detail if detail else ''}"
    result = task.result()
    if isinstance(result, str) and result.lstrip().startswith("Error"):
        return FAILED, result
    return DONE, result


def notification_blocks(record: BackgroundTask, status: str, result: Any) -> list[dict]:
    """``<task_notification>`` as one text block (capped at ``max_chars``), the result's images after it."""
    from agents import content, transcript

    media: list[dict] = []
    if isinstance(result, list):
        parts: list[str] = []
        for block in content.cap_blocks(result, record.max_chars):
            if not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif content.is_media_block(block):
                media.append(dict(block))
            else:  # browser_state, a media ref, anything else: its one-line text
                parts.append(content.result_preview([block]))
        body = "\n".join(p for p in parts if p)
    else:
        body = transcript.cap_text("" if result is None else str(result), record.max_chars)
    text = (
        f'<task_notification id="{_attr(record.id)}" tool="{_attr(record.tool)}" '
        f'status="{_NOTIFY_STATUS.get(status, status)}" elapsed="{record.elapsed()}s">\n'
        f"<result>\n{body}\n</result>\n</task_notification>"
    )
    return [content.text_block(text), *media]


# ---- notification turns (no turn of that topic is running) -------------------------------------------
@dataclass(eq=False)
class _Batch:
    """A queued "background task finished" turn; notifications join it until it starts."""

    user_id: str
    thread_id: int | None
    blocks: list[dict] = field(default_factory=list)
    started: bool = False


_batches: dict[tuple[str, int | None], _Batch] = {}


def submit_notification(user_id: str, thread_id: int | None, blocks: list[dict], *, seq: int | None = None) -> None:
    """Queue a notification turn for ``blocks`` (or add them to the one still waiting). Never suspends;
    raises only when the queue is closed."""
    from bot import runtime

    key = (str(user_id), thread_id)
    batch = _batches.get(key)
    if batch is not None and not batch.started:
        batch.blocks.extend(blocks)
        return
    batch = _Batch(str(user_id), thread_id, list(blocks))
    host = _hosts.get(str(user_id))
    job = Job(
        user_id=str(user_id),
        label=NOTIFICATION_LABEL,
        run=partial(_run_notification, host.bot if host else None, batch),
        thread_id=thread_id,
        **({"seq": seq} if seq is not None else {}),
    )
    runtime.queue.submit_nowait(job)
    _batches[key] = batch


async def _run_notification(bot: Any, batch: _Batch, ctx: JobContext) -> None:
    """The queued turn: what finished, framed so the agent tells the user what matters (or stays quiet)."""
    from agents import content
    from bot.agent_runner import QUIET_MARKER, run_turn
    from bot.limits import check_user_limits

    batch.started = True
    if _batches.get((batch.user_id, batch.thread_id)) is batch:
        del _batches[(batch.user_id, batch.thread_id)]
    host = _hosts.get(batch.user_id)
    bot = bot or (host.bot if host else None)
    if bot is None:
        logger.error("No bot to tell %s about a finished background task", batch.user_id)
        return
    _, _, limit = await asyncio.to_thread(check_user_limits, batch.user_id)
    await run_turn(
        bot=bot, user_id=batch.user_id, chat_id=int(batch.user_id), prompt=[content.text_block(NOTIFICATION_PROMPT), *batch.blocks],
        thread_id=batch.thread_id, ctx=ctx, limit=limit, header=NOTIFICATION_HEADER, reasoning_caption=NOTIFICATION_CAPTION,
        quiet_marker=QUIET_MARKER,
    )


# ---- the host -------------------------------------------------------------------------------------------
class BackgroundHost:
    """One user's background tasks (the ``BackgroundHost`` protocol of the agent core).

    ``bind`` (``run_turn``, at the start of each turn) records the bot, the chat and the topic
    of the running turn: a task adopted now reports to that topic.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)
        self.bot: Any = None
        self.chat_id: int | None = None
        self.thread_id: int | None = None
        self._counter = 0
        self._tasks: dict[str, BackgroundTask] = {}

    def bind(self, *, bot: Any, chat_id: int | None, thread_id: int | None) -> "BackgroundHost":
        self.bot = bot
        self.chat_id = chat_id
        self.thread_id = thread_id
        return self

    # -- the protocol -------------------------------------------------------------------------------
    def running(self) -> list[BackgroundTask]:
        return [t for t in self._tasks.values() if t.status == RUNNING]

    def can_adopt(self) -> bool:
        return len(self.running()) < MAX_BACKGROUND_TASKS

    def adopt(
        self, task: asyncio.Task, *, tool_name: str, tool_use_id: str, display_args: str,
        started: float, max_chars: int | None,
    ) -> str:
        self._counter += 1
        task_id = f"t{self._counter}"
        record = BackgroundTask(
            id=task_id, tool=str(tool_name), tool_use_id=str(tool_use_id), args=str(display_args or ""), task=task,
            started_wall=_wall(float(started)), max_chars=max_chars, thread_id=self.thread_id,
        )
        self._tasks[task_id] = record
        self._prune()
        task.add_done_callback(partial(self._finished, record))
        return task_id

    def list(self) -> list[dict]:
        return [t.entry() for t in self._tasks.values()]

    def cancel(self, task_id: str) -> bool:
        record = self._tasks.get(str(task_id).strip())
        if record is None or record.status != RUNNING or record.task.done() or record.task.cancelling():
            return False
        record.task.cancel()
        return True

    # -- the rest -------------------------------------------------------------------------------------
    def cancel_all(self) -> int:
        """Cancel every running task without announcing it (/cancel, shutdown)."""
        count = 0
        for record in self.running():
            record.silent = True  # also one the agent is already cancelling: /cancel announces nothing
            if not record.task.done() and not record.task.cancelling():
                record.task.cancel()
                count += 1
        return count

    def _prune(self) -> None:
        finished = [k for k, t in self._tasks.items() if t.status != RUNNING]
        for key in finished[: max(0, len(finished) - MAX_LISTED)]:
            del self._tasks[key]

    def _finished(self, record: BackgroundTask, task: asyncio.Task) -> None:
        status, result = _outcome(task)
        record.status = status
        record.finished_wall = time.time()
        self._prune()
        if record.silent:
            return
        try:
            blocks = notification_blocks(record, status, result)
        except Exception as e:  # noqa: BLE001 - the notification must never break the loop's callbacks
            logger.exception("Could not format background task %s of %s: %s", record.id, self.user_id, e)
            return
        item = turn_inbox.InboxItem(
            kind=turn_inbox.TASK, future=turn_inbox.ready_future(blocks), thread_id=record.thread_id,
            label=f"task {record.id}",
        )
        item.requeue = partial(submit_notification, self.user_id, record.thread_id, blocks, seq=item.seq)
        if turn_inbox.deliver_task(self.user_id, record.thread_id, item):
            return
        if status == CANCELLED:
            return  # the agent cancelled it itself: worth a line in a running turn, not a turn of its own
        try:
            submit_notification(self.user_id, record.thread_id, blocks)
        except Exception as e:  # noqa: BLE001 - the queue is closed (shutdown)
            logger.warning("Could not queue the result of background task %s for %s: %s", record.id, self.user_id, e)


_hosts: dict[str, BackgroundHost] = {}
_registry_loop: asyncio.AbstractEventLoop | None = None


def _same_loop() -> None:
    """Tasks belong to the event loop they run on: a new loop (every test) starts with no hosts."""
    global _registry_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if loop is not _registry_loop:
        _hosts.clear()
        _batches.clear()
        _registry_loop = loop


def host_for(user_id: str, *, bot: Any = None, chat_id: int | None = None, thread_id: int | None = None) -> BackgroundHost:
    """The user's host, bound to the turn that starts now (``run_turn``)."""
    _same_loop()
    host = _hosts.get(str(user_id))
    if host is None:
        host = _hosts[str(user_id)] = BackgroundHost(user_id)
    return host.bind(bot=bot, chat_id=chat_id, thread_id=thread_id)


def cancel_all(user_id: str) -> int:
    """/cancel: the user's running tasks are cancelled silently and a queued notification turn forgotten."""
    _same_loop()
    for key in [k for k in _batches if k[0] == str(user_id)]:
        del _batches[key]
    host = _hosts.get(str(user_id))
    return host.cancel_all() if host is not None else 0


async def shutdown(grace: float = SHUTDOWN_GRACE) -> None:
    """Cancel every user's tasks (nothing announced) and give them ``grace`` seconds to end."""
    _same_loop()
    tasks = []
    for host in _hosts.values():
        tasks += [r.task for r in host.running()]
        host.cancel_all()
    _batches.clear()
    if tasks:
        await asyncio.wait(tasks, timeout=grace)


def reset() -> None:
    """Forget every host and queued notification (tests)."""
    _hosts.clear()
    _batches.clear()
