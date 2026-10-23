"""Tool calls that outlive their turn step: the per-user background task host.

The agent loop (``agents/main.py``) moves a tool call that runs too long, or that a user
message interrupts, to the background: it answers the call with a placeholder and hands
the still-running asyncio task to ``BackgroundHost.adopt`` (the ``BackgroundHost``
protocol of the agent core). From then on the host owns it, across turns:

* when it ends, its result (the tool's raw result: a string, a list of content blocks, or
  an exception) is kept on its record until the agent reads it with ``get_task_result`` (or
  ``wait_for_tasks`` hands it over): ``result_blocks``, text capped at the call's
  ``max_chars``, images kept. Tool output stays inside a ``tool_result`` that way, where the
  model treats it as untrusted, and never sits next to the user's words;
* what announces the end is a notice of facts only (``notice_text``: id, tool, outcome,
  time), which the agent core sends as a mid-conversation ``system`` message: it goes to the
  inbox of the user's running turn in the topic the task was started from
  (``bot.inbox.deliver_task``), read at the next tool boundary; with no such turn it becomes a
  "background task finished" turn through the user's queue, like a scheduled task
  (``NOTIFICATION_PROMPT`` with the facts, header ``NOTIFICATION_HEADER``, the quiet marker
  allowed). Tasks that finish while such a turn is still queued join it instead of queueing
  another, and a result the agent has read meanwhile is announced no more;
* ``/cancel`` cancels the user's tasks (``cancel_all``, nothing is announced), the Stop
  button does not; a turn that fails cancels the ones it adopted (``cancel_turn``: its Retry
  starts them again); shutdown cancels everyone's (``shutdown``) and refuses new ones.

At most ``MAX_BACKGROUND_TASKS`` run per user (``can_adopt``), whatever their topic. Task ids
are "t1", "t2", … per user, increasing for the life of the process. What the agent's task tools
see is the bound turn's topic only (``running``, ``list``, ``cancel``, ``unread``, ``read``): a
turn in one forum topic neither waits for, lists, reads nor cancels the tasks of another.

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
MAX_LISTED = 20  # finished tasks kept for list() and get_task_result (the read ones go first)
SHUTDOWN_GRACE = 3.0
NOTIFICATION_PROMPT = (
    "[Background task finished] A task you started earlier finished while no turn was running:\n{facts}\n"
    "Read its result with get_task_result, then tell the user what matters from it. If there is nothing worth "
    "telling, reply with exactly <no_update/>."
)
NOTIFICATION_HEADER = "🔔 *Background task finished*"
NOTIFICATION_LABEL = "background task result"
NOTIFICATION_CAPTION = "Background task reasoning history."

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"


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
    silent: bool = False  # cancelled by /cancel, a failed turn or shutdown: nothing is announced
    turn: int = 0  # the host's turn number when it was adopted (cancel_turn)
    result: Any = None  # the raw outcome once finished (``_outcome``), until the record is pruned
    read: bool = False  # the agent has the result (get_task_result / wait_for_tasks); a cancel has nothing to read

    @property
    def unread(self) -> bool:
        return self.status in (DONE, FAILED) and not self.read and not self.silent

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
            "unread": self.unread,
        }


# ---- the notice and the result ---------------------------------------------------------------
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


def _facts(record: BackgroundTask) -> str:
    """``background task t3 (run_command) completed after 42 s``: what a notice may say (no output)."""
    outcome = {DONE: "completed after", FAILED: "failed after", CANCELLED: "was cancelled after"}.get(record.status, record.status)
    return f"background task {record.id} ({record.tool}) {outcome} {record.elapsed()} s"


def notice_text(record: BackgroundTask) -> str:
    """The notice the agent gets when a task ends: facts only, never the output (that is read
    with ``get_task_result``, inside a tool result). A cancelled task has nothing to read."""
    if record.status == CANCELLED:
        return f"The {_facts(record)}."
    what = "the error" if record.status == FAILED else "its result"
    return f'The {_facts(record)}. Read {what} with get_task_result (task_id "{record.id}").'


def notice_blocks(record: BackgroundTask) -> list[dict]:
    """The inbox's blocks for a task that ended: one notice block (``agents.task_tools.NOTICE_BLOCK``,
    which the agent core sends as a ``system`` message), or nothing once the agent has read the
    result anyway (``wait_for_tasks`` handed it over before the notice was drained)."""
    from agents.task_tools import NOTICE_BLOCK

    if record.status in (DONE, FAILED) and record.read:
        return []
    return [{"type": NOTICE_BLOCK, "text": notice_text(record)}]


def result_blocks(record: BackgroundTask) -> list[dict]:
    """What ``get_task_result`` returns: a header line, the result's text (capped at the call's
    ``max_chars``) and its images, as tool-result content."""
    from agents import content, transcript

    result = record.result
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
    head = f"Result of the {_facts(record)}:"
    return [content.text_block(f"{head}\n\n{body}" if body else head), *media]


# ---- notification turns (no turn of that topic is running) -------------------------------------------
@dataclass(eq=False)
class _Batch:
    """A queued "background task finished" turn; tasks that end join it until it starts."""

    user_id: str
    thread_id: int | None
    records: list[BackgroundTask] = field(default_factory=list)
    started: bool = False


_batches: dict[tuple[str, int | None], _Batch] = {}


def submit_notification(user_id: str, thread_id: int | None, records: list[BackgroundTask], *, seq: int | None = None) -> None:
    """Queue a notification turn for the tasks in ``records`` (or add them to the one still waiting).
    Never suspends; raises only when the queue is closed."""
    from bot import runtime

    key = (str(user_id), thread_id)
    batch = _batches.get(key)
    if batch is not None and not batch.started:
        batch.records.extend(r for r in records if r not in batch.records)
        return
    batch = _Batch(str(user_id), thread_id, list(records))
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


def notification_prompt(records: list[BackgroundTask]) -> str:
    """The framing of a notification turn: the facts of each task, no output (read with get_task_result)."""
    return NOTIFICATION_PROMPT.format(facts="\n".join(f"- The {_facts(r)}." for r in records))


async def _run_notification(bot: Any, batch: _Batch, ctx: JobContext) -> None:
    """The queued turn: what finished, framed so the agent reads it and tells the user what matters
    (or stays quiet). A task whose result the agent has read meanwhile is left out; with none left
    there is no turn."""
    from bot.agent_runner import QUIET_MARKER, run_turn
    from bot.limits import check_user_limits

    batch.started = True
    if _batches.get((batch.user_id, batch.thread_id)) is batch:
        del _batches[(batch.user_id, batch.thread_id)]
    records = [r for r in batch.records if r.unread]
    if not records:
        return
    host = _hosts.get(batch.user_id)
    bot = bot or (host.bot if host else None)
    if bot is None:
        logger.error("No bot to tell %s about a finished background task", batch.user_id)
        return
    _, _, limit = await asyncio.to_thread(check_user_limits, batch.user_id)
    await run_turn(
        bot=bot, user_id=batch.user_id, chat_id=int(batch.user_id), prompt=notification_prompt(records),
        thread_id=batch.thread_id, ctx=ctx, limit=limit, header=NOTIFICATION_HEADER, reasoning_caption=NOTIFICATION_CAPTION,
        quiet_marker=QUIET_MARKER,
    )


# ---- the host -------------------------------------------------------------------------------------------
class BackgroundHost:
    """One user's background tasks (the ``BackgroundHost`` protocol of the agent core).

    ``bind`` (``run_turn``, at the start of each turn) records the bot, the chat and the topic
    of the running turn and starts a new turn number: a task adopted now reports to that topic,
    and the task tools of the turn (``running``, ``list``, ``cancel``) see that topic's tasks
    only. The cap (``can_adopt``), ``cancel_all`` and shutdown count every topic.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = str(user_id)
        self.bot: Any = None
        self.chat_id: int | None = None
        self.thread_id: int | None = None
        self._counter = 0
        self._turn = 0
        self._tasks: dict[str, BackgroundTask] = {}

    def bind(self, *, bot: Any, chat_id: int | None, thread_id: int | None) -> "BackgroundHost":
        self.bot = bot
        self.chat_id = chat_id
        self.thread_id = thread_id
        self._turn += 1
        return self

    @property
    def turn(self) -> int:
        """The number of the bound turn (``cancel_turn``)."""
        return self._turn

    def _all_running(self) -> list[BackgroundTask]:
        return [t for t in self._tasks.values() if t.status == RUNNING]

    def _here(self, record: BackgroundTask) -> bool:
        return record.thread_id == self.thread_id

    # -- the protocol -------------------------------------------------------------------------------
    def running(self) -> list[BackgroundTask]:
        """The running tasks of the bound turn's topic."""
        return [t for t in self._all_running() if self._here(t)]

    def can_adopt(self) -> bool:
        return not _closed and len(self._all_running()) < MAX_BACKGROUND_TASKS

    def adopt(
        self, task: asyncio.Task, *, tool_name: str, tool_use_id: str, display_args: str,
        started: float, max_chars: int | None,
    ) -> str | None:
        if _closed:
            # Shutting down: nothing may outlive the process. The call ends here (None: not taken,
            # so the turn sees it end as cancelled).
            if not task.done():
                task.cancel()
            return None
        self._counter += 1
        task_id = f"t{self._counter}"
        record = BackgroundTask(
            id=task_id, tool=str(tool_name), tool_use_id=str(tool_use_id), args=str(display_args or ""), task=task,
            started_wall=_wall(float(started)), max_chars=max_chars, thread_id=self.thread_id, turn=self._turn,
        )
        self._tasks[task_id] = record
        self._prune()
        task.add_done_callback(partial(self._finished, record))
        return task_id

    def list(self) -> list[dict]:
        return [t.entry() for t in self._tasks.values() if self._here(t)]

    def unread(self) -> list[str]:
        """Ids of the bound topic's finished tasks whose result the agent has not read, oldest end first."""
        done = [t for t in self._tasks.values() if self._here(t) and t.unread]
        return [t.id for t in sorted(done, key=lambda t: t.finished_wall or 0.0)]

    def read(self, task_id: str) -> list[dict] | None:
        """The result of a finished task of the bound topic as tool-result content, marking it read;
        None for an unknown, pruned or still running one."""
        record = self._tasks.get(str(task_id).strip())
        if record is None or not self._here(record) or record.status == RUNNING:
            return None
        record.read = True
        return result_blocks(record)

    def cancel(self, task_id: str) -> bool:
        record = self._tasks.get(str(task_id).strip())
        if record is None or not self._here(record):
            return False
        if record.status != RUNNING or record.task.done() or record.task.cancelling():
            return False
        record.task.cancel()
        return True

    # -- the rest -------------------------------------------------------------------------------------
    @staticmethod
    def _cancel_silently(records: list[BackgroundTask]) -> int:
        count = 0
        for record in records:
            record.silent = True  # also one the agent is already cancelling: nothing is announced
            if not record.task.done() and not record.task.cancelling():
                record.task.cancel()
                count += 1
        return count

    def cancel_all(self) -> int:
        """Cancel every running task of the user, in every topic, without announcing it (/cancel, shutdown)."""
        return self._cancel_silently(self._all_running())

    def cancel_turn(self, turn: int) -> int:
        """Cancel, without announcing it, every running task adopted during turn ``turn`` (a turn
        that failed: its Retry runs the calls again)."""
        return self._cancel_silently([t for t in self._all_running() if t.turn == turn])

    def _prune(self) -> None:
        """Keep at most ``MAX_LISTED`` finished tasks: the ones already read go first, oldest first."""
        finished = [k for k, t in self._tasks.items() if t.status != RUNNING]
        excess = len(finished) - MAX_LISTED
        if excess <= 0:
            return
        order = [k for k in finished if not self._tasks[k].unread] + [k for k in finished if self._tasks[k].unread]
        for key in order[:excess]:
            del self._tasks[key]

    def _requeue_notice(self, record: BackgroundTask, seq: int) -> None:
        """A notice the turn did not read: a notification turn, unless the result was read anyway."""
        if record.unread:
            submit_notification(self.user_id, record.thread_id, [record], seq=seq)

    def _finished(self, record: BackgroundTask, task: asyncio.Task) -> None:
        status, result = _outcome(task)
        record.status = status
        record.result = result
        record.read = status == CANCELLED  # nothing to read
        record.finished_wall = time.time()
        self._prune()
        if record.silent:
            return
        # The notice is rendered when the turn reads it: by then the agent may have read the result.
        item = turn_inbox.InboxItem(
            kind=turn_inbox.TASK, future=turn_inbox.ready_future(partial(notice_blocks, record)), thread_id=record.thread_id,
            label=f"task {record.id}",
        )
        item.requeue = partial(self._requeue_notice, record, item.seq)
        if turn_inbox.deliver_task(self.user_id, record.thread_id, item):
            return
        if status == CANCELLED:
            return  # the agent cancelled it itself: worth a line in a running turn, not a turn of its own
        try:
            submit_notification(self.user_id, record.thread_id, [record])
        except Exception as e:  # noqa: BLE001 - the queue is closed (shutdown)
            logger.warning("Could not queue the result of background task %s for %s: %s", record.id, self.user_id, e)


_hosts: dict[str, BackgroundHost] = {}
_registry_loop: asyncio.AbstractEventLoop | None = None
_closed = False  # shutdown() ran: no host adopts anything any more


def _same_loop() -> None:
    """Tasks belong to the event loop they run on: a new loop (every test) starts with no hosts."""
    global _registry_loop, _closed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if loop is not _registry_loop:
        _hosts.clear()
        _batches.clear()
        _closed = False
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
    """Cancel every user's tasks (nothing announced) and give them ``grace`` seconds to end. From
    now on no host adopts a call: a turn still running hands none over (``can_adopt`` is False,
    and a task given to ``adopt`` anyway is cancelled)."""
    global _closed
    _same_loop()
    _closed = True
    tasks = []
    for host in _hosts.values():
        tasks += [r.task for r in host._all_running()]
        host.cancel_all()
    _batches.clear()
    if tasks:
        await asyncio.wait(tasks, timeout=grace)


def reset() -> None:
    """Forget every host and queued notification (tests)."""
    global _closed
    _hosts.clear()
    _batches.clear()
    _closed = False
