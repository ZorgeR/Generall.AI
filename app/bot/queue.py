"""Per-user job queues.

Every chat has its own FIFO queue and a worker task that runs one job at a
time, so a user's messages are processed strictly in order while different
users never wait for each other. Long jobs are bounded by a deadline, a global
semaphore caps how many turns run at once, and ``cancel()`` implements
``/cancel``.

Every job has a random id. The Stop buttons of a running turn (the one in its
status message and Telegram's own button on its streaming draft, whose
``draft_id`` is the job id) name that id, and ``cancel_current()`` stops the
running job only when it is still the one the button belongs to: a late press
never hits the next turn, and queued messages are kept.

Jobs must be cooperative: they run as asyncio tasks, so blocking work inside
them has to be offloaded with ``asyncio.to_thread``. A job that ignores
cancellation cannot be stopped.

Every job also carries ``seq``, a process-wide arrival number (``next_seq``), and a
queued job is inserted before every pending job with a larger one. Jobs created
when they are submitted simply go to the end; a job submitted later for something
that arrived earlier (a message the running turn's inbox did not consume,
``bot.inbox``, or an album flushed after a text) keeps its place. ``thread_id`` and
``from_user`` say whether a pending job holds a user's message in a forum topic:
``has_pending_user_jobs`` is how the handlers know a new message must queue behind
it instead of joining the running turn. ``on_drop`` is called when ``cancel()`` drops a
job before it ran (a requeued inbox message cancels its preparation still in flight).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Callback data of the Stop button on a running turn: "queue_stop_<Job.token>".
STOP_CALLBACK_PREFIX = "queue_stop_"
_JOB_ID_LIMIT = 2**31  # ids are 1 .. 2**31 - 1
_TOKEN_RE = re.compile(r"[0-9a-f]{1,8}")
_SEQ = itertools.count(1)


def next_seq() -> int:
    """A process-wide increasing arrival number (jobs, inbox items): the order the user sent things in."""
    return next(_SEQ)


def new_job_id() -> int:
    """A random non-zero 31-bit id: a valid Telegram ``draft_id`` and short in callback data."""
    return secrets.randbelow(_JOB_ID_LIMIT - 1) + 1


def job_id_from_token(token: str | None) -> int | None:
    """The inverse of ``Job.token``; None for anything that is not one (stale or forged callback data)."""
    if not token or not _TOKEN_RE.fullmatch(token):
        return None
    value = int(token, 16)
    return value if 0 < value < _JOB_ID_LIMIT else None


@dataclass
class Job:
    user_id: str
    label: str
    run: Callable[["JobContext"], Awaitable[Any]]
    created_at: float = field(default_factory=time.monotonic)
    progress: str = ""
    started_at: float | None = None  # set when the job actually starts (after the slot wait)
    id: int = field(default_factory=new_job_id)  # also the draft_id of the turn's streaming drafts
    seq: int = field(default_factory=next_seq)  # arrival order: a queued job goes before pending jobs with a larger one
    thread_id: int | None = None  # the forum topic of the message it answers (None: none / not a topic)
    from_user: bool = False  # holds a user's message or button press (not a scheduled task or a notification)
    on_drop: Callable[[], Any] | None = None  # cancel() dropped it before it ran (never raises into cancel)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0

    @property
    def token(self) -> str:
        """The id as lowercase hex, for callback data (see ``STOP_CALLBACK_PREFIX``)."""
        return format(self.id, "x")


class JobContext:
    """Handed to a running job so it can publish progress and observe cancellation."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.cancel_reason: str | None = None
        # Cleared by the job once its answer exists and is being delivered: from then on
        # its Stop buttons (cancel_current) no longer apply; /cancel (cancel()) still does.
        self.stoppable = True

    def set_progress(self, text: str) -> None:
        self.job.progress = text


@dataclass
class SubmitResult:
    was_busy: bool
    position: int  # 0 = will run immediately, N = N jobs ahead (including the running one)
    current: Job | None


@dataclass
class QueueStatus:
    current: Job | None
    pending: int


class _UserQueue:
    def __init__(self) -> None:
        # One token per pending job: the worker waits on it, then takes ``pending[0]``. The jobs
        # themselves live in ``pending`` (sorted by ``seq``), so one can be inserted in its place.
        self.queue: asyncio.Queue[None] = asyncio.Queue()
        self.pending: list[Job] = []
        self.worker: asyncio.Task | None = None
        self.current: Job | None = None
        self.current_task: asyncio.Task | None = None
        self.context: JobContext | None = None


def _worker_is_being_cancelled() -> bool:
    task = asyncio.current_task()
    return bool(task and task.cancelling())


class QueueManager:
    def __init__(
        self,
        *,
        max_concurrent_turns: int = 8,
        turn_timeout: float = 1800.0,
        on_timeout: Callable[[Job], Awaitable[None]] | None = None,
        on_error: Callable[[Job, BaseException], Awaitable[None]] | None = None,
        shutdown_grace: float = 10.0,
    ) -> None:
        self._users: dict[str, _UserQueue] = {}
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_turns))
        self.turn_timeout = turn_timeout
        self.on_timeout = on_timeout
        self.on_error = on_error
        self.shutdown_grace = shutdown_grace
        self._closed = False

    # ---- public API ------------------------------------------------------
    def status(self, user_id: str) -> QueueStatus:
        uq = self._users.get(str(user_id))
        if uq is None:
            return QueueStatus(current=None, pending=0)
        return QueueStatus(current=uq.current, pending=len(uq.pending))

    def is_busy(self, user_id: str) -> bool:
        st = self.status(user_id)
        return st.current is not None or st.pending > 0

    def has_pending_user_jobs(self, user_id: str, thread_id: int | None) -> bool:
        """A queued (not yet running) job holds a message or button press of the user in that topic."""
        uq = self._users.get(str(user_id))
        return uq is not None and any(j.from_user and j.thread_id == thread_id for j in uq.pending)

    async def submit(self, job: Job) -> SubmitResult:
        return self.submit_nowait(job)

    def submit_nowait(self, job: Job) -> SubmitResult:
        """``submit`` without a suspension point: callable from cleanup code (``finally``, a cancelled
        task) and atomic with whatever the caller checked just before."""
        if self._closed:
            raise RuntimeError("queue manager is closed")
        uq = self._get(job.user_id)
        index = len(uq.pending)
        while index > 0 and uq.pending[index - 1].seq > job.seq:
            index -= 1  # something that arrived after this job is waiting: go before it
        position = index + (1 if uq.current is not None else 0)
        was_busy = position > 0
        # The job the user is waiting on: the running one, or the head of the queue
        # when the worker has not picked it up yet.
        current = uq.current or (uq.pending[0] if uq.pending else None)
        uq.pending.insert(index, job)
        uq.queue.put_nowait(None)
        if uq.worker is None or uq.worker.done():
            uq.worker = asyncio.create_task(self._worker(job.user_id, uq), name=f"queue-worker-{job.user_id}")
        return SubmitResult(was_busy=was_busy, position=position, current=current)

    async def cancel(self, user_id: str, reason: str = "user") -> tuple[bool, int]:
        """Cancel the running (or slot-waiting) job and drop queued ones. Returns (cancelled_current, dropped)."""
        uq = self._users.get(str(user_id))
        if uq is None:
            return False, 0
        dropped = 0
        while not uq.queue.empty():
            try:
                uq.queue.get_nowait()
                uq.queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        dropped_jobs, uq.pending = uq.pending, []
        for job in dropped_jobs:
            if job.on_drop is not None:
                try:
                    job.on_drop()
                except Exception as e:  # noqa: BLE001
                    logger.debug("on_drop of %r for %s failed: %s", job.label, user_id, e)
        cancelled = False
        if uq.current_task is not None and not uq.current_task.done():
            # A job already stopping (a Stop button, the deadline) is left to finish its
            # cleanup: a second cancel would cut the partial answer and the "Stopped." edit.
            if not uq.current_task.cancelling():
                if uq.context is not None:
                    uq.context.cancel_reason = reason
                uq.current_task.cancel()
            cancelled = True
        return cancelled, dropped

    async def cancel_current(self, user_id: str, job_id: int | None = None, reason: str = "user") -> bool:
        """Cancel only the running (or slot-waiting) job; queued jobs stay and run next.

        With ``job_id`` the job is cancelled only if it is still the running one, so a
        Stop button pressed after its turn ended never stops the next turn. A job that
        cleared ``ctx.stoppable`` (its answer is being delivered) is not cancelled, and a
        job already being cancelled is not cancelled again. Returns True when the job is
        (now or already) stopping; ``ctx.cancel_reason`` is set as in ``cancel()``.
        """
        uq = self._users.get(str(user_id))
        if uq is None or uq.current is None or uq.current_task is None or uq.current_task.done():
            return False
        if job_id is not None and uq.current.id != job_id:
            return False
        if uq.context is not None and not uq.context.stoppable:
            return False  # the answer is complete and being delivered; nothing left to stop
        if uq.current_task.cancelling():
            # Already stopping (e.g. the draft's Stop and the status's Stop both pressed):
            # a second cancel would interrupt the job's cleanup (partial answer, "Stopped.").
            return True
        if uq.context is not None:
            uq.context.cancel_reason = reason
        uq.current_task.cancel()
        return True

    async def shutdown(self) -> None:
        self._closed = True
        workers = [uq.worker for uq in self._users.values() if uq.worker and not uq.worker.done()]
        for uq in self._users.values():
            if uq.current_task and not uq.current_task.done():
                if uq.context is not None:
                    uq.context.cancel_reason = "shutdown"
                uq.current_task.cancel()
        for w in workers:
            w.cancel()
        if workers:
            try:
                await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=self.shutdown_grace)
            except asyncio.TimeoutError:
                logger.warning("Queue workers did not stop within %.0fs; continuing shutdown", self.shutdown_grace)

    # ---- internals -------------------------------------------------------
    def _get(self, user_id: str) -> _UserQueue:
        return self._users.setdefault(str(user_id), _UserQueue())

    async def _run_job(self, job: Job, ctx: JobContext) -> None:
        """Wait for a global slot, then run the job under the deadline.

        Runs inside the cancellable task so /cancel also works while waiting for a slot.
        """
        async with self._semaphore:
            job.started_at = time.monotonic()
            await asyncio.wait_for(job.run(ctx), timeout=self.turn_timeout)

    async def _worker(self, user_id: str, uq: _UserQueue) -> None:
        while True:
            await uq.queue.get()
            if not uq.pending:  # a token whose job was dropped (cannot happen: cancel() drains both)
                uq.queue.task_done()
                continue
            job = uq.pending.pop(0)
            ctx = JobContext(job)
            uq.current = job
            uq.context = ctx
            task = asyncio.create_task(self._run_job(job, ctx), name=f"job-{user_id}-{job.label}")
            uq.current_task = task
            try:
                await task
            except asyncio.TimeoutError:
                logger.warning("Job %r for %s timed out after %.0fs", job.label, user_id, self.turn_timeout)
                if self.on_timeout:
                    await self._safe(self.on_timeout(job))
            except asyncio.CancelledError:
                if _worker_is_being_cancelled():
                    raise  # the worker itself is being cancelled (shutdown)
                logger.info("Job %r for %s cancelled (%s)", job.label, user_id, ctx.cancel_reason)
            except Exception as e:  # noqa: BLE001
                # A job may swallow CancelledError and raise something else instead
                # (for example a Telegram error while editing "🛑 Stopped."). If the
                # worker itself was asked to stop, honour that instead of looping on.
                if _worker_is_being_cancelled():
                    raise asyncio.CancelledError() from e
                logger.exception("Job %r for %s failed: %s", job.label, user_id, e)
                if self.on_error:
                    await self._safe(self.on_error(job, e))
            else:
                if _worker_is_being_cancelled():
                    raise asyncio.CancelledError()
            finally:
                uq.current = None
                uq.current_task = None
                uq.context = None
                uq.queue.task_done()

    @staticmethod
    async def _safe(coro: Awaitable[None]) -> None:
        try:
            await coro
        except Exception as e:  # noqa: BLE001
            logger.error("Queue callback failed: %s", e)
