"""Per-user job queues.

Every chat has its own FIFO queue and a worker task that runs one job at a
time, so a user's messages are processed strictly in order while different
users never wait for each other. Long jobs are bounded by a deadline, a global
semaphore caps how many turns run at once, and ``cancel()`` implements
``/cancel``.

Jobs must be cooperative: they run as asyncio tasks, so blocking work inside
them has to be offloaded with ``asyncio.to_thread``. A job that ignores
cancellation cannot be stopped.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class Job:
    user_id: str
    label: str
    run: Callable[["JobContext"], Awaitable[Any]]
    created_at: float = field(default_factory=time.monotonic)
    progress: str = ""
    started_at: float | None = None  # set when the job actually starts (after the slot wait)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0


class JobContext:
    """Handed to a running job so it can publish progress and observe cancellation."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.cancel_reason: str | None = None

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
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.pending: list[Job] = []  # mirror of the queue contents, in order
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

    async def submit(self, job: Job) -> SubmitResult:
        if self._closed:
            raise RuntimeError("queue manager is closed")
        uq = self._get(job.user_id)
        position = len(uq.pending) + (1 if uq.current is not None else 0)
        was_busy = position > 0
        # The job the user is waiting on: the running one, or the head of the queue
        # when the worker has not picked it up yet.
        current = uq.current or (uq.pending[0] if uq.pending else None)
        uq.pending.append(job)
        await uq.queue.put(job)
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
        uq.pending.clear()
        cancelled = False
        if uq.current_task is not None and not uq.current_task.done():
            if uq.context is not None:
                uq.context.cancel_reason = reason
            uq.current_task.cancel()
            cancelled = True
        return cancelled, dropped

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
            job = await uq.queue.get()
            if uq.pending and uq.pending[0] is job:
                uq.pending.pop(0)
            elif job in uq.pending:
                uq.pending.remove(job)
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
