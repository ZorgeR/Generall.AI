import asyncio

import pytest

from bot.queue import Job, JobContext, QueueManager


def make_job(user_id: str, label: str, coro_factory):
    return Job(user_id=user_id, label=label, run=coro_factory)


async def test_same_user_jobs_run_in_order_and_second_is_reported_busy():
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5)
    order: list[str] = []

    async def job(name: str, delay: float, ctx: JobContext) -> None:
        await asyncio.sleep(delay)
        order.append(name)

    r1 = await qm.submit(make_job("u1", "first", lambda ctx: job("a", 0.05, ctx)))
    r2 = await qm.submit(make_job("u1", "second", lambda ctx: job("b", 0.0, ctx)))
    assert r1.was_busy is False and r1.position == 0
    assert r2.was_busy is True and r2.position == 1
    assert r2.current is not None and r2.current.label == "first"
    await asyncio.sleep(0.3)
    assert order == ["a", "b"]
    assert qm.is_busy("u1") is False
    await qm.shutdown()


async def test_one_user_never_blocks_another():
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5)
    done: list[str] = []

    async def slow(ctx: JobContext) -> None:
        await asyncio.sleep(0.5)
        done.append("slow-user")

    async def fast(ctx: JobContext) -> None:
        await asyncio.sleep(0.01)
        done.append("fast-user")

    await qm.submit(make_job("slow", "video generation", slow))
    await asyncio.sleep(0.05)
    result = await qm.submit(make_job("fast", "text", fast))
    assert result.was_busy is False  # a different user is never "busy" because of someone else
    await asyncio.sleep(0.2)
    assert done == ["fast-user"]
    await asyncio.sleep(0.5)
    assert done == ["fast-user", "slow-user"]
    await qm.shutdown()


async def test_cancel_stops_current_and_drops_queued():
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5)
    events: list[str] = []

    async def long_job(ctx: JobContext) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append(f"cancelled:{ctx.cancel_reason}")
            raise

    async def queued_job(ctx: JobContext) -> None:
        events.append("queued-ran")

    await qm.submit(make_job("u1", "long", long_job))
    await qm.submit(make_job("u1", "queued", queued_job))
    await asyncio.sleep(0.05)
    cancelled, dropped = await qm.cancel("u1", reason="user")
    assert cancelled is True and dropped == 1
    await asyncio.sleep(0.1)
    assert events == ["cancelled:user"]
    assert qm.status("u1").current is None

    # The worker is still alive and accepts new jobs after a cancel.
    async def after(ctx: JobContext) -> None:
        events.append("after")

    await qm.submit(make_job("u1", "after", after))
    await asyncio.sleep(0.1)
    assert events[-1] == "after"
    await qm.shutdown()


async def test_timeout_notifies_and_worker_continues():
    timed_out: list[str] = []

    async def on_timeout(job: Job) -> None:
        timed_out.append(job.label)

    qm = QueueManager(max_concurrent_turns=4, turn_timeout=0.1, on_timeout=on_timeout)
    ran: list[str] = []

    async def stuck(ctx: JobContext) -> None:
        await asyncio.sleep(5)

    async def next_job(ctx: JobContext) -> None:
        ran.append("next")

    await qm.submit(make_job("u1", "stuck", stuck))
    await qm.submit(make_job("u1", "next", next_job))
    await asyncio.sleep(0.4)
    assert timed_out == ["stuck"]
    assert ran == ["next"]
    await qm.shutdown()


async def test_job_exception_is_reported_and_does_not_kill_worker():
    errors: list[str] = []

    async def on_error(job: Job, exc: BaseException) -> None:
        errors.append(f"{job.label}:{type(exc).__name__}")

    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5, on_error=on_error)
    ran: list[str] = []

    async def boom(ctx: JobContext) -> None:
        raise ValueError("bad")

    async def ok(ctx: JobContext) -> None:
        ran.append("ok")

    await qm.submit(make_job("u1", "boom", boom))
    await qm.submit(make_job("u1", "ok", ok))
    await asyncio.sleep(0.2)
    assert errors == ["boom:ValueError"]
    assert ran == ["ok"]
    await qm.shutdown()


async def test_progress_is_visible_in_status():
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5)

    async def job(ctx: JobContext) -> None:
        ctx.set_progress("executing tool search_web")
        await asyncio.sleep(0.2)

    await qm.submit(make_job("u1", "text", job))
    await asyncio.sleep(0.05)
    st = qm.status("u1")
    assert st.current is not None and st.current.progress == "executing tool search_web"
    assert st.current.elapsed >= 0
    await qm.shutdown()


async def test_global_concurrency_cap_limits_parallel_turns():
    qm = QueueManager(max_concurrent_turns=2, turn_timeout=5)
    running = 0
    peak = 0

    async def job(ctx: JobContext) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.1)
        running -= 1

    for uid in ("a", "b", "c", "d"):
        await qm.submit(make_job(uid, "t", job))
    await asyncio.sleep(0.5)
    assert peak == 2
    await qm.shutdown()


@pytest.mark.parametrize("n", [1, 3])
async def test_shutdown_cancels_running_jobs(n: int):
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5)
    cancelled = 0

    async def job(ctx: JobContext) -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled += 1
            raise

    for i in range(n):
        await qm.submit(make_job(f"u{i}", "t", job))
    await asyncio.sleep(0.05)
    await qm.shutdown()
    assert cancelled == n


async def test_cancel_works_while_waiting_for_a_global_slot():
    """/cancel must stop a job that is still waiting for a free turn slot."""
    qm = QueueManager(max_concurrent_turns=1, turn_timeout=5)
    started: list[str] = []

    async def hog(ctx: JobContext) -> None:
        started.append("hog")
        await asyncio.sleep(0.4)

    async def waiter(ctx: JobContext) -> None:
        started.append("waiter")

    await qm.submit(make_job("a", "hog", hog))
    await asyncio.sleep(0.02)
    await qm.submit(make_job("b", "waiter", waiter))
    await asyncio.sleep(0.02)
    assert qm.status("b").current is not None
    assert qm.status("b").current.started_at is None  # waiting for a slot, not started
    cancelled, dropped = await qm.cancel("b", reason="user")
    assert cancelled is True and dropped == 0
    await asyncio.sleep(0.5)
    assert started == ["hog"]  # the waiting job never ran
    assert qm.is_busy("b") is False
    await qm.shutdown()


async def test_shutdown_completes_even_if_job_raises_on_cancel():
    """A job that turns CancelledError into another exception must not keep the worker alive."""
    qm = QueueManager(max_concurrent_turns=4, turn_timeout=5, shutdown_grace=1)

    async def bad_job(ctx: JobContext) -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise RuntimeError("telegram edit failed while stopping")

    await qm.submit(make_job("u1", "bad", bad_job))
    await asyncio.sleep(0.02)
    await asyncio.wait_for(qm.shutdown(), timeout=2)
    assert qm.is_busy("u1") is False


async def test_timeout_fires_only_after_job_actually_starts():
    """The deadline is measured from the moment the job gets a slot, not from submission."""
    fired: list[str] = []

    async def on_timeout(job: Job) -> None:
        fired.append(job.label)

    qm = QueueManager(max_concurrent_turns=1, turn_timeout=0.3, on_timeout=on_timeout)

    async def hog(ctx: JobContext) -> None:
        await asyncio.sleep(0.25)

    async def quick(ctx: JobContext) -> None:
        await asyncio.sleep(0.1)

    await qm.submit(make_job("a", "hog", hog))
    await qm.submit(make_job("b", "quick", quick))  # waits ~0.25s for a slot, then runs 0.1s
    await asyncio.sleep(0.6)
    assert fired == []
    await qm.shutdown()
