import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from reminders_store import RemindersStore, calculate_next_trigger, new_reminder_id


def test_next_trigger_is_in_the_future_and_steps_by_period():
    past = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    nxt = calculate_next_trigger(past, "daily", 1)
    assert nxt > datetime.now(timezone.utc)
    assert nxt - datetime.now(timezone.utc) < timedelta(days=1)


def test_unknown_period_type_raises_instead_of_unbound_variable():
    with pytest.raises(ValueError):
        calculate_next_trigger(datetime.now(timezone.utc).isoformat(), "fortnightly", 1)
    with pytest.raises(ValueError):
        calculate_next_trigger(datetime.now(timezone.utc).isoformat(), "daily", 0)


def test_ids_never_collide():
    existing = [{"id": "aaaaaaaa"}, {"id": "bbbbbbbb"}]
    ids = {new_reminder_id(existing) for _ in range(50)}
    assert not ids & {"aaaaaaaa", "bbbbbbbb"}


async def test_add_update_and_listing(tmp_path):
    store = RemindersStore(base_dir=tmp_path)
    assert list(store.user_ids()) == []
    r = await store.add("77", {"text": "hello", "status": "pending", "type": "user"})
    assert r["id"]
    assert list(store.user_ids()) == ["77"]

    def toggle(reminders):
        for item in reminders:
            if item["id"] == r["id"]:
                item["enabled"] = False
                return True
        return False

    assert await store.update("77", toggle) is True
    loaded = await store.load("77")
    assert loaded[0]["enabled"] is False


async def test_concurrent_updates_do_not_lose_writes(tmp_path):
    store = RemindersStore(base_dir=tmp_path)
    await store.save("5", [])

    async def add_one(i: int) -> None:
        await store.add("5", {"text": f"r{i}", "status": "pending", "type": "user"})

    await asyncio.gather(*(add_one(i) for i in range(20)))
    loaded = await store.load("5")
    assert len(loaded) == 20
    assert len({r["id"] for r in loaded}) == 20


async def test_update_does_not_rewrite_file_when_nothing_changed(tmp_path):
    store = RemindersStore(tmp_path)
    await store.add("1", {"text": "x", "time": "2030-01-01T00:00:00+00:00", "type": "user", "status": "pending"})
    path = store.path("1")
    before = path.stat().st_mtime_ns
    await store.update("1", lambda reminders: None)
    assert path.stat().st_mtime_ns == before
    await store.update("1", lambda reminders: reminders[0].__setitem__("status", "completed"))
    assert path.stat().st_mtime_ns != before
