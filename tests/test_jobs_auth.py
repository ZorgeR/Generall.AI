from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import bot.jobs as jobs
from bot.queue import QueueManager
from reminders_store import RemindersStore


class FakeAuth:
    def __init__(self, allowed):
        self.allowed = set(allowed)

    def is_authorized(self, user_id):
        return str(user_id) in self.allowed


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(message_id=len(self.sent))


async def test_reminders_of_unauthorized_users_are_not_delivered(tmp_path, monkeypatch):
    store = RemindersStore(tmp_path)
    monkeypatch.setattr(jobs, "reminders_store", store)
    monkeypatch.setattr(jobs, "auth", FakeAuth({"1"}))
    due = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    await store.add("1", {"text": "hello", "time": due, "type": "user", "status": "pending"})
    await store.add("2", {"text": "blocked user", "time": due, "type": "user", "status": "pending"})
    await store.add("2", {"text": "agent task", "time": due, "type": "agent", "status": "pending"})
    bot = FakeBot()
    qm = QueueManager(max_concurrent_turns=1, turn_timeout=5)

    await jobs.check_reminders(bot, qm)

    assert [kw["chat_id"] for kw in bot.sent] == [1]
    assert (await store.load("1"))[0]["status"] == "completed"
    assert [r["status"] for r in await store.load("2")] == ["pending", "pending"]  # untouched, not queued
    assert qm.is_busy("2") is False
    await qm.shutdown()
