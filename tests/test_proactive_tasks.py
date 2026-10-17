"""Scheduled tasks and local-time reminders: the set_user_timezone tool,
schedule_reminder's local-time conversion and weekdays, calculate_next_trigger, the scheduler's
framing prompt / quiet marker, and the snooze buttons on fired user reminders."""
import datetime as dt
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

import bot.jobs as jobs
import reminders_store as rs
from agents.user_interactions import UserInteractions
from bot import timezones
from bot.agent_runner import QUIET_MARKER, TurnResult
from bot.queue import Job, JobContext, QueueManager
from bot.settings import UserSettings
from reminders_store import RemindersStore, calculate_next_trigger, first_allowed_trigger

USER = "42"
BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Settings under tmp_path/data; the agent tool, the scheduler and the test share one store there."""
    monkeypatch.chdir(tmp_path)
    store = RemindersStore(base_dir=tmp_path / "data")
    monkeypatch.setattr("agents.user_interactions.reminders_store", store)
    monkeypatch.setattr(jobs, "reminders_store", store)
    monkeypatch.setattr(jobs, "check_user_limits", lambda uid: (True, 0, None))
    return store


class FakeSender:
    def __init__(self):
        self.sent = []

    async def send_markdown(self, text):
        self.sent.append(text)


def tool(sender=None):
    return UserInteractions(USER, sender or FakeSender())


def set_zone(name):
    UserSettings(USER).set("timezone", name, "name")


def freeze_now(monkeypatch, moment: datetime):
    """``datetime.now`` inside reminders_store returns ``moment``."""

    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(rs, "datetime", FrozenDatetime)


# ---- set_user_timezone ----------------------------------------------------------------------
async def test_set_user_timezone_saves_the_zone_and_returns_the_local_time():
    result = await tool().execute_tool("set_user_timezone", {"timezone": "berlin"})
    assert result.startswith("Time zone saved: Europe/Berlin. The user's local time is now ")
    assert "(Europe/Berlin, UTC+" in result
    assert UserSettings(USER).get("timezone", "name") == "Europe/Berlin"
    result = await tool().execute_tool("set_user_timezone", {"timezone": "America/New_York"})
    assert UserSettings(USER).get("timezone", "name") == "America/New_York" and "America/New_York" in result


async def test_set_user_timezone_rejects_unknown_names():
    set_zone("Asia/Tokyo")
    result = await tool().execute_tool("set_user_timezone", {"timezone": "Narnia"})
    assert result.startswith("Error: unknown time zone 'Narnia'")
    assert UserSettings(USER).get("timezone", "name") == "Asia/Tokyo"
    assert (await tool().execute_tool("set_user_timezone", {})).startswith("Error")


def test_tool_schemas_and_prompt_notes_describe_local_time_weekdays_and_the_quiet_marker():
    schemas = {t["name"]: t for t in tool().tools_schema}
    assert "set_user_timezone" in schemas and schemas["set_user_timezone"]["input_schema"]["required"] == ["timezone"]
    reminder = schemas["schedule_reminder"]
    props = reminder["input_schema"]["properties"]
    assert set(props) >= {"reminder_text", "reminder_time", "timezone", "reminder_type", "is_periodic", "period_type",
                          "period_interval", "weekdays"}
    assert props["weekdays"]["items"] == {"type": "integer", "minimum": 0, "maximum": 6}
    assert "local time" in reminder["description"] and QUIET_MARKER in reminder["description"]
    assert "local time" in props["reminder_time"]["description"]
    assert reminder["input_schema"]["required"] == ["reminder_text", "reminder_time", "reminder_type"]
    notes = UserInteractions.prompt_notes
    assert "set_user_timezone" in notes and QUIET_MARKER in notes and "local time" in notes and "ask once" in notes


# ---- schedule_reminder ----------------------------------------------------------------------
async def test_schedule_reminder_reads_the_time_in_the_users_zone(env):
    set_zone("Europe/Berlin")
    sender = FakeSender()
    result = await tool(sender).execute_tool("schedule_reminder", {
        "reminder_text": "Call Alex", "reminder_time": "2030-01-05 09:00", "reminder_type": "user",
    })
    assert result == "Successfully scheduled user reminder for 2030-01-05 09:00 (Europe/Berlin, UTC+1)"
    (stored,) = await env.load(USER)
    assert stored["time"] == "2030-01-05T08:00:00+00:00"  # UTC in the file
    assert stored["timezone"] == "Europe/Berlin" and stored["weekdays"] is None and stored["status"] == "pending"
    assert sender.sent == ["✅ Reminder scheduled for 2030-01-05 09:00 (Europe/Berlin, UTC+1):\nCall Alex"]


async def test_schedule_reminder_keeps_an_explicit_offset_and_honours_the_timezone_param(env):
    set_zone("Europe/Berlin")
    await tool().execute_tool("schedule_reminder", {
        "reminder_text": "a", "reminder_time": "2030-01-05 09:00Z", "reminder_type": "user",
    })
    await tool().execute_tool("schedule_reminder", {
        "reminder_text": "b", "reminder_time": "2030-01-05 09:00", "reminder_type": "agent", "timezone": "Asia/Tokyo",
    })
    result = await tool().execute_tool("schedule_reminder", {
        "reminder_text": "c", "reminder_time": "2030-01-05 09:00", "reminder_type": "user", "timezone": "Atlantis",
    })
    assert result.startswith("Error: unknown time zone 'Atlantis'")
    a, b = await env.load(USER)
    assert a["time"] == "2030-01-05T09:00:00+00:00" and a["timezone"] == "Europe/Berlin"  # Z wins, the zone is still noted
    assert b["time"] == "2030-01-05T00:00:00+00:00" and b["timezone"] == "Asia/Tokyo" and b["type"] == "agent"


async def test_schedule_reminder_without_a_zone_is_utc(env):
    result = await tool().execute_tool("schedule_reminder", {
        "reminder_text": "a", "reminder_time": "2030-01-05 09:00", "reminder_type": "user",
    })
    assert result == "Successfully scheduled user reminder for 2030-01-05 09:00 (UTC)"
    (stored,) = await env.load(USER)
    assert stored["time"] == "2030-01-05T09:00:00+00:00" and stored["timezone"] == ""


async def test_schedule_reminder_errors(env):
    bad = await tool().execute_tool("schedule_reminder", {"reminder_text": "a", "reminder_time": "soon", "reminder_type": "user"})
    assert bad.startswith("Error: Invalid datetime format")
    past = await tool().execute_tool("schedule_reminder", {"reminder_text": "a", "reminder_time": "2001-01-01 09:00", "reminder_type": "user"})
    assert past == "Error: Reminder time must be in the future"
    assert await env.load(USER) == []


async def test_weekdays_store_the_list_and_move_the_first_run_to_an_allowed_day(env):
    set_zone("Europe/Berlin")
    sender = FakeSender()
    result = await tool(sender).execute_tool("schedule_reminder", {
        "reminder_text": "Check the news", "reminder_time": "2030-01-05 08:00",  # a Saturday
        "reminder_type": "agent", "is_periodic": True, "period_type": "daily", "period_interval": 1,
        "weekdays": [4, 0, 1, 2, 3, 0],
    })
    (stored,) = await env.load(USER)
    assert stored["weekdays"] == [0, 1, 2, 3, 4]
    first = datetime.fromisoformat(stored["time"]).astimezone(BERLIN)
    assert (first.strftime("%a %Y-%m-%d %H:%M")) == "Mon 2030-01-07 08:00"
    assert stored["next_trigger"] == stored["time"]
    assert result.endswith("repeat every 1 daily on Mon, Tue, Wed, Thu, Fri")
    assert "2030-01-07 08:00 (Europe/Berlin, UTC+1)" in result
    assert sender.sent[0].startswith("✅ Agent task scheduled for 2030-01-07 08:00 (Europe/Berlin, UTC+1)")


async def test_weekdays_are_validated(env):
    base = {"reminder_text": "a", "reminder_time": "2030-01-05 08:00", "reminder_type": "agent"}
    assert (await tool().execute_tool("schedule_reminder", {**base, "weekdays": [0, 7]})).startswith("Error: weekdays must be")
    assert (await tool().execute_tool("schedule_reminder", {**base, "weekdays": []})).startswith("Error: weekdays must name")
    not_daily = await tool().execute_tool("schedule_reminder", {
        **base, "is_periodic": True, "period_type": "weekly", "period_interval": 1, "weekdays": [0],
    })
    assert not_daily.startswith("Error: weekdays only apply to periodic daily reminders")
    one_shot = await tool().execute_tool("schedule_reminder", {**base, "weekdays": [0]})
    assert one_shot.startswith("Error: weekdays only apply")
    assert await env.load(USER) == []


# ---- calculate_next_trigger ---------------------------------------------------------------
def test_next_trigger_skips_weekdays_outside_the_list(monkeypatch):
    freeze_now(monkeypatch, timezones.to_utc("2026-10-16 10:00", BERLIN))  # Friday
    friday = timezones.to_utc("2026-10-16 09:00", BERLIN).isoformat()
    nxt = calculate_next_trigger(friday, "daily", 1, weekdays=[0, 1, 2, 3, 4], tz="Europe/Berlin")
    assert nxt.astimezone(BERLIN).strftime("%a %Y-%m-%d %H:%M") == "Mon 2026-10-19 09:00"
    weekend = calculate_next_trigger(friday, "daily", 1, weekdays=[5, 6], tz="Europe/Berlin")
    assert weekend.astimezone(BERLIN).strftime("%a %Y-%m-%d %H:%M") == "Sat 2026-10-17 09:00"
    every_day = calculate_next_trigger(friday, "daily", 1, tz="Europe/Berlin")
    assert every_day.astimezone(BERLIN).strftime("%a %Y-%m-%d %H:%M") == "Sat 2026-10-17 09:00"


def test_weekday_is_judged_in_the_reminders_zone(monkeypatch):
    # 01:00 Monday in Tokyo is still Sunday 16:00 UTC: the Tokyo weekday must count
    tokyo = ZoneInfo("Asia/Tokyo")
    freeze_now(monkeypatch, timezones.to_utc("2026-10-18 10:00", tokyo))  # Sunday
    sunday = timezones.to_utc("2026-10-18 01:00", tokyo).isoformat()
    nxt = calculate_next_trigger(sunday, "daily", 1, weekdays=[0], tz="Asia/Tokyo")
    assert nxt.astimezone(tokyo).strftime("%a %Y-%m-%d %H:%M") == "Mon 2026-10-19 01:00"
    assert nxt.astimezone(timezone.utc).strftime("%a %H:%M") == "Sun 16:00"


def test_daily_steps_keep_the_local_time_across_dst(monkeypatch):
    freeze_now(monkeypatch, timezones.to_utc("2026-10-24 10:00", BERLIN))
    before = timezones.to_utc("2026-10-24 09:00", BERLIN)  # CEST, 07:00Z
    nxt = calculate_next_trigger(before.isoformat(), "daily", 1, tz="Europe/Berlin")
    assert nxt == timezones.to_utc("2026-10-25 09:00", BERLIN)  # CET, 08:00Z: the clock time stays 09:00
    assert nxt.hour == 8 and nxt.tzinfo == timezone.utc
    hourly = calculate_next_trigger(before.isoformat(), "hourly", 24, tz="Europe/Berlin")
    assert hourly == before + timedelta(hours=24)  # hourly periods are absolute


def test_next_trigger_without_a_zone_behaves_as_before(monkeypatch):
    freeze_now(monkeypatch, datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc))
    past = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc).isoformat()
    assert calculate_next_trigger(past, "daily", 1) == datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    assert calculate_next_trigger(past, "weekly", 1) == datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
    assert calculate_next_trigger(past, "daily", 1, tz="Nowhere/Zone") == datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        calculate_next_trigger(past, "daily", 1, weekdays=[9])


def test_first_allowed_trigger():
    saturday = timezones.to_utc("2030-01-05 08:00", BERLIN)
    assert first_allowed_trigger(saturday, None) == saturday
    assert first_allowed_trigger(saturday, [5], "Europe/Berlin") == saturday
    monday = first_allowed_trigger(saturday, [0, 1, 2, 3, 4], "Europe/Berlin")
    assert monday.astimezone(BERLIN).strftime("%a %H:%M") == "Mon 08:00"
    assert rs.clean_weekdays(None) is None and rs.clean_weekdays((6, 0)) == [0, 6]
    with pytest.raises(ValueError):
        rs.clean_weekdays("0,1")
    with pytest.raises(ValueError):
        rs.clean_weekdays([True])


# ---- the scheduler: scheduled tasks -------------------------------------------------------
class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(message_id=len(self.sent))


def ctx():
    return JobContext(Job(user_id=USER, label="scheduled agent task", run=None))


async def test_agent_reminder_runs_with_the_framing_prompt_and_the_quiet_marker(env, monkeypatch):
    set_zone("Europe/Berlin")
    run_turn = AsyncMock(return_value=TurnResult(response="", messages=[], turn_id=None, quiet=True))
    monkeypatch.setattr(jobs, "run_turn", run_turn)
    reminder = await env.add(USER, {
        "text": "Check the news for me", "type": "agent", "status": "processing", "is_periodic": True,
        "period_type": "daily", "period_interval": 1, "weekdays": [0, 1, 2, 3, 4], "timezone": "Europe/Berlin",
        "time": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    })
    bot = FakeBot()
    await jobs.run_agent_reminder(bot, USER, reminder, ctx())

    run_turn.assert_awaited_once()
    kw = run_turn.await_args.kwargs
    prompt = kw["prompt"]
    assert prompt.startswith(f"[Scheduled task] This is a task you scheduled earlier (id {reminder['id']}; it is ")
    assert "(Europe/Berlin, UTC+" in prompt
    assert ("Task: Check the news for me. Do it now. If there is nothing new or worth telling the user, reply with exactly "
            f"{QUIET_MARKER} and nothing else.") in prompt
    assert kw["quiet_marker"] == QUIET_MARKER and kw["header"] == jobs.TASK_HEADER
    assert kw["user_id"] == USER and kw["chat_id"] == 42 and "status" not in kw
    assert bot.sent == []  # no separate announcement message

    (stored,) = await env.load(USER)
    assert stored["status"] == "pending" and stored["agent_response"] == QUIET_MARKER
    assert stored["last_result"] == "quiet" and stored["quiet_runs"] == 1
    nxt = datetime.fromisoformat(stored["time"])
    assert nxt > datetime.now(timezone.utc) and nxt.astimezone(BERLIN).weekday() < 5

    # a second quiet run counts up; an answered run records the answer and keeps the counter
    stored["status"] = "processing"
    await jobs.run_agent_reminder(bot, USER, stored, ctx())
    (stored,) = await env.load(USER)
    assert stored["quiet_runs"] == 2 and stored["last_result"] == "quiet"
    run_turn.return_value = TurnResult(response="Two new items: ...", messages=[], turn_id="abc")
    stored["status"] = "processing"
    await jobs.run_agent_reminder(bot, USER, stored, ctx())
    (stored,) = await env.load(USER)
    assert stored["last_result"] == "answered" and stored["agent_response"] == "Two new items: ..." and stored["quiet_runs"] == 2


async def test_one_shot_task_completes_and_a_failed_run_is_marked(env, monkeypatch):
    run_turn = AsyncMock(return_value=TurnResult(response="Done.", messages=[], turn_id="x"))
    monkeypatch.setattr(jobs, "run_turn", run_turn)
    reminder = await env.add(USER, {"text": "Report", "type": "agent", "status": "processing", "time": datetime.now(timezone.utc).isoformat()})
    await jobs.run_agent_reminder(FakeBot(), USER, reminder, ctx())
    (stored,) = await env.load(USER)
    assert stored["status"] == "completed" and stored["agent_response"] == "Done." and stored["last_result"] == "answered"
    assert "(UTC)" in run_turn.await_args.kwargs["prompt"]  # no zone known: UTC
    assert "quiet_runs" not in stored

    run_turn.return_value = None  # run_turn reported an error
    failed = await env.add(USER, {"text": "Report 2", "type": "agent", "status": "processing", "time": datetime.now(timezone.utc).isoformat()})
    await jobs.run_agent_reminder(FakeBot(), USER, failed, ctx())
    stored = next(r for r in await env.load(USER) if r["id"] == failed["id"])
    assert stored["status"] == "failed" and "agent_response" not in stored


def test_task_prompt_is_the_framing_text():
    now = datetime(2026, 10, 17, 12, 5, tzinfo=timezone.utc)
    prompt = jobs.task_prompt({"id": "ab12cd34", "text": "Water the plants"}, now, BERLIN)
    assert prompt == (
        "[Scheduled task] This is a task you scheduled earlier (id ab12cd34; it is 2026-10-17 14:05 (Europe/Berlin, UTC+2)). "
        "Task: Water the plants. Do it now. If there is nothing new or worth telling the user, reply with exactly "
        "<no_update/> and nothing else."
    )


# ---- the scheduler: user reminders --------------------------------------------------------
class FakeAuth:
    def is_authorized(self, user_id):
        return True


async def test_fired_user_reminders_carry_snooze_buttons_and_reschedule_with_weekdays(env, monkeypatch):
    monkeypatch.setattr(jobs, "auth", FakeAuth())
    due = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    once = await env.add(USER, {"text": "Call Alex", "type": "user", "status": "pending", "time": due})
    daily = await env.add(USER, {"text": "Stand-up", "type": "user", "status": "pending", "time": due, "is_periodic": True,
                                 "period_type": "daily", "period_interval": 1, "weekdays": [0, 1, 2, 3, 4], "timezone": "Europe/Berlin"})
    bot = FakeBot()
    qm = QueueManager(max_concurrent_turns=1, turn_timeout=5)
    try:
        await jobs.check_reminders(bot, qm)
    finally:
        await qm.shutdown()

    assert [kw["text"] for kw in bot.sent] == ["🔔 *Reminder*\n\nCall Alex", "🔔 *Reminder*\n\nStand-up"]
    for kw, reminder in zip(bot.sent, (once, daily)):
        data = [b.callback_data for row in kw["reply_markup"].inline_keyboard for b in row]
        assert data == [f"reminder_snooze_{reminder['id']}_10", f"reminder_snooze_{reminder['id']}_60", f"reminder_snooze_{reminder['id']}_tomorrow"]
        assert [b.text for row in kw["reply_markup"].inline_keyboard for b in row] == ["⏰ 10 min", "1 h", "tomorrow 09:00"]
    stored = {r["id"]: r for r in await env.load(USER)}
    assert stored[once["id"]]["status"] == "completed"
    nxt = datetime.fromisoformat(stored[daily["id"]]["time"])
    assert stored[daily["id"]]["status"] == "pending" and nxt > datetime.now(timezone.utc)
    assert nxt.astimezone(BERLIN).weekday() < 5


def test_snooze_keyboard_matches_the_ui_family():
    data = [b.callback_data for row in jobs.snooze_keyboard("ab12").inline_keyboard for b in row]
    assert data == ["reminder_snooze_ab12_10", "reminder_snooze_ab12_60", "reminder_snooze_ab12_tomorrow"]
