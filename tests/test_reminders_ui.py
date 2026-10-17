"""/timezone (command, picker, location) and the upgraded /reminders UI.

Handlers are called directly with real aiogram ``Message`` objects whose Telegram calls are
mocks; the store lives under tmp_path and settings are read from the tmp_path cwd.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from aiogram.filters import CommandObject
from aiogram.types import Chat, Location, Message

from bot import timezones
from bot.handlers import commands, messages
from bot.handlers import reminders_ui as ui
from bot.settings import UserSettings
from reminders_store import RemindersStore

USER = "42"
BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Settings under tmp_path/data, a reminders store there too, no pending edits."""
    monkeypatch.chdir(tmp_path)
    store = RemindersStore(base_dir=tmp_path / "data")
    monkeypatch.setattr(ui, "reminders_store", store)
    ui.reset()
    yield store
    ui.reset()


def make_message(text="/reminders", *, chat_id=int(USER), location=None):
    msg = Message(message_id=7, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type="private"),
                  text=text, location=location)
    object.__setattr__(msg, "answer", AsyncMock(return_value=msg))
    object.__setattr__(msg, "edit_text", AsyncMock(return_value=msg))
    object.__setattr__(msg, "delete", AsyncMock(return_value=True))
    return msg


def make_callback(data, message=None):
    return SimpleNamespace(data=data, message=message if message is not None else make_message(), answer=AsyncMock())


def buttons(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


def labels(markup):
    return [b.text for row in markup.inline_keyboard for b in row]


def set_zone(name):
    UserSettings(USER).set("timezone", name, "name")


def utc(text):
    return timezones.to_utc(text, None)


def last_text(mock):
    return mock.await_args.args[0]


def last_markup(mock):
    return mock.await_args.kwargs.get("reply_markup")


# ---- /timezone ----------------------------------------------------------------------------
def tz_command(text):
    args = text.split(" ", 1)[1] if " " in text else None
    return CommandObject(prefix="/", command="timezone", args=args)


async def test_timezone_without_args_shows_the_state_and_the_picker():
    message = make_message("/timezone")
    await commands.timezone_command(message, user_id=USER, command=tz_command("/timezone"))
    text, kw = last_text(message.answer), message.answer.await_args.kwargs
    assert "not set" in text and "UTC" in text
    assert "send me your location" in text and "/timezone Europe/Berlin" in text and "city name" in text
    markup = kw["reply_markup"]
    assert buttons(markup) == [f"tz_set_{i}" for i in range(len(timezones.COMMON_ZONES))]
    assert all(len(row) == 2 for row in markup.inline_keyboard)
    assert labels(markup) == list(timezones.COMMON_ZONES)  # nothing marked yet


async def test_timezone_with_a_name_saves_it_and_confirms_in_local_time():
    message = make_message("/timezone berlin")
    await commands.timezone_command(message, user_id=USER, command=tz_command("/timezone berlin"))
    assert UserSettings(USER).get("timezone", "name") == "Europe/Berlin"
    text = last_text(message.answer)
    assert "Time zone set to *Europe/Berlin*" in text
    expected = timezones.format_local(datetime.now(timezone.utc), BERLIN, with_zone=False)
    assert expected[:13] in text  # the date and hour of the local time (the minute may just tick over)
    assert message.answer.await_args.kwargs["parse_mode"] == "Markdown"

    # with a zone set, the overview names it and marks it in the picker
    message = make_message("/timezone")
    await commands.timezone_command(message, user_id=USER, command=tz_command("/timezone"))
    text = last_text(message.answer)
    assert "Europe/Berlin" in text and "Local time:" in text and "not set" not in text
    assert "✓ Europe/Berlin" in labels(last_markup(message.answer))


async def test_timezone_with_an_unknown_name_keeps_the_setting():
    set_zone("Asia/Tokyo")
    message = make_message("/timezone Mars_Base")
    await commands.timezone_command(message, user_id=USER, command=tz_command("/timezone Mars_Base"))
    assert "don't know the time zone" in last_text(message.answer) and "Mars Base" in last_text(message.answer)
    assert UserSettings(USER).get("timezone", "name") == "Asia/Tokyo"


async def test_picker_button_saves_the_zone_and_edits_the_message():
    message = make_message("/timezone")
    callback = make_callback("tz_set_1", message)
    await commands.timezone_button(callback, user_id=USER)
    callback.answer.assert_awaited_once()
    assert UserSettings(USER).get("timezone", "name") == timezones.COMMON_ZONES[1]
    text = last_text(message.edit_text)
    assert f"Time zone set to *{timezones.COMMON_ZONES[1]}*" in text
    assert f"✓ {timezones.COMMON_ZONES[1]}" in labels(last_markup(message.edit_text))


async def test_picker_ignores_garbage_and_missing_messages():
    callback = make_callback("tz_set_999")
    await commands.timezone_button(callback, user_id=USER)
    assert UserSettings(USER).get("timezone", "name") == ""
    callback.message.edit_text.assert_not_awaited()
    await commands.timezone_button(SimpleNamespace(data="tz_set_0", message=None, answer=AsyncMock()), user_id=USER)
    assert UserSettings(USER).get("timezone", "name") == ""


# ---- location -----------------------------------------------------------------------------
async def test_a_shared_location_sets_the_zone(monkeypatch):
    seen = []

    def finder(lat, lon):
        seen.append((lat, lon))
        return "Asia/Tbilisi"

    monkeypatch.setattr(timezones, "zone_from_location", finder)
    message = make_message(None, location=Location(latitude=41.7151, longitude=44.8271))
    await messages.on_location(message, user_id=USER)
    assert seen == [(41.7151, 44.8271)]
    assert UserSettings(USER).get("timezone", "name") == "Asia/Tbilisi"
    assert "Time zone set to *Asia/Tbilisi*" in last_text(message.answer)


async def test_a_location_without_timezonefinder_asks_for_the_name(monkeypatch):
    monkeypatch.setattr(timezones, "zone_from_location", lambda lat, lon: None)
    message = make_message(None, location=Location(latitude=1.0, longitude=2.0))
    await messages.on_location(message, user_id=USER)
    assert UserSettings(USER).get("timezone", "name") == ""
    text = last_text(message.answer)
    assert "/timezone Europe/Berlin" in text and "city name" in text


# ---- the list -----------------------------------------------------------------------------
async def seed(store, *, agent=True):
    r1 = await store.add(USER, {"text": "Call Alex", "type": "user", "status": "pending", "time": utc("2030-01-05 07:00").isoformat()})
    r2 = await store.add(USER, {"text": "Dentist", "type": "user", "status": "pending", "time": utc("2030-01-03 15:30").isoformat(),
                                "is_periodic": True, "period_type": "weekly", "period_interval": 1})
    done = await store.add(USER, {"text": "Old", "type": "user", "status": "completed", "time": utc("2026-01-01 10:00").isoformat()})
    task = None
    if agent:
        task = await store.add(USER, {"text": "Check the news", "type": "agent", "status": "pending",
                                      "time": utc("2030-01-04 06:00").isoformat(), "is_periodic": True,
                                      "period_type": "daily", "period_interval": 1, "weekdays": [0, 1, 2, 3, 4],
                                      "timezone": "Europe/Berlin", "last_result": "quiet", "quiet_runs": 3})
    return r1, r2, done, task


async def test_list_groups_reminders_and_tasks_in_local_time(env):
    set_zone("Europe/Berlin")
    r1, r2, _, task = await seed(env)
    message = make_message()
    await ui.reminders_command(message)
    text, markup = last_text(message.answer), last_markup(message.answer)
    assert "⏰ Reminders: 2 | 🤖 Tasks: 1 | Completed: 1" in text
    assert "Times in Europe/Berlin." in text
    rows = labels(markup)
    assert rows.index("⏰ Reminders") < rows.index("🤖 Tasks")
    # soonest first inside the group; hours are local (15:30Z -> 16:30 CET in January)
    assert rows[1] == "⏰ 16:30 03/01 ↻1w : Dentist"
    assert rows[rows.index("🤖 Tasks") + 1] == "🤖 07:00 04/01 ↻1d : Check the news"
    data = buttons(markup)
    assert f"reminder_info_{r2['id']}" in data and f"reminder_info_{r1['id']}" in data and f"reminder_info_{task['id']}" in data
    assert f"reminder_delete_{r1['id']}" in data and f"reminder_toggle_{r1['id']}" in data  # the old rows stay
    assert data[-2:] == ["reminders_new", "reminders_summary"]


async def test_list_without_a_zone_says_so_and_shows_utc(env):
    r1, *_ = await seed(env, agent=False)
    message = make_message()
    await ui.reminders_command(message)
    text, markup = last_text(message.answer), last_markup(message.answer)
    assert "Times in UTC. Set your zone with /timezone." in text and "🤖 Tasks: 0" in text
    assert "⏰ 07:00 05/01 : Call Alex" in labels(markup)
    assert "🤖 Tasks" not in labels(markup)


async def test_no_reminders_yet_explains_how_to_create_one(env):
    message = make_message()
    await ui.reminders_command(message)
    text = last_text(message.answer)
    assert text.startswith("No reminders found.") and "remind me tomorrow at 9" in text


async def test_pagination_still_works(env):
    for i in range(7):
        await env.add(USER, {"text": f"r{i}", "type": "user", "status": "pending", "time": utc(f"2030-01-{10 + i:02d} 10:00").isoformat()})
    message = make_message()
    await ui.reminders_command(message)
    markup = last_markup(message.answer)
    assert buttons(markup).count("reminders_page_2") == 1 and "1/2" in labels(markup)
    assert sum(1 for d in buttons(markup) if d.startswith("reminder_info_")) == 5

    callback = make_callback("reminders_page_2", message)
    await ui.reminder_button(callback)
    markup = last_markup(message.answer)
    assert "reminders_page_1" in buttons(markup) and "2/2" in labels(markup)
    assert sum(1 for d in buttons(markup) if d.startswith("reminder_info_")) == 2
    message.delete.assert_awaited()  # the old list message is replaced


# ---- the detail page ----------------------------------------------------------------------
async def test_info_page_shows_local_times_and_the_new_buttons(env):
    set_zone("Europe/Berlin")
    r1, _, _, task = await seed(env)
    message = make_message()
    await ui.reminder_button(make_callback(f"reminder_info_{r1['id']}", message))
    text, markup = last_text(message.edit_text), last_markup(message.edit_text)
    assert "*Reminder Details*" in text and "Scheduled: 2030-01-05 08:00 (Europe/Berlin, UTC+1)" in text
    assert buttons(markup) == [
        f"reminder_delay_{r1['id']}_60", f"reminder_delay_{r1['id']}_1440", f"reminder_edit_{r1['id']}",
        f"reminder_delete_{r1['id']}", f"reminder_toggle_{r1['id']}", "reminders_page_1",
    ]
    assert labels(markup)[:3] == ["⏰ +1h", "📅 +1d", "✏️ Edit text"]

    await ui.reminder_button(make_callback(f"reminder_info_{task['id']}", message))
    text = last_text(message.edit_text)
    assert "*Task Details*" in text and "On: Mon, Tue, Wed, Thu, Fri" in text
    assert "Last run: quiet (nothing new), 3 quiet run(s) so far" in text


async def test_delay_buttons_push_the_time(env):
    set_zone("Europe/Berlin")
    r1, r2, *_ = await seed(env)
    message = make_message()
    await ui.reminder_button(make_callback(f"reminder_delay_{r1['id']}_60", message))
    stored = {r["id"]: r for r in await env.load(USER)}
    assert stored[r1["id"]]["time"] == utc("2030-01-05 08:00").isoformat()
    assert "Scheduled: 2030-01-05 09:00 (Europe/Berlin, UTC+1)" in last_text(message.edit_text)
    await ui.reminder_button(make_callback(f"reminder_delay_{r2['id']}_1440", message))
    stored = {r["id"]: r for r in await env.load(USER)}
    assert stored[r2["id"]]["time"] == utc("2030-01-04 15:30").isoformat()
    assert stored[r2["id"]]["next_trigger"] == utc("2030-01-04 15:30").isoformat()  # periodic: the series moves


async def test_delaying_a_past_reminder_counts_from_now(env):
    _, _, done, _ = await seed(env)
    message = make_message()
    before = datetime.now(timezone.utc)
    await ui.reminder_button(make_callback(f"reminder_delay_{done['id']}_60", message))
    stored = next(r for r in await env.load(USER) if r["id"] == done["id"])
    when = datetime.fromisoformat(stored["time"])
    assert before + timedelta(minutes=59) < when < before + timedelta(minutes=61)
    assert stored["status"] == "pending" and "completed_at" not in stored


async def test_unknown_reminder_on_the_detail_actions(env):
    message = make_message()
    await ui.reminder_button(make_callback("reminder_delay_nope_60", message))
    assert last_text(message.edit_text) == "Reminder not found."
    await ui.reminder_button(make_callback("reminder_edit_nope", message))
    assert last_text(message.edit_text) == "Reminder not found." and ui.pending_edit(USER) is None


# ---- edit text ----------------------------------------------------------------------------
async def test_edit_text_takes_the_next_message_as_the_new_text(env):
    r1, *_ = await seed(env)
    page = make_message()
    await ui.reminder_button(make_callback(f"reminder_edit_{r1['id']}", page))
    assert ui.pending_edit(USER) == r1["id"]
    assert "Current text: Call Alex" in last_text(page.edit_text)
    assert buttons(last_markup(page.edit_text)) == [f"reminder_editcancel_{r1['id']}"]

    # the filter matches this chat only
    typed = make_message("Call Alex about the invoice")
    assert ui._edit_is_pending(typed) is True
    assert ui._edit_is_pending(make_message("hello", chat_id=7)) is False

    await ui.reminder_edit_text(typed)
    assert ui.pending_edit(USER) is None and ui._edit_is_pending(typed) is False  # consumed
    stored = next(r for r in await env.load(USER) if r["id"] == r1["id"])
    assert stored["text"] == "Call Alex about the invoice"
    texts = [c.args[0] for c in typed.answer.await_args_list]
    assert texts[0] == "✅ Reminder text updated." and "Text: Call Alex about the invoice" in texts[1]
    assert buttons(typed.answer.await_args_list[1].kwargs["reply_markup"])[2] == f"reminder_edit_{r1['id']}"


async def test_cancel_edit_and_expiry_clear_the_pending_state(env, monkeypatch):
    r1, *_ = await seed(env)
    page = make_message()
    await ui.reminder_button(make_callback(f"reminder_edit_{r1['id']}", page))
    assert ui.pending_edit(USER) == r1["id"]
    await ui.reminder_button(make_callback(f"reminder_editcancel_{r1['id']}", page))
    assert ui.pending_edit(USER) is None
    assert "*Reminder Details*" in last_text(page.edit_text)  # back on the detail page

    await ui.reminder_button(make_callback(f"reminder_edit_{r1['id']}", page))
    monkeypatch.setattr(ui.time, "monotonic", lambda: ui.time.time() + ui.EDIT_TIMEOUT * 10)
    assert ui.pending_edit(USER) is None  # expired: the next text goes to the agent again
    typed = make_message("some chat")
    assert ui._edit_is_pending(typed) is False


async def test_the_reminders_command_ends_a_pending_edit(env):
    r1, *_ = await seed(env)
    await ui.reminder_button(make_callback(f"reminder_edit_{r1['id']}"))
    await ui.reminders_command(make_message("/reminders"))
    assert ui.pending_edit(USER) is None


# ---- snooze -------------------------------------------------------------------------------
async def test_snooze_reschedules_a_fired_one_shot_reminder(env):
    set_zone("Europe/Berlin")
    done = await env.add(USER, {"text": "Take the pills", "type": "user", "status": "completed",
                                "time": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                                "completed_at": datetime.now(timezone.utc).isoformat()})
    fired = make_message("🔔 Reminder\n\nTake the pills")
    before = datetime.now(timezone.utc)
    await ui.reminder_button(make_callback(f"reminder_snooze_{done['id']}_10", fired))
    (stored,) = await env.load(USER)
    when = datetime.fromisoformat(stored["time"])
    assert stored["status"] == "pending" and "completed_at" not in stored and stored["enabled"] is True
    assert before + timedelta(minutes=9) < when < before + timedelta(minutes=11)
    text = last_text(fired.edit_text)
    assert text.startswith("🔔 *Reminder*\n\nTake the pills\n\n⏰ Snoozed until ") and "Europe/Berlin" in text
    assert last_markup(fired.edit_text) is None  # the snooze buttons are gone


async def test_snooze_tomorrow_means_nine_local(env):
    set_zone("Europe/Berlin")
    done = await env.add(USER, {"text": "x", "type": "user", "status": "completed", "time": datetime.now(timezone.utc).isoformat()})
    await ui.reminder_button(make_callback(f"reminder_snooze_{done['id']}_tomorrow", make_message("x")))
    (stored,) = await env.load(USER)
    local = datetime.fromisoformat(stored["time"]).astimezone(BERLIN)
    tomorrow = (datetime.now(BERLIN) + timedelta(days=1)).date()
    assert (local.date(), local.hour, local.minute) == (tomorrow, 9, 0)


async def test_snoozing_a_periodic_reminder_adds_a_copy_and_keeps_the_series(env):
    _, r2, *_ = await seed(env, agent=False)
    await ui.reminder_button(make_callback(f"reminder_snooze_{r2['id']}_60", make_message("x")))
    stored = await env.load(USER)
    series = next(r for r in stored if r["id"] == r2["id"])
    assert series["time"] == utc("2030-01-03 15:30").isoformat() and series["is_periodic"] is True
    copy = next(r for r in stored if r.get("snoozed_from") == r2["id"])
    assert copy["type"] == "user" and copy["is_periodic"] is False and copy["text"] == "Dentist" and copy["status"] == "pending"
    assert copy["id"] != r2["id"]


async def test_snooze_of_a_deleted_reminder_or_bad_value_is_reported(env):
    fired = make_message("🔔 Reminder\n\ngone")
    await ui.reminder_button(make_callback("reminder_snooze_nope_10", fired))
    assert "Could not snooze" in last_text(fired.edit_text)
    assert ui.snooze_time(datetime.now(timezone.utc), "soon", None) is None
    assert ui.snooze_time(datetime.now(timezone.utc), "-5", None) is None


# ---- Markdown escaping ----------------------------------------------------------------------
def unescaped_underscores(text):
    """Positions of ``_`` not preceded by a backslash: legacy Markdown would read them as italics."""
    return [i for i, ch in enumerate(text) if ch == "_" and (i == 0 or text[i - 1] != "\\")]


async def test_zone_names_with_underscores_are_escaped_on_every_page(env):
    set_zone("America/New_York")
    r1, _, _, task = await seed(env)
    zone = ZoneInfo("America/New_York")

    # the list header
    message = make_message()
    await ui.reminders_command(message)
    text = last_text(message.answer)
    assert "Times in America/New\\_York." in text and not unescaped_underscores(text)
    assert "⏰ 02:00 05/01 : Call Alex" in labels(last_markup(message.answer))  # button labels stay plain

    # the detail page: Created, Scheduled, Next Trigger, Last Triggered
    def _stamp(reminders):
        for r in reminders:
            if r["id"] == task["id"]:
                r.update(created_at=utc("2026-10-17 12:00").isoformat(), last_triggered=utc("2030-01-02 10:00").isoformat(),
                         next_trigger=utc("2030-01-04 06:00").isoformat())

    await env.update(USER, _stamp)
    for reminder_id in (r1["id"], task["id"]):
        await ui.reminder_button(make_callback(f"reminder_info_{reminder_id}", message))
        text = last_text(message.edit_text)
        assert "(America/New\\_York, UTC-5)" in text and not unescaped_underscores(text), text
    assert "Created: 2026-10-17 08:00 (America/New\\_York, UTC-4)" in text  # EDT in October, EST in January
    assert "Next Trigger: 2030-01-04 01:00 (America/New\\_York, UTC-5)" in text
    assert "Last Triggered: 2030-01-02 05:00 (America/New\\_York, UTC-5)" in text

    # the snooze confirmation and the /timezone overview
    text = ui.snooze_text("Take the pills", utc("2030-01-05 14:00"), zone)
    assert text.endswith("⏰ Snoozed until 2030-01-05 09:00 (America/New\\_York, UTC-5)") and not unescaped_underscores(text)
    text = commands.timezone_overview("America/New_York")
    assert "*Your time zone*: America/New\\_York" in text and not unescaped_underscores(text)
    # inside a bold span legacy Markdown takes "_" literally and forbids escaping, so the confirmation keeps it bare
    assert "*America/New_York*" in commands.timezone_set_text("America/New_York")


async def test_snooze_handler_uses_the_escaped_text(env):
    set_zone("America/New_York")
    done = await env.add(USER, {"text": "x_y", "type": "user", "status": "completed", "time": datetime.now(timezone.utc).isoformat()})
    fired = make_message("x_y")
    await ui.reminder_button(make_callback(f"reminder_snooze_{done['id']}_60", fired))
    text = last_text(fired.edit_text)
    assert "x\\_y" in text and "America/New\\_York" in text and not unescaped_underscores(text)


# ---- new, summary, legacy callbacks -------------------------------------------------------
async def test_new_explains_how_to_create_a_reminder(env):
    await seed(env)
    message = make_message()
    await ui.reminder_button(make_callback("reminders_new", message))
    text = last_text(message.edit_text)
    assert "remind me tomorrow at 9 to call Alex" in text and "every weekday at 8" in text
    assert buttons(last_markup(message.edit_text)) == ["reminders_page_1"]


async def test_summary_counts_quiet_runs(env):
    await seed(env)
    message = make_message()
    await ui.reminder_button(make_callback("reminders_summary", message))
    text = last_text(message.edit_text)
    assert "Active User Reminders: 2" in text and "Active Agent Tasks: 1" in text and "Quiet task runs (nothing new): 3" in text


async def test_delete_and_toggle_keep_working(env):
    r1, r2, *_ = await seed(env, agent=False)
    message = make_message()
    await ui.reminder_button(make_callback(f"reminder_toggle_{r1['id']}", message))
    assert next(r for r in await env.load(USER) if r["id"] == r1["id"])["enabled"] is False
    assert "⏰ Reminders: 2" in last_text(message.answer)  # the menu was re-sent
    await ui.reminder_button(make_callback(f"reminder_delete_{r1['id']}", message))
    assert [r["id"] for r in await env.load(USER) if r.get("status") == "pending"] == [r2["id"]]
    assert message.delete.await_count == 2


async def test_callbacks_without_a_message_and_noop_are_harmless(env):
    callback = SimpleNamespace(data="reminder_info_x", message=None, answer=AsyncMock())
    await ui.reminder_button(callback)
    callback.answer.assert_awaited_once()
    noop = make_callback("noop")
    await ui.noop_button(noop)
    noop.answer.assert_awaited_once()
