"""The user profile: the store, the agent's tools, the <profile> block in
the per-turn context and the /profile command with its buttons."""
import copy
import importlib
import os
import re
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery
from aiogram.types import Chat, Message

from agents import transcript as T
from agents.profile_tools import ProfileTools, last_shown_profile_fingerprint, profile_block
from bot import profile as P
from bot.handlers import profile_ui
from bot.profile import MAX_BYTES, ProfileStore, fingerprint, parse

USER = "42"
TODAY = date(2026, 10, 17)


@pytest.fixture
def store(tmp_path):
    return ProfileStore(base_dir=tmp_path)


def context_turn(context_text: str, question: str = "hello") -> dict:
    return {"role": "user", "content": [{"type": "text", "text": context_text}, {"type": "text", "text": question}]}


# ---- the store -----------------------------------------------------------------
def test_round_trip_keeps_sections_in_order_and_dates_notes(store):
    store.add(USER, "notes", "Preparing a talk on beekeeping", today=TODAY)
    store.add(USER, "about", "Name: Anna")
    store.add(USER, "Projects", "GALL.AI bot")
    store.add(USER, "preferences", "- Short answers")  # a bullet the agent wrote is stripped
    text = store.text(USER)
    assert text == (
        "## About\n- Name: Anna\n\n"
        "## Preferences\n- Short answers\n\n"
        "## Notes\n- [2026-10-17] Preparing a talk on beekeeping\n\n"
        "## Projects\n- GALL.AI bot\n"
    )
    profile = store.load(USER)
    assert list(profile.sections) == ["About", "Preferences", "Notes", "Projects"]
    assert profile.entries("ABOUT") == ["Name: Anna"]
    assert parse(text).render() == text  # parse ∘ render is stable
    assert store.path(USER) == store.base_dir / USER / "profile.md"


def test_an_entry_is_not_written_twice(store):
    first = store.add(USER, "About", "Lives in Lisbon")
    again = store.add(USER, "About", "lives in lisbon")
    assert not first.duplicate and again.duplicate
    assert store.load(USER).entries("About") == ["Lives in Lisbon"]
    # a note already there with another date is the same note
    store.add(USER, "Notes", "Learning Portuguese", today=date(2026, 1, 1))
    assert store.add(USER, "Notes", "Learning Portuguese", today=TODAY).duplicate


def test_remove_by_text_substring_and_index(store):
    for entry in ("Name: Anna", "Lives in Lisbon", "Speaks English and Portuguese"):
        store.add(USER, "About", entry)
    assert not store.remove(USER, "About", "s")  # ambiguous (Lisbon, Speaks): matches nothing on purpose
    assert store.remove(USER, "About", "- NAME: ANNA")  # exact, any case, bullet tolerated
    assert store.remove(USER, "About", "lisbon")  # unique substring
    assert not store.remove(USER, "About", "nothing like this")
    assert not store.remove(USER, "Hobbies", "x")  # unknown section
    assert store.load(USER).entries("About") == ["Speaks English and Portuguese"]
    store.add(USER, "About", "Second")
    assert store.remove(USER, "About", 2) and store.load(USER).entries("About") == ["Speaks English and Portuguese"]
    assert store.remove(USER, "About", "1")  # a 1-based position as text
    assert store.text(USER) == "" and not store.path(USER).exists()  # an empty profile has no file


def test_remove_by_needle_against_a_dated_note(store):
    store.add(USER, "Notes", "Call the dentist", today=TODAY)
    assert store.remove(USER, "Notes", "call the dentist")  # the date is not part of the needle
    assert store.load(USER).is_empty()


def test_replace_rewrites_in_place_and_keeps_a_note_date(store):
    store.add(USER, "About", "Lives in Lisbon")
    store.add(USER, "About", "Name: Anna")
    store.add(USER, "Notes", "Talk in November", today=date(2026, 9, 1))
    assert store.replace(USER, "About", "lisbon", "Lives in Porto")
    assert store.load(USER).entries("About") == ["Lives in Porto", "Name: Anna"]
    assert store.replace(USER, "Notes", 1, "Talk moved to December")
    assert store.load(USER).entries("Notes") == ["[2026-09-01] Talk moved to December"]
    assert store.replace(USER, "Notes", 1, "[2026-10-17] Talk cancelled")  # an own date wins
    assert store.load(USER).entries("Notes") == ["[2026-10-17] Talk cancelled"]
    assert not store.replace(USER, "About", "unknown", "x")
    assert not store.replace(USER, "About", "porto", "   ")


def test_the_cap_rotates_the_oldest_notes_and_keeps_about_and_preferences(store):
    store.add(USER, "About", "Name: Anna")
    store.add(USER, "Preferences", "Metric units")
    rotated = []
    for i in range(40):  # 40 × ~300 bytes is well over 8 KB
        saved = store.add(USER, "Notes", f"note {i:02d} " + "x" * 280, today=TODAY)
        rotated += saved.rotated
    assert rotated and rotated[0].startswith("[2026-10-17] note 00 ")
    profile = store.load(USER)
    assert profile.size() <= MAX_BYTES
    assert profile.entries("About") == ["Name: Anna"] and profile.entries("Preferences") == ["Metric units"]
    notes = profile.entries("Notes")
    assert notes[-1].startswith("[2026-10-17] note 39 ") and not any("note 00 " in n for n in notes)
    assert len(notes) + len(rotated) == 40


def test_long_entries_are_cut_and_empty_ones_refused(store):
    saved = store.add(USER, "About", "y" * 2000)
    assert len(saved.entry) == P.MAX_ENTRY_CHARS and saved.entry.endswith("…")
    with pytest.raises(ValueError):
        store.add(USER, "About", "   ")


def test_fingerprint_is_stable_and_changes_with_the_text(store):
    store.add(USER, "About", "Name: Anna")
    text = store.text(USER)
    assert re.fullmatch(r"[0-9a-f]{12}", fingerprint(text))
    assert fingerprint(text) == fingerprint(text) == store.fingerprint(USER)
    store.add(USER, "About", "Lives in Lisbon")
    assert store.fingerprint(USER) != fingerprint(text)
    assert fingerprint("") == fingerprint(None)


def test_a_corrupt_or_odd_file_is_tolerated(store, caplog):
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe not utf-8 at all \xc3\x28")
    assert store.text(USER) == "" and store.load(USER).is_empty()
    assert "Unreadable profile" in caplog.text
    store.add(USER, "About", "Fresh start")  # overwrites the garbage
    assert store.load(USER).entries("About") == ["Fresh start"]

    path.write_text("# Profile\nstray line\n\n## About\n* Name: Anna\nLives in Lisbon\n\n## Notes\n\n##  Projects  \n- Bot\n", encoding="utf-8")
    profile = store.load(USER)
    assert profile.entries("About") == ["stray line", "Name: Anna", "Lives in Lisbon"]
    assert profile.entries("Notes") == [] and profile.entries("projects") == ["Bot"]
    assert list(profile.sections) == ["About", "Preferences", "Notes", "Projects"]


def test_writes_are_atomic(store):
    store.add(USER, "About", "Name: Anna")
    assert [p.name for p in store.path(USER).parent.iterdir()] == ["profile.md"]  # no tmp file left behind
    assert store.path(USER).stat().st_mode & 0o777 == 0o644
    store.clear(USER)
    assert not store.path(USER).exists()
    store.clear(USER)  # idempotent


@pytest.mark.parametrize("target", ["outside", "sibling", "environ"])
async def test_a_linked_profile_is_never_followed(store, tmp_path, caplog, target):
    """data/<uid> is the sandbox's workspace: `ln -s /proc/self/environ profile.md` (or a link to
    another user's profile) must not put that file into the <context> block, user_profile_read or
    /profile. The link reads as an empty profile and the next write replaces the link itself."""
    secret = tmp_path / "outside.env"
    secret.write_text("## About\n- TELEGRAM_BOT_TOKEN=123:supersecret\n", encoding="utf-8")
    store.add("999", "About", "Victim lives at 12 Secret Street")
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    os.symlink({"outside": str(secret), "sibling": "../999/profile.md", "environ": "/proc/self/environ"}[target], path)

    assert store.text(USER) == "" and store.load(USER).is_empty() and "Unreadable profile" in caplog.text
    assert (await ProfileTools(USER, store).execute_tool("user_profile_read", {})).startswith("The profile is empty.")
    store.add(USER, "About", "Name: Anna")
    assert not path.is_symlink() and store.load(USER).entries("About") == ["Name: Anna"]
    assert secret.read_text(encoding="utf-8").endswith("supersecret\n")  # the target was not written either
    assert store.load("999").entries("About") == ["Victim lives at 12 Secret Street"]


def test_a_planted_tmp_link_is_not_written_through(store, tmp_path):
    """The old fixed tmp name (profile.md.tmp) as a link to a file the bot can write: a profile
    write must not land in that file."""
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    store.path(USER).parent.mkdir(parents=True)
    os.symlink(str(victim), store.path(USER).with_suffix(".md.tmp"))
    store.add(USER, "About", "Name: Anna")
    assert victim.read_text(encoding="utf-8") == "untouched"
    assert store.load(USER).entries("About") == ["Name: Anna"] and not store.path(USER).is_symlink()


def test_a_fifo_profile_does_not_hang_the_read(store):
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    os.mkfifo(path)
    assert store.text(USER) == ""  # returns at once: not a regular file


# ---- the tools -----------------------------------------------------------------
@pytest.fixture
def tools(store):
    return ProfileTools(USER, store)


def test_tool_schema_and_prompt_notes(tools):
    names = [t["name"] for t in tools.tools_schema]
    assert names == ["user_profile_read", "user_profile_add", "user_profile_remove", "user_profile_replace"]
    assert all(t["input_schema"]["type"] == "object" for t in tools.tools_schema)
    notes = ProfileTools.prompt_notes
    assert "user_profile_add" in notes and "forget" in notes and "<profile>" in notes and "quietly" in notes


async def test_add_remove_replace_and_read_confirm_and_show_the_section(tools, store):
    assert (await tools.execute_tool("user_profile_read", {})).startswith("The profile is empty.")
    out = await tools.execute_tool("user_profile_add", {"section": "preferences", "text": "Short answers"})
    assert out.startswith("Saved to Preferences: Short answers.") and "## Preferences\n1. Short answers" in out
    out = await tools.execute_tool("user_profile_add", {"section": "Preferences", "text": "short answers"})
    assert out.startswith("Already in Preferences:")
    out = await tools.execute_tool("user_profile_add", {"section": "Preferences", "text": "Metric units"})
    assert "1. Short answers\n2. Metric units" in out
    out = await tools.execute_tool("user_profile_replace", {"section": "Preferences", "old": "metric", "new": "Imperial units"})
    assert out.startswith("Replaced in Preferences: metric -> Imperial units") and "2. Imperial units" in out
    out = await tools.execute_tool("user_profile_remove", {"section": "Preferences", "needle": "1"})
    assert out.startswith("Removed from Preferences: 1") and "1. Imperial units" in out
    out = await tools.execute_tool("user_profile_remove", {"section": "Preferences", "needle": "zzz"})
    assert out.startswith("Nothing removed") and "1. Imperial units" in out
    assert (await tools.execute_tool("user_profile_read", {})) == store.text(USER).strip()
    assert (await tools.execute_tool("user_profile_add", {"section": "About", "text": " "})).startswith("Error:")
    assert (await tools.execute_tool("user_profile_replace", {"section": "About", "old": "x", "new": ""})).startswith("Error:")
    assert (await tools.execute_tool("user_profile_remove", {"section": "Hobbies", "needle": "x"})).startswith("Nothing removed")
    assert await tools.execute_tool("something_else", {}) == "Unknown tool: something_else"


async def test_the_rotation_is_reported_in_the_tool_result(tools):
    for i in range(40):
        out = await tools.execute_tool("user_profile_add", {"section": "Notes", "text": f"note {i:02d} " + "x" * 280})
    assert "oldest note(s) were dropped" in out and "8 KB" in out


async def test_a_store_failure_is_a_tool_result(tools, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tools.store, "add", boom)
    out = await tools.execute_tool("user_profile_add", {"section": "About", "text": "x"})
    assert out == "Error: OSError: disk full"


# ---- the context block ---------------------------------------------------------
def test_last_shown_fingerprint_reads_the_last_context_block_only():
    assert last_shown_profile_fingerprint([]) is None
    messages = [
        context_turn("<context>\ntime\n" + profile_block("## About\n- a\n", "aaaaaaaaaaaa") + "\n</context>"),
        {"role": "assistant", "content": [{"type": "text", "text": '<profile fingerprint="bbbbbbbbbbbb">'}]},  # not a user turn
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": '<profile fingerprint="cccccccccccc">'}]},
        context_turn("<context>\ntime\n</context>", 'I typed <profile fingerprint="dddddddddddd"> myself'),  # not a context block
    ]
    assert last_shown_profile_fingerprint(messages) == "aaaaaaaaaaaa"
    messages.append(context_turn("<context>\n" + profile_block("## About\n- b\n", "eeeeeeeeeeee") + "\n</context>"))
    assert last_shown_profile_fingerprint(messages) == "eeeeeeeeeeee"


def test_profile_block_shape():
    block = profile_block("## About\n- Name: Anna\n")
    assert block == f'<profile fingerprint="{fingerprint("## About\n- Name: Anna\n")}">\n## About\n- Name: Anna\n</profile>'


MEMORY_LINE = "• [2026-09-30] weather: the user lives in Lisbon"
HIT_LINE = "• [2026-09-29] Q: favourite colour? / A: green"


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """A ChainOfThoughtAgent under tmp_path with fixed memory lines; the profile store is the
    default one, so it writes tmp_path/data/42/profile.md like the agent reads it."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    from bot.settings import DEFAULT_SETTINGS

    agents_main = importlib.import_module("agents.main")
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    agent = agents_main.ChainOfThoughtAgent(user_id=USER, user_settings=settings)

    async def hits(question, k):
        return [HIT_LINE]

    monkeypatch.setattr(agent, "_semantic_hits", hits)
    monkeypatch.setattr(agent, "_recent_summaries", lambda limit: [MEMORY_LINE])
    return agent


def profile_tags(text: str) -> list[str]:
    return re.findall(r'<profile fingerprint="([0-9a-f]{12})">', text)


async def test_the_context_shows_the_profile_only_when_it_changed(agent):
    P.profile_store.add(USER, "About", "Name: Anna")
    fp = P.profile_store.fingerprint(USER)

    full, new = await agent._request_context("hello", None, [])
    for text in (full, new):
        assert T.is_context_block({"type": "text", "text": text}) and text.endswith("\n</context>")
        assert profile_tags(text) == [fp] and "\n## About\n- Name: Anna\n</profile>" in text
        assert text.index("</memory>") < text.index("<profile")  # memory first, then the profile
    assert MEMORY_LINE in full and HIT_LINE in full and MEMORY_LINE in new

    # the turn that showed it is in the transcript: the memory lines and the profile are not repeated
    shown = [context_turn(new, "hello"), {"role": "assistant", "content": [{"type": "text", "text": "Hi Anna"}]}]
    full2, new2 = await agent._request_context("and now?", None, shown)
    assert "<profile" not in new2 and MEMORY_LINE not in new2 and HIT_LINE not in new2
    assert profile_tags(full2) == [fp] and MEMORY_LINE in full2  # the quick path sees everything
    assert T.shown_memory_lines(shown) == {MEMORY_LINE, HIT_LINE}  # profile entries are not memory lines

    # the profile changed: the new version goes into the next turn
    P.profile_store.add(USER, "Preferences", "Short answers")
    fp2 = P.profile_store.fingerprint(USER)
    assert fp2 != fp
    _, new3 = await agent._request_context("and now?", None, shown)
    assert profile_tags(new3) == [fp2] and "- Short answers" in new3

    # an older fingerprint in an earlier turn does not count: the LAST shown one does
    shown.append(context_turn(new3, "and now?"))
    _, new4 = await agent._request_context("more", None, shown)
    assert "<profile" not in new4


async def test_no_profile_block_when_empty_or_disabled(agent):
    full, new = await agent._request_context("hello", None, [])
    assert "<profile" not in full and "<profile" not in new
    P.profile_store.add(USER, "About", "Name: Anna")
    agent.user_settings["profile"]["enabled"] = False
    full, new = await agent._request_context("hello", None, [])
    assert "<profile" not in full and "<profile" not in new
    # the rest of the block is what it always was
    assert re.fullmatch(
        r"<context>\nCurrent time in UTC\+0: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\n"
        r"User timezone: unknown \(ask once and save it with set_user_timezone\)\n<memory>\n"
        r"Recent conversations \(newest first\):\n" + re.escape(MEMORY_LINE) + r"\n\n"
        r"Related earlier conversations \(semantic search\):\n" + re.escape(HIT_LINE) + r"\n</memory>\n</context>",
        new,
    )


def test_render_context_without_a_profile_is_unchanged(agent):
    plain = agent._render_context(["• a"], ["• b"])
    assert plain == agent._render_context(["• a"], ["• b"], "")
    assert plain.endswith("<memory>\nRecent conversations (newest first):\n• a\n\nRelated earlier conversations (semantic search):\n• b\n</memory>\n</context>")
    assert agent._render_context([], []).count("\n") == 3  # UTC time and the timezone line only


async def test_the_legacy_path_sends_the_context_block_too(agent, monkeypatch):
    """Transcript off ("Remember this chat in full"): the static <capabilities> notes still
    promise a <context> block with the local time, the zone and the profile, so the legacy user
    turn carries one (its memory stays in the legacy prompt) and the quick path the same text;
    the stored short-term memory keeps the bare question, never a stale block."""
    P.profile_store.add(USER, "About", "Name: Kenji")
    settings = agent.user_settings
    settings["transcript"]["enabled"] = False
    settings["semantic_search"]["enabled"] = False  # no embeddings call
    settings["timezone"]["name"] = "Asia/Tokyo"
    sent: list[list[dict]] = []
    complexity = ["complex"]

    async def classify(question, history=None):
        return complexity[0]

    async def no_save(*args, **kwargs):
        return ""

    async def loop(messages, **kwargs):
        sent.append(copy.deepcopy(messages))
        thread = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return "answer", thread + [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]

    async def quick(context_memory, system_context, question, on_text_chunk=None):
        sent.append(copy.deepcopy(context_memory))
        return "quick", [{"role": "user", "content": [{"type": "text", "text": question}]},
                         {"role": "assistant", "content": [{"type": "text", "text": "quick"}]}]

    monkeypatch.setattr(agent, "_classify_complexity", classify)
    monkeypatch.setattr(agent, "_save_conversation", no_save)
    monkeypatch.setattr(agent.agent, "generate_response", loop)
    monkeypatch.setattr(agent, "_simple_response", quick)

    assert (await agent.generate_response("remind me in 2 hours"))[0] == "answer"
    stored = [(m["role"], b["text"]) for m in agent._load_short_term_memory() for b in m["content"]]
    assert not any("<context>" in t for _, t in stored)
    assert [r for r, t in stored if t.endswith("remind me in 2 hours")] == ["user"]  # the full agent's turn, bare

    complexity[0] = "simple"
    assert (await agent.generate_response("thanks"))[0] == "quick"
    for request, asked in zip(sent, ("remind me in 2 hours", "thanks")):
        turn = request[-1]["content"]
        assert turn.startswith("<context>\nCurrent time in UTC+0: ") and "\nLocal time (Asia/Tokyo): " in turn
        assert profile_tags(turn) and "- Name: Kenji" in turn and "<memory>" not in turn
        assert turn.endswith(asked)


# ---- /profile ------------------------------------------------------------------
@pytest.fixture
def ui(store, monkeypatch):
    monkeypatch.setattr(profile_ui, "profile_store", store)
    return store


def make_message():
    msg = Message(message_id=7, date=datetime.now(timezone.utc), chat=Chat(id=int(USER), type="private"))
    object.__setattr__(msg, "answer", AsyncMock(return_value=msg))
    object.__setattr__(msg, "edit_text", AsyncMock(return_value=msg))
    return msg


def make_callback(data, message=None, *, answer=None):
    return SimpleNamespace(data=data, message=message if message is not None else make_message(), answer=answer or AsyncMock())


def _buttons(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


async def test_profile_command_shows_an_empty_profile_with_the_hint(ui):
    message = make_message()
    await profile_ui.profile_command(message, user_id=USER)
    text, kw = message.answer.await_args.args[0], message.answer.await_args.kwargs
    assert "Your profile is empty." in text and 'remember that' in text and 'forget' in text
    assert kw["parse_mode"] == "Markdown" and _buttons(kw["reply_markup"]) == ["profile_refresh"]


async def test_profile_command_renders_sections_escaped_with_both_buttons(ui):
    ui.add(USER, "About", "Name: Anna_Maria *the* [one]")
    ui.add(USER, "Notes", "Talk in November", today=TODAY)
    message = make_message()
    await profile_ui.profile_command(message, user_id=USER)
    text, kw = message.answer.await_args.args[0], message.answer.await_args.kwargs
    assert "*About*\n• Name: Anna\\_Maria \\*the\\* \\[one]" in text
    assert "*Notes*\n• \\[2026-10-17] Talk in November" in text
    assert text.startswith(profile_ui.HEADER) and text.endswith(profile_ui.HINT)
    assert _buttons(kw["reply_markup"]) == ["profile_refresh", "profile_clear"]


async def test_forget_everything_asks_first_and_then_deletes(ui):
    ui.add(USER, "About", "Name: Anna")
    message = make_message()
    callback = make_callback("profile_clear", message)
    await profile_ui.profile_button(callback, user_id=USER)
    callback.answer.assert_awaited_once()
    text, kw = message.edit_text.await_args.args[0], message.edit_text.await_args.kwargs
    assert text == profile_ui.CONFIRM and _buttons(kw["reply_markup"]) == ["profile_clear_yes", "profile_cancel"]
    assert ui.path(USER).exists()  # nothing deleted yet

    await profile_ui.profile_button(make_callback("profile_cancel", message), user_id=USER)
    text, kw = message.edit_text.await_args.args[0], message.edit_text.await_args.kwargs
    assert "• Name: Anna" in text and _buttons(kw["reply_markup"]) == ["profile_refresh", "profile_clear"]

    await profile_ui.profile_button(make_callback("profile_clear_yes", message), user_id=USER)
    text, kw = message.edit_text.await_args.args[0], message.edit_text.await_args.kwargs
    assert text.startswith(profile_ui.CLEARED) and _buttons(kw["reply_markup"]) == ["profile_refresh"]
    assert not ui.path(USER).exists() and message.answer.await_count == 0


async def test_refresh_reads_the_file_again(ui):
    message = make_message()
    await profile_ui.profile_button(make_callback("profile_refresh", message), user_id=USER)
    assert "Your profile is empty." in message.edit_text.await_args.args[0]
    ui.add(USER, "Preferences", "Metric units")  # the agent wrote meanwhile
    too_old = TelegramBadRequest(method=AnswerCallbackQuery(callback_query_id="1"), message="Bad Request: query is too old")
    await profile_ui.profile_button(make_callback("profile_refresh", message, answer=AsyncMock(side_effect=too_old)), user_id=USER)
    assert "• Metric units" in message.edit_text.await_args.args[0]


def big_profile(store):
    """A profile just under MAX_BYTES (About, Preferences, as many ~600-char Notes as fit): the
    store accepts it, one Telegram message (4096 characters) cannot show it."""
    store.add(USER, "About", "Name: Anna_Maria, lives in Lisbon")
    store.add(USER, "Preferences", "Short answers, *metric* units")
    i = 0
    while store.load(USER).size() + 640 <= MAX_BYTES:
        i += 1
        saved = store.add(USER, "Notes", f"Note {i:02d}: " + ("lorem ipsum " * 48).strip(), today=TODAY)
        assert not saved.rotated and not saved.duplicate
    profile = store.load(USER)
    assert MAX_BYTES - 700 < profile.size() <= MAX_BYTES and len(profile.entries("Notes")) == i >= 10
    return profile


def test_render_pages_packs_sections_whole_and_marks_continuations():
    profile = parse("## About\n- a1\n- a2\n\n## Preferences\n- p1\n\n## Notes\n- n1\n- n2\n- n3\n")
    assert profile_ui.render_pages(profile) == [profile_ui.render(profile)]  # it fits: one message, as before
    first = f"{profile_ui.HEADER}\n\n*About*\n• a1\n• a2"
    limit = profile_ui._FOOTER_RESERVE + profile_ui._utf16_len(first)  # About fits exactly, Preferences does not
    assert profile_ui.render_pages(profile, limit) == [
        first,
        "*Preferences*\n• p1\n\n*Notes*\n• n1",  # a new section opens a page without "(continued)"
        f"*Notes*{profile_ui.CONTINUED}\n• n2\n• n3\n\n{profile_ui.HINT}",
    ]
    assert profile_ui.render_one(profile, limit) == f"{first}\n\n{profile_ui.MORE.format(n=4)}\n\n{profile_ui.HINT}"
    assert profile_ui.render_one(profile) == profile_ui.render(profile)
    empty = parse("")
    assert profile_ui.render_pages(empty) == [profile_ui.render(empty)] and "empty" in profile_ui.render_one(empty)


def test_render_pages_splits_a_profile_near_the_cap_without_cutting_entries(store):
    profile = big_profile(store)
    assert len(profile_ui.render(profile)) > 4096  # the reviewer's case: one message cannot carry it
    pages = profile_ui.render_pages(profile)
    assert len(pages) >= 2 and all(profile_ui._utf16_len(p) <= profile_ui.PAGE_CHARS for p in pages)
    assert pages[0].startswith(profile_ui.HEADER) and pages[-1].endswith(profile_ui.HINT)
    assert profile_ui.HINT not in "".join(pages[:-1]) and all(not p.startswith(profile_ui.HEADER) for p in pages[1:])
    bullets = [line for p in pages for line in p.splitlines() if line.startswith("• ")]
    expected = [f"• {profile_ui.escape_markdown(e)}" for entries in profile.sections.values() for e in entries]
    assert bullets == expected  # every entry once, in order, whole
    assert "\n*Notes*\n• " in pages[0] and pages[1].startswith(f"*Notes*{profile_ui.CONTINUED}\n• ")


async def test_profile_command_sends_a_big_profile_in_pages_with_the_buttons_on_the_last(ui):
    profile = big_profile(ui)
    pages = profile_ui.render_pages(profile)
    message = make_message()
    await profile_ui.profile_command(message, user_id=USER)
    calls = message.answer.await_args_list
    assert [c.args[0] for c in calls] == pages and len(pages) >= 2
    assert all(c.kwargs["parse_mode"] == "Markdown" and len(c.args[0]) <= 4096 for c in calls)
    assert [c.kwargs.get("reply_markup") is not None for c in calls] == [False] * (len(pages) - 1) + [True]
    assert _buttons(calls[-1].kwargs["reply_markup"]) == ["profile_refresh", "profile_clear"]


async def test_refresh_shows_one_page_of_a_big_profile_and_counts_the_rest(ui):
    profile = big_profile(ui)
    message = make_message()
    await profile_ui.profile_button(make_callback("profile_refresh", message), user_id=USER)
    message.edit_text.assert_awaited_once()
    text, kw = message.edit_text.await_args.args[0], message.edit_text.await_args.kwargs
    assert profile_ui._utf16_len(text) <= 4096 and text.startswith(profile_ui.HEADER) and text.endswith(profile_ui.HINT)
    shown = sum(1 for line in text.splitlines() if line.startswith("• "))
    total = sum(len(entries) for entries in profile.sections.values())
    assert 0 < shown < total and profile_ui.MORE.format(n=total - shown) in text
    assert _buttons(kw["reply_markup"]) == ["profile_refresh", "profile_clear"]
    assert message.answer.await_count == 0  # edited in place, no extra messages

    await profile_ui.profile_button(make_callback("profile_cancel", message), user_id=USER)  # "Keep it" shows the same page
    assert message.edit_text.await_args.args[0] == text


async def test_unknown_profile_callbacks_and_inaccessible_messages_are_ignored(ui):
    message = make_message()
    await profile_ui.profile_button(make_callback("profile_something", message), user_id=USER)
    await profile_ui.profile_button(make_callback("profile_refresh", SimpleNamespace(chat=None)), user_id=USER)
    assert message.edit_text.await_count == 0


def test_the_ui_module_imports_neither_stats_nor_agents():
    import ast

    tree = ast.parse(open(profile_ui.__file__, encoding="utf-8").read())
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = [alias.name for node in top_level if isinstance(node, ast.Import) for alias in node.names]
    modules += [node.module for node in top_level if isinstance(node, ast.ImportFrom)]
    assert not any(m == "stats" or m.startswith("agents") for m in modules), modules
