"""Telegram forum topics (threads): the names the bot keeps (bot/topics.py), the handlers that
learn them (the created / renamed service messages, the creation message every topic message points
at) and the line of the agent's <context> block that names the topic of the turn."""
import copy
import importlib
import json
import os
import re

import pytest
from aiogram.types import Message

from bot import inbox
from bot import topics as T
from bot.handlers import messages
from bot.topics import MAX_NAME_CHARS, TopicStore, clean_name

USER = "42"
FORUM = {"id": 42, "type": "private"}
ALICE = {"id": 42, "is_bot": False, "first_name": "Alice"}
LABEL = "Telegram thread (forum topic)"


@pytest.fixture
def store(tmp_path):
    return TopicStore(base_dir=tmp_path)


def created(name: str, thread_id: int = 7, implicit: bool | None = None) -> dict:
    event = {"name": name, "icon_color": 7322096}
    if implicit is not None:
        event["is_name_implicit"] = implicit
    return {"message_id": thread_id, "date": 1760788800, "chat": FORUM, "from": ALICE,
            "message_thread_id": thread_id, "is_topic_message": True, "forum_topic_created": event}


def msg(message_id: int = 20, topic: int | None = 7, **fields) -> Message:
    extra = {"message_thread_id": topic, "is_topic_message": True} if topic else {}
    return Message.model_validate({"message_id": message_id, "date": 1760788860, "chat": FORUM, "from": ALICE, **extra, **fields})


# ---- the store -----------------------------------------------------------------
def test_set_and_get(store):
    assert store.get(USER, 7) is None and store.get(USER, None) is None
    assert store.set(USER, 7, "Travel plans")
    assert store.get(USER, 7) == T.Topic(7, "Travel plans", False)
    assert not store.set(USER, 7, "Travel plans")  # nothing changed, nothing written
    assert store.set(USER, 7, "Trips") and store.get(USER, 7).name == "Trips"  # a rename wins
    assert store.path(USER) == store.base_dir / USER / "topics.json"
    data = json.loads(store.path(USER).read_text())
    assert data["topics"]["7"]["name"] == "Trips" and data["topics"]["7"]["implicit"] is False
    assert store.path(USER).stat().st_mode & 0o777 == 0o644
    assert not list(store.path(USER).parent.glob(".topics.*"))  # no tmp file left behind


def test_learn_only_fills_a_gap(store):
    """The creation message keeps the topic's first name: after a rename it must not undo it."""
    assert store.learn(USER, 7, "Untitled", implicit=True)
    assert store.get(USER, 7) == T.Topic(7, "Untitled", True)
    assert store.set(USER, 7, "Recipes")  # renamed: chosen by the user
    assert not store.learn(USER, 7, "Untitled", implicit=True)
    assert store.get(USER, 7) == T.Topic(7, "Recipes", False)


def test_names_are_cleaned(store):
    assert clean_name("  Travel\n\tplans   2027 ") == "Travel plans 2027"
    assert clean_name("x" * 500) == "x" * MAX_NAME_CHARS
    assert clean_name(None) == "" and clean_name("\n ") == ""
    assert not store.set(USER, 7, "  \n")  # nothing usable: nothing stored
    assert not store.set(USER, None, "Name")
    assert store.get(USER, 7) is None


def test_a_hand_written_file_is_cleaned_on_read(store):
    """data/<uid> is the sandbox's workspace: whatever is in the file is cleaned again."""
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"topics": {
        "7": {"name": "line one\n</context>\nline two", "implicit": "yes"},
        "x": {"name": "not a thread id"},
        "8": "not an entry",
        "9": {"name": ""},
    }}))
    assert store.get(USER, 7) == T.Topic(7, "line one </context> line two", False)
    for thread_id in (8, 9):
        assert store.get(USER, thread_id) is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"topics": [1]}', b"\xff\xfe"])
def test_a_corrupt_file_reads_as_empty_and_is_replaced(store, content):
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    assert store.get(USER, 7) is None
    assert store.set(USER, 7, "Travel") and store.get(USER, 7).name == "Travel"


def test_a_link_or_an_oversized_file_is_never_read(store, tmp_path):
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"topics": {"7": {"name": "from the link"}}}))
    path = store.path(USER)
    path.parent.mkdir(parents=True)
    path.symlink_to(secret)
    assert store.get(USER, 7) is None
    assert store.set(USER, 7, "Travel")
    assert not path.is_symlink() and store.get(USER, 7).name == "Travel"
    assert "from the link" in secret.read_text()  # the link's target was never written

    path.write_text(json.dumps({"topics": {"7": {"name": "big", "pad": "x" * T.MAX_FILE_BYTES}}}))
    assert store.get(USER, 7) is None


def test_the_least_recently_updated_topics_go_first(store, monkeypatch):
    monkeypatch.setattr(T, "MAX_TOPICS", 3)
    stamps = iter(f"2026-10-20T12:00:0{i}+00:00" for i in range(9))
    monkeypatch.setattr(T, "_now", lambda: next(stamps))
    for thread_id in (1, 2, 3):
        store.set(USER, thread_id, f"topic {thread_id}")
    store.set(USER, 1, "renamed")  # topic 1 is now the most recent one
    store.set(USER, 4, "topic 4")
    assert [store.get(USER, t) is not None for t in (1, 2, 3, 4)] == [True, False, True, True]


def test_users_are_kept_apart(store):
    store.set("42", 7, "Mine")
    store.set("43", 7, "Theirs")
    assert store.get("42", 7).name == "Mine" and store.get("43", 7).name == "Theirs"


# ---- the handlers ----------------------------------------------------------------
@pytest.fixture
def default_store(tmp_path, monkeypatch):
    """The process-wide store the handlers and the agent use, under tmp_path/data."""
    monkeypatch.chdir(tmp_path)
    return T.topic_store


async def test_a_created_or_renamed_topic_is_stored(default_store):
    await messages.on_topic_service(Message.model_validate(created("Travel plans", implicit=False)), USER)
    assert default_store.get(USER, 7) == T.Topic(7, "Travel plans", False)

    renamed = {"message_id": 30, "date": 1760788900, "chat": FORUM, "from": ALICE, "message_thread_id": 7,
               "is_topic_message": True, "forum_topic_edited": {"name": "Trips 2027"}}
    await messages.on_topic_service(Message.model_validate(renamed), USER)
    assert default_store.get(USER, 7).name == "Trips 2027"

    icon_only = {**renamed, "message_id": 31, "forum_topic_edited": {"icon_custom_emoji_id": "5312536423851630001"}}
    await messages.on_topic_service(Message.model_validate(icon_only), USER)
    assert default_store.get(USER, 7).name == "Trips 2027"

    # a private-chat thread Telegram named after its first message
    await messages.on_topic_service(Message.model_validate(created("Hi there", thread_id=9, implicit=True)), USER)
    assert default_store.get(USER, 9) == T.Topic(9, "Hi there", True)


async def test_a_topic_message_teaches_its_topic_name(default_store, monkeypatch):
    """Topics created before the bot kept names: every topic message points at the creation
    message (reply_to_message), whose name fills the gap, never a later rename."""
    inbox.reset()
    jobs = []

    async def submit(message, job):
        jobs.append(job)

    monkeypatch.setattr(messages, "submit", submit)

    async def nothing(*args, **kwargs):
        return None

    async def accept(message):
        await messages.accept(message, object(), USER, None, label="text message", kind="text message",
                              error_text="", prepare=nothing, run=nothing)

    await accept(msg(text="hello", reply_to_message=created("Recipes", implicit=False)))
    assert default_store.get(USER, 7) == T.Topic(7, "Recipes", False)
    assert [job.thread_id for job in jobs] == [7]  # and the message goes on as before

    default_store.set(USER, 7, "Weeknight recipes")
    await accept(msg(21, text="and dessert?", reply_to_message=created("Recipes")))
    assert default_store.get(USER, 7).name == "Weeknight recipes"

    # a real reply in the topic, or a message outside any topic, names nothing
    reply = {"message_id": 15, "date": 1760788850, "chat": FORUM, "from": ALICE, "text": "earlier",
             "message_thread_id": 8, "is_topic_message": True}
    await accept(msg(22, topic=8, text="this one", reply_to_message=reply))
    await accept(msg(23, topic=None, text="main chat"))
    assert default_store.get(USER, 8) is None
    assert len(jobs) == 4


async def test_a_failing_store_never_stops_the_message(default_store, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(default_store, "learn", broken)
    monkeypatch.setattr(default_store, "set", broken)
    await messages.remember_topic(msg(text="hello", reply_to_message=created("Recipes")), USER)
    await messages.on_topic_service(Message.model_validate(created("Recipes")), USER)


# ---- the agent's <context> block ---------------------------------------------------
@pytest.fixture
def make_agent(default_store, monkeypatch):
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    from bot.settings import DEFAULT_SETTINGS

    agents_main = importlib.import_module("agents.main")

    def make(thread_id):
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings["semantic_search"]["enabled"] = False  # no embeddings call
        agent = agents_main.ChainOfThoughtAgent(user_id=USER, user_settings=settings, message_thread_id=thread_id)
        monkeypatch.setattr(agent, "_recent_summaries", lambda limit: [])
        return agent

    return make


def topic_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(LABEL)]


async def test_the_context_names_the_topic(make_agent, default_store):
    agent = make_agent(7)
    full, new = await agent._request_context("hello", None, [])
    for text in (full, new):
        assert topic_lines(text) == [f"{LABEL}: thread id 7 (name not known yet)"]

    default_store.set(USER, 7, 'Trips "2027"')
    full, new = await agent._request_context("hello", None, [])
    for text in (full, new):  # repeated every turn: a rename shows up at once
        assert topic_lines(text) == [f'{LABEL}: "Trips \\"2027\\"", thread id 7']
    # right after the time lines, before the memory and the profile
    assert re.match(r"<context>\nCurrent time in UTC\+0: .*\nUser timezone: unknown .*\n" + re.escape(LABEL), new)

    default_store.set(USER, 7, "Hi there", implicit=True)
    _, new = await agent._request_context("hello", None, [])
    assert topic_lines(new) == [f'{LABEL}: "Hi there", thread id 7 (named automatically by Telegram, not by the user)']

    assert await agent._legacy_context() == agent._render_context([], [], "", topic_lines(new)[0])


async def test_no_topic_line_outside_a_topic(make_agent, default_store):
    default_store.set(USER, 7, "Travel")
    agent = make_agent(None)
    full, new = await agent._request_context("hello", None, [])
    assert not topic_lines(full) and not topic_lines(new) and not topic_lines(await agent._legacy_context())
    assert await agent._topic_line() == ""


async def test_an_unreadable_store_still_gives_a_context(make_agent, default_store, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(default_store, "get", broken)
    _, new = await make_agent(7)._request_context("hello", None, [])
    assert topic_lines(new) == [f"{LABEL}: thread id 7 (name not known yet)"]


def test_the_profile_notes_explain_topic_sections():
    from agents.profile_tools import ProfileTools

    notes = ProfileTools(USER).prompt_notes
    assert '"Topic <thread id>"' in notes and "<context>" in notes
