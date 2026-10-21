"""Telegram forum topics (threads): the names the bot keeps (bot/topics.py), the handlers that
learn them (the created / renamed service messages, the creation message every topic message points
at) and the line of the agent's <context> block that names the topic of the turn."""
import copy
import importlib
import json
import os
import re
from datetime import datetime, timezone

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


# ---- the other conversations: memory lines, memory_search, list_threads / read_thread ----------------
from agents.search_tools import SearchTools  # noqa: E402
from agents.thread_tools import READ_BUDGET_CHARS, ThreadTools, conversation_thread  # noqa: E402


def context(time: str) -> dict:
    return {"type": "text", "text": f"<context>\nCurrent time in UTC+0: {time}\n</context>"}


def user(text: str, time: str | None = None) -> dict:
    return {"role": "user", "content": ([context(time)] if time else []) + [{"type": "text", "text": text}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def write_transcript(base, thread_id, messages):
    name = f"topic_{thread_id}_transcript.json" if thread_id else "transcript.json"
    folder = base / USER / "transcripts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"messages": messages, "created": "2026-10-01T00:00:00+00:00"}))
    return folder / name


def write_conversation(base, timestamp, topic, thread_id, summary="", question="q", response="r"):
    folder = base / USER / "conversations"
    folder.mkdir(parents=True, exist_ok=True)
    data = {"timestamp": timestamp, "thread_id": thread_id, "topic": topic, "summary": summary,
            "question": question, "response": response, "full_history": []}
    (folder / f"conversation_{timestamp}_{topic}.json").write_text(json.dumps(data))


@pytest.fixture
def chats(default_store, tmp_path):
    """The main chat and two threads: "Travel plans" (42) and "Recipes" (51); 51 has never been talked in."""
    base = tmp_path / "data"
    default_store.set(USER, 42, "Travel plans")
    default_store.set(USER, 51, "Recipes")
    write_transcript(base, None, [
        user("What is the capital of Portugal?", "2026-10-18 09:00:00"),
        assistant("Lisbon.\n<follow_ups>\n- Tell me more\n</follow_ups>"),
    ])
    write_transcript(base, 42, [
        user("Book me a flight to Lisbon on Friday", "2026-10-19 10:00:00"),
        {"role": "assistant", "content": [{"type": "text", "text": "Searching."},
                                         {"type": "tool_use", "id": "t1", "name": "search_web", "input": {"query": "flights"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "TAP 8:05 SECRET-RESULT"}]},
        assistant("The TAP flight at 8:05 is the cheapest."),
    ])
    return base


def test_conversation_thread_finds_the_file_of_a_hit(chats):
    folder = chats / USER / "conversations"
    write_conversation(chats, "20261019_100500", "lisbon_flight", 42)
    write_conversation(chats, "20261018_090500", "portugal", None)
    assert conversation_thread(folder, "20261019_100500", "lisbon_flight") == (True, 42)
    assert conversation_thread(folder, "20261018_090500", "portugal") == (True, None)
    assert conversation_thread(folder, "20261019_100500", "renamed_topic") == (True, 42)  # the only file of that second
    assert conversation_thread(folder, "20990101_000000", "x") == (False, None)
    assert conversation_thread(folder, "../../x", "x") == (False, None)  # not a timestamp: no glob outside


async def test_memory_lines_from_other_conversations_say_where_they_come_from(make_agent, chats, monkeypatch):
    write_conversation(chats, "20261019_100500", "lisbon_flight", 42, summary="Booked the TAP flight to Lisbon.")
    write_conversation(chats, "20261018_090500", "portugal", None, summary="Capital of Portugal.")
    write_conversation(chats, "20261017_090500", "pasta", 51, summary="Carbonara recipe.")
    agent = make_agent(42)
    recent = type(agent)._recent_summaries  # make_agent stubs it out on the instance
    lines = recent(agent, 5)
    assert lines == [
        "• [20261019_100500] lisbon_flight: Booked the TAP flight to Lisbon.",  # this thread: no mark
        "• [20261018_090500] portugal: Capital of Portugal. [from the main chat]",
        '• [20261017_090500] pasta: Carbonara recipe. [from thread "Recipes" (id 51)]',
    ]
    # outside any thread, the main chat's lines are this conversation's own
    assert recent(make_agent(None), 5)[0].endswith('[from thread "Travel plans" (id 42)]')

    hits = [{"question": "Capital of Portugal?", "answer": "Lisbon", "timestamp": "20261018_090500", "topic": "portugal"},
            {"question": "Lost one", "answer": "?", "timestamp": "20200101_000000", "topic": "gone"}]
    monkeypatch.setattr(agent.conversation_embeddings, "search_conversations", lambda query, k: hits)
    assert await agent._semantic_hits("portugal", 3) == [
        "• [20261018_090500] Q: Capital of Portugal? / A: Lisbon [from the main chat]",
        "• [20200101_000000] Q: Lost one / A: ?",  # its file is gone: no mark
    ]


def test_memory_search_names_the_thread_of_every_hit(chats):
    write_conversation(chats, "20261019_100500", "lisbon_flight", 42, summary="Booked the TAP flight to Lisbon.")
    found = json.loads(SearchTools(USER, thread_id=42).memory_search("lisbon"))
    threads = {(h["source"], h["thread"]) for h in found}
    assert threads == {
        ("conversation", 'thread "Travel plans" (id 42) (current)'),
        ("transcript", 'thread "Travel plans" (id 42) (current)'),
        ("transcript", "the main chat"),  # no longer "current conversation" from inside a thread
    }
    assert not any("SECRET-RESULT" in h["snippet"] for h in found)  # tool results stay out, as before


def test_list_threads(chats):
    def at(path, when):
        stamp = datetime.fromisoformat(when).replace(tzinfo=timezone.utc).timestamp()
        os.utime(path, (stamp, stamp))

    folder = chats / USER / "transcripts"
    at(folder / "transcript.json", "2026-10-18T09:00:00")
    at(folder / "topic_42_transcript.json", "2026-10-19T10:00:00")
    lines = ThreadTools(USER, 51).list_threads().splitlines()
    assert lines[1:4] == [
        '- thread "Travel plans" (id 42) · last message 2026-10-19 10:00 UTC',
        "- the main chat · last message 2026-10-18 09:00 UTC",
        '- thread "Recipes" (id 51) · no messages kept · current',  # named, never talked in
    ]
    assert lines[-1].startswith("Read one with read_thread")


def test_read_thread(chats):
    tools = ThreadTools(USER, 51)
    text = tools.read_thread("42")
    assert text == (
        'thread "Travel plans" (id 42): the last 2 of 2 messages, oldest first.\n\n'
        "[2026-10-19 10:00 UTC] User: Book me a flight to Lisbon on Friday\n\n"
        "You: Searching.\n\nThe TAP flight at 8:05 is the cheapest."
    )
    assert "SECRET-RESULT" not in text  # tool calls and their results are left out
    for name in ("Travel plans", "travel", "#42", "thread 42"):
        assert tools.read_thread(name) == text
    main = tools.read_thread("main")
    assert main.startswith("the main chat: the last 2 of 2 messages")
    assert "Lisbon." in main and "follow_ups" not in main and "Tell me more" not in main
    assert tools.read_thread("42", max_messages=1).startswith('thread "Travel plans" (id 42): the last 1 of 2 messages')
    assert tools.read_thread("Recipes").startswith("That is the current conversation")
    assert tools.read_thread("pasta") == 'Error: no thread named "pasta". Use list_threads to see them.'
    assert tools.read_thread("99") == "No messages are kept for thread 99."
    T.topic_store.set(USER, 60, "Travel diary")
    assert tools.read_thread("travel").startswith("Error: several threads match")
    assert tools.execute_tool("read_thread", {"thread": "Travel plans"}) == text
    assert tools.execute_tool("list_threads", {}).startswith("Conversations with this user")


def test_read_thread_keeps_the_newest_messages_within_its_budget(chats):
    long = "x" * 1990
    write_transcript(chats, 77, [m for i in range(40) for m in (user(f"{i} {long}", "2026-10-19 10:00:00"), assistant(f"answer {i}"))])
    text = ThreadTools(USER).read_thread("77", max_messages=100)
    assert len(text) <= READ_BUDGET_CHARS + 200
    assert text.rstrip().endswith("You: answer 39")  # the end of the conversation is always there
    assert "of 80 messages" in text


def test_read_thread_falls_back_to_the_dialog_history_and_never_follows_a_link(chats, tmp_path):
    folder = chats / USER / "short_term_memory"
    folder.mkdir(parents=True)
    (folder / "topic_88_dialog_history.json").write_text(json.dumps(
        [{"role": "user", "content": "legacy question"}, {"role": "assistant", "content": "legacy answer"}]))
    assert ThreadTools(USER).read_thread("88").endswith("User: legacy question\n\nYou: legacy answer")

    secret = tmp_path / "other_user_transcript.json"
    secret.write_text(json.dumps({"messages": [user("someone else's secret")]}))
    (chats / USER / "transcripts" / "topic_99_transcript.json").symlink_to(secret)
    assert ThreadTools(USER).read_thread("99") == "No messages are kept for thread 99."


def test_the_agent_gets_the_thread_tools(make_agent):
    agent = make_agent(42)
    names = {tool.get("name") for tool in agent.agent.get_tools_schema()}
    assert {"list_threads", "read_thread", "memory_search"} <= names
    assert agent.search_tools.thread_id == 42 and agent.thread_tools.thread_id == 42
    assert "read_thread" in ThreadTools.prompt_notes and "[from ...]" in ThreadTools.prompt_notes
