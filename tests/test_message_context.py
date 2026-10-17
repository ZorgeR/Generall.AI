"""Reply / forward context (F1): the preamble text, and every content job putting it in front of its prompt."""
import shutil
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message

from bot import media
from bot.handlers import messages
from bot.message_context import ASSISTANT, END_OF_REPLY, QUOTE_LIMIT, TRUNCATED, describe_reference, with_reference
from bot.queue import Job, JobContext

USER = "42"
BOT_ID = 777
TS = 1760700000  # 2025-10-17 11:20 UTC
REPLY_TS = TS - 3600
REPLY_WHEN = "2025-10-17 10:20 UTC"
CHAT = {"id": 42, "type": "private"}
ALICE = {"id": 42, "is_bot": False, "first_name": "Alice", "last_name": "Smith"}
BOB = {"id": 5, "is_bot": False, "first_name": "Bob"}
BOT = {"id": BOT_ID, "is_bot": True, "first_name": "GenerALL"}
PHOTO = [{"file_id": "p", "file_unique_id": "pu", "width": 10, "height": 10}]


def msg(**fields) -> Message:
    """A message from Alice in her private chat, in Bot API JSON shape (``from``, unix ``date``)."""
    return Message.model_validate({"message_id": 10, "date": TS, "chat": CHAT, "from": ALICE, **fields})


def replied(**fields) -> dict:
    """The ``reply_to_message`` of a message: sent an hour earlier, by Bob unless ``from`` says otherwise."""
    return {"message_id": 9, "date": REPLY_TS, "chat": CHAT, "from": BOB, **fields}


def origin(kind: str, **fields) -> dict:
    return {"type": kind, "date": REPLY_TS, **fields}


# ---- describe_reference ---------------------------------------------------------------------
def test_a_plain_message_has_no_preamble():
    m = msg(text="hi")
    assert describe_reference(m, BOT_ID) == ""
    assert with_reference("hi", m, BOT_ID) == "hi"


def test_reply_to_a_user_text():
    m = msg(text="translate this", reply_to_message=replied(text="Let's meet at 5"))
    assert describe_reference(m, BOT_ID) == (
        f"[Replying to a message from Bob, sent {REPLY_WHEN}]\nLet's meet at 5\n{END_OF_REPLY}"
    )
    assert with_reference("translate this", m, BOT_ID) == describe_reference(m, BOT_ID) + "\n\ntranslate this"


def test_reply_to_the_bots_own_message_names_the_assistant():
    m = msg(text="shorter", reply_to_message=replied(**{"from": BOT}, text="Here is a long answer."))
    text = describe_reference(m, BOT_ID)
    assert text.startswith(f"[Replying to a message from {ASSISTANT}, sent {REPLY_WHEN}]\nHere is a long answer.")
    # Without a bot id (the test stand-ins) any bot's message counts as the assistant's.
    assert ASSISTANT in describe_reference(m, None)
    # A different bot is not the assistant when the bot id is known.
    other = msg(text="x", reply_to_message=replied(**{"from": {**BOT, "id": 1}}, text="y"))
    assert "GenerALL" in describe_reference(other, BOT_ID) and ASSISTANT not in describe_reference(other, BOT_ID)


def test_reply_to_the_users_own_earlier_message_marks_the_user():
    m = msg(text="I meant this", reply_to_message=replied(**{"from": ALICE}, text="Book the flight"))
    assert describe_reference(m, BOT_ID).startswith(f"[Replying to a message from Alice Smith (the user), sent {REPLY_WHEN}]")


def test_reply_with_a_quote_names_the_quoted_part():
    m = msg(text="why?", reply_to_message=replied(text="We leave at 5 because of the traffic."),
            quote={"text": "because of the traffic", "position": 14})
    lines = describe_reference(m, BOT_ID).split("\n")
    assert lines[1] == "We leave at 5 because of the traffic."
    assert lines[2] == '[The user quoted this part of it: "because of the traffic"]'
    assert lines[3] == END_OF_REPLY


def test_reply_to_a_photo_with_caption():
    m = msg(text="which breed?", reply_to_message=replied(photo=PHOTO, caption="Our cat"))
    assert describe_reference(m, BOT_ID) == f"[Replying to a photo from Bob, sent {REPLY_WHEN}]\nOur cat\n{END_OF_REPLY}"


@pytest.mark.parametrize("fields, kind", [
    ({"document": {"file_id": "d", "file_unique_id": "du", "file_name": "report.pdf"}}, "a document: report.pdf"),
    ({"document": {"file_id": "d", "file_unique_id": "du"}}, "a document"),
    ({"voice": {"file_id": "v", "file_unique_id": "vu", "duration": 3}}, "a voice message"),
    ({"video": {"file_id": "f", "file_unique_id": "fu", "width": 1, "height": 1, "duration": 2}}, "a video"),
    ({"sticker": {"file_id": "s", "file_unique_id": "su", "type": "regular", "width": 1, "height": 1,
                  "is_animated": False, "is_video": False, "emoji": "😀"}}, "a sticker 😀"),
    ({"location": {"latitude": 52.52, "longitude": 13.405}}, "a location 52.52, 13.405"),
    ({"audio": {"file_id": "a", "file_unique_id": "au", "duration": 61, "title": "Song"}}, "an audio file: Song"),
    ({"poll": {"id": "1", "question": "Lunch?", "options": [{"text": "yes", "voter_count": 0, "persistent_id": "o1"}],
               "total_voter_count": 0, "is_closed": False, "is_anonymous": True, "type": "regular",
               "allows_multiple_answers": False, "allows_revoting": False, "members_only": False}}, "a poll: Lunch?"),
])
def test_reply_media_kinds(fields, kind):
    m = msg(text="?", reply_to_message=replied(**fields))
    assert describe_reference(m, BOT_ID) == f"[Replying to {kind} from Bob, sent {REPLY_WHEN}]\n{END_OF_REPLY}"


def test_long_quoted_text_is_truncated():
    m = msg(text="tl;dr", reply_to_message=replied(text="x" * 2000))
    body = describe_reference(m, BOT_ID).split("\n")[1]
    assert body == "x" * QUOTE_LIMIT + TRUNCATED and "(truncated)" in body


def test_reply_to_a_channel_post_names_the_channel():
    post = {"message_id": 9, "date": REPLY_TS, "chat": CHAT, "sender_chat": {"id": -100, "type": "channel", "title": "Daily News"},
            "text": "Markets up"}
    m = msg(text="summarize", reply_to_message=post)
    assert describe_reference(m, BOT_ID).startswith(f"[Replying to a message from Daily News, sent {REPLY_WHEN}]")


def test_reply_to_a_message_of_another_chat_uses_its_origin():
    m = msg(text="what is this?", external_reply={"origin": origin("user", sender_user=BOB), "photo": PHOTO})
    assert describe_reference(m, BOT_ID) == f"[Replying to a photo from Bob, sent {REPLY_WHEN}, in another chat]\n{END_OF_REPLY}"


# A forum topic: Telegram sets reply_to_message on EVERY message of a topic to the topic's creation
# service message (message_id == message_thread_id), which is not a reply (reviewer defect 1).
FORUM = {"id": -100, "type": "supergroup", "title": "Team", "is_forum": True}
TOPIC_ROOT = {"message_id": 3, "date": REPLY_TS, "chat": FORUM, "from": ALICE,
              "forum_topic_created": {"name": "Work", "icon_color": 7322096}}


def topic_msg(**fields) -> Message:
    return Message.model_validate({"message_id": 10, "date": TS, "chat": FORUM, "from": ALICE, "is_topic_message": True,
                                   "message_thread_id": 3, **fields})


def test_a_topic_message_is_not_a_reply_to_the_topic_service_message():
    m = topic_msg(text="hi", reply_to_message=TOPIC_ROOT)
    assert describe_reference(m, BOT_ID) == ""
    assert with_reference("hi", m, BOT_ID) == "hi"


def test_a_topic_message_whose_reply_is_the_thread_root_is_not_a_reply():
    # Older servers / edited roots may omit forum_topic_created; the id match is enough.
    root = {"message_id": 3, "date": REPLY_TS, "chat": FORUM, "from": ALICE}
    assert describe_reference(topic_msg(text="hi", reply_to_message=root), BOT_ID) == ""


def test_a_real_reply_inside_a_topic_is_described():
    m = topic_msg(text="why?", reply_to_message={"message_id": 7, "date": REPLY_TS, "chat": FORUM, "from": BOB, "text": "Because."})
    assert describe_reference(m, BOT_ID) == f"[Replying to a message from Bob, sent {REPLY_WHEN}]\nBecause.\n{END_OF_REPLY}"
    # A forwarded message in a topic still describes the forward, not the topic root.
    f = topic_msg(text="x", reply_to_message=TOPIC_ROOT, forward_origin=origin("user", sender_user=BOB))
    assert describe_reference(f, BOT_ID) == f"[Forwarded from Bob ({REPLY_WHEN})]"


# The bot answers with rich messages: no text/caption, the content is in rich_message.blocks (reviewer defect 2).
def rich(blocks: list) -> dict:
    return replied(**{"from": BOT}, rich_message={"blocks": blocks})


def para(text) -> dict:
    return {"type": "paragraph", "text": text}


def test_reply_to_the_bots_rich_answer_quotes_its_text():
    m = msg(text="shorter", reply_to_message=rich([para("Here is a long answer.")]))
    assert describe_reference(m, BOT_ID) == (
        f"[Replying to a message from {ASSISTANT}, sent {REPLY_WHEN}]\nHere is a long answer.\n{END_OF_REPLY}"
    )


def test_rich_answer_blocks_are_flattened_to_plain_text():
    cell = {"align": "left", "valign": "top"}
    m = msg(text="explain", reply_to_message=rich([
        {"type": "heading", "text": "Plan", "size": 2},
        para(["Here is ", {"type": "bold", "text": "the"}, " ", {"type": "url", "text": "answer", "url": "https://x"}]),
        {"type": "list", "items": [{"label": "1.", "blocks": [para("first")]}, {"label": "•", "blocks": [para("second")]}]},
        {"type": "pre", "text": "print(1)", "language": "python"},
        {"type": "table", "cells": [[{**cell, "text": "A", "is_header": True}, {**cell, "text": "B"}],
                                    [{**cell, "text": "1"}, {**cell, "text": "2"}]]},
        {"type": "blockquote", "blocks": [para("quoted")], "credit": "Someone"},
        {"type": "details", "summary": "More", "blocks": [para("extra")]},
        para([{"type": "custom_emoji", "custom_emoji_id": "1", "alternative_text": "🙂"}, " done"]),
        {"type": "divider"},
    ]))
    body = describe_reference(m, BOT_ID).split("\n")[1:-1]
    assert body == ["Plan", "Here is the answer", "1. first", "• second", "print(1)", "A | B", "1 | 2", "quoted", "Someone",
                    "More", "extra", "🙂 done"]


def test_rich_summary_dumps_are_left_out_of_the_quote():
    """Replying to a turn summary: thinking, buttons and the tool-call / thinking sections are not the answer."""
    m = msg(text="again", reply_to_message=rich([
        para([{"type": "bold", "text": "Done in 3 s"}, ": search_web"]),
        {"type": "details", "summary": "Tool calls (1)", "blocks": [
            {"type": "details", "summary": "🔍 search_web · 1 s", "blocks": [{"type": "pre", "text": "{\"query\": \"x\"}"}]}]},
        {"type": "details", "summary": "💭 Thinking", "blocks": [para([{"type": "italic", "text": "secret plan"}])]},
        {"type": "thinking", "text": "more secret"},
        para("1.2k tokens · ≈$0.01"),
        {"type": "buttons", "buttons": [{"text": "🔁 Retry", "callback_data": "turn_retry_abc"}]},
    ]))
    text = describe_reference(m, BOT_ID)
    assert text.split("\n")[1:-1] == ["Done in 3 s: search_web", "1.2k tokens · ≈$0.01"]
    for hidden in ("query", "secret", "Retry", "Tool calls", "Thinking"):
        assert hidden not in text


def test_long_rich_answer_is_truncated():
    m = msg(text="tl;dr", reply_to_message=rich([para("x" * 1000), para("y" * 1000)]))
    body = describe_reference(m, BOT_ID).split("\n", 1)[1].rsplit("\n", 1)[0]
    assert body.endswith(TRUNCATED) and len(body) == QUOTE_LIMIT + len(TRUNCATED)


# sender_chat is the real sender when from_user is a placeholder account (reviewer defect 3).
def test_sender_chat_wins_over_placeholder_accounts():
    group = {"id": -1, "type": "supergroup", "title": "Team"}
    telegram = {"id": 777000, "is_bot": False, "first_name": "Telegram"}
    auto_forward = {"message_id": 9, "date": REPLY_TS, "chat": group, "from": telegram,
                    "sender_chat": {"id": -100, "type": "channel", "title": "Daily News"}, "text": "Markets up"}
    assert describe_reference(msg(text="?", reply_to_message=auto_forward), BOT_ID).startswith(
        f"[Replying to a message from Daily News, sent {REPLY_WHEN}]")
    anonymous = {"id": 1087968824, "is_bot": True, "first_name": "Group", "username": "GroupAnonymousBot"}
    admin_post = {"message_id": 9, "date": REPLY_TS, "chat": group, "from": anonymous, "sender_chat": group, "text": "Rule 1"}
    for bot_id in (BOT_ID, None):  # even without a bot id the anonymous bot is not the assistant
        header = describe_reference(msg(text="?", reply_to_message=admin_post), bot_id).split("\n")[0]
        assert header == f"[Replying to a message from Team, sent {REPLY_WHEN}]"
    # The assistant's own message still wins over any sender_chat.
    own = {"message_id": 9, "date": REPLY_TS, "chat": group, "from": BOT, "sender_chat": group, "text": "My answer"}
    assert ASSISTANT in describe_reference(msg(text="?", reply_to_message=own), BOT_ID)


def test_forwarded_from_a_user():
    m = msg(text="hello", forward_origin=origin("user", sender_user=BOB))
    assert describe_reference(m, BOT_ID) == f"[Forwarded from Bob ({REPLY_WHEN})]"
    assert with_reference("hello", m, BOT_ID) == f"[Forwarded from Bob ({REPLY_WHEN})]\n\nhello"


def test_forwarded_from_a_hidden_user():
    m = msg(text="hello", forward_origin=origin("hidden_user", sender_user_name="Carol"))
    assert describe_reference(m, BOT_ID) == f"[Forwarded from Carol ({REPLY_WHEN})]"


def test_forwarded_from_a_channel_with_signature():
    m = msg(text="Markets up", forward_origin=origin(
        "channel", chat={"id": -100, "type": "channel", "title": "Daily News"}, message_id=1, author_signature="Editor",
    ))
    assert describe_reference(m, BOT_ID) == f"[Forwarded from Daily News, signed Editor ({REPLY_WHEN})]"


def test_forwarded_from_a_group_chat():
    m = msg(text="x", forward_origin=origin("chat", sender_chat={"id": -1, "type": "supergroup", "title": "Team"}))
    assert describe_reference(m, BOT_ID) == f"[Forwarded from Team ({REPLY_WHEN})]"


def test_a_forwarded_reply_describes_both():
    m = msg(text="fyi", reply_to_message=replied(text="earlier"), forward_origin=origin("user", sender_user=BOB))
    text = describe_reference(m, BOT_ID)
    assert text.index("[Replying to") < text.index(END_OF_REPLY) < text.index("[Forwarded from Bob")


# ---- every content job puts the preamble before its prompt ------------------------------------
REPLY = replied(text="Let's meet at 5")
HEADER = f"[Replying to a message from Bob, sent {REPLY_WHEN}]"


class FakeBot:
    """Records sends/edits; ``download`` writes a small file so the jobs' copies succeed."""

    id = BOT_ID

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return self._msg()

    async def download(self, file, destination=None, **kw):
        with open(destination, "wb") as f:
            f.write(b"data")


class _Stats:
    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = _Stats()
    monkeypatch.setitem(sys.modules, "stats", stats)


@pytest.fixture
def run_turn(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr(messages, "run_turn", mock)
    return mock


@pytest.fixture
def quiet_media(monkeypatch):
    """No ffmpeg, Whisper or vision calls: canned transcriptions and descriptions."""
    monkeypatch.setattr(media, "convert_audio", AsyncMock())
    monkeypatch.setattr(media, "transcribe_audio", AsyncMock(return_value="call mom tonight"))
    monkeypatch.setattr(media, "extract_video_screenshots", AsyncMock(return_value=[]))
    monkeypatch.setattr(media, "describe_image_anthropic", AsyncMock(return_value="a cat"))
    monkeypatch.setattr(media, "describe_image_openai", AsyncMock(return_value="a tabby cat"))
    monkeypatch.setattr(media, "describe_document", AsyncMock(return_value="quarterly numbers"))
    monkeypatch.setattr(media, "prepare_downloaded_image_for_vision", lambda src, dst: shutil.copy(src, dst))
    monkeypatch.setattr(media, "save_vision_copy", lambda src, dst, max_px=None: (shutil.copy(src, dst), (1, 1))[1])


def ctx() -> JobContext:
    return JobContext(Job(user_id=USER, label="test", run=None))


def prompt_of(run_turn: AsyncMock) -> str:
    """The prompt's text: the string itself, or the first text block of a block prompt (media jobs)."""
    run_turn.assert_awaited_once()
    prompt = run_turn.await_args.kwargs["prompt"]
    if isinstance(prompt, list):
        return next(b["text"] for b in prompt if b.get("type") == "text")
    return prompt


def assert_preamble_first(prompt: str, own_text: str) -> None:
    assert prompt.startswith(HEADER)
    assert prompt.index(END_OF_REPLY) < prompt.index(own_text)
    assert "\n\n" in prompt[prompt.index(END_OF_REPLY):prompt.index(own_text)]


async def test_text_job(run_turn):
    await messages._run_text(FakeBot(), msg(text="translate this", reply_to_message=REPLY), USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert prompt == f"{HEADER}\nLet's meet at 5\n{END_OF_REPLY}\n\ntranslate this"


async def test_text_job_without_a_reference_is_unchanged(run_turn):
    await messages._run_text(FakeBot(), msg(text="hi"), USER, None, ctx())
    assert prompt_of(run_turn) == "hi"


async def test_voice_job(run_turn, quiet_media):
    m = msg(voice={"file_id": "v", "file_unique_id": "vu", "duration": 3}, reply_to_message=REPLY)
    await messages._run_voice(FakeBot(), m, USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "call mom tonight")
    assert prompt.endswith("\n\ncall mom tonight") and run_turn.await_args.kwargs["speak"] is True


async def test_video_job(run_turn, quiet_media):
    m = msg(video={"file_id": "f", "file_unique_id": "fu", "width": 1, "height": 1, "duration": 2}, caption="look",
            reply_to_message=REPLY)
    await messages._run_video(FakeBot(), m, USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "User sent a video file.")
    assert "User caption: look" in prompt and "call mom tonight" in prompt


async def test_audio_job(run_turn, quiet_media):
    m = msg(audio={"file_id": "a", "file_unique_id": "au", "duration": 61, "file_name": "song.mp3"}, caption="what song?",
            reply_to_message=REPLY)
    await messages._run_audio(FakeBot(), m, USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "User sent an audio file.")
    assert "File: song.mp3" in prompt and "User message: what song?" in prompt


async def test_photo_job(run_turn, quiet_media):
    m = msg(photo=PHOTO, caption="which breed?", reply_to_message=REPLY)
    await messages._run_images(FakeBot(), m, USER, None, [m.photo[-1]], m.caption, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "which breed?")
    assert "User attached 1 image(s)" in prompt and "Image 1: " in prompt


async def test_album_job(run_turn, quiet_media):
    """An album is flushed into one _run_images call with every photo (see _flush_album)."""
    m = msg(photo=PHOTO, media_group_id="g1", reply_to_message=REPLY)
    refs = [m.photo[-1], m.photo[-1]]
    await messages._run_images(FakeBot(), m, USER, None, refs, None, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "User sent 2 photo(s).")
    assert "Image 2: " in prompt


async def test_document_job(run_turn, quiet_media):
    m = msg(document={"file_id": "d", "file_unique_id": "du", "file_name": "Report.PDF"}, caption="summarize",
            reply_to_message=REPLY)
    await messages._run_document(FakeBot(), m, USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert_preamble_first(prompt, "summarize")
    assert "File: report.pdf" in prompt and "quarterly numbers" in prompt


async def test_forwarded_text_job(run_turn):
    m = msg(text="Markets up", forward_origin=origin("channel", chat={"id": -100, "type": "channel", "title": "Daily News"},
                                                      message_id=1))
    await messages._run_text(FakeBot(), m, USER, None, ctx())
    assert prompt_of(run_turn) == f"[Forwarded from Daily News ({REPLY_WHEN})]\n\nMarkets up"
