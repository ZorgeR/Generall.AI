import asyncio
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramNotFound
from aiogram.methods import SendMessage

from bot import rich
from bot.streaming import create_streaming_callback


class FakeBot:
    def __init__(self, rich_error=None):
        self.calls = []
        self.rich_error = rich_error

    async def send_rich_message_draft(self, **kw):
        self.calls.append(("rich", kw))
        if self.rich_error:
            raise self.rich_error
        return True

    async def send_message_draft(self, **kw):
        self.calls.append(("plain", kw))
        return True


@pytest.fixture(autouse=True)
def _reset():
    rich.reset()
    yield
    rich.reset()


async def test_rich_draft_uses_markdown_and_thinking_block():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, enabled=True, rich=True)
    await cb("thinking hard", True)
    await asyncio.sleep(0.4)
    await cb("# Answer\n\n| a |\n|---|\n| 1 |", False)
    await asyncio.sleep(0.4)
    kinds = [k for k, _ in bot.calls]
    assert kinds == ["rich", "rich"]
    assert bot.calls[0][1]["rich_message"].html.startswith("<tg-thinking>")
    assert bot.calls[1][1]["rich_message"].markdown.startswith("# Answer")
    assert bot.calls[1][1]["chat_id"] == 42
    # one random non-zero draft id for every draft of the turn, exposed on the callback
    assert cb.draft_id > 0 and {kw["draft_id"] for _, kw in bot.calls} == {cb.draft_id}
    assert "can_stop" not in bot.calls[0][1] and "keep_on_stop" not in bot.calls[0][1]


async def test_unsupported_server_falls_back_to_plain_draft():
    bot = FakeBot(rich_error=TelegramNotFound(method=SendMessage(chat_id=1, text="x"), message="Not Found"))
    cb = create_streaming_callback(bot, 42, enabled=True, rich=True)
    await cb("partial", False)
    await asyncio.sleep(0.4)
    await cb("partial more", False)
    await asyncio.sleep(0.4)
    kinds = [k for k, _ in bot.calls]
    assert kinds == ["rich", "plain", "plain"]  # second flush skips rich entirely
    assert rich.is_available() is False


async def test_rich_off_uses_plain_draft():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, enabled=True, rich=False)
    await cb("hello", False)
    assert [k for k, _ in bot.calls] == ["plain"]
    assert bot.calls[0][1]["text"] == "hello"


async def test_draft_id_and_stop_button_are_forwarded_to_rich_and_plain_drafts():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, 7, enabled=True, rich=True, draft_id=123456, can_stop=True)
    assert cb.draft_id == 123456
    await cb("thinking", True)
    await asyncio.sleep(0.4)
    rich.mark_unavailable("old server")  # the next draft goes out as a plain one
    await cb("answer", False)
    assert [k for k, _ in bot.calls] == ["rich", "plain"]
    for _, kw in bot.calls:
        assert kw["draft_id"] == 123456 and kw["message_thread_id"] == 7
        assert kw["can_stop"] is True and kw["keep_on_stop"] is True


async def test_plain_drafts_carry_the_stop_button_too():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, enabled=True, rich=False, draft_id=99, can_stop=True)
    await cb("hello", False)
    assert bot.calls == [("plain", {"chat_id": 42, "draft_id": 99, "text": "hello", "can_stop": True, "keep_on_stop": True})]


async def test_partial_text_tracks_only_the_answer_not_the_thinking():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, enabled=True, rich=True)
    assert cb.partial_text() == ""
    await cb("let me think", True)
    assert cb.partial_text() == ""  # thinking is never the answer
    await cb("The answer", False)
    await cb("The answer is 42", False)
    await cb("more thinking", True)
    assert cb.partial_text() == "The answer is 42"
    cb.close()


async def test_close_cancels_the_pending_flush_and_ignores_later_chunks():
    bot = FakeBot()
    cb = create_streaming_callback(bot, 42, enabled=True, rich=False)
    await cb("one", False)  # flushed at once
    await cb("one two", False)  # throttled: a delayed flush is pending
    cb.close()
    await cb("one two three", False)
    await asyncio.sleep(0.4)
    assert [kw["text"] for _, kw in bot.calls] == ["one"]  # nothing reappears after close()
    assert cb.partial_text() == "one two"


def test_streaming_disabled_returns_none():
    assert create_streaming_callback(FakeBot(), 42, enabled=False, draft_id=5, can_stop=True) is None
