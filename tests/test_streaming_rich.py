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
    assert bot.calls[1][1]["draft_id"] == 1 and bot.calls[1][1]["chat_id"] == 42


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
