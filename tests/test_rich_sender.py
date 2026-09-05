"""ChatSender.send_markdown rendering tiers, exercised against a fake Bot."""
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage

from bot import rich
from bot.sender import ChatSender

_METHOD = SendMessage(chat_id=1, text="x")


def not_found():
    return TelegramNotFound(method=_METHOD, message="Not Found")


def bad_request(msg="Bad Request: can't parse"):
    return TelegramBadRequest(method=_METHOD, message=msg)


class FakeBot:
    def __init__(self, *, rich_error=None, rich_html_error=None, mdv2_error=None):
        self.calls: list[tuple[str, dict]] = []
        self.rich_error = rich_error
        self.rich_html_error = rich_html_error
        self.mdv2_error = mdv2_error
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        rm = kw["rich_message"]
        if rm.markdown is not None and self.rich_error is not None:
            raise self.rich_error
        if rm.html is not None and self.rich_html_error is not None:
            raise self.rich_html_error
        return self._msg()

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        if kw.get("parse_mode") == "MarkdownV2" and self.mdv2_error is not None:
            raise self.mdv2_error
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return self._msg()

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    def names(self):
        return [n for n, _ in self.calls]


@pytest.fixture(autouse=True)
def _reset_rich_flag():
    rich.reset()
    yield
    rich.reset()


TEXT = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nsome `code` and snake_case_name"
STATUS = SimpleNamespace(message_id=7)


async def test_rich_mode_sends_one_rich_message_and_deletes_status():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True)
    sent = await sender.send_markdown(TEXT, edit=STATUS)
    assert bot.names() == ["send_rich_message", "delete_message"]
    kw = bot.calls[0][1]
    assert kw["rich_message"].markdown == TEXT and kw["chat_id"] == 42
    assert bot.calls[1][1]["message_id"] == 7
    assert len(sent) == 1


async def test_rich_disabled_keeps_legacy_edit_in_place():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=False)
    await sender.send_markdown(TEXT, edit=STATUS)
    assert bot.names() == ["edit_message_text"]
    assert bot.calls[0][1]["parse_mode"] == "Markdown"


async def test_unsupported_server_falls_back_to_markdown_v2_and_remembers():
    bot = FakeBot(rich_error=not_found())
    sender = ChatSender(bot, 42, rich=True)
    await sender.send_markdown(TEXT, edit=STATUS)
    assert bot.names() == ["send_rich_message", "send_message", "delete_message"]
    assert bot.calls[1][1]["parse_mode"] == "MarkdownV2"
    assert "snake\\_case\\_name" in bot.calls[1][1]["text"]  # escaped, not stripped
    assert rich.is_available() is False

    bot.calls.clear()
    await sender.send_markdown("second answer")
    assert bot.names() == ["send_message"]  # no wasted rich round-trip any more


async def test_rejected_markdown_is_retried_as_rich_html():
    bot = FakeBot(rich_error=bad_request())
    sender = ChatSender(bot, 42, rich=True)
    await sender.send_markdown(TEXT)
    assert bot.names() == ["send_rich_message", "send_rich_message"]
    first, second = bot.calls[0][1]["rich_message"], bot.calls[1][1]["rich_message"]
    assert first.markdown == TEXT and first.html is None
    assert second.html and "<table>" in second.html
    assert rich.is_available() is True  # a bad message is not an unsupported server


async def test_everything_rejected_ends_in_legacy_markdown_then_raw():
    bot = FakeBot(rich_error=bad_request(), rich_html_error=bad_request(), mdv2_error=bad_request())
    sender = ChatSender(bot, 42, rich=True)
    await sender.send_markdown(TEXT)
    names = bot.names()
    assert names[:2] == ["send_rich_message", "send_rich_message"]
    assert names[2] == "send_message" and bot.calls[2][1]["parse_mode"] == "MarkdownV2"
    assert names[3] == "send_message" and bot.calls[3][1]["parse_mode"] == "Markdown"
    assert bot.calls[3][1]["text"] == TEXT  # legacy tier gets the untouched source


async def test_long_answer_is_split_into_several_rich_messages():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True)
    text = "\n\n".join(f"Paragraph {i} " + "word " * 400 for i in range(40))  # ~ 90 KB
    sent = await sender.send_markdown(text)
    assert len(sent) >= 3 and bot.names() == ["send_rich_message"] * len(sent)
    for _, kw in bot.calls:
        assert len(kw["rich_message"].markdown.encode("utf-8")) <= rich.RICH_BYTE_LIMIT


async def test_forum_thread_id_is_passed_to_rich_send():
    bot = FakeBot()
    sender = ChatSender(bot, 42, thread_id=5, rich=True)
    await sender.send_markdown("hi")
    assert bot.calls[0][1]["message_thread_id"] == 5


async def test_empty_answer_still_produces_a_message():
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True)
    sent = await sender.send_markdown("   ")
    assert len(sent) == 1 and "No response" in bot.calls[0][1]["rich_message"].markdown
