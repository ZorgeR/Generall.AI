"""Tests for the Telegram delivery helpers (fallback chain and splitting)."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telegram_md  # noqa: E402
from telegram_md import (  # noqa: E402
    BadRequest,
    HTML,
    MARKDOWN_V2,
    edit_rich,
    edit_status,
    reply_rich,
    reply_status,
    send_rich,
)


class FakeMessage:
    """Records every call the helpers make, and can reject parse modes."""

    def __init__(self, reject=()):
        self.reject = set(reject)
        self.edits = []
        self.replies = []

    async def _handle(self, sink, text, parse_mode=None, **kwargs):
        if parse_mode in self.reject:
            raise BadRequest("Bad Request: can't parse entities: unexpected end")
        sink.append({"text": text, "parse_mode": parse_mode, **kwargs})
        return self

    async def edit_text(self, text, parse_mode=None, **kwargs):
        return await self._handle(self.edits, text, parse_mode, **kwargs)

    async def reply_text(self, text, parse_mode=None, **kwargs):
        return await self._handle(self.replies, text, parse_mode, **kwargs)


class FakeBot:
    def __init__(self, reject=()):
        self.reject = set(reject)
        self.sent = []

    async def send_message(self, chat_id, text, parse_mode=None, **kwargs):
        if parse_mode in self.reject:
            raise BadRequest("Bad Request: can't parse entities")
        self.sent.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode, **kwargs})
        return self.sent[-1]


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class DeliveryTests(unittest.TestCase):
    def test_reply_uses_markdown_v2(self):
        message = FakeMessage()
        run(reply_rich(message, "**hi** there"))
        self.assertEqual(len(message.replies), 1)
        self.assertEqual(message.replies[0]["parse_mode"], MARKDOWN_V2)
        self.assertEqual(message.replies[0]["text"], "*hi* there")

    def test_falls_back_to_html(self):
        message = FakeMessage(reject={MARKDOWN_V2})
        run(reply_rich(message, "**hi** there"))
        self.assertEqual(len(message.replies), 1)
        self.assertEqual(message.replies[0]["parse_mode"], HTML)
        self.assertEqual(message.replies[0]["text"], "<b>hi</b> there")

    def test_falls_back_to_plain_text(self):
        message = FakeMessage(reject={MARKDOWN_V2, HTML})
        run(reply_rich(message, "**hi** there"))
        self.assertEqual(len(message.replies), 1)
        self.assertIsNone(message.replies[0]["parse_mode"])
        self.assertEqual(message.replies[0]["text"], "hi there")

    def test_non_formatting_errors_propagate(self):
        class Rejecting(FakeMessage):
            async def reply_text(self, text, parse_mode=None, **kwargs):
                raise BadRequest("Bad Request: chat not found")

        with self.assertRaises(BadRequest):
            run(reply_rich(Rejecting(), "hello"))

    def test_extra_kwargs_are_forwarded(self):
        message = FakeMessage()
        run(reply_rich(message, "hello", reply_to_message_id=7))
        self.assertEqual(message.replies[0]["reply_to_message_id"], 7)

    def test_long_answer_is_split(self):
        message = FakeMessage()
        run(reply_rich(message, "\n\n".join("Paragraph %d %s" % (i, "w " * 300) for i in range(20))))
        self.assertGreater(len(message.replies), 1)
        for reply in message.replies:
            self.assertLessEqual(len(reply["text"]), telegram_md.TELEGRAM_MAX_MESSAGE_LENGTH)

    def test_edit_rich_edits_then_replies(self):
        message = FakeMessage()
        run(edit_rich(message, "\n\n".join("Paragraph %d %s" % (i, "w " * 300) for i in range(20))))
        self.assertEqual(len(message.edits), 1)
        self.assertGreater(len(message.replies), 0)

    def test_edit_rich_uses_reply_target(self):
        status = FakeMessage()
        origin = FakeMessage()
        run(edit_rich(status, "\n\n".join("Para %d %s" % (i, "w " * 300) for i in range(20)), reply_to=origin))
        self.assertEqual(len(status.edits), 1)
        self.assertEqual(len(status.replies), 0)
        self.assertGreater(len(origin.replies), 0)

    def test_edit_status_swallows_errors(self):
        class Broken(FakeMessage):
            async def edit_text(self, text, parse_mode=None, **kwargs):
                raise RuntimeError("network down")

        run(edit_status(Broken(), "**status**"))  # must not raise

    def test_not_modified_is_ignored(self):
        class NotModified(FakeMessage):
            async def edit_text(self, text, parse_mode=None, **kwargs):
                raise BadRequest("Bad Request: message is not modified")

        run(edit_status(NotModified(), "**status**"))  # must not raise

    def test_reply_status_returns_message(self):
        message = FakeMessage()
        result = run(reply_status(message, "**working...**"))
        self.assertIs(result, message)
        self.assertEqual(message.replies[0]["text"], "*working\\.\\.\\.*")

    def test_send_rich_targets_chat_and_thread(self):
        bot = FakeBot()
        run(send_rich(bot, 42, "**hi**", message_thread_id=5))
        self.assertEqual(bot.sent[0]["chat_id"], 42)
        self.assertEqual(bot.sent[0]["message_thread_id"], 5)
        self.assertEqual(bot.sent[0]["text"], "*hi*")

    def test_empty_text_sends_nothing(self):
        message = FakeMessage()
        run(reply_rich(message, "   "))
        self.assertEqual(message.replies, [])


if __name__ == "__main__":
    unittest.main()
