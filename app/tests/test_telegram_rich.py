"""Tests for the Bot API 10.1 Rich Messages layer."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telegram_rich  # noqa: E402
from telegram_rich import (  # noqa: E402
    RICH_MAX_CHARS,
    build_rich_message,
    build_thinking_message,
    edit_rich_message,
    rich_enabled,
    send_rich_draft,
    send_rich_message,
    split_rich,
)


class FakeBot:
    """Records do_api_request calls; can fail a given method."""

    def __init__(self, fail_with=None, fail_methods=()):
        self.calls = []
        self.fail_with = fail_with
        self.fail_methods = set(fail_methods)

    async def do_api_request(self, endpoint, api_kwargs=None, return_type=None):
        self.calls.append({"method": endpoint, "payload": api_kwargs})
        if self.fail_with and endpoint in self.fail_methods:
            raise self.fail_with
        return {"message_id": len(self.calls)}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class RichTestCase(unittest.TestCase):
    def setUp(self):
        # Each test starts from "support unknown".
        telegram_rich._supported = None

    def tearDown(self):
        telegram_rich._supported = None


class PayloadTests(RichTestCase):
    def test_markdown_is_passed_through_untouched(self):
        source = "# Title\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n- [x] done $$E=mc^2$$"
        self.assertEqual(build_rich_message(source), {"markdown": source})

    def test_skip_entity_detection_is_optional(self):
        self.assertNotIn("skip_entity_detection", build_rich_message("hi"))
        self.assertTrue(build_rich_message("hi", True)["skip_entity_detection"])

    def test_thinking_block_is_html_and_escaped(self):
        payload = build_thinking_message("weighing <a> & <b>")
        self.assertEqual(
            payload, {"html": "<tg-thinking>weighing &lt;a&gt; &amp; &lt;b&gt;</tg-thinking>"}
        )

    def test_split_respects_the_documented_limit(self):
        self.assertEqual(split_rich("  "), [])
        self.assertEqual(split_rich("short"), ["short"])
        long_source = "\n\n".join("Paragraph %d %s" % (i, "word " * 200) for i in range(80))
        pieces = split_rich(long_source)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), RICH_MAX_CHARS)


class SendTests(RichTestCase):
    def test_send_uses_send_rich_message(self):
        bot = FakeBot()
        result = run(send_rich_message(bot, 42, "# Hello\n\nworld"))
        self.assertEqual(len(result), 1)
        call = bot.calls[0]
        self.assertEqual(call["method"], "sendRichMessage")
        self.assertEqual(call["payload"]["chat_id"], 42)
        self.assertEqual(call["payload"]["rich_message"]["markdown"], "# Hello\n\nworld")

    def test_extra_params_are_forwarded(self):
        bot = FakeBot()
        run(send_rich_message(bot, 42, "hi", message_thread_id=9))
        self.assertEqual(bot.calls[0]["payload"]["message_thread_id"], 9)

    def test_edit_uses_edit_message_text_with_rich_message(self):
        bot = FakeBot()
        run(edit_rich_message(bot, 42, 7, "**done**"))
        call = bot.calls[0]
        self.assertEqual(call["method"], "editMessageText")
        self.assertEqual(call["payload"]["message_id"], 7)
        self.assertEqual(call["payload"]["rich_message"]["markdown"], "**done**")

    def test_draft_streams_partial_text(self):
        bot = FakeBot()
        self.assertTrue(run(send_rich_draft(bot, "42", 3, "partial **answer**")))
        call = bot.calls[0]
        self.assertEqual(call["method"], "sendRichMessageDraft")
        self.assertEqual(call["payload"]["chat_id"], 42)  # int, per the API
        self.assertEqual(call["payload"]["draft_id"], 3)
        self.assertEqual(call["payload"]["rich_message"]["markdown"], "partial **answer**")

    def test_draft_uses_a_thinking_block_for_reasoning(self):
        bot = FakeBot()
        run(send_rich_draft(bot, 42, 1, "considering options", thinking=True))
        self.assertIn("<tg-thinking>", bot.calls[0]["payload"]["rich_message"]["html"])

    def test_draft_id_is_never_zero(self):
        bot = FakeBot()
        run(send_rich_draft(bot, 42, 0, "text"))
        self.assertEqual(bot.calls[0]["payload"]["draft_id"], 1)


class FallbackTests(RichTestCase):
    def test_unknown_method_latches_support_off(self):
        bot = FakeBot(fail_with=Exception("Bad Request: method not found"),
                      fail_methods={"sendRichMessage"})
        self.assertIsNone(run(send_rich_message(bot, 42, "hi")))
        self.assertFalse(rich_enabled())

        # A second send must not spend another API call.
        self.assertIsNone(run(send_rich_message(bot, 42, "hi again")))
        self.assertEqual(len(bot.calls), 1)

    def test_other_errors_do_not_disable_rich_messages(self):
        bot = FakeBot(fail_with=Exception("Bad Request: message is too long"),
                      fail_methods={"sendRichMessage"})
        self.assertIsNone(run(send_rich_message(bot, 42, "hi")))
        self.assertTrue(rich_enabled())

    def test_draft_failure_reports_false_for_fallback(self):
        bot = FakeBot(fail_with=Exception("Bad Request: unknown method"),
                      fail_methods={"sendRichMessageDraft"})
        self.assertFalse(run(send_rich_draft(bot, 42, 1, "text")))
        self.assertFalse(rich_enabled())

    def test_edit_failure_returns_none_for_fallback(self):
        bot = FakeBot(fail_with=Exception("Bad Request: method not found"),
                      fail_methods={"editMessageText"})
        self.assertIsNone(run(edit_rich_message(bot, 42, 7, "text")))

    def test_disabled_by_flag_skips_every_call(self):
        telegram_rich._supported = False
        bot = FakeBot()
        self.assertIsNone(run(send_rich_message(bot, 42, "hi")))
        self.assertIsNone(run(edit_rich_message(bot, 42, 7, "hi")))
        self.assertFalse(run(send_rich_draft(bot, 42, 1, "hi")))
        self.assertEqual(bot.calls, [])


if __name__ == "__main__":
    unittest.main()
