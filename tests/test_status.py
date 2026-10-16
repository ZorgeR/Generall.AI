"""bot/status.py: rich progress/summary blocks and the StatusMessage lifecycle."""
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import InlineKeyboardMarkup, InputRichBlockButtons, InputRichBlockDetails, InputRichBlockParagraph

from agents.trace import ToolTrace
from bot import rich
from bot.sender import ChatSender
from bot.status import StatusMessage, plain_header, progress_blocks, summary_blocks, trace_summary, usage_text

_M = SendMessage(chat_id=1, text="x")


def make_trace(calls=3, with_usage=True, thinking=True):
    trace = ToolTrace()
    for i in range(calls):
        c = trace.start("search_web" if i % 2 == 0 else "run_command", {"query": f"q{i}", "n": i})
        c.done("result line one\nline two" if i else "Error: boom", ok=bool(i))
    if with_usage:
        trace.add_usage({"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 500}, model="claude-sonnet-5-5")
        trace.add_usage({"input_tokens": 100, "output_tokens": 50}, model="mystery-model")
    if thinking:
        trace.add_thinking("Let me think about pings.")
    return trace


def test_usage_text_and_cost():
    trace = make_trace(calls=0)
    text = usage_text(trace)
    assert text.startswith("🧮 2 calls · in 10.6k (") and "% cached" in text and "out 250" in text
    assert "≈$" in text  # sonnet-5 is priced; the unknown model only adds tokens
    assert trace.usage_by_model["claude-sonnet-5-5"]["cache_read_tokens"] == 9000
    assert usage_text(ToolTrace()) == ""
    assert ToolTrace().cost_usd is None


def test_plain_header():
    assert plain_header("💭 *Thinking...*") == "💭 Thinking..."
    assert plain_header("🎬 _Processing video..._") == "🎬 Processing video..."


def test_progress_blocks_structure():
    trace = make_trace()
    blocks = progress_blocks("💭 *Thinking...*", "used 3/50\n", "executing-tools", "Running: search_web", 3, 0, trace)
    assert isinstance(blocks[0], InputRichBlockParagraph)
    details = [b for b in blocks if isinstance(b, InputRichBlockDetails)]
    assert len(details) == 1 and details[0].is_open is True and details[0].summary.startswith("🔧 Tools (3, ")
    assert len(details[0].blocks[0].items) == 3
    assert "🧮" in blocks[-1].text


def test_summary_blocks_have_expandable_calls_thinking_and_fit():
    trace = make_trace(calls=60)
    blocks = summary_blocks(trace)
    assert "60 tool calls" in blocks[0].text[0].text and "1 failed" in blocks[0].text[0].text
    details = [b for b in blocks if isinstance(b, InputRichBlockDetails)]
    assert details[0].summary == "Tool calls (60)"
    entries = details[0].blocks
    assert all(isinstance(e, InputRichBlockDetails) for e in entries[:-1]) and entries[-1].text.startswith("… ")
    first = entries[0]
    assert first.summary.startswith("❌ search_web · ") and first.blocks[0].language == "json" and '"query": "q0"' in first.blocks[0].text
    assert first.blocks[1].text.startswith("Error: boom")
    assert details[1].summary == "💭 Thinking" and "pings" in details[1].blocks[0].text[0].text
    total = sum(len(b.model_dump_json(exclude_none=True).encode()) for b in blocks)
    assert total <= 28000


def test_summary_blocks_without_tools():
    trace = make_trace(calls=0, thinking=False)
    blocks = summary_blocks(trace)
    assert blocks[0].text[0].text.startswith("✅ Done in ") and not any(isinstance(b, InputRichBlockDetails) for b in blocks)
    assert trace_summary(trace).startswith("*✅ Done in ")


class FakeBot:
    def __init__(self, rich_error=None):
        self.calls = []
        self.rich_error = rich_error

    async def edit_message_text(self, **kw):
        kind = "rich" if "rich_message" in kw else "text"
        self.calls.append((kind, kw))
        if kind == "rich" and self.rich_error:
            raise self.rich_error
        return SimpleNamespace(message_id=kw["message_id"])

    async def delete_message(self, **kw):
        self.calls.append(("delete", kw))
        return True

    def kinds(self):
        return [k for k, _ in self.calls]


@pytest.fixture(autouse=True)
def _reset():
    rich.reset()
    yield
    rich.reset()


async def test_status_edits_rich_then_summarizes_in_place():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="💭 *Thinking...*")
    trace = make_trace()
    await status.update(step="executing-tools", details="x", iteration=1, critique=0, trace=trace, quota="")
    await status.finish(trace, keep=True)
    assert bot.kinds() == ["rich", "rich"]
    assert bot.calls[0][1]["rich_message"].blocks[0].text[0].text == "💭 Thinking..."
    assert bot.calls[1][1]["rich_message"].blocks[0].text[0].text.startswith("🔧 3 tool calls")
    assert bot.calls[1][1]["message_id"] == 7


async def test_status_falls_back_to_plain_when_rich_edit_is_rejected():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: message can't be edited"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="💭 *Thinking...*")
    trace = make_trace()
    for i in range(4):
        await status.update(step="s", details=f"d{i}", iteration=i, critique=0, trace=trace)
    kinds = bot.kinds()
    # every rejected rich update is followed by the plain one, until rich progress is given up on
    assert kinds.count("rich") == StatusMessage.MAX_PROGRESS_FAILURES
    assert kinds[-1] == "text" and kinds.count("text") == 4
    assert bot.calls[1][1]["parse_mode"] == "Markdown" and "*Step:*" in bot.calls[1][1]["text"]
    assert rich.is_available() is True  # a rejected edit is not an unsupported server


async def test_status_unsupported_server_disables_rich_globally_and_delete_when_not_kept():
    bot = FakeBot(rich_error=TelegramNotFound(method=_M, message="Not Found"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="💭 *Thinking...*")
    await status.update(step="s", details="d", iteration=0, critique=0, trace=ToolTrace())
    assert rich.is_available() is False
    await status.finish(ToolTrace(), keep=False)
    assert bot.kinds()[-1] == "delete"


async def test_status_with_no_message_is_a_noop():
    status = StatusMessage(ChatSender(FakeBot(), 42), None, rich=True, header="x")
    await status.update(step="s", details="d", iteration=0, critique=0, trace=ToolTrace())
    await status.finish(ToolTrace(), keep=True)
    await status.set_text("🛑 Stopped.")


def test_is_not_modified_matches_both_phrasings():
    from bot.ui import is_not_modified

    assert is_not_modified(TelegramBadRequest(method=_M, message="Bad Request: message is not modified"))
    assert is_not_modified(TelegramBadRequest(method=_M, message="Bad Request: MESSAGE_NOT_MODIFIED"))
    assert not is_not_modified(TelegramBadRequest(method=_M, message="Bad Request: can't parse entities"))


async def test_identical_rich_edit_is_success_and_keeps_rich_mode():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: MESSAGE_NOT_MODIFIED"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="💭 *Thinking...*")
    trace = make_trace()
    await status.update(step="s", details="d", iteration=0, critique=0, trace=trace)
    await status.update(step="s", details="d", iteration=0, critique=0, trace=trace)
    assert bot.kinds() == ["rich", "rich"]  # never fell back to text
    assert status.rich is True and status._progress_failures == 0


async def test_rejected_progress_edit_still_gets_a_rich_summary():
    """A refused progress payload says nothing about the summary, which is different content."""
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: BLOCK_INVALID"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="💭 *Thinking...*")
    trace = make_trace()
    for _ in range(4):
        await status.update(step="s", details="d", iteration=0, critique=0, trace=trace)
    kinds = bot.kinds()
    assert kinds.count("rich") == StatusMessage.MAX_PROGRESS_FAILURES  # stops retrying rich progress
    assert status.rich is True

    bot.rich_error = None
    await status.finish(trace, keep=True)
    assert bot.kinds()[-1] == "rich"
    assert bot.calls[-1][1]["rich_message"].blocks[0].text[0].text.startswith("🔧 3 tool calls")


async def test_concurrent_updates_are_coalesced():
    """A batch of parallel tools must not fire a burst of edits at the same message."""
    import asyncio

    class SlowBot(FakeBot):
        async def edit_message_text(self, **kw):
            await asyncio.sleep(0.05)
            return await super().edit_message_text(**kw)

    bot = SlowBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    trace = make_trace()
    await asyncio.gather(*(status.update(step="s", details=f"d{i}", iteration=i, critique=0, trace=trace) for i in range(7)))
    assert len(bot.calls) == 1  # one edit in flight, the rest dropped in favour of newer state


async def test_summary_falls_back_to_text_when_rich_summary_is_refused():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: BLOCK_INVALID"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    trace = make_trace()
    await status.finish(trace, keep=True)
    assert bot.kinds() == ["rich", "text"]
    assert bot.calls[1][1]["text"].startswith("*🔧 3 tool calls")
    assert status.rich is False


# ---- the Stop button ---------------------------------------------------------
STOP = "queue_stop_1a2b3c"


def has_row(kw) -> bool:
    return any(isinstance(b, InputRichBlockButtons) for b in kw["rich_message"].blocks)


def keyboard_data(kw):
    markup = kw.get("reply_markup")
    if markup is None:
        return None
    assert isinstance(markup, InlineKeyboardMarkup)
    return [[(b.text, b.style, b.callback_data) for b in row] for row in markup.inline_keyboard]


class RowRefusingBot(FakeBot):
    """Refuses rich edits that contain a button row, like a server without rich buttons."""

    async def edit_message_text(self, **kw):
        if "rich_message" in kw and has_row(kw):
            self.calls.append(("rich", kw))
            raise TelegramBadRequest(method=_M, message="Bad Request: BUTTON_TYPE_INVALID")
        return await super().edit_message_text(**kw)


def test_progress_blocks_end_with_the_stop_row_only_when_asked():
    trace = make_trace()
    blocks = progress_blocks("h", "", "s", "d", 1, 0, trace, stop_callback=STOP)
    row = blocks[-1]
    assert isinstance(row, InputRichBlockButtons) and len(row.buttons) == 1
    button = row.buttons[0]
    assert (button.text, button.style, button.callback_data) == ("🛑 Stop", "danger", STOP)
    plain = progress_blocks("h", "", "s", "d", 1, 0, trace)
    assert not any(isinstance(b, InputRichBlockButtons) for b in plain)
    assert len(blocks) == len(plain) + 1


async def test_rich_progress_carries_the_stop_row_and_the_summary_drops_it():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h", stop_callback=STOP)
    trace = make_trace()
    await status.update(step="s", details="d", iteration=1, critique=0, trace=trace)
    kw = bot.calls[0][1]
    assert has_row(kw) and kw["rich_message"].blocks[-1].buttons[0].callback_data == STOP
    assert kw.get("reply_markup") is None  # the row replaces the inline keyboard of the first status

    await status.finish(trace, keep=True)
    kw = bot.calls[-1][1]
    assert bot.kinds() == ["rich", "rich"] and not has_row(kw) and kw.get("reply_markup") is None


async def test_refused_row_falls_back_to_an_inline_keyboard_and_is_remembered():
    bot = RowRefusingBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h", stop_callback=STOP)
    trace = make_trace()
    await status.update(step="s", details="d1", iteration=1, critique=0, trace=trace)
    assert bot.kinds() == ["rich", "rich"]
    assert has_row(bot.calls[0][1])
    retry = bot.calls[1][1]
    assert not has_row(retry) and keyboard_data(retry) == [[("🛑 Stop", "danger", STOP)]]
    assert rich.buttons_available() is False and rich.is_available() is True
    assert status.rich is True and status._progress_failures == 0

    # later edits go straight to the inline keyboard, with no refused round-trip first
    await status.update(step="s", details="d2", iteration=2, critique=0, trace=trace)
    assert bot.kinds() == ["rich", "rich", "rich"]
    assert not has_row(bot.calls[2][1]) and keyboard_data(bot.calls[2][1]) == [[("🛑 Stop", "danger", STOP)]]

    await status.set_text("🛑 Stopped.")
    assert bot.calls[-1][1]["rich_message"].blocks[0].text == "🛑 Stopped." and bot.calls[-1][1].get("reply_markup") is None


async def test_refused_row_and_refused_retry_count_one_failure_and_use_the_text_keyboard():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: message can't be edited"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h", stop_callback=STOP)
    await status.update(step="s", details="d", iteration=1, critique=0, trace=make_trace())
    assert bot.kinds() == ["rich", "rich", "text"]
    assert has_row(bot.calls[0][1]) and not has_row(bot.calls[1][1])
    assert keyboard_data(bot.calls[2][1]) == [[("🛑 Stop", "danger", STOP)]]
    assert status._progress_failures == 1
    assert rich.buttons_available() is True  # the row was not the problem


async def test_legacy_text_status_has_the_stop_keyboard_until_the_end():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=False, header="💭 *Thinking...*", stop_callback=STOP)
    trace = make_trace()
    await status.update(step="s", details="d", iteration=1, critique=0, trace=trace)
    kw = bot.calls[0][1]
    assert bot.kinds() == ["text"] and "*Step:*" in kw["text"]
    assert keyboard_data(kw) == [[("🛑 Stop", "danger", STOP)]]

    await status.finish(trace, keep=True)
    assert bot.kinds() == ["text", "text"] and bot.calls[-1][1].get("reply_markup") is None
    await status.set_text("❌ An error occurred.")
    assert bot.calls[-1][1]["text"] == "❌ An error occurred." and bot.calls[-1][1].get("reply_markup") is None


async def test_no_stop_callback_means_no_button_anywhere():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    await status.update(step="s", details="d", iteration=1, critique=0, trace=make_trace())
    assert not has_row(bot.calls[0][1]) and bot.calls[0][1].get("reply_markup") is None
    text_status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=8), rich=False, header="h")
    await text_status.update(step="s", details="d", iteration=1, critique=0, trace=make_trace())
    assert bot.calls[-1][1].get("reply_markup") is None


# ---- action buttons on the summary ---------------------------------------------
ACTIONS = [("▶️ Continue", "turn_cont_0123456789ab", "primary"), ("🔁 Retry", "turn_retry_0123456789ab", None)]
AS_KEYBOARD = [[(text, style, data) for text, data, style in ACTIONS]]


def row_data(kw):
    row = kw["rich_message"].blocks[-1]
    assert isinstance(row, InputRichBlockButtons)
    return [(b.text, b.callback_data, b.style) for b in row.buttons]


async def test_summary_ends_with_the_action_row():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h", stop_callback=STOP)
    trace = make_trace()
    await status.update(step="s", details="d", iteration=1, critique=0, trace=trace)
    assert await status.finish(trace, keep=True, actions=ACTIONS) is True
    kw = bot.calls[-1][1]
    assert bot.kinds() == ["rich", "rich"] and kw["message_id"] == 7
    assert kw["rich_message"].blocks[0].text[0].text.startswith("🔧 3 tool calls")
    assert row_data(kw) == ACTIONS and kw.get("reply_markup") is None
    assert STOP not in kw["rich_message"].model_dump_json()  # the Stop button is gone


async def test_refused_action_row_moves_to_an_inline_keyboard_and_is_remembered():
    bot = RowRefusingBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is True
    assert bot.kinds() == ["rich", "rich"]
    retry = bot.calls[1][1]
    assert not has_row(retry) and keyboard_data(retry) == AS_KEYBOARD
    assert rich.buttons_available() is False and rich.is_available() is True and status.rich is True


async def test_known_refused_rows_go_straight_to_the_inline_keyboard():
    rich.mark_buttons_unavailable("BUTTON_TYPE_INVALID")
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is True
    assert bot.kinds() == ["rich"] and not has_row(bot.calls[0][1]) and keyboard_data(bot.calls[0][1]) == AS_KEYBOARD


async def test_refused_rich_summary_falls_back_to_text_with_the_keyboard():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: BLOCK_INVALID"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is True
    assert bot.kinds() == ["rich", "rich", "text"]  # with the row, without it, then text
    text = bot.calls[-1][1]
    assert text["text"].startswith("*🔧 3 tool calls") and keyboard_data(text) == AS_KEYBOARD
    assert status.rich is False and rich.buttons_available() is True  # the row was not the problem


async def test_unsupported_server_summary_uses_text_with_the_keyboard():
    bot = FakeBot(rich_error=TelegramNotFound(method=_M, message="Not Found"))
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is True
    assert bot.kinds() == ["rich", "text"] and keyboard_data(bot.calls[-1][1]) == AS_KEYBOARD
    assert rich.is_available() is False


async def test_text_status_summary_carries_the_keyboard():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=False, header="h", stop_callback=STOP)
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is True
    assert bot.kinds() == ["text"] and keyboard_data(bot.calls[0][1]) == AS_KEYBOARD


async def test_actions_are_not_placed_when_deleted_absent_or_every_edit_fails():
    bot = FakeBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.finish(make_trace(), keep=False, actions=ACTIONS) is False
    assert bot.kinds() == ["delete"]

    assert await StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=8), rich=True, header="h").finish(make_trace(), keep=True) is False
    assert not has_row(bot.calls[-1][1])  # no actions: the summary exactly as before

    class BrokenBot(FakeBot):
        async def edit_message_text(self, **kw):
            self.calls.append(("rich" if "rich_message" in kw else "text", kw))
            raise TelegramBadRequest(method=_M, message="Bad Request: message to edit not found")

    broken = BrokenBot()
    status = StatusMessage(ChatSender(broken, 42), SimpleNamespace(message_id=9), rich=True, header="h")
    assert await status.finish(make_trace(), keep=True, actions=ACTIONS) is False
    assert await StatusMessage(ChatSender(bot, 42), None, rich=True, header="h").finish(make_trace(), keep=True, actions=ACTIONS) is False
