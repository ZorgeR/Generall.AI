"""bot/status.py: rich progress/summary blocks and the StatusMessage lifecycle."""
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import InputRichBlockDetails, InputRichBlockParagraph

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
        trace.add_usage({"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 500}, model="claude-sonnet-5")
        trace.add_usage({"input_tokens": 100, "output_tokens": 50}, model="mystery-model")
    if thinking:
        trace.add_thinking("Let me think about pings.")
    return trace


def test_usage_text_and_cost():
    trace = make_trace(calls=0)
    text = usage_text(trace)
    assert text.startswith("🧮 2 calls · in 10.6k (") and "% cached" in text and "out 250" in text
    assert "≈$" in text  # sonnet-5 is priced; the unknown model only adds tokens
    assert trace.usage_by_model["claude-sonnet-5"]["cache_read_tokens"] == 9000
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
    await status.update(step="s", details="d", iteration=0, critique=0, trace=trace)
    await status.update(step="s", details="d2", iteration=1, critique=0, trace=trace)
    kinds = bot.kinds()
    assert kinds[0] == "rich" and kinds[1] == "text" and kinds[2] == "text"  # rich tried once, then plain only
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
