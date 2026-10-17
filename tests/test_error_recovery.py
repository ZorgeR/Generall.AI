"""F2 error recovery, the bot side: a failed turn is never a dead end.

``friendly_error`` turns an exception into the plain-language line the user sees, ``run_turn``
saves a turn record with ``error`` and renders the status as "❌ <reason>" with a Retry
button (``StatusMessage.fail``), ``record_actions`` offers Retry alone for such a record, and
pressing Retry replays the stored prompt like for a successful turn.
"""
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import anthropic
import httpx
import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage
from aiogram.types import Chat, InlineKeyboardMarkup, InputRichBlockButtons, InputRichBlockParagraph, Message

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import rich, turn_actions
from bot.agent_runner import friendly_error, run_turn
from bot.handlers import messages as chat_messages
from bot.handlers import turn_actions_ui as ui
from bot.queue import Job, JobContext, SubmitResult
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.status import StatusMessage
from bot.turn_actions import RETRY_LABEL, TurnRecord

USER = "42"
_M = SendMessage(chat_id=1, text="x")
_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls, status: int, message: str = "api error", *, body=None, headers=None):
    """An Anthropic SDK status error as the SDK would raise it."""
    return cls(message, response=httpx.Response(status, request=_REQUEST, headers=headers or {}), body=body)


# ---- friendly_error ---------------------------------------------------------------------
@pytest.mark.parametrize("error, expected", [
    (api_error(anthropic.RateLimitError, 429, "rate_limit_error: too many requests"), agent_runner.ERROR_RATE_LIMIT),
    (api_error(anthropic.OverloadedError, 529, "overloaded_error: Overloaded"), agent_runner.ERROR_OVERLOADED),
    (api_error(anthropic.InternalServerError, 500, "internal server error"), agent_runner.ERROR_OVERLOADED),
    (api_error(anthropic.APIStatusError, 503, "service unavailable"), agent_runner.ERROR_OVERLOADED),
    (api_error(anthropic.APIStatusError, 400, "x", body={"error": {"type": "overloaded_error"}}), agent_runner.ERROR_OVERLOADED),
    (anthropic.APIConnectionError(request=_REQUEST), agent_runner.ERROR_NETWORK),
    (anthropic.APITimeoutError(request=_REQUEST), agent_runner.ERROR_NETWORK),
    (api_error(anthropic.BadRequestError, 400, "prompt is too long: 210000 tokens > 200000 maximum"), agent_runner.ERROR_TOO_LARGE),
    (api_error(anthropic.BadRequestError, 400, "max_tokens: 64000 > 32000, which is the maximum allowed"), agent_runner.ERROR_TOO_LARGE),
    (api_error(anthropic.BadRequestError, 400, "input length and max_tokens exceed context limit"), agent_runner.ERROR_TOO_LARGE),
    (api_error(anthropic.RequestTooLargeError, 413, "request_too_large"), agent_runner.ERROR_TOO_LARGE),
    (api_error(anthropic.BadRequestError, 400, "messages.0.content: field required"), agent_runner.ERROR_GENERIC),
    (api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"), agent_runner.ERROR_GENERIC),
    (TelegramBadRequest(method=_M, message="Bad Request: message is too long"), "Telegram refused the message: Bad Request: message is too long"),
    (TelegramNetworkError(method=_M, message="Connection reset"), "Telegram refused the message: Connection reset"),
    (RuntimeError("boom"), agent_runner.ERROR_GENERIC),
    (ValueError(), agent_runner.ERROR_GENERIC),
])
def test_friendly_error_mapping(error, expected):
    assert friendly_error(error) == expected
    assert friendly_error(error, "abc-123") == f"{expected} (trace abc-123)"


def test_telegram_detail_is_one_short_line():
    text = friendly_error(TelegramBadRequest(method=_M, message="Bad Request:\n  " + "x" * 300))
    assert "\n" not in text and text.endswith("…") and len(text) < len("Telegram refused the message: ") + 125


# ---- run_turn: the error path -------------------------------------------------------------
class FakeBot:
    """Records the Bot API calls; ``row_error`` refuses every rich edit that carries a button row."""

    def __init__(self, *, row_error: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.row_error = row_error
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        message = kw.get("rich_message")
        if self.row_error is not None and message is not None and any(isinstance(b, InputRichBlockButtons) for b in message.blocks):
            raise self.row_error
        return SimpleNamespace(message_id=kw["message_id"])

    async def edit_message_reply_markup(self, **kw):
        self.calls.append(("edit_message_reply_markup", kw))
        return True

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    async def send_document(self, **kw):
        self.calls.append(("send_document", kw))
        return self._msg()

    async def send_voice(self, **kw):
        self.calls.append(("send_voice", kw))
        return self._msg()

    async def send_message_draft(self, **kw):
        return True

    async def send_rich_message_draft(self, **kw):
        return True

    def of(self, name):
        return [kw for n, kw in self.calls if n == name]


def buttons(kw) -> list[str]:
    """Callback data of every button on one edit: rich button rows and the inline keyboard."""
    data = []
    message = kw.get("rich_message")
    if message is not None and message.blocks:
        data += [b.callback_data for blk in message.blocks if isinstance(blk, InputRichBlockButtons) for b in blk.buttons]
    if kw.get("reply_markup") is not None:
        data += [b.callback_data for row in kw["reply_markup"].inline_keyboard for b in row]
    return data


def edit_text(kw) -> str:
    if kw.get("rich_message") is not None:
        return str(kw["rich_message"].blocks[0].text)
    return kw.get("text") or ""


def use_agent(monkeypatch, *, error: Exception | None = None, response: str = "Here is X.", usage: bool = True):
    """A ChainOfThoughtAgent stand-in that spends tokens, then fails with ``error`` or answers."""

    class Agent:
        def __init__(self, **kw):
            self.agent = SimpleNamespace(limit_reached=False)

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            await update_status("thinking", "Generating", 0, 0)
            if usage and trace is not None:
                trace.add_usage({"input_tokens": 100, "output_tokens": 10}, model="claude-sonnet-5-5")
            if error is not None:
                raise error
            return response, [{"role": "assistant", "content": [{"type": "text", "text": response}]}]

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Workspace under tmp_path, a stats tracker that records usage rows, drafts off, rich on, TTS configured."""
    monkeypatch.chdir(tmp_path)
    usage: list[dict] = []
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None,
        track_usage=lambda *a, **k: usage.append(k),
        get_user_action_count=lambda *a, **k: 0,
        track_message_received=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key="k", data_dir="data"))
    rich.reset()
    yield SimpleNamespace(usage=usage)
    rich.reset()


def records() -> list[dict]:
    directory = Path("data") / USER / "turns"
    return [json.loads(p.read_text()) for p in directory.glob("*.json")] if directory.is_dir() else []


async def run(bot, *, ctx=True, **kw):
    context = JobContext(Job(user_id=USER, label="text", run=None)) if ctx else None
    return await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Tell me about X", ctx=context, **kw)


async def test_a_failed_turn_shows_the_reason_and_a_retry_button(monkeypatch, env, caplog):
    use_agent(monkeypatch, error=api_error(anthropic.RateLimitError, 429, "rate limited"))
    bot = FakeBot()
    assert await run(bot) is None

    last = bot.of("edit_message_text")[-1]
    text = edit_text(last)
    assert text.startswith(f"❌ {agent_runner.ERROR_RATE_LIMIT} (trace ") and isinstance(last["rich_message"].blocks[0], InputRichBlockParagraph)
    trace_id = text.rsplit("(trace ", 1)[1].rstrip(")")
    assert any(trace_id in r.getMessage() for r in caplog.records if r.levelname == "ERROR")  # the log entry support can find

    (record,) = records()
    assert buttons(last) == [f"turn_retry_{record['id']}"]  # Retry alone: no Stop, no Listen, no Reasoning
    assert record["error"] == text[2:] and record["response"] == "" and record["prompt"] == "Tell me about X"
    assert record["has_reasoning"] is False and not list((Path("data") / USER / "turns").glob("*.reasoning.txt"))
    assert bot.of("send_document") == [] and bot.of("send_rich_message") == [] and bot.of("edit_message_reply_markup") == []
    assert len(env.usage) == 1 and env.usage[0]["model"] == "claude-sonnet-5-5"  # the tokens were spent


async def test_a_media_prompt_keeps_its_blocks_for_retry(monkeypatch):
    use_agent(monkeypatch, error=RuntimeError("boom"))
    blocks = [{"type": "text", "text": "User sent a photo"}, {"type": "media_ref", "kind": "image", "path": "data/42/images/x.jpg", "title": "x.jpg"}]
    ctx = JobContext(Job(user_id=USER, label="photo", run=None))
    assert await run_turn(bot=FakeBot(), user_id=USER, chat_id=42, prompt=blocks, ctx=ctx, speak=True, thread_id=5) is None
    (record,) = records()
    assert record["prompt_blocks"] == blocks and record["speak"] is True and record["thread_id"] == 5
    assert record["error"].startswith(agent_runner.ERROR_GENERIC)


async def test_legacy_mode_error_is_plain_text_with_an_inline_keyboard(monkeypatch):
    UserSettings(USER).set("rich_messages", False, "enabled")
    use_agent(monkeypatch, error=anthropic.APITimeoutError(request=_REQUEST))
    bot = FakeBot()
    assert await run(bot) is None
    last = bot.of("edit_message_text")[-1]
    assert "rich_message" not in last and "parse_mode" not in last  # plain text: the reason is not Markdown
    assert last["text"].startswith(f"❌ {agent_runner.ERROR_NETWORK} (trace ")
    assert isinstance(last["reply_markup"], InlineKeyboardMarkup)
    assert [b.text for row in last["reply_markup"].inline_keyboard for b in row] == [RETRY_LABEL]


async def test_a_refused_button_row_falls_back_to_an_inline_keyboard(monkeypatch):
    use_agent(monkeypatch, error=RuntimeError("boom"))
    bot = FakeBot(row_error=TelegramBadRequest(method=_M, message="Bad Request: BUTTON_TYPE_INVALID"))
    assert await run(bot, ctx=False) is None  # no Stop row on the progress edits: the first row is the Retry row
    with_row, without = bot.of("edit_message_text")[-2:]
    assert any(isinstance(b, InputRichBlockButtons) for b in with_row["rich_message"].blocks)
    assert not any(isinstance(b, InputRichBlockButtons) for b in without["rich_message"].blocks)
    assert edit_text(without).startswith(f"❌ {agent_runner.ERROR_GENERIC}") and len(buttons(without)) == 1
    assert rich.buttons_available() is False  # remembered for the process, like the summary's row


async def test_a_failed_record_save_shows_the_reason_without_buttons(monkeypatch):
    use_agent(monkeypatch, error=RuntimeError("boom"))

    async def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(turn_actions, "save", broken)
    bot = FakeBot()
    assert await run(bot) is None
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last).startswith(f"❌ {agent_runner.ERROR_GENERIC} (trace ") and buttons(last) == []
    assert records() == [] and bot.of("send_document") == []


async def test_a_failure_after_the_agent_answered_counts_the_usage_once(monkeypatch, env):
    """The voice reply blows up after the agent returned: usage was already recorded, Retry is still offered."""
    use_agent(monkeypatch)

    async def broken(text, voice_id):
        raise RuntimeError("tts down")

    monkeypatch.setattr(agent_runner, "synthesize_speech", broken)
    monkeypatch.setitem(sys.modules, "voice", SimpleNamespace(VoiceManager=lambda: SimpleNamespace(get_user_voice=lambda uid: "v")))
    bot = FakeBot()
    assert await run(bot, speak=True) is None
    assert len(env.usage) == 1
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last).startswith(f"❌ {agent_runner.ERROR_GENERIC}") and len(buttons(last)) == 1
    assert sum(1 for r in records() if r["error"]) == 1  # the successful record is there too, from before the failure


async def test_without_a_job_the_error_still_gets_retry(monkeypatch):
    use_agent(monkeypatch, error=api_error(anthropic.OverloadedError, 529, "Overloaded"))
    bot = FakeBot()
    assert await run(bot, ctx=False) is None
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last).startswith(f"❌ {agent_runner.ERROR_OVERLOADED}") and buttons(last)[0].startswith("turn_retry_")
    assert not any(d.startswith("queue_stop_") for kw in bot.of("edit_message_text") for d in buttons(kw))


async def test_a_cancel_is_not_an_error(monkeypatch):
    """The CancelledError path is unchanged: "🛑 Stopped.", no record, no Retry."""
    gate = asyncio.Event()

    class Agent:
        def __init__(self, **kw):
            self.agent = SimpleNamespace(limit_reached=False)

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            await update_status("thinking", "Generating", 0, 0)
            await gate.wait()

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    bot = FakeBot()
    ctx = JobContext(Job(user_id=USER, label="text", run=None))
    task = asyncio.create_task(run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Tell me about X", ctx=ctx))
    for _ in range(100):
        if bot.of("edit_message_text"):
            break
        await asyncio.sleep(0.01)
    ctx.cancel_reason = "user"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    last = bot.of("edit_message_text")[-1]
    assert edit_text(last) == "🛑 Stopped." and buttons(last) == [] and records() == []


# ---- the record and its actions -----------------------------------------------------------
def test_error_records_offer_retry_only():
    record = TurnRecord(id=turn_actions.new_turn_id(), prompt="p", response="", error="Something went wrong. (trace x)",
                        limit_reached=True, has_reasoning=True)
    assert turn_actions.record_actions(record, tts=True) == [(RETRY_LABEL, f"turn_retry_{record.id}", None)]
    fine = TurnRecord(id=turn_actions.new_turn_id(), prompt="p", response="r", limit_reached=True, has_reasoning=True)
    assert [a[0] for a in turn_actions.record_actions(fine, tts=True)] == ["▶️ Continue", RETRY_LABEL, "🔊 Listen", "📄 Reasoning"]


async def test_error_round_trips_and_old_records_read_as_none():
    record = TurnRecord(id=turn_actions.new_turn_id(), prompt="p", response="", error="Network problem reaching the AI provider. (trace t)")
    await turn_actions.save(USER, record)
    assert turn_actions.load(USER, record.id) == record
    data = record.to_dict()
    assert TurnRecord.from_dict(data).error == record.error
    del data["error"]
    assert TurnRecord.from_dict(data).error is None  # a record written before this field existed
    assert TurnRecord.from_dict({**data, "error": ""}).error is None


# ---- pressing Retry on an error record ----------------------------------------------------
class FakeCallback:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.answers: list[tuple] = []

    async def answer(self, text=None, show_alert=None, **kw):
        self.answers.append((text, bool(show_alert)))


class FakeQueue:
    def __init__(self):
        self.jobs: list[Job] = []

    async def submit(self, job):
        self.jobs.append(job)
        return SubmitResult(was_busy=False, position=0, current=None)


def chat_message() -> Message:
    from datetime import datetime, timezone

    return Message(message_id=500, date=datetime.now(timezone.utc), chat=Chat(id=42, type="private"))


async def test_retry_on_a_failed_turn_replays_its_prompt(monkeypatch):
    use_agent(monkeypatch, error=api_error(anthropic.RateLimitError, 429, "rate limited"))
    bot = FakeBot()
    assert await run(bot, speak=True) is None
    (retry_data,) = buttons(bot.of("edit_message_text")[-1])

    queue, turns = FakeQueue(), []

    async def fake_run_turn(**kw):
        turns.append(kw)

    monkeypatch.setattr(chat_messages, "queue", queue)
    monkeypatch.setattr(ui, "run_turn", fake_run_turn)
    cb = FakeCallback(retry_data, chat_message())
    await ui.on_retry(cb, bot=bot, user_id=USER, limit=50)
    assert cb.answers == [(None, False)]
    assert bot.of("send_message")[-1]["text"] == "🔁 Retry"
    (job,) = queue.jobs
    assert job.label == "retry"
    await job.run(JobContext(job))
    (turn,) = turns
    assert turn["prompt"] == "Tell me about X" and turn["speak"] is True and turn["limit"] == 50 and turn["chat_id"] == 42


async def test_retry_on_a_failed_media_turn_replays_the_blocks(monkeypatch):
    blocks = [{"type": "text", "text": "User sent a photo"}, {"type": "media_ref", "kind": "image", "path": "data/42/images/x.jpg", "title": "x.jpg"}]
    record = TurnRecord(id=turn_actions.new_turn_id(), prompt="User sent a photo [image: x.jpg]", prompt_blocks=blocks, response="", error="boom")
    await turn_actions.save(USER, record)
    queue, turns = FakeQueue(), []

    async def fake_run_turn(**kw):
        turns.append(kw)

    monkeypatch.setattr(chat_messages, "queue", queue)
    monkeypatch.setattr(ui, "run_turn", fake_run_turn)
    await ui.on_retry(FakeCallback(f"turn_retry_{record.id}", chat_message()), bot=FakeBot(), user_id=USER, limit=None)
    (job,) = queue.jobs
    await job.run(JobContext(job))
    assert turns[0]["prompt"] == blocks


# ---- StatusMessage.fail -------------------------------------------------------------------
class StatusBot:
    def __init__(self, rich_error=None, row_error=None):
        self.calls = []
        self.rich_error = rich_error
        self.row_error = row_error

    async def edit_message_text(self, **kw):
        kind = "rich" if "rich_message" in kw else "text"
        self.calls.append((kind, kw))
        if kind == "rich":
            if self.rich_error is not None:
                raise self.rich_error
            if self.row_error is not None and any(isinstance(b, InputRichBlockButtons) for b in kw["rich_message"].blocks):
                raise self.row_error
        return SimpleNamespace(message_id=kw["message_id"])

    def kinds(self):
        return [k for k, _ in self.calls]


RETRY = [(RETRY_LABEL, "turn_retry_0123456789ab", None)]


async def test_fail_edits_the_status_into_the_reason_with_the_action_row():
    bot = StatusBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h", stop_callback="queue_stop_1f")
    assert await status.fail("❌ Something went wrong. (trace t)", actions=RETRY) is True
    (kw,) = [kw for _, kw in bot.calls]
    paragraph, row = kw["rich_message"].blocks
    assert isinstance(paragraph, InputRichBlockParagraph) and paragraph.text == "❌ Something went wrong. (trace t)"
    assert isinstance(row, InputRichBlockButtons) and [(b.text, b.callback_data) for b in row.buttons] == [(RETRY_LABEL, "turn_retry_0123456789ab")]
    assert kw.get("reply_markup") is None and "queue_stop_" not in kw["rich_message"].model_dump_json()


async def test_fail_without_actions_or_message():
    bot = StatusBot()
    status = StatusMessage(ChatSender(bot, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.fail("❌ x") is False
    assert bot.kinds() == ["rich"] and len(bot.calls[0][1]["rich_message"].blocks) == 1
    assert await StatusMessage(ChatSender(bot, 42), None, rich=True, header="h").fail("❌ x", actions=RETRY) is False
    assert len(bot.calls) == 1


async def test_fail_falls_back_like_finish():
    refused_row = StatusBot(row_error=TelegramBadRequest(method=_M, message="Bad Request: BUTTON_TYPE_INVALID"))
    status = StatusMessage(ChatSender(refused_row, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.fail("❌ x", actions=RETRY) is True
    assert refused_row.kinds() == ["rich", "rich"]
    retry = refused_row.calls[1][1]
    assert len(retry["rich_message"].blocks) == 1 and [b.text for r in retry["reply_markup"].inline_keyboard for b in r] == [RETRY_LABEL]
    assert rich.buttons_available() is False
    rich.reset()

    no_rich = StatusBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: BLOCK_INVALID"))
    status = StatusMessage(ChatSender(no_rich, 42), SimpleNamespace(message_id=7), rich=True, header="h")
    assert await status.fail("❌ x", actions=RETRY) is True
    assert no_rich.kinds() == ["rich", "rich", "text"]  # with the row, without it, then plain text
    text = no_rich.calls[-1][1]
    assert text["text"] == "❌ x" and "parse_mode" not in text and [b.text for r in text["reply_markup"].inline_keyboard for b in r] == [RETRY_LABEL]

    legacy = StatusBot()
    status = StatusMessage(ChatSender(legacy, 42), SimpleNamespace(message_id=7), rich=False, header="h", stop_callback="queue_stop_1f")
    assert await status.fail("❌ x", actions=RETRY) is True
    assert legacy.kinds() == ["text"] and legacy.calls[0][1]["reply_markup"] is not None
    assert "queue_stop_" not in str(legacy.calls[0][1]["reply_markup"])
