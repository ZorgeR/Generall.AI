"""ask_user_choice: the question registry, rendering, the tool call and the tap handler."""
import asyncio
import copy
import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import (
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputRichBlockButtons,
    InputRichBlockParagraph,
    Message,
    RichBlockButtons,
    RichBlockParagraph,
    RichMessage,
    RichMessageButton,
)

from agents.user_interactions import UserInteractions
from bot import questions, rich
from bot.handlers.questions_ui import ALREADY_ANSWERED_TEXT, EXPIRED_TEXT, on_question_answer
from bot.sender import ChatSender

_M = SendMessage(chat_id=1, text="x")
CHAT = 42


@pytest.fixture(autouse=True)
def _reset():
    rich.reset()
    questions.reset()
    yield
    rich.reset()
    questions.reset()


class FakeBot:
    def __init__(self, *, rich_error=None, edit_error=None):
        self.calls: list[tuple[str, dict]] = []
        self.rich_error = rich_error
        self.edit_error = edit_error
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        if self.rich_error is not None:
            raise self.rich_error
        return self._msg()

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        if self.edit_error is not None:
            raise self.edit_error
        return self._msg()

    async def edit_message_reply_markup(self, **kw):
        self.calls.append(("edit_message_reply_markup", kw))
        return True

    def names(self):
        return [n for n, _ in self.calls]

    def sent(self):
        return [kw for n, kw in self.calls if n in ("send_rich_message", "send_message")]

    def edits(self):
        return [kw for n, kw in self.calls if n == "edit_message_text"]


def qid_of(payload: dict) -> str:
    """The question id carried by the buttons of a sent question."""
    if "rich_message" in payload:
        button = next(b for b in payload["rich_message"].blocks if isinstance(b, InputRichBlockButtons)).buttons[0]
    else:
        button = payload["reply_markup"].inline_keyboard[0][0]
    return questions.parse_callback_data(button.callback_data)[0]


async def wait_sent(bot: FakeBot, count: int = 1) -> None:
    for _ in range(500):
        if len(bot.sent()) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError("the question was never sent")


def tg_message(message_id=5, chat_id=CHAT, **kw) -> Message:
    return Message(message_id=message_id, date=datetime.now(timezone.utc), chat=Chat(id=chat_id, type="private"), **kw)


class FakeCallback:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.answers: list[str] = []

    async def answer(self, text=None, **kw):
        self.answers.append(text)


# ---- registry -------------------------------------------------------------------
async def test_resolve_answers_once_and_returns_the_option():
    q = questions.register(CHAT, "Send it?", ["Yes", "No"])
    assert questions.get(q.id) is q and questions.state(q.id) == "pending"
    assert questions.resolve(q.id, CHAT, 1) == "No"
    assert q.future.result() == "No"
    assert questions.state(q.id) == "answered"
    assert questions.resolve(q.id, CHAT, 0) is None  # already answered
    questions.discard(q.id)
    assert questions.get(q.id) is None and questions.state(q.id) == "answered"


async def test_resolve_rejects_other_chat_bad_index_and_unknown_ids():
    q = questions.register(CHAT, "Pick", ["A", "B", "C"])
    assert questions.resolve(q.id, 7, 0) is None  # another chat
    assert questions.resolve(q.id, CHAT, 3) is None
    assert questions.resolve(q.id, CHAT, -1) is None
    assert questions.resolve("deadbeef", CHAT, 0) is None
    assert not q.future.done()
    assert questions.resolve(q.id, str(CHAT), 2) == "C"  # the chat id may come as a string


async def test_discarded_question_is_expired():
    q = questions.register(CHAT, "Pick", ["A", "B"])
    questions.discard(q.id)
    assert questions.resolve(q.id, CHAT, 0) is None
    assert questions.state(q.id) == "closed"
    assert questions.state("0badf00d") == "unknown"
    questions.discard("0badf00d")  # unknown ids are a no-op


async def test_concurrent_questions_get_distinct_ids():
    ids = {questions.register(CHAT, f"q{i}", ["A", "B"]).id for i in range(50)}
    assert len(ids) == 50


# ---- callback data and arguments -------------------------------------------------
def test_callback_data_fits_and_round_trips():
    data = questions.callback_data("deadbeef", 7)
    assert data == "ask_deadbeef_7" and len(data.encode()) <= 64
    assert questions.parse_callback_data(data) == ("deadbeef", 7)


@pytest.mark.parametrize(
    "garbage",
    [None, 12, "", "ask_", "ask_deadbeef", "ask_deadbeef_", "ask_xyzxyzxy_1", "ask_deadbeef_x", "ask_deadbeef_1_2",
     "ask_DEADBEEF_1", "turn_deadbeef_1", "ask_deadbeef_123"],
)
def test_parse_callback_data_tolerates_garbage(garbage):
    assert questions.parse_callback_data(garbage) is None


def test_clean_options_strips_dedupes_and_truncates():
    long = "x" * 100
    cleaned = questions.clean_options(["  Yes ", "", "yes", "No\nway", None, long, long + "y", 3])
    assert cleaned == ["Yes", "No way", "x" * 63 + "…", "3"]
    assert all(len(o) <= questions.MAX_LABEL for o in cleaned)
    assert questions.clean_options('["A", "B"]') == ["A", "B"]  # array sent as a JSON string


@pytest.mark.parametrize("bad", [["only one"], ["A", "a", " "], [], "not json", {"a": 1}, [str(i) for i in range(9)]])
def test_clean_options_rejects_too_few_or_too_many(bad):
    with pytest.raises(ValueError):
        questions.clean_options(bad)


def test_timeout_default_and_clamp():
    assert questions.clamp_timeout(None) == 300
    assert questions.clamp_timeout(5) == 30
    assert questions.clamp_timeout(5000) == 900
    assert questions.clamp_timeout("120") == 120
    with pytest.raises(ValueError):
        questions.clamp_timeout("soon")


def test_timeout_infinite_is_clamped_and_nan_is_an_error():
    # JSON "1e999" parses to inf; int(inf) used to raise OverflowError past the tool's "Error:" handling
    assert questions.clamp_timeout(float("inf")) == 900
    assert questions.clamp_timeout("-Infinity") == 30
    with pytest.raises(ValueError):
        questions.clamp_timeout(float("nan"))


async def test_tool_with_overflowing_timeout_does_not_raise(monkeypatch):
    seen = {}

    async def fake_ask(sender, question, options, timeout):
        seen["timeout"] = timeout
        return "A"

    monkeypatch.setattr(questions, "ask", fake_ask)
    args = {"question": "Q?", "options": ["A", "B"], "timeout_seconds": float("inf")}
    assert await run_tool(ChatSender(FakeBot(), CHAT), args) == 'User chose: "A"'
    assert seen["timeout"] == 900
    args["timeout_seconds"] = float("nan")
    assert (await run_tool(ChatSender(FakeBot(), CHAT), args)).startswith("Error:")


# ---- rendering ---------------------------------------------------------------------
async def test_rows_two_per_row_or_one_when_a_label_is_long():
    assert questions.button_rows(["A", "B", "C"]) == [[0, 1], [2]]
    assert questions.button_rows(["Short", "A label longer than twenty-four"]) == [[0], [1]]
    assert all(len(r) <= questions.ROW_LIMIT for r in questions.button_rows([str(i) for i in range(8)]))


async def test_rich_and_legacy_forms_carry_the_same_rows():
    q = questions.register(CHAT, "Which one?\n\nDetails here.", ["A", "B", "C"])
    blocks = questions.question_blocks(q)
    assert [type(b) for b in blocks] == [InputRichBlockParagraph, InputRichBlockParagraph, InputRichBlockButtons, InputRichBlockButtons]
    assert blocks[0].text == "❓ Which one?" and blocks[1].text == "Details here."
    assert [[b.text for b in blk.buttons] for blk in blocks[2:]] == [["A", "B"], ["C"]]
    assert blocks[3].buttons[0].callback_data == f"ask_{q.id}_2"
    kb = questions.keyboard(q)
    assert [[b.text for b in row] for row in kb.inline_keyboard] == [["A", "B"], ["C"]]
    assert kb.inline_keyboard[0][1].callback_data == f"ask_{q.id}_1"
    final = questions.final_blocks(q, questions.answered_line("B"))
    assert [b.text for b in final] == ["❓ Which one?", "Details here.", "✅ B"]
    assert questions.final_text(q, questions.NO_ANSWER_LINE) == "❓ Which one?\n\nDetails here.\n\n⌛ No answer"


# ---- the tool ----------------------------------------------------------------------
async def run_tool(sender, args):
    return await UserInteractions(str(CHAT), sender).execute_tool("ask_user_choice", args)


async def test_tool_rich_happy_path():
    bot = FakeBot()
    task = asyncio.create_task(run_tool(ChatSender(bot, CHAT, thread_id=9, rich=True), {"question": "Send the SMS?", "options": ["Send", "Cancel"]}))
    await wait_sent(bot)
    payload = bot.sent()[0]
    assert bot.names() == ["send_rich_message"] and payload["message_thread_id"] == 9
    qid = qid_of(payload)
    assert questions.get(qid).rich is True and questions.get(qid).message_id == 101
    assert questions.resolve(qid, CHAT, 0) == "Send"
    assert await task == 'User chose: "Send"'
    assert questions.get(qid) is None and questions.state(qid) == "answered"
    assert bot.edits() == []  # on an answer the tap handler edits the message, not the tool


async def test_tool_legacy_happy_path_with_inline_keyboard():
    bot = FakeBot()
    task = asyncio.create_task(run_tool(ChatSender(bot, CHAT, rich=False), {"question": "Use snake_case?", "options": ["Yes", "No"]}))
    await wait_sent(bot)
    payload = bot.sent()[0]
    assert bot.names() == ["send_message"]
    assert payload["text"] == "❓ Use snake_case?" and "parse_mode" not in payload  # plain text, underscores kept
    assert isinstance(payload["reply_markup"], InlineKeyboardMarkup)
    questions.resolve(qid_of(payload), CHAT, 1)
    assert await task == 'User chose: "No"'


async def test_two_questions_at_once():
    bot = FakeBot()
    sender = ChatSender(bot, CHAT, rich=True)
    tasks = [
        asyncio.create_task(run_tool(sender, {"question": "First?", "options": ["A", "B"]})),
        asyncio.create_task(run_tool(sender, {"question": "Second?", "options": ["C", "D"]})),
    ]
    await wait_sent(bot, 2)
    first, second = (qid_of(p) for p in bot.sent())
    assert first != second
    questions.resolve(second, CHAT, 0)
    questions.resolve(first, CHAT, 1)
    assert await asyncio.gather(*tasks) == ['User chose: "B"', 'User chose: "C"']


async def test_generic_rich_rejection_falls_back_without_disabling_buttons():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: can't parse rich message"))
    task = asyncio.create_task(questions.ask(ChatSender(bot, CHAT, rich=True), "Q?", ["A", "B"], 5))
    await wait_sent(bot, 2)
    assert bot.names() == ["send_rich_message", "send_message"]
    qid = qid_of(bot.sent()[1])
    assert questions.get(qid).rich is False
    questions.resolve(qid, CHAT, 0)
    assert await task == "A"
    assert rich.is_available() and rich.buttons_available()


async def test_button_rejection_marks_buttons_unavailable():
    bot = FakeBot(rich_error=TelegramBadRequest(method=_M, message="Bad Request: BUTTON_TYPE_INVALID"))
    task = asyncio.create_task(questions.ask(ChatSender(bot, CHAT, rich=True), "Q?", ["A", "B"], 5))
    await wait_sent(bot, 2)
    questions.resolve(qid_of(bot.sent()[1]), CHAT, 1)
    assert await task == "B"
    assert rich.is_available() and not rich.buttons_available()


async def test_unsupported_server_marks_rich_unavailable_and_falls_back():
    bot = FakeBot(rich_error=TelegramNotFound(method=_M, message="Not Found"))
    task = asyncio.create_task(questions.ask(ChatSender(bot, CHAT, rich=True), "Q?", ["A", "B"], 5))
    await wait_sent(bot, 2)
    questions.resolve(qid_of(bot.sent()[1]), CHAT, 0)
    await task
    assert rich.is_available() is False


async def test_buttons_flag_off_goes_straight_to_the_keyboard():
    rich.mark_buttons_unavailable("test")
    bot = FakeBot()
    task = asyncio.create_task(questions.ask(ChatSender(bot, CHAT, rich=True), "Q?", ["A", "B"], 5))
    await wait_sent(bot)
    assert bot.names() == ["send_message"]
    questions.resolve(qid_of(bot.sent()[0]), CHAT, 0)
    await task


@pytest.mark.parametrize("rich_mode", [True, False])
async def test_timeout_edits_into_no_answer_and_expires(rich_mode):
    bot = FakeBot()
    answer = await questions.ask(ChatSender(bot, CHAT, rich=rich_mode), "Q?", ["A", "B"], 0.01)
    assert answer is None
    qid = qid_of(bot.sent()[0])
    assert questions.get(qid) is None and questions.state(qid) == "closed"
    assert questions.resolve(qid, CHAT, 0) is None  # a late tap answers nothing
    edit = bot.edits()[-1]
    assert edit["message_id"] == 101
    if rich_mode:
        assert [b.text for b in edit["rich_message"].blocks] == ["❓ Q?", "⌛ No answer"]
    else:
        assert edit["text"] == "❓ Q?\n\n⌛ No answer" and edit.get("reply_markup") is None


async def test_tool_reports_the_timeout(monkeypatch):
    seen = {}

    async def fake_ask(sender, question, options, timeout):
        seen["timeout"] = timeout
        return None

    monkeypatch.setattr(questions, "ask", fake_ask)
    result = await run_tool(ChatSender(FakeBot(), CHAT), {"question": "Q?", "options": ["A", "B"], "timeout_seconds": 5000})
    assert seen["timeout"] == 900
    assert result == "No answer within 900 seconds. Continue without it: take the safe default or ask in your final reply."


async def test_cancellation_discards_edits_and_reraises():
    bot = FakeBot()
    task = asyncio.create_task(run_tool(ChatSender(bot, CHAT, rich=True), {"question": "Q?", "options": ["A", "B"]}))
    await wait_sent(bot)
    qid = qid_of(bot.sent()[0])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert questions.get(qid) is None and questions.state(qid) == "closed"
    assert [b.text for b in bot.edits()[-1]["rich_message"].blocks] == ["❓ Q?", "🛑 Cancelled"]


async def test_failed_rich_final_edit_falls_back_to_plain_text():
    bot = FakeBot(edit_error=TelegramBadRequest(method=_M, message="Bad Request: message can't be edited"))
    assert await questions.ask(ChatSender(bot, CHAT, rich=True), "Q?", ["A", "B"], 0.01) is None
    assert bot.names() == ["send_rich_message", "edit_message_text", "edit_message_text"]
    assert bot.edits()[-1]["text"] == "❓ Q?\n\n⌛ No answer"


@pytest.mark.parametrize(
    "args",
    [
        {"options": ["A", "B"]},
        {"question": "  ", "options": ["A", "B"]},
        {"question": "Q?", "options": ["A"]},
        {"question": "Q?", "options": ["A", "a"]},
        {"question": "Q?", "options": [str(i) for i in range(9)]},
        {"question": "Q?"},
        {"question": "Q?", "options": ["A", "B"], "timeout_seconds": "later"},
    ],
)
async def test_invalid_arguments_are_errors(args):
    bot = FakeBot()
    result = await run_tool(ChatSender(bot, CHAT), args)
    assert result.startswith("Error:")
    assert bot.calls == [] and questions._pending == {}


async def test_missing_sender_is_an_error():
    assert (await run_tool(None, {"question": "Q?", "options": ["A", "B"]})).startswith("Error:")


async def test_send_failure_is_an_error_and_leaves_nothing_pending():
    class Broken(FakeBot):
        async def send_message(self, **kw):
            raise RuntimeError("network down")

    result = await run_tool(ChatSender(Broken(), CHAT), {"question": "Q?", "options": ["A", "B"]})
    assert result.startswith("Error:") and "network down" in result
    assert questions._pending == {}


# ---- the tap handler ---------------------------------------------------------------
async def test_handler_resolves_and_edits_rich_question():
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    q.rich, q.message_id = True, 5
    callback = FakeCallback(questions.callback_data(q.id, 1), tg_message(5))
    await on_question_answer(callback, bot)
    assert callback.answers == ["✅ B"]
    assert q.future.result() == "B"
    edit = bot.edits()[0]
    assert edit["chat_id"] == CHAT and edit["message_id"] == 5
    assert [b.text for b in edit["rich_message"].blocks] == ["❓ Q?", "✅ B"]


async def test_handler_resolves_and_edits_legacy_question_without_keyboard():
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    q.message_id = 5
    callback = FakeCallback(questions.callback_data(q.id, 0), tg_message(5))
    await on_question_answer(callback, bot)
    assert callback.answers == ["✅ A"]
    edit = bot.edits()[0]
    assert edit["text"] == "❓ Q?\n\n✅ A" and edit.get("reply_markup") is None and "rich_message" not in edit


async def test_handler_drives_a_waiting_tool_call_end_to_end():
    bot = FakeBot()
    task = asyncio.create_task(run_tool(ChatSender(bot, CHAT, rich=True), {"question": "Delete 3 files?", "options": ["Delete", "Keep"]}))
    await wait_sent(bot)
    callback = FakeCallback(questions.callback_data(qid_of(bot.sent()[0]), 1), tg_message(101))
    await on_question_answer(callback, bot)
    assert await task == 'User chose: "Keep"'
    assert callback.answers == ["✅ Keep"]
    assert [b.text for b in bot.edits()[0]["rich_message"].blocks] == ["❓ Delete 3 files?", "✅ Keep"]


async def test_handler_second_tap_is_already_answered_and_does_not_edit():
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    questions.resolve(q.id, CHAT, 0)
    callback = FakeCallback(questions.callback_data(q.id, 1), tg_message(5))
    await on_question_answer(callback, bot)
    assert callback.answers == [ALREADY_ANSWERED_TEXT] and bot.calls == []
    questions.discard(q.id)  # the tool call woke up and left
    await on_question_answer(callback, bot)
    assert callback.answers[-1] == ALREADY_ANSWERED_TEXT and bot.calls == []


async def test_handler_timed_out_question_is_expired_without_edit():
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    questions.discard(q.id)
    callback = FakeCallback(questions.callback_data(q.id, 0), tg_message(5))
    await on_question_answer(callback, bot)
    assert callback.answers == [EXPIRED_TEXT] and bot.calls == []


async def test_handler_closed_question_with_leftover_keyboard_removes_it():
    # cancelled mid-send (or the "No answer" edit failed): the buttons are still on the message
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    questions.discard(q.id)
    data = questions.callback_data(q.id, 0)
    markup = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="A", callback_data=data)]])
    callback = FakeCallback(data, tg_message(5, reply_markup=markup))
    await on_question_answer(callback, bot)
    assert callback.answers == [EXPIRED_TEXT]
    assert bot.calls == [("edit_message_reply_markup", {"chat_id": CHAT, "message_id": 5, "reply_markup": None})]


async def test_cancel_during_send_then_late_tap_removes_the_buttons():
    class SlowBot(FakeBot):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()

        async def send_rich_message(self, **kw):
            self.calls.append(("send_rich_message", kw))
            self.entered.set()
            await asyncio.sleep(3600)  # the message reaches Telegram, the reply never reaches us

    bot = SlowBot()
    task = asyncio.create_task(run_tool(ChatSender(bot, CHAT, rich=True), {"question": "Q?", "options": ["A", "B"]}))
    await bot.entered.wait()
    payload = bot.calls[0][1]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bot.edits() == []  # nothing to edit: the tool never got the message
    qid = qid_of(payload)
    rm = RichMessage(blocks=[
        RichBlockParagraph(text="❓ Q?"),
        RichBlockButtons(buttons=[RichMessageButton(text="A", callback_data=questions.callback_data(qid, 0))]),
    ])
    callback = FakeCallback(questions.callback_data(qid, 0), tg_message(101, rich_message=rm))
    await on_question_answer(callback, bot)
    assert callback.answers == [EXPIRED_TEXT]
    assert [b.text for b in bot.edits()[0]["rich_message"].blocks] == ["❓ Q?", questions.EXPIRED_LINE]


async def test_handler_unknown_question_removes_inline_keyboard():
    bot = FakeBot()
    markup = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="A", callback_data="ask_deadbeef_0")]])
    callback = FakeCallback("ask_deadbeef_0", tg_message(5, reply_markup=markup))
    await on_question_answer(callback, bot)
    assert callback.answers == [EXPIRED_TEXT]
    assert bot.calls == [("edit_message_reply_markup", {"chat_id": CHAT, "message_id": 5, "reply_markup": None})]


async def test_handler_unknown_rich_question_is_rerendered_without_buttons():
    bot = FakeBot()
    rm = RichMessage(blocks=[
        RichBlockParagraph(text="❓ Old question"),
        RichBlockButtons(buttons=[RichMessageButton(text="A", callback_data="ask_deadbeef_0")]),
    ])
    callback = FakeCallback("ask_deadbeef_0", tg_message(5, rich_message=rm))
    await on_question_answer(callback, bot)
    assert callback.answers == [EXPIRED_TEXT]
    assert [b.text for b in bot.edits()[0]["rich_message"].blocks] == ["❓ Old question", questions.EXPIRED_LINE]


async def test_handler_other_chat_or_bad_index_leaves_the_question_pending():
    bot = FakeBot()
    q = questions.register(CHAT, "Q?", ["A", "B"])
    for data, chat in ((questions.callback_data(q.id, 0), 7), (questions.callback_data(q.id, 5), CHAT)):
        callback = FakeCallback(data, tg_message(5, chat_id=chat))
        await on_question_answer(callback, bot)
        assert callback.answers == [EXPIRED_TEXT]
    assert bot.calls == [] and not q.future.done()


async def test_handler_garbage_or_inaccessible_message():
    bot = FakeBot()
    for callback in (FakeCallback("ask_nonsense", tg_message()), FakeCallback("ask_deadbeef_0", None)):
        await on_question_answer(callback, bot)
        assert callback.answers == [EXPIRED_TEXT]
    assert bot.calls == []


# ---- schema and wiring ---------------------------------------------------------------
def test_schema_is_valid():
    schema = next(t for t in UserInteractions("1", None).tools_schema if t["name"] == "ask_user_choice")
    props = schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["question", "options"]
    assert props["options"]["type"] == "array" and props["options"]["items"]["type"] == "string"
    assert (props["options"]["minItems"], props["options"]["maxItems"]) == (questions.MIN_OPTIONS, questions.MAX_OPTIONS)
    assert props["options"]["items"]["maxLength"] == questions.MAX_LABEL
    assert (props["timeout_seconds"]["minimum"], props["timeout_seconds"]["maximum"]) == (questions.MIN_TIMEOUT, questions.MAX_TIMEOUT)
    assert props["timeout_seconds"]["default"] == questions.DEFAULT_TIMEOUT
    assert "irreversible" in schema["description"] and "scheduled" in schema["description"]


def test_tool_name_is_unique_across_providers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "x")
    module = importlib.import_module("agents.main")
    from bot.settings import DEFAULT_SETTINGS

    agent = module.ChainOfThoughtAgent(user_id="42", user_settings=copy.deepcopy(DEFAULT_SETTINGS))
    names = [t["name"] for t in agent.agent.get_tools_schema()]
    # the sandbox patcher adds these two to TerminalTools at startup
    names += [n for n in ("run_shell_script", "install_package") if n not in names]
    assert names.count("ask_user_choice") == 1
    assert len(names) == len(set(names))


def test_sandbox_patcher_leaves_user_interactions_alone():
    root = Path(__file__).resolve().parents[1] / "app" / "secure_container"
    for name in ("tool_integrator.py", "secure_tool_wrapper.py"):
        source = (root / name).read_text()
        for needle in ("UserInteractions", "user_interactions", "ask_user_choice", "_ask_user_choice"):
            assert needle not in source, f"{name} mentions {needle}"
