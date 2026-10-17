"""Buttons on a finished turn: follow-up parsing, the turn store, keyboards, run_turn placement, handlers."""
import asyncio
import copy
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aiogram.types import Chat, InlineKeyboardMarkup, InputRichBlockButtons, Message

import bot.agent_runner as agent_runner
import bot.streaming as streaming
from bot import media, rich, turn_actions
from bot.agent_runner import run_turn
from bot.handlers import messages as chat_messages
from bot.handlers import turn_actions_ui as ui
from bot.queue import Job, JobContext, SubmitResult
from bot.turn_actions import TurnRecord, parse_follow_ups, strip_follow_ups_draft

USER = "42"
MESSAGES = [
    {"role": "user", "content": "Tell me about X"},
    {"role": "assistant", "content": [{"type": "text", "text": "Here is X."}]},
]
ANSWER = "Here is X.\n\n<follow_ups>\n- Make it shorter\n- Compare with 2024\n</follow_ups>"


# ---- parse_follow_ups / strip_follow_ups_draft ------------------------------------
def test_no_block_returns_the_text_untouched():
    assert parse_follow_ups("Just an answer.\n") == ("Just an answer.\n", [])
    assert parse_follow_ups("") == ("", [])


def test_closed_block_at_the_end_is_removed_with_its_whitespace():
    assert parse_follow_ups(ANSWER) == ("Here is X.", ["Make it shorter", "Compare with 2024"])


def test_unclosed_block_runs_to_the_end():
    assert parse_follow_ups("Answer.\n<follow_ups>\n- One\n- Two") == ("Answer.", ["One", "Two"])


def test_block_in_the_middle_keeps_the_text_after_it():
    text = "Intro.\n<FOLLOW_UPS>\n- Next\n</Follow_Ups>\n\nOutro."
    assert parse_follow_ups(text) == ("Intro.\n\nOutro.", ["Next"])


def test_only_the_last_block_counts():
    text = "Use <follow_ups>a</follow_ups> tags.\n\n<follow_ups>\n- Real one\n</follow_ups>"
    assert parse_follow_ups(text) == ("Use <follow_ups>a</follow_ups> tags.", ["Real one"])


def test_markers_and_quotes_are_stripped_and_items_are_capped_and_deduped():
    body = "\n".join(['- "Make it shorter"', "* «Add sources»", "• make it shorter", "1. Translate", "2) Summarize", "", "  - Fifth"])
    clean, items = parse_follow_ups(f"Done.\n<follow_ups>\n{body}\n</follow_ups>")
    assert clean == "Done."
    assert items == ["Make it shorter", "Add sources", "Translate", "Summarize"]  # deduped, max 4


def test_long_items_are_cut_and_an_empty_block_gives_no_items():
    _, items = parse_follow_ups("x<follow_ups>- " + "y" * 300 + "</follow_ups>")
    assert len(items[0]) == turn_actions.MAX_FOLLOW_UP_CHARS and items[0].endswith("…")
    assert parse_follow_ups("Answer.\n\n<follow_ups>\n\n</follow_ups>\n") == ("Answer.", [])
    assert parse_follow_ups("<follow_ups>- Only buttons</follow_ups>") == ("", ["Only buttons"])


@pytest.mark.parametrize("tail", ["<", "<f", "<fol", "<follow_", "<FOLLOW_UPS", "<follow_ups"])
def test_draft_cuts_a_half_typed_opening_tag(tail):
    assert strip_follow_ups_draft(f"Answer\n\n{tail}") == "Answer\n\n"


def test_draft_removes_the_block_and_keeps_everything_else():
    assert strip_follow_ups_draft("Answer\n\n<follow_ups>\n- Sho") == "Answer"
    assert strip_follow_ups_draft(ANSWER) == "Here is X."
    assert strip_follow_ups_draft("a < b") == "a < b"
    assert strip_follow_ups_draft("x <b>") == "x <b>"
    assert strip_follow_ups_draft("") == ""


# ---- the store ------------------------------------------------------------------------
@pytest.fixture
def workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key=None, data_dir="data"))
    return tmp_path


def make_record(**kw) -> TurnRecord:
    base = dict(id=turn_actions.new_turn_id(), prompt="Tell me about X", response="Here is X.")
    base.update(kw)
    return TurnRecord(**base)


async def test_save_and_load_round_trip(workdir):
    record = make_record(speak=True, thread_id=7, limit_reached=True, follow_ups=["a", "b"], reasoning_caption="Agent task reasoning history.")
    await turn_actions.save(USER, record, "the transcript")
    loaded = turn_actions.load(USER, record.id)
    assert loaded == record and loaded.has_reasoning is True
    assert datetime.fromisoformat(loaded.created).tzinfo is not None
    assert turn_actions.load_reasoning(USER, record.id) == "the transcript"
    files = sorted(p.name for p in (workdir / "data" / USER / "turns").iterdir())
    assert files == [f"{record.id}.json", f"{record.id}.reasoning.txt"]  # no temp files left behind

    plain = make_record()
    await turn_actions.save(USER, plain)
    assert turn_actions.load(USER, plain.id).has_reasoning is False
    assert turn_actions.load_reasoning(USER, plain.id) is None


@pytest.mark.parametrize("turn_id", ["../../settings", "ABCDEF012345", "abc", "0123456789abc", "", None, 12])
async def test_invalid_ids_never_build_a_path(workdir, turn_id):
    assert turn_actions.load(USER, turn_id) is None
    assert turn_actions.load_reasoning(USER, turn_id) is None
    if isinstance(turn_id, str):
        with pytest.raises(ValueError):
            await turn_actions.save(USER, make_record(id=turn_id))
    assert not (workdir / "data").exists()


async def test_invalid_user_and_junk_files_load_as_none(workdir):
    record = make_record()
    await turn_actions.save(USER, record)
    assert turn_actions.load("../42", record.id) is None
    path = workdir / "data" / USER / "turns" / f"{record.id}.json"
    path.write_text("{not json")
    assert turn_actions.load(USER, record.id) is None
    other = make_record()
    path.write_text(json.dumps({**record.to_dict(), "id": other.id}))  # a record under the wrong name
    assert turn_actions.load(USER, record.id) is None


async def test_prune_keeps_the_newest_turns_and_removes_both_files(workdir):
    directory = workdir / "data" / USER / "turns"
    ids = []
    for i in range(turn_actions.MAX_TURNS + 2):
        record = make_record()
        await turn_actions.save(USER, record, f"reasoning {i}")
        ids.append(record.id)
        os.utime(directory / f"{record.id}.json", ns=(10**18 + i, 10**18 + i))  # strictly ascending age
    newest = make_record()
    await turn_actions.save(USER, newest, "latest")
    kept = {p.stem for p in directory.glob("*.json")}
    assert len(kept) == turn_actions.MAX_TURNS and newest.id in kept
    assert kept.isdisjoint(ids[:3])  # the three oldest are gone ...
    for old in ids[:3]:
        assert not (directory / f"{old}.reasoning.txt").exists()  # ... with their reasoning files
    assert turn_actions.load_reasoning(USER, ids[3]) == "reasoning 3"


# ---- keyboards ------------------------------------------------------------------------
def labels(actions):
    return [a[0] for a in actions]


def test_actions_order_and_conditions(workdir):
    full = make_record(limit_reached=True, has_reasoning=True)
    actions = turn_actions.record_actions(full, tts=True)
    assert labels(actions) == ["▶️ Continue", "🔁 Retry", "🔊 Listen", "📄 Reasoning"]
    assert actions[0][2] == "primary" and all(style is None for _, _, style in actions[1:])
    assert [a[1] for a in actions] == [f"turn_{kind}_{full.id}" for kind in ("cont", "retry", "listen", "reason")]
    assert labels(turn_actions.record_actions(make_record(), tts=False)) == ["🔁 Retry"]

    # Listen follows the ElevenLabs key when not given
    assert "🔊 Listen" not in labels(turn_actions.record_actions(full))
    turn_actions.config.elevenlabs_api_key = "key"
    assert "🔊 Listen" in labels(turn_actions.record_actions(full))


def test_callback_data_fits_and_round_trips():
    turn_id = turn_actions.new_turn_id()
    for action in ("retry", "listen", "reason", "cont"):
        data = turn_actions.callback_data(action, turn_id)
        assert len(data.encode()) <= 64 and turn_actions.parse_callback(data) == (action, turn_id, None)
    data = turn_actions.callback_data("fu", turn_id, 3)
    assert data == f"turn_fu_{turn_id}_3" and len(data.encode()) <= 64
    assert turn_actions.parse_callback(data) == ("fu", turn_id, 3)
    for junk in (None, "", "turn_retry_", "turn_retry_XYZ", f"turn_fu_{turn_id}", f"turn_nope_{turn_id}", f"queue_stop_{turn_id}"):
        assert turn_actions.parse_callback(junk) is None


def test_answer_keyboard_puts_follow_ups_first_one_per_row():
    record = make_record(follow_ups=["Short one", "A much longer suggestion that will not fit on a button"])
    actions = turn_actions.record_actions(record, tts=False)
    markup = turn_actions.answer_keyboard(record, actions)
    rows = [[(b.text, b.callback_data) for b in row] for row in markup.inline_keyboard]
    assert rows[0] == [("Short one", f"turn_fu_{record.id}_0")]
    label = rows[1][0][0]
    assert len(label) <= turn_actions.FOLLOW_UP_LABEL and label.endswith("…") and label.startswith("A much longer")
    assert rows[2] == [("🔁 Retry", f"turn_retry_{record.id}")]
    assert len(turn_actions.answer_keyboard(record).inline_keyboard) == 2  # actions placed elsewhere
    assert turn_actions.answer_keyboard(make_record()) is None


# ---- run_turn -------------------------------------------------------------------------
class Sent(SimpleNamespace):
    """A sent message; ``answer`` is what the busy notice calls."""

    async def answer(self, text, **kw):
        self.bot.calls.append(("answer", {"text": text, **kw}, None))


class FakeBot:
    def __init__(self, *, markup_error=None):
        self.calls: list[tuple[str, dict, object]] = []
        self.markup_error = markup_error
        self._id = 100

    def _record(self, name, kw, result=None):
        self.calls.append((name, kw, result))
        return result

    def _msg(self):
        self._id += 1
        return Sent(message_id=self._id, bot=self)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        return self._record("send_message", kw, self._msg())

    async def send_rich_message(self, **kw):
        return self._record("send_rich_message", kw, self._msg())

    async def edit_message_text(self, **kw):
        return self._record("edit_message_text", kw, SimpleNamespace(message_id=kw["message_id"]))

    async def edit_message_reply_markup(self, **kw):
        self._record("edit_message_reply_markup", kw)
        if self.markup_error is not None:
            raise self.markup_error
        return True

    async def delete_message(self, **kw):
        return self._record("delete_message", kw, True)

    async def send_document(self, **kw):
        return self._record("send_document", kw, self._msg())

    async def send_voice(self, **kw):
        return self._record("send_voice", kw, self._msg())

    def of(self, name):
        return [kw for n, kw, _ in self.calls if n == name]

    def results(self, name):
        return [r for n, _, r in self.calls if n == name]


def keyboard(markup: InlineKeyboardMarkup):
    return [[(b.text, b.callback_data) for b in row] for row in markup.inline_keyboard]


def use_agent(monkeypatch, response=ANSWER, messages=MESSAGES, *, limit_reached=False):
    prompts: list[str] = []

    class Agent:
        def __init__(self, **kw):
            self.agent = SimpleNamespace(limit_reached=limit_reached)

        async def generate_response(self, prompt, update_status, on_text_chunk=None, trace=None):
            prompts.append(prompt)
            await update_status("thinking", "Generating", 0, 0)
            return response, messages

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    return prompts


def write_settings(**categories):
    path = Path("data") / USER / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(categories))


@pytest.fixture
def env(monkeypatch, workdir):
    """Workspace under tmp_path, a no-op stats tracker, drafts off, rich support on."""
    received: list[tuple] = []
    stats = ModuleType("stats")
    stats.stats_tracker = SimpleNamespace(
        track_message_sent=lambda *a, **k: None,
        track_usage=lambda *a, **k: None,
        get_user_action_count=lambda *a, **k: 0,
        track_message_received=lambda *a, **k: received.append(a),
    )
    monkeypatch.setitem(sys.modules, "stats", stats)
    monkeypatch.setattr(streaming, "config", SimpleNamespace(streaming_enabled=False))
    rich.reset()
    yield SimpleNamespace(received=received)
    rich.reset()


async def test_rich_turn_actions_go_on_the_summary_and_follow_ups_under_the_answer(env, monkeypatch):
    prompts = use_agent(monkeypatch)
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="Tell me about X")
    assert prompts == ["Tell me about X"]
    assert result.response == "Here is X." and turn_actions.parse_follow_ups(result.response)[1] == []

    record = turn_actions.load(USER, result.turn_id)
    assert record.prompt == "Tell me about X" and record.response == "Here is X."
    assert record.follow_ups == ["Make it shorter", "Compare with 2024"] and record.has_reasoning

    assert [kw["rich_message"].markdown for kw in bot.of("send_rich_message")] == ["Here is X."]
    summary = bot.of("edit_message_text")[-1]
    row = summary["rich_message"].blocks[-1]
    assert isinstance(row, InputRichBlockButtons) and summary.get("reply_markup") is None
    assert [(b.text, b.callback_data) for b in row.buttons] == [
        ("🔁 Retry", f"turn_retry_{record.id}"), ("📄 Reasoning", f"turn_reason_{record.id}"),
    ]

    answer = bot.results("send_rich_message")[-1]
    attached = bot.of("edit_message_reply_markup")
    assert len(attached) == 1 and attached[0]["message_id"] == answer.message_id and attached[0]["chat_id"] == 42
    assert keyboard(attached[0]["reply_markup"]) == [
        [("Make it shorter", f"turn_fu_{record.id}_0")], [("Compare with 2024", f"turn_fu_{record.id}_1")],
    ]
    assert bot.of("send_document") == []  # the reasoning file waits for its button


async def test_without_the_summary_the_actions_go_under_the_answer(env, monkeypatch):
    use_agent(monkeypatch, limit_reached=True)
    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key="key", data_dir="data"))
    write_settings(trace={"keep_summary": False})
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert bot.of("delete_message")  # the status is gone
    rows = keyboard(bot.of("edit_message_reply_markup")[0]["reply_markup"])
    tid = result.turn_id
    assert rows == [
        [("Make it shorter", f"turn_fu_{tid}_0")], [("Compare with 2024", f"turn_fu_{tid}_1")],
        [("▶️ Continue", f"turn_cont_{tid}"), ("🔁 Retry", f"turn_retry_{tid}"), ("🔊 Listen", f"turn_listen_{tid}"), ("📄 Reasoning", f"turn_reason_{tid}")],
    ]
    assert bot.of("edit_message_reply_markup")[0]["reply_markup"].inline_keyboard[2][0].style == "primary"
    assert turn_actions.load(USER, tid).limit_reached is True


async def test_legacy_mode_answer_replaces_the_status_and_carries_every_button(env, monkeypatch):
    use_agent(monkeypatch)
    write_settings(rich_messages={"enabled": False})
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    status = bot.results("send_message")[0]
    answer = bot.of("edit_message_text")[-1]
    assert answer["message_id"] == status.message_id and answer["text"] == "Here is X."
    attached = bot.of("edit_message_reply_markup")
    assert len(attached) == 1 and attached[0]["message_id"] == status.message_id
    rows = keyboard(attached[0]["reply_markup"])
    assert len(rows) == 3 and rows[2] == [("🔁 Retry", f"turn_retry_{result.turn_id}"), ("📄 Reasoning", f"turn_reason_{result.turn_id}")]


async def test_no_follow_ups_and_actions_on_the_summary_means_no_answer_keyboard(env, monkeypatch):
    use_agent(monkeypatch, response="Plain answer.")
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert result.response == "Plain answer." and bot.of("edit_message_reply_markup") == []
    assert isinstance(bot.of("edit_message_text")[-1]["rich_message"].blocks[-1], InputRichBlockButtons)


async def test_follow_ups_off_hides_the_block_but_makes_no_buttons(env, monkeypatch):
    use_agent(monkeypatch)
    write_settings(follow_ups={"enabled": False})
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert result.response == "Here is X." and bot.of("send_rich_message")[0]["rich_message"].markdown == "Here is X."
    assert turn_actions.load(USER, result.turn_id).follow_ups == []
    assert bot.of("edit_message_reply_markup") == []


async def test_a_failed_save_sends_the_reasoning_file_and_no_buttons(env, monkeypatch):
    use_agent(monkeypatch)

    async def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(turn_actions, "save", broken)
    bot = FakeBot()
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert result.turn_id is None and result.response == "Here is X."
    assert len(bot.of("send_document")) == 1
    assert bot.of("edit_message_reply_markup") == []
    summary = bot.of("edit_message_text")[-1]
    assert not any(isinstance(b, InputRichBlockButtons) for b in summary["rich_message"].blocks)


async def test_a_refused_keyboard_never_breaks_the_turn(env, monkeypatch):
    use_agent(monkeypatch)
    bot = FakeBot(markup_error=RuntimeError("message to edit not found"))
    result = await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi")
    assert result is not None and len(bot.of("edit_message_reply_markup")) == 1


async def test_errors_get_only_a_retry_button(env, monkeypatch):
    class Agent:
        def __init__(self, **kw):
            pass

        async def generate_response(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(agent_runner, "_agents", lambda: SimpleNamespace(ChainOfThoughtAgent=Agent))
    bot = FakeBot()
    assert await run_turn(bot=bot, user_id=USER, chat_id=42, prompt="hi") is None
    assert bot.of("edit_message_reply_markup") == []  # no answer message to hang buttons under
    row = bot.of("edit_message_text")[-1]["rich_message"].blocks[-1]  # the status: reason + Retry (tests/test_error_recovery.py)
    assert isinstance(row, InputRichBlockButtons) and [b.text for b in row.buttons] == ["🔁 Retry"]
    (record_file,) = (Path("data") / USER / "turns").glob("*.json")  # the failed turn is saved so Retry can replay it
    assert json.loads(record_file.read_text())["error"].startswith("Something went wrong.")


# ---- streaming drafts --------------------------------------------------------------------
async def test_drafts_and_partial_text_never_show_the_block(monkeypatch):
    monkeypatch.setattr(streaming, "DRAFT_THROTTLE", 0)
    drafts: list[str] = []

    class DraftBot:
        async def send_message_draft(self, **kw):
            drafts.append(kw["text"])

    cb = streaming.create_streaming_callback(DraftBot(), 42, enabled=True)
    for chunk in ("Answer", "Answer\n\n<fol", "Answer\n\n<follow_ups>\n- Sho", "Answer\n\n<follow_ups>\n- Shorter\n</follow_ups>"):
        await cb(chunk, False)
    await cb("<follow", True)  # thinking is shown as it is
    assert drafts == ["Answer", "Answer\n\n", "Answer", "💭 <follow"]
    assert cb.partial_text() == "Answer"


# ---- the limit_reached flag ------------------------------------------------------------
@pytest.fixture
def agents_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(name, "x")
    return importlib.import_module("agents.main")


def reply(*blocks, stop="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=None)


def text_block(t):
    return SimpleNamespace(type="text", text=t)


def tool_block(n):
    return SimpleNamespace(type="tool_use", id=f"tu{n}", name="get_time_in_timezone", input={"timezone": "UTC"})


def scripted_agent(agents_main, monkeypatch, script):
    agent = agents_main.AgentAnthropic(user_id=None)
    replies = list(script)

    async def stream(kwargs, on_event=None):
        return replies.pop(0)

    async def run_tools(blocks, trace=None, refresh=None):
        return [{"type": "tool_result", "tool_use_id": b.id, "content": "12:00"} for b in blocks]

    monkeypatch.setattr(agent, "_stream_message", stream)
    monkeypatch.setattr(agent, "run_tool_batch", run_tools)
    return agent, replies


def settings(**tools):
    from bot.settings import DEFAULT_SETTINGS

    s = copy.deepcopy(DEFAULT_SETTINGS)
    s["tools"].update(tools)
    return s


async def test_limit_flag_on_a_refused_tool_call(agents_main, monkeypatch):
    from agents.trace import TurnBudget

    agent, _ = scripted_agent(agents_main, monkeypatch, [
        reply(tool_block(1), stop="tool_use"),
        reply(text_block("I checked one thing; two remain."), tool_block(2), stop="tool_use"),
    ])
    agent.limit_reached = True  # reset at the start of every run
    text, _ = await agent.generate_response(prompt="go", user_settings=settings(max_iteration=5), budget=TurnBudget(1))
    assert text == "I checked one thing; two remain." and agent.limit_reached is True


async def test_limit_flag_on_loop_exhaustion(agents_main, monkeypatch):
    agent, replies = scripted_agent(agents_main, monkeypatch, [
        reply(tool_block(1), stop="tool_use"),
        reply(text_block("Final summary, remaining steps listed.")),  # the SYSTEM NOTICE call
    ])
    text, messages = await agent.generate_response(prompt="go", user_settings=settings(max_iteration=1))
    assert text == "Final summary, remaining steps listed." and replies == []
    assert any("SYSTEM NOTICE" in str(m["content"]) for m in messages)
    assert agent.limit_reached is True


async def test_no_limit_flag_for_a_normal_answer_or_disabled_tools(agents_main, monkeypatch):
    agent, _ = scripted_agent(agents_main, monkeypatch, [reply(tool_block(1), stop="tool_use"), reply(text_block("done"))])
    await agent.generate_response(prompt="go", user_settings=settings(max_iteration=5))
    assert agent.limit_reached is False

    agent, _ = scripted_agent(agents_main, monkeypatch, [reply(text_block("no tools for me"), tool_block(1), stop="tool_use")])
    await agent.generate_response(prompt="go", user_settings=settings(enabled=False, max_iteration=5))
    assert agent.limit_reached is False


@pytest.mark.parametrize("enabled", [True, False])
async def test_follow_ups_guide_follows_the_setting(agents_main, monkeypatch, enabled):
    from agents.subagent import SUBAGENT_SYSTEM

    s = settings()
    s["follow_ups"] = {"enabled": enabled}
    agent = agents_main.ChainOfThoughtAgent(user_id=USER, user_settings=s)
    seen: list[str] = []

    async def capture(question, system_context, *a, **k):
        seen.append(system_context)
        return "ok", []

    monkeypatch.setattr(agent, "_generate_with_transcript", capture)
    await agent.generate_response("hi")
    # right after the formatting guide, and only when the setting is on; the providers'
    # <capabilities> notes (ChainOfThoughtAgent._provider_notes) come after the guides
    tail = agents_main.RICH_FORMATTING_GUIDE + (agents_main.FOLLOW_UPS_GUIDE if enabled else "")
    before_notes = seen[0].split("\n<capabilities>")[0]
    assert before_notes.endswith(tail) and (agents_main.FOLLOW_UPS_GUIDE in seen[0]) is enabled
    assert "follow_ups" not in SUBAGENT_SYSTEM  # subagents report to the main agent, never to the user


# ---- handlers --------------------------------------------------------------------------
class FakeCallback:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.answers: list[tuple] = []

    async def answer(self, text=None, show_alert=None, **kw):
        self.answers.append((text, bool(show_alert)))


def chat_message(*, topic: int | None = None) -> Message:
    extra = {"message_thread_id": topic, "is_topic_message": True} if topic else {}
    return Message(message_id=500, date=datetime.now(timezone.utc), chat=Chat(id=42, type="supergroup" if topic else "private"), **extra)


class FakeQueue:
    def __init__(self, busy=False):
        self.jobs: list[Job] = []
        self.busy = busy

    async def submit(self, job):
        self.jobs.append(job)
        current = Job(user_id=USER, label="text message", run=None) if self.busy else None
        return SubmitResult(was_busy=self.busy, position=1 if self.busy else 0, current=current)


@pytest.fixture
def handlers(env, monkeypatch):
    queue = FakeQueue()
    monkeypatch.setattr(chat_messages, "queue", queue)
    turns: list[dict] = []

    async def fake_run_turn(**kw):
        turns.append(kw)

    monkeypatch.setattr(ui, "run_turn", fake_run_turn)
    return SimpleNamespace(queue=queue, turns=turns, received=env.received)


async def saved(reasoning: str | None = "the transcript", **kw) -> TurnRecord:
    record = make_record(**kw)
    await turn_actions.save(USER, record, reasoning)
    return record


async def test_retry_submits_the_stored_prompt_as_a_new_turn(handlers):
    record = await saved(prompt="User sent a photo... describe it", speak=True, thread_id=None)
    bot = FakeBot()
    cb = FakeCallback(f"turn_retry_{record.id}", chat_message())
    await ui.on_retry(cb, bot=bot, user_id=USER, limit=50)
    assert cb.answers == [(None, False)]
    echo = bot.of("send_message")[0]
    assert echo["text"] == "🔁 Retry" and "parse_mode" not in echo
    assert handlers.received == [(USER, "button")]

    job = handlers.queue.jobs[0]
    assert job.user_id == USER and job.label == "retry"
    await job.run(JobContext(job))
    turn = handlers.turns[0]
    assert turn["prompt"] == "User sent a photo... describe it" and turn["speak"] is True and turn["limit"] == 50
    assert turn["reply_to_message_id"] == bot.results("send_message")[0].message_id
    assert turn["chat_id"] == 42 and turn["thread_id"] is None


async def test_continue_and_follow_up_prompts_and_topics(handlers):
    record = await saved(follow_ups=["Make it shorter"], thread_id=3)
    bot = FakeBot()
    await ui.on_continue(FakeCallback(f"turn_cont_{record.id}", chat_message()), bot=bot, user_id=USER, limit=None)
    await ui.on_follow_up(FakeCallback(f"turn_fu_{record.id}_0", chat_message(topic=9)), bot=bot, user_id=USER, limit=None)
    assert [kw["text"] for kw in bot.of("send_message")] == ["▶️ Continue", "💬 Make it shorter"]
    assert bot.of("send_message")[0]["message_thread_id"] == 3  # the record's topic
    assert bot.of("send_message")[1]["message_thread_id"] == 9  # the pressed message's topic wins
    for job in handlers.queue.jobs:
        await job.run(JobContext(job))
    assert [t["prompt"] for t in handlers.turns] == [ui.CONTINUE_PROMPT, "Make it shorter"]
    assert [t["thread_id"] for t in handlers.turns] == [3, 9]
    assert [j.label for j in handlers.queue.jobs] == ["continue", "follow-up"]


async def test_a_busy_queue_gets_the_busy_notice(handlers):
    handlers.queue.busy = True
    record = await saved()
    bot = FakeBot()
    await ui.on_retry(FakeCallback(f"turn_retry_{record.id}", chat_message()), bot=bot, user_id=USER, limit=None)
    assert len(handlers.queue.jobs) == 1 and "previous request" in bot.of("answer")[0]["text"]


@pytest.mark.parametrize("make_data", [
    lambda r: "turn_retry_000000000000",  # unknown (pruned) turn
    lambda r: f"turn_fu_{r.id}_1",  # follow-up index out of range
    lambda r: f"turn_retry_{r.id}xx",  # malformed
])
async def test_expired_presses_get_an_alert_and_start_nothing(handlers, make_data):
    record = await saved(follow_ups=["only one"])
    bot = FakeBot()
    data = make_data(record)
    handler = ui.on_follow_up if data.startswith("turn_fu_") else ui.on_retry
    cb = FakeCallback(data, chat_message())
    await handler(cb, bot=bot, user_id=USER, limit=None)
    assert cb.answers == [(ui.EXPIRED_TEXT, True)]
    assert bot.calls == [] and handlers.queue.jobs == []


async def test_a_press_without_a_message_or_from_another_chat_expires(handlers):
    record = await saved()
    cb = FakeCallback(f"turn_retry_{record.id}", None)
    await ui.on_retry(cb, bot=FakeBot(), user_id=USER, limit=None)
    assert cb.answers == [(ui.EXPIRED_TEXT, True)]
    cb = FakeCallback(f"turn_retry_{record.id}", chat_message())
    await ui.on_retry(cb, bot=FakeBot(), user_id="77", limit=None)  # records are per chat
    assert cb.answers == [(ui.EXPIRED_TEXT, True)]
    cb = FakeCallback("turn_bogus", chat_message())
    await ui.on_unknown(cb)
    assert cb.answers == [(ui.EXPIRED_TEXT, True)]


async def test_reasoning_button_sends_the_stored_file(handlers):
    record = await saved(reasoning_caption="Agent task reasoning history.")
    bot = FakeBot()
    cb = FakeCallback(f"turn_reason_{record.id}", chat_message())
    await ui.on_reasoning(cb, bot=bot, user_id=USER)
    assert cb.answers == [(None, False)]
    doc = bot.of("send_document")[0]
    assert doc["document"].filename == f"reasoning_42_{record.id}.txt"
    assert doc["document"].data == b"the transcript" and doc["caption"] == "Agent task reasoning history."

    plain = await saved(reasoning=None)
    cb = FakeCallback(f"turn_reason_{plain.id}", chat_message())
    await ui.on_reasoning(cb, bot=bot, user_id=USER)
    assert cb.answers == [(ui.EXPIRED_TEXT, True)] and len(bot.of("send_document")) == 1


@pytest.fixture
def tts(monkeypatch):
    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key="key", data_dir="data"))
    voice = ModuleType("voice")
    voice.VoiceManager = lambda: SimpleNamespace(get_user_voice=lambda user_id: f"voice-of-{user_id}")
    monkeypatch.setitem(sys.modules, "voice", voice)
    calls: list[tuple] = []
    gate = asyncio.Event()
    gate.set()

    async def synthesize(text, voice_id):
        calls.append((text, voice_id))
        await gate.wait()
        return b"mp3" if text != "fail" else None

    monkeypatch.setattr(media, "synthesize_speech", synthesize)
    return SimpleNamespace(calls=calls, gate=gate)


async def test_listen_reads_the_stored_answer_once_per_press(handlers, tts):
    record = await saved(response="Here is X.")
    bot = FakeBot()
    tts.gate.clear()
    first = FakeCallback(f"turn_listen_{record.id}", chat_message())
    task = asyncio.create_task(ui.on_listen(first, bot=bot, user_id=USER))
    await asyncio.sleep(0.01)
    second = FakeCallback(f"turn_listen_{record.id}", chat_message())
    await ui.on_listen(second, bot=bot, user_id=USER)  # ignored while the first one works
    tts.gate.set()
    await task
    assert first.answers == [(ui.PREPARING_AUDIO_TEXT, False)] and second.answers == [(ui.PREPARING_AUDIO_TEXT, False)]
    assert tts.calls == [("Here is X.", f"voice-of-{USER}")]
    assert len(bot.of("send_voice")) == 1 and bot.of("send_voice")[0]["voice"].data == b"mp3"

    await ui.on_listen(FakeCallback(f"turn_listen_{record.id}", chat_message()), bot=bot, user_id=USER)
    assert len(tts.calls) == 2  # free again once the first finished


async def test_listen_failures(handlers, tts, monkeypatch):
    record = await saved(response="fail")
    bot = FakeBot()
    cb = FakeCallback(f"turn_listen_{record.id}", chat_message())
    await ui.on_listen(cb, bot=bot, user_id=USER)
    assert cb.answers == [(ui.PREPARING_AUDIO_TEXT, False)]
    assert bot.of("send_voice") == [] and bot.of("send_message")[0]["text"] == f"❌ {ui.AUDIO_FAILED_TEXT}"

    monkeypatch.setattr(turn_actions, "config", SimpleNamespace(elevenlabs_api_key=None, data_dir="data"))
    cb = FakeCallback(f"turn_listen_{record.id}", chat_message())
    await ui.on_listen(cb, bot=bot, user_id=USER)
    assert cb.answers == [(ui.AUDIO_FAILED_TEXT, True)] and len(tts.calls) == 1
