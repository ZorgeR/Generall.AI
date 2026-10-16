"""Multiple-choice questions the agent asks with buttons (the ``ask_user_choice`` tool).

The tool call sends the question with one button per option and waits on an
``asyncio.Future`` until the user taps one (``bot/handlers/questions_ui.py``
resolves it), the timeout passes, or the turn is cancelled. Pending questions
live in an in-memory registry keyed by a short random id, which is also what the
buttons carry (``ask_<id>_<index>``), so a restart forgets them: a tap on an old
question only gets "expired" and the buttons are taken off.

Rendering follows the answer tiers: with rich messages on (and the server
accepting button rows inside them, ``bot.rich.buttons_available``) the question
is a rich message of a paragraph plus ``InputRichBlockButtons`` rows; otherwise
it is plain text with an inline keyboard of the same rows. Either way the
message ends in a final state without buttons: the question plus
"✅ <option>", "⌛ No answer" or "🛑 Cancelled".

The question text is shown as plain text in both forms (no Markdown parsing),
so what the model wrote is what the user reads. All tool calls of one assistant
message run concurrently, so several questions may be pending in one chat.
No import-time side effects and no ``agents`` imports.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputRichBlockButtons,
    InputRichBlockParagraph,
    InputRichMessage,
    Message,
    RichBlockButtons,
    RichBlockParagraph,
    RichMessageButton,
)

from bot import rich as rich_render
from bot.ui import is_not_modified

logger = logging.getLogger(__name__)

CALLBACK_PREFIX = "ask_"
CALLBACK_MAX_BYTES = 64  # Bot API limit for callback_data
MIN_OPTIONS = 2
MAX_OPTIONS = 8
MAX_LABEL = 64  # characters per button label (longer ones are cut with "…")
MAX_QUESTION = 3500  # keeps the question plus its final line under one 4096-char message
DEFAULT_TIMEOUT = 300
MIN_TIMEOUT = 30
MAX_TIMEOUT = 900
WIDE_LABEL = 24  # a label longer than this puts every option on a row of its own
PER_ROW = 2
ROW_LIMIT = 8  # buttons per InputRichBlockButtons row (Bot API: 1-8)
MAX_CLOSED = 256  # recently closed ids remembered, so a late tap gets the right notice

ANSWERED_PREFIX = "✅ "
NO_ANSWER_LINE = "⌛ No answer"
CANCELLED_LINE = "🛑 Cancelled"
EXPIRED_LINE = "⌛ This question has expired."

_CALLBACK_RE = re.compile(r"ask_([0-9a-f]{8})_(\d{1,2})")


@dataclass
class PendingQuestion:
    id: str
    chat_id: int
    question: str
    options: list[str]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    message_id: int | None = None  # set once the question message is sent
    rich: bool = False  # sent as a rich message with button blocks (else text + inline keyboard)


_pending: dict[str, PendingQuestion] = {}
# id -> "answered" | "closed" (timed out / cancelled) for questions that are gone.
_closed: OrderedDict[str, str] = OrderedDict()


# ---- registry ---------------------------------------------------------------
def register(chat_id: int, question: str, options: list[str]) -> PendingQuestion:
    """A new pending question (call from inside the event loop that will await it)."""
    qid = secrets.token_hex(4)
    while qid in _pending or qid in _closed:
        qid = secrets.token_hex(4)
    pending = PendingQuestion(
        id=qid,
        chat_id=int(chat_id),
        question=question,
        options=list(options),
        future=asyncio.get_running_loop().create_future(),
    )
    _pending[qid] = pending
    return pending


def get(qid: str) -> PendingQuestion | None:
    return _pending.get(qid)


def resolve(qid: str, chat_id: int, index: int) -> str | None:
    """Answer question ``qid`` with option ``index``; the option, or None.

    None when the id is unknown or expired, the tap came from another chat, the
    index is out of range, or the question was already answered.
    """
    pending = _pending.get(qid)
    if pending is None or pending.future.done():
        return None
    try:
        if int(chat_id) != pending.chat_id:
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(index, int) or not 0 <= index < len(pending.options):
        return None
    option = pending.options[index]
    pending.future.set_result(option)
    return option


def discard(qid: str) -> None:
    """Forget a pending question; a later tap on it gets "already answered" or "expired"."""
    pending = _pending.pop(qid, None)
    if pending is None:
        return
    answered = pending.future.done() and not pending.future.cancelled()
    _closed[qid] = "answered" if answered else "closed"
    while len(_closed) > MAX_CLOSED:
        _closed.popitem(last=False)


def state(qid: str) -> str:
    """One of: pending, answered (still registered or gone), closed (timed out / cancelled), unknown."""
    pending = _pending.get(qid)
    if pending is not None:
        return "answered" if pending.future.done() else "pending"
    return _closed.get(qid, "unknown")


# ---- callback data ----------------------------------------------------------
def callback_data(qid: str, index: int) -> str:
    data = f"{CALLBACK_PREFIX}{qid}_{index}"
    assert len(data.encode("utf-8")) <= CALLBACK_MAX_BYTES, data
    return data


def parse_callback_data(data: Any) -> tuple[str, int] | None:
    """``"ask_<id>_<index>"`` -> (id, index); None for anything else."""
    if not isinstance(data, str):
        return None
    m = _CALLBACK_RE.fullmatch(data)
    if m is None:
        return None
    return m.group(1), int(m.group(2))


# ---- argument cleaning --------------------------------------------------------
def _label(item: Any) -> str:
    if item is None or isinstance(item, (dict, list, tuple, bool)):
        return ""
    label = " ".join(str(item).split())  # a button label is one line
    if len(label) > MAX_LABEL:
        label = label[: MAX_LABEL - 1].rstrip() + "…"
    return label


def clean_options(options: Any) -> list[str]:
    """Strip, drop empty and duplicate (case-insensitive) labels, cut to ``MAX_LABEL``.

    Raises ValueError when the result is not 2..8 options.
    """
    if isinstance(options, str):
        try:
            options = json.loads(options)  # a model sometimes sends the array as a JSON string
        except ValueError:
            raise ValueError("options must be an array of strings") from None
    if not isinstance(options, (list, tuple)):
        raise ValueError("options must be an array of strings")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in options:
        label = _label(item)
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
    if len(cleaned) < MIN_OPTIONS:
        raise ValueError(f"at least {MIN_OPTIONS} different, non-empty options are needed")
    if len(cleaned) > MAX_OPTIONS:
        raise ValueError(f"at most {MAX_OPTIONS} options are allowed (got {len(cleaned)})")
    return cleaned


def clamp_timeout(value: Any) -> int:
    """Seconds to wait: default when missing, clamped to ``MIN_TIMEOUT..MAX_TIMEOUT``."""
    if value is None or value == "":
        return DEFAULT_TIMEOUT
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("timeout_seconds must be a number of seconds") from None
    if seconds != seconds:  # NaN
        raise ValueError("timeout_seconds must be a number of seconds")
    # clamp before int(): "inf" / 1e999 (JSON parses it to inf) would raise OverflowError
    return int(max(MIN_TIMEOUT, min(MAX_TIMEOUT, seconds)))


def clean_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    question = question.strip()
    if len(question) > MAX_QUESTION:
        question = question[: MAX_QUESTION - 1].rstrip() + "…"
    return question


# ---- rendering ----------------------------------------------------------------
def button_rows(options: list[str]) -> list[list[int]]:
    """Option indexes per row: two per row, one per row when any label is long."""
    per_row = 1 if any(len(o) > WIDE_LABEL for o in options) else PER_ROW
    per_row = max(1, min(per_row, ROW_LIMIT))
    indexes = list(range(len(options)))
    return [indexes[i:i + per_row] for i in range(0, len(indexes), per_row)]


def answered_line(option: str) -> str:
    return f"{ANSWERED_PREFIX}{option}"


def question_text(pending: PendingQuestion) -> str:
    return f"❓ {pending.question}"


def final_text(pending: PendingQuestion, line: str) -> str:
    return f"{question_text(pending)}\n\n{line}"


def _question_paragraphs(pending: PendingQuestion) -> list[InputRichBlockParagraph]:
    parts = [p.strip() for p in pending.question.split("\n\n") if p.strip()] or [pending.question]
    parts[0] = f"❓ {parts[0]}"
    return [InputRichBlockParagraph(text=p) for p in parts]


def question_blocks(pending: PendingQuestion) -> list:
    """Rich form: the question, then one ``InputRichBlockButtons`` block per row of options."""
    blocks: list = _question_paragraphs(pending)
    for row in button_rows(pending.options):
        blocks.append(
            InputRichBlockButtons(
                buttons=[
                    RichMessageButton(text=pending.options[i], callback_data=callback_data(pending.id, i)) for i in row
                ]
            )
        )
    return blocks


def final_blocks(pending: PendingQuestion, line: str) -> list:
    """Rich final state: the question and the outcome line, no buttons."""
    return [*_question_paragraphs(pending), InputRichBlockParagraph(text=line)]


def keyboard(pending: PendingQuestion) -> InlineKeyboardMarkup:
    """Legacy form: the same rows as an inline keyboard under a plain-text message."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=pending.options[i], callback_data=callback_data(pending.id, i)) for i in row]
            for row in button_rows(pending.options)
        ]
    )


# ---- sending and editing --------------------------------------------------------
def _about_buttons(error: BaseException) -> bool:
    return "button" in str(error).lower() and not rich_render.is_chat_error(error)


async def _send(sender, pending: PendingQuestion) -> Message:
    """Rich message with button rows when possible, else text with an inline keyboard.

    A 404 / unknown method marks rich messages unsupported for the process (as every
    other rich send does). Any other 400 falls back to the inline keyboard for this
    question, and the process-wide "buttons inside rich messages" flag is set only
    when Telegram's error names the buttons: a generic 400 may be about this one
    message, the flag also disables the answer buttons of other features until a
    restart, and questions are rare enough that retrying rich next time is cheap.
    """
    if getattr(sender, "rich", False) and rich_render.is_available() and rich_render.buttons_available():
        try:
            message = await sender.bot.send_rich_message(
                rich_message=InputRichMessage(blocks=question_blocks(pending)), **sender._kw()
            )
            pending.rich = True
            return message
        except (TelegramBadRequest, TelegramNotFound) as e:
            if rich_render.is_unsupported_error(e):
                rich_render.mark_unavailable(str(e))
            else:
                logger.warning("Rich question rejected (%s); sending it with an inline keyboard", e)
                if _about_buttons(e):
                    rich_render.mark_buttons_unavailable(str(e))
    pending.rich = False
    return await sender.send_text(question_text(pending), markdown=False, reply_markup=keyboard(pending))


async def show_outcome(sender, pending: PendingQuestion, line: str, message) -> None:
    """Edit the question ``message`` into its final state without buttons. Never raises.

    ``sender`` is a ``ChatSender`` for the question's chat; a rich question is edited as
    rich blocks, a plain one through ``sender.edit_text`` (no reply_markup, so the
    inline keyboard goes away).
    """
    if message is None:
        return
    if pending.rich:
        try:
            await sender.bot.edit_message_text(
                chat_id=sender.chat_id,
                message_id=message.message_id,
                rich_message=InputRichMessage(blocks=final_blocks(pending, line)),
            )
            return
        except TelegramBadRequest as e:
            if is_not_modified(e):
                return
            logger.warning("Rich question edit rejected (%s); trying plain text", e)
        except Exception as e:  # noqa: BLE001 - best effort
            logger.warning("Rich question edit failed (%s); trying plain text", e)
    try:
        await sender.edit_text(message, final_text(pending, line), markdown=False)
    except Exception as e:  # noqa: BLE001 - edit_text never raises, but a stand-in sender might
        logger.warning("Question edit failed: %s", e)


async def expire_message(bot, message: Message) -> None:
    """Best-effort: take the buttons off a question nobody waits for any more (e.g. after a restart)."""
    try:
        if message.reply_markup is not None:
            await bot.edit_message_reply_markup(chat_id=message.chat.id, message_id=message.message_id, reply_markup=None)
            return
        blocks = message.rich_message.blocks if message.rich_message is not None else None
        if not blocks or not any(isinstance(b, RichBlockButtons) for b in blocks):
            return
        kept = [InputRichBlockParagraph(text=b.text) for b in blocks if isinstance(b, RichBlockParagraph)]
        kept.append(InputRichBlockParagraph(text=EXPIRED_LINE))
        await bot.edit_message_text(
            chat_id=message.chat.id, message_id=message.message_id, rich_message=InputRichMessage(blocks=kept)
        )
    except Exception as e:  # noqa: BLE001
        if not is_not_modified(e):
            logger.debug("Could not remove the buttons of an expired question: %s", e)


async def ask(sender, question: str, options: list[str], timeout: float) -> str | None:
    """Send the question and wait for a tap: the chosen option, or None after ``timeout`` seconds.

    ``question`` and ``options`` are expected cleaned (``clean_question`` / ``clean_options``).
    The tap handler edits the message on an answer; here a timeout edits it into
    "⌛ No answer" and a cancellation (/cancel, turn deadline) into "🛑 Cancelled" before
    re-raising, unless a tap already answered it. A failed send raises. The question
    always leaves the registry on the way out.
    """
    pending = register(sender.chat_id, question, options)
    message = None
    try:
        message = await _send(sender, pending)
        pending.message_id = message.message_id
        try:
            # shield: the timeout cancels only the wait, so a tap in the same instant still counts
            return await asyncio.wait_for(asyncio.shield(pending.future), timeout)
        except asyncio.TimeoutError:
            if pending.future.done() and not pending.future.cancelled():
                return pending.future.result()
            discard(pending.id)  # from here on a late tap reads "expired"
            await show_outcome(sender, pending, NO_ANSWER_LINE, message)
            return None
    except asyncio.CancelledError:
        answered = pending.future.done() and not pending.future.cancelled()
        discard(pending.id)
        if not answered:  # a tap that came first keeps its "✅ <option>" (its handler edits it)
            await show_outcome(sender, pending, CANCELLED_LINE, message)
        raise
    finally:
        discard(pending.id)


def reset() -> None:
    """Forget every pending and closed question (tests)."""
    _pending.clear()
    _closed.clear()
