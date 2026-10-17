"""Buttons on a finished turn: Continue, Retry, Listen, Reasoning, and follow-up suggestions.

The model may end an answer with a block of likely next requests::

    <follow_ups>
    - Make it shorter
    - Compare with 2024
    </follow_ups>

``parse_follow_ups`` removes it from the text the user sees and returns the
suggestions, which become buttons under the answer (one per row). Streaming
drafts hide the block as it is written (``strip_follow_ups_draft``), including a
half-typed opening tag, so it never flickers on screen.

Every successful turn is saved as a small record (``data/<uid>/turns/<id>.json``,
plus ``<id>.reasoning.txt`` when there is a reasoning transcript), newest
``MAX_TURNS`` per user. The buttons carry only the turn id (``turn_<action>_<id>``
and ``turn_fu_<id>_<index>``), so a press loads the record: Retry re-runs the
stored prompt, Continue asks the agent to finish a turn that hit the tool
budget, Listen reads the answer aloud, Reasoning sends the transcript file on
demand (it is no longer sent after every turn). A record that is gone (pruned,
or a forged id) turns the press into "expired".

Placement is decided by ``bot/agent_runner.py``: the action row goes at the end
of the rich turn summary when there is one, otherwise into an inline keyboard
under the last answer message, below the follow-up rows. The handlers live in
``bot/handlers/turn_actions_ui.py``. No import-time side effects.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.config import config
from bot.status import inline_buttons

logger = logging.getLogger(__name__)

# ---- follow-up suggestions ---------------------------------------------------
MAX_FOLLOW_UPS = 4
MAX_FOLLOW_UP_CHARS = 200  # a suggestion is sent back as the next prompt
FOLLOW_UP_LABEL = 40  # characters on the button

_OPEN_TAG = "<follow_ups>"
_OPEN_RE = re.compile(r"<follow_ups\s*>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</follow_ups\s*>", re.IGNORECASE)
_MARKER_RE = re.compile(r"^(?:[-*•]|\d{1,2}[.)])\s+")
_QUOTES = "\"'`“”„«»‘’"


def _split_block(text: str) -> tuple[str, str | None]:
    """(text without the LAST <follow_ups> block, the block's body or None when there is none)."""
    openings = list(_OPEN_RE.finditer(text))
    if not openings:
        return text, None
    opening = openings[-1]
    closing = _CLOSE_RE.search(text, opening.end())
    body_end, after_start = (closing.start(), closing.end()) if closing else (len(text), len(text))
    before, after = text[: opening.start()].rstrip(), text[after_start:].strip()
    clean = f"{before}\n\n{after}" if before and after else before or after
    return clean, text[opening.end():body_end]


def _suggestion(line: str) -> str:
    item = _MARKER_RE.sub("", line.strip()).strip().strip(_QUOTES).strip()
    if len(item) > MAX_FOLLOW_UP_CHARS:
        item = item[: MAX_FOLLOW_UP_CHARS - 1].rstrip() + "…"
    return item


def parse_follow_ups(text: str) -> tuple[str, list[str]]:
    """Split an answer into (visible text, follow-up suggestions).

    Uses the LAST ``<follow_ups>`` block (case-insensitive; a missing closing tag means
    "until the end"). Items are its non-empty lines with list markers and surrounding
    quotes removed: at most ``MAX_FOLLOW_UPS``, each at most ``MAX_FOLLOW_UP_CHARS``,
    duplicates dropped. No block -> ``(text, [])``.
    """
    clean, body = _split_block(text or "")
    if body is None:
        return text or "", []
    suggestions: list[str] = []
    seen: set[str] = set()
    for line in body.splitlines():
        item = _suggestion(line)
        if not item or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        suggestions.append(item)
        if len(suggestions) >= MAX_FOLLOW_UPS:
            break
    return clean, suggestions


def strip_follow_ups_draft(text: str) -> str:
    """The visible part of a streaming draft: no follow-ups block, no half-typed opening tag.

    The block arrives at the end of the answer a few characters at a time, so a draft
    ending in "<", "<fol" … "<follow_ups" is cut there too (the next chunk shows a
    plain "<" again if it was not the tag after all).
    """
    if not text:
        return text
    text, _ = _split_block(text)
    lower = text.lower()
    for n in range(min(len(_OPEN_TAG) - 1, len(text)), 0, -1):
        if lower.endswith(_OPEN_TAG[:n]):
            return text[:-n]
    return text


# ---- the turn record -------------------------------------------------------------
MAX_TURNS = 50  # records kept per user; older ones are pruned on save
TURNS_DIR = "turns"
DEFAULT_REASONING_CAPTION = "Reasoning history."
_TURN_ID_RE = re.compile(r"[0-9a-f]{12}")
_USER_ID_RE = re.compile(r"-?\d{1,20}")


def new_turn_id() -> str:
    return secrets.token_hex(6)


def valid_turn_id(turn_id: Any) -> bool:
    return isinstance(turn_id, str) and bool(_TURN_ID_RE.fullmatch(turn_id))


@dataclass
class TurnRecord:
    id: str
    prompt: str  # the text view of what run_turn received: Retry sends it again (or prompt_blocks)
    response: str  # the displayed answer (follow-ups removed): what Listen reads
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    speak: bool = False
    thread_id: int | None = None
    limit_reached: bool = False  # the tool budget ran out: offer Continue
    follow_ups: list[str] = field(default_factory=list)
    reasoning_caption: str = DEFAULT_REASONING_CAPTION
    has_reasoning: bool = False
    prompt_blocks: list | None = None  # the prompt as content blocks (media refs, agents.content) when it had media

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "TurnRecord | None":
        """A record from stored JSON, or None when it is not one (junk never raises)."""
        if not isinstance(data, dict) or not valid_turn_id(data.get("id")):
            return None
        thread_id = data.get("thread_id")
        follow_ups = data.get("follow_ups")
        prompt_blocks = data.get("prompt_blocks")
        return cls(
            id=data["id"],
            prompt=str(data.get("prompt") or ""),
            response=str(data.get("response") or ""),
            created=str(data.get("created") or ""),
            speak=data.get("speak") is True,
            thread_id=thread_id if isinstance(thread_id, int) and not isinstance(thread_id, bool) else None,
            limit_reached=data.get("limit_reached") is True,
            follow_ups=[str(s) for s in follow_ups if isinstance(s, str) and s] if isinstance(follow_ups, list) else [],
            reasoning_caption=str(data.get("reasoning_caption") or DEFAULT_REASONING_CAPTION),
            has_reasoning=data.get("has_reasoning") is True,
            prompt_blocks=[b for b in prompt_blocks if isinstance(b, dict)] if isinstance(prompt_blocks, list) and prompt_blocks else None,
        )


def _turns_dir(user_id: str) -> Path | None:
    user_id = str(user_id)
    if not _USER_ID_RE.fullmatch(user_id):
        return None
    return Path(config.data_dir) / user_id / TURNS_DIR


def _paths(user_id: str, turn_id: str) -> tuple[Path, Path] | None:
    """(record, reasoning file) — only for a valid id; a path is never built from anything else."""
    directory = _turns_dir(user_id)
    if directory is None or not valid_turn_id(turn_id):
        return None
    return directory / f"{turn_id}.json", directory / f"{turn_id}.reasoning.txt"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def prune(user_id: str, keep: int = MAX_TURNS, protect: str | None = None) -> int:
    """Delete all but the newest ``keep`` records (by mtime) and their reasoning files. Returns the count."""
    directory = _turns_dir(user_id)
    if directory is None or not directory.is_dir():
        return 0
    records: list[tuple[int, str, Path]] = []
    for path in directory.glob("*.json"):
        if valid_turn_id(path.stem):
            try:
                records.append((path.stat().st_mtime_ns, path.stem, path))
            except OSError:
                continue
    records.sort(reverse=True)
    removed = 0
    for _, turn_id, path in records[max(0, keep):]:
        if turn_id == protect:
            continue
        for stale in (path, directory / f"{turn_id}.reasoning.txt"):
            try:
                stale.unlink(missing_ok=True)
            except OSError as e:
                logger.debug("Could not prune %s: %s", stale, e)
        removed += 1
    return removed


def _save_sync(user_id: str, record: TurnRecord, reasoning: str | None) -> None:
    paths = _paths(user_id, record.id)
    if paths is None:
        raise ValueError(f"invalid user or turn id: {user_id!r} / {record.id!r}")
    record_path, reasoning_path = paths
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record.has_reasoning = bool(reasoning)
    # The reasoning file first: a record that says has_reasoning always has its file.
    if reasoning:
        _write_atomic(reasoning_path, reasoning)
    try:
        _write_atomic(record_path, json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    except BaseException:
        reasoning_path.unlink(missing_ok=True)
        raise
    try:
        prune(user_id, protect=record.id)
    except Exception as e:  # noqa: BLE001 - a failed prune must not lose the turn's buttons
        logger.warning("Could not prune old turns of %s: %s", user_id, e)


async def save(user_id: str, record: TurnRecord, reasoning: str | None = None) -> None:
    """Store the record (and its reasoning text) atomically, then prune old turns. Raises on failure."""
    await asyncio.to_thread(_save_sync, str(user_id), record, reasoning)


def load(user_id: str, turn_id: str) -> TurnRecord | None:
    """The stored record, or None (unknown, pruned, invalid id or unreadable). Blocking: use to_thread."""
    paths = _paths(user_id, turn_id)
    if paths is None:
        return None
    try:
        data = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.debug("Turn %s of %s not loaded: %s", turn_id, user_id, e)
        return None
    record = TurnRecord.from_dict(data)
    return record if record is not None and record.id == turn_id else None


def load_reasoning(user_id: str, turn_id: str) -> str | None:
    """The stored reasoning transcript, or None. Blocking: use to_thread."""
    paths = _paths(user_id, turn_id)
    if paths is None:
        return None
    try:
        return paths[1].read_text(encoding="utf-8") or None
    except OSError:
        return None


# ---- buttons -----------------------------------------------------------------------
CALLBACK_PREFIX = "turn_"
RETRY, LISTEN, REASONING, CONTINUE, FOLLOW_UP = "retry", "listen", "reason", "cont", "fu"
CONTINUE_LABEL = "▶️ Continue"
RETRY_LABEL = "🔁 Retry"
LISTEN_LABEL = "🔊 Listen"
REASONING_LABEL = "📄 Reasoning"

_CALLBACK_RE = re.compile(r"turn_(retry|listen|reason|cont)_([0-9a-f]{12})|turn_fu_([0-9a-f]{12})_(\d{1,2})")

# (label, callback_data, style or None): the shape StatusMessage.finish(actions=...) takes.
Action = tuple[str, str, "str | None"]


def callback_data(action: str, turn_id: str, index: int | None = None) -> str:
    """``turn_<action>_<id>`` or, for a follow-up, ``turn_fu_<id>_<index>`` (at most 24 bytes)."""
    if action == FOLLOW_UP:
        return f"{CALLBACK_PREFIX}{FOLLOW_UP}_{turn_id}_{int(index or 0)}"
    return f"{CALLBACK_PREFIX}{action}_{turn_id}"


def parse_callback(data: str | None) -> tuple[str, str, int | None] | None:
    """(action, turn_id, follow-up index or None), or None for anything that is not ours."""
    match = _CALLBACK_RE.fullmatch(data or "")
    if match is None:
        return None
    if match.group(1):
        return match.group(1), match.group(2), None
    return FOLLOW_UP, match.group(3), int(match.group(4))


def tts_available() -> bool:
    return bool(config.elevenlabs_api_key)


def record_actions(record: TurnRecord, *, tts: bool | None = None) -> list[Action]:
    """Continue (only after the tool budget ran out), Retry, Listen (TTS configured), Reasoning (if stored)."""
    actions: list[Action] = []
    if record.limit_reached:
        actions.append((CONTINUE_LABEL, callback_data(CONTINUE, record.id), "primary"))
    actions.append((RETRY_LABEL, callback_data(RETRY, record.id), None))
    if tts_available() if tts is None else tts:
        actions.append((LISTEN_LABEL, callback_data(LISTEN, record.id), None))
    if record.has_reasoning:
        actions.append((REASONING_LABEL, callback_data(REASONING, record.id), None))
    return actions


def follow_up_label(text: str) -> str:
    return text if len(text) <= FOLLOW_UP_LABEL else text[: FOLLOW_UP_LABEL - 1].rstrip() + "…"


def answer_keyboard(record: TurnRecord, actions: list[Action] | None = None) -> InlineKeyboardMarkup | None:
    """The keyboard under the last answer message: one row per follow-up, then ``actions`` (if any)."""
    rows = [
        [InlineKeyboardButton(text=follow_up_label(text), callback_data=callback_data(FOLLOW_UP, record.id, i))]
        for i, text in enumerate(record.follow_ups[:MAX_FOLLOW_UPS])
    ]
    if actions:
        rows.append(inline_buttons(actions))
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


async def attach_keyboard(bot, chat_id: int, message, markup: InlineKeyboardMarkup) -> bool:
    """Put ``markup`` under an already sent message. Best-effort: never raises."""
    try:
        await bot.edit_message_reply_markup(chat_id=chat_id, message_id=message.message_id, reply_markup=markup)
        return True
    except Exception as e:  # noqa: BLE001 - the answer is delivered; the buttons are a bonus
        logger.warning("Could not attach the turn buttons in chat %s: %s", chat_id, e)
        return False
