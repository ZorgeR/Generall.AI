"""One real Messages API transcript per chat (or forum topic), replayed as is.

This replaces the summaries + dialog-history + reasoning-context triad that used
to be rebuilt into a fake conversation every turn. The file holds the exact API
blocks of every turn (``text``, ``tool_use``, ``tool_result``, ``thinking`` with
its signature), so the next turn sees what tools returned and what the model
thought, and the request prefix stays byte-stable for prompt caching.

Append-only. A thinking block is bound to the conversation that produced it: the
API rejects (or, with the drop policy, drops) a block whose system prompt, tools
or earlier messages changed since. So every turn stores exactly what was sent,
the per-turn ``<context>`` block included, and the stored history is never
edited in place, with two deliberate exceptions, both of which strip every
thinking block (``strip_thinking``) because the old blocks no longer match:

* a compaction boundary (``prune``): only when the estimated size exceeds
  ``max_context_tokens``, tool results older than the last
  ``keep_tool_results_turns`` user turns are cleared to a short marker and, if
  still above the target, the oldest half is summarized by the caller-supplied
  ``summarize`` coroutine into one user message, until the transcript is below
  ``PRUNE_TARGET_RATIO`` of the budget (so the next boundary is many turns away);
* a prefix change (``prefix_fingerprint``): the model, the system prompt or the
  tool definitions differ from the ones the transcript was built with (a settings
  change or a deploy).

Tool results are capped at ``max_tool_result_chars`` when the tool returns
(``cap_text`` in the agent loop), before the model sees them.

Files: ``data/<uid>/transcripts/[topic_<thread>_]transcript.json`` written
atomically (tmp + rename). Per-user turns never overlap (per-user queue), so no
lock is needed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

TRANSCRIPT_VERSION = 1
CLEARED_MARKER = "[tool result cleared to save context; call the tool again if you need it]"
SUMMARY_TAG = "earlier_conversation_summary"
CONTEXT_OPEN = "<context>\n"  # the per-turn context block stored first in each user turn
MEMORY_BULLET = "• "  # one memory line (summary or semantic hit) inside a context block
CHARS_PER_TOKEN = 3.2  # conservative for the Sonnet 5.x tokenizer and non-Latin scripts
PRUNE_TARGET_RATIO = 0.7  # a compaction brings the transcript down to this share of the budget
THINKING_TYPES = ("thinking", "redacted_thinking")


@dataclass
class Transcript:
    user_id: str
    thread_id: int | None = None
    messages: list[dict] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    model: str = ""
    seeded_from: str | None = None
    fingerprint: str = ""  # prefix_fingerprint of the model/system/tools the stored blocks were made with

    def to_json(self) -> dict:
        return {
            "version": TRANSCRIPT_VERSION,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "created": self.created,
            "updated": self.updated,
            "model": self.model,
            "seeded_from": self.seeded_from,
            "fingerprint": self.fingerprint,
            "messages": self.messages,
        }

    @classmethod
    def from_json(cls, data: dict, user_id: str, thread_id: int | None) -> "Transcript":
        messages = data.get("messages")
        return cls(
            user_id=user_id,
            thread_id=thread_id,
            messages=messages if isinstance(messages, list) else [],
            created=data.get("created", ""),
            updated=data.get("updated", ""),
            model=data.get("model", ""),
            seeded_from=data.get("seeded_from"),
            fingerprint=data.get("fingerprint") or "",
        )


# ---- message helpers ---------------------------------------------------------
def is_tool_result_message(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def is_user_turn(message: dict) -> bool:
    """A real user message (the start of a turn), as opposed to a tool-result message."""
    return message.get("role") == "user" and not is_tool_result_message(message)


def is_context_block(block) -> bool:
    """The per-turn ``<context>`` block (time and memory) stored at the start of a user turn."""
    return isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).startswith(CONTEXT_OPEN)


def message_text(message: dict) -> str:
    """The message's own text: text blocks without the per-turn context block."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text" and not is_context_block(b)
        )
    return ""


def shown_memory_lines(messages: list[dict]) -> set[str]:
    """Memory lines already shown in this transcript's context blocks. A new turn repeats
    none of them: the earlier context blocks stay in the history, so the model still has
    them, and the transcript does not grow by the same summaries every turn."""
    shown: set[str] = set()
    for m in messages:
        content = m.get("content")
        if m.get("role") != "user" or not isinstance(content, list):
            continue
        for block in content:
            if is_context_block(block):
                shown.update(line for line in block["text"].splitlines() if line.startswith(MEMORY_BULLET))
    return shown


def strip_thinking(messages: list[dict]) -> int:
    """Remove every thinking / redacted_thinking block in place (for a history that was
    edited, where they would no longer match). An assistant message left empty is
    dropped. Returns the number of blocks removed."""
    removed = 0
    kept: list[dict] = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, list):
            rest = [b for b in content if not (isinstance(b, dict) and b.get("type") in THINKING_TYPES)]
            removed += len(content) - len(rest)
            if not rest:
                continue
            if len(rest) != len(content):
                m = {**m, "content": rest}
        kept.append(m)
    messages[:] = kept
    return removed


def prefix_fingerprint(model: str, system: str, tools: list[dict]) -> str:
    """Hash of what a thinking block is bound to besides the messages: the model, the
    system prompt and the tool definitions (as a name-sorted set)."""
    ordered = sorted(tools or [], key=lambda t: str(t.get("name", "")))
    payload = json.dumps({"model": model, "system": system, "tools": ordered}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_only_view(messages: list[dict]) -> list[dict]:
    """User/assistant text turns only (for models or paths that cannot take tool blocks)."""
    out: list[dict] = []
    for m in messages:
        if is_tool_result_message(m):
            continue
        text = message_text(m).strip()
        if not text:
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + text}
        else:
            out.append({"role": m["role"], "content": text})
    if out and out[0]["role"] != "user":
        out = out[1:]
    return out


def estimate_tokens(messages: list[dict]) -> int:
    return int(len(json.dumps(messages, ensure_ascii=False)) / CHARS_PER_TOKEN)


def sanitize_turn(messages: list[dict], final_text: str) -> list[dict]:
    """Make one turn's messages safe to store: no empty text blocks or empty messages, no
    tool_use block without its tool_result (budget exhausted mid-batch), a leading user
    message, and the final answer as the last assistant message."""
    cleaned: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "text" and not (b.get("text") or "").strip())]
            if not content:
                continue
            m = {**m, "content": content}
        elif isinstance(content, str):
            if not content.strip():
                continue
        else:
            continue
        cleaned.append(m)
    answered = {b.get("tool_use_id") for m in cleaned if isinstance(m.get("content"), list)
                for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"}
    out: list[dict] = []
    for m in cleaned:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            content = [b for b in m["content"] if not (isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in answered)]
            if not content:
                continue
            m = {**m, "content": content}
        out.append(m)
    while out and out[0].get("role") != "user":
        out.pop(0)
    final = (final_text or "").strip()
    last_text = message_text(out[-1]).strip() if out and out[-1].get("role") == "assistant" else None
    if not out or out[-1].get("role") != "assistant" or (final and last_text != final):
        out.append({"role": "assistant", "content": [{"type": "text", "text": final or "(no answer)"}]})
    return out


def strip_private_keys(messages: list[dict]) -> list[dict]:
    """Copy without the keys the API must not see (``_ephemeral`` markers and the like)."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


# ---- size control ------------------------------------------------------------
def cap_text(text: str, max_chars: int | None) -> str:
    """A tool result cut to ``max_chars`` (with a note), applied before the model sees it."""
    if not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated {len(text) - max_chars} characters]"


def clear_old_tool_results(messages: list[dict], keep_turns: int) -> int:
    """Replace tool results older than the last ``keep_turns`` user turns with a marker."""
    turn_starts = [i for i, m in enumerate(messages) if is_user_turn(m)]
    if len(turn_starts) <= keep_turns:
        return 0
    cutoff = turn_starts[-keep_turns] if keep_turns > 0 else len(messages)
    cleared = 0
    for m in messages[:cutoff]:
        if not is_tool_result_message(m):
            continue
        for block in m["content"]:
            if block.get("content") != CLEARED_MARKER:
                block["content"] = CLEARED_MARKER
                block.pop("is_error", None)
                cleared += 1
    return cleared


def split_for_summary(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Oldest part (to summarize) and the rest, cut at a user-turn boundary near the middle."""
    turn_starts = [i for i, m in enumerate(messages) if is_user_turn(m)]
    if len(turn_starts) < 3:
        return [], messages
    mid = turn_starts[len(turn_starts) // 2]
    if mid <= 0:
        return [], messages
    return messages[:mid], messages[mid:]


def summary_message(summary: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"<{SUMMARY_TAG}>\n{summary.strip()}\n</{SUMMARY_TAG}>"}],
    }


async def prune(
    messages: list[dict],
    *,
    max_context_tokens: int,
    keep_tool_results_turns: int,
    summarize: Callable[[list[dict]], Awaitable[str]] | None = None,
) -> dict:
    """Compaction boundary, in place. Below ``max_context_tokens`` nothing changes (the
    history stays append-only). Above it: clear old tool results, then summarize the
    oldest half until under ``PRUNE_TARGET_RATIO`` of the budget; if anything was edited,
    strip every thinking block. Returns counters for logging."""
    stats = {"cleared": 0, "summarized": 0, "thinking_stripped": 0}
    if estimate_tokens(messages) <= max_context_tokens:
        return stats
    target = int(max_context_tokens * PRUNE_TARGET_RATIO)
    stats["cleared"] = clear_old_tool_results(messages, keep_tool_results_turns)
    if estimate_tokens(messages) > target and summarize is not None:
        for _ in range(4):  # a few rounds at most; each halves the transcript
            old, rest = split_for_summary(messages)
            if not old:
                break
            try:
                summary = await summarize(old)
            except Exception as e:  # noqa: BLE001 - keep the transcript rather than lose it
                logger.error("Transcript summarization failed, keeping full history: %s", e)
                break
            messages[:] = [summary_message(summary)] + rest
            stats["summarized"] += len(old)
            if estimate_tokens(messages) <= target:
                break
    if stats["cleared"] or stats["summarized"]:
        stats["thinking_stripped"] = strip_thinking(messages)
    return stats


# ---- store -------------------------------------------------------------------
class TranscriptStore:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)

    def path(self, user_id: str, thread_id: int | None) -> Path:
        name = f"topic_{thread_id}_transcript.json" if thread_id else "transcript.json"
        return self.base_dir / str(user_id) / "transcripts" / name

    def exists(self, user_id: str, thread_id: int | None) -> bool:
        return self.path(user_id, thread_id).exists()

    def load(self, user_id: str, thread_id: int | None) -> Transcript:
        path = self.path(user_id, thread_id)
        if not path.exists():
            return Transcript(user_id=str(user_id), thread_id=thread_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            return Transcript.from_json(data, str(user_id), thread_id)
        except Exception as e:  # noqa: BLE001 - a corrupt file must not lock the user out
            logger.error("Unreadable transcript %s (%s); starting a fresh one", path, e)
            return Transcript(user_id=str(user_id), thread_id=thread_id)

    def save(self, transcript: Transcript) -> None:
        path = self.path(transcript.user_id, transcript.thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        transcript.created = transcript.created or now
        transcript.updated = now
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(transcript.to_json(), f, ensure_ascii=False, indent=1)
        tmp.replace(path)

    def seed_from_dialog_history(self, user_id: str, thread_id: int | None, dialog_history: list[dict]) -> Transcript:
        """First transcript for a user: start it from the legacy question/answer pairs."""
        transcript = Transcript(user_id=str(user_id), thread_id=thread_id, seeded_from="dialog_history")
        for m in dialog_history:
            role = m.get("role")
            text = message_text(m).strip() if isinstance(m, dict) else ""
            if role in ("user", "assistant") and text:
                transcript.messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        # the API wants the conversation to start with a user turn
        while transcript.messages and transcript.messages[0]["role"] != "user":
            transcript.messages.pop(0)
        return transcript


def clone(messages: list[dict]) -> list[dict]:
    return copy.deepcopy(messages)


transcript_store = TranscriptStore()
