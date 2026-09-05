"""One real Messages API transcript per chat (or forum topic), replayed as is.

This replaces the summaries + dialog-history + reasoning-context triad that used
to be rebuilt into a fake conversation every turn. The file holds the exact API
blocks of every turn (``text``, ``tool_use``, ``tool_result``, ``thinking`` with
its signature), so the next turn sees what tools returned and what the model
thought, and the request prefix stays byte-stable for prompt caching.

Size control, applied after every turn (see ``prune``):

1. every stored ``tool_result`` is capped at ``max_tool_result_chars`` (the model
   already saw the full text during the turn);
2. tool results older than the last ``keep_tool_results_turns`` user turns are
   cleared to a short marker (the tool_use/tool_result pairing stays intact);
3. when the estimated size still exceeds ``max_context_tokens`` the oldest half
   is summarized by the caller-supplied ``summarize`` coroutine into one user
   message that replaces it.

Files: ``data/<uid>/transcripts/[topic_<thread>_]transcript.json`` written
atomically (tmp + rename). Per-user turns never overlap (per-user queue), so no
lock is needed here.
"""
from __future__ import annotations

import copy
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
CHARS_PER_TOKEN = 3.2  # conservative for Sonnet 5's tokenizer and non-Latin scripts


@dataclass
class Transcript:
    user_id: str
    thread_id: int | None = None
    messages: list[dict] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    model: str = ""
    seeded_from: str | None = None

    def to_json(self) -> dict:
        return {
            "version": TRANSCRIPT_VERSION,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "created": self.created,
            "updated": self.updated,
            "model": self.model,
            "seeded_from": self.seeded_from,
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


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


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
def cap_tool_results(messages: list[dict], max_chars: int) -> int:
    """Truncate oversized tool results in place. Returns the number truncated."""
    truncated = 0
    for m in messages:
        if not is_tool_result_message(m):
            continue
        for block in m["content"]:
            content = block.get("content")
            if isinstance(content, str) and len(content) > max_chars:
                block["content"] = content[:max_chars] + f"\n…[truncated {len(content) - max_chars} characters]"
                truncated += 1
    return truncated


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
    max_tool_result_chars: int,
    summarize: Callable[[list[dict]], Awaitable[str]] | None = None,
) -> dict:
    """Apply the three size-control steps in place. Returns counters for logging."""
    stats = {"truncated": cap_tool_results(messages, max_tool_result_chars), "cleared": 0, "summarized": 0}
    if estimate_tokens(messages) <= max_context_tokens:
        return stats
    stats["cleared"] = clear_old_tool_results(messages, keep_tool_results_turns)
    if estimate_tokens(messages) <= max_context_tokens or summarize is None:
        return stats
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
        if estimate_tokens(messages) <= max_context_tokens:
            break
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
