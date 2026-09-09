"""Run one agent turn for a user and deliver the answer.

This replaces the old ``get_answer`` / ``send_response_to_user`` /
``send_reasoning_file`` trio and the copy of it that lived in the reminder
job. Everything user-facing about a turn (status message, streaming draft,
voice reply, text reply, reasoning file, error and cancel notices) happens
here, so every entry point behaves the same.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from aiogram.types import Message

from bot.limits import usage_line
from bot.media import synthesize_speech, user_dir
from bot.queue import JobContext
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.status import StatusMessage, entity_safe, render_trace, trace_summary, usage_text  # noqa: F401 (re-exported)
from bot.streaming import create_streaming_callback
from models import ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

THINKING = "💭 *Thinking...*"


@dataclass
class TurnResult:
    response: str
    messages: list


def _agents():
    """Import the agent package lazily: it must be imported after the sandbox patcher ran."""
    import agents.main as agents

    return agents


ARGS_CHARS = 2000  # per tool call, in the reasoning file
RESULT_CHARS = 2000  # per tool result
THINKING_CHARS = 4000  # per thinking block


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} chars total)"


def _result_text(content: Any) -> str:
    """A tool result is a string or a list of content blocks."""
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def reasoning_text(messages: list) -> str:
    """The turn as a readable transcript: prompts, thinking, tool calls WITH their arguments, results.

    Every block of every message is rendered, not just the first one: with thinking on, an
    assistant message starts with a ``thinking`` block and its tool calls follow it, so reading
    only ``content[0]`` hid the calls and the parameters they were made with.
    """
    names: dict[str, str] = {}  # tool_use_id -> tool name, so results can name their call
    for msg in messages:
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names[str(block.get("id"))] = str(block.get("name", "unknown"))

    out: list[str] = []
    for msg in messages:
        try:
            role = str(msg.get("role", "")).capitalize() or "Message"
            content = msg.get("content", [])
            if isinstance(content, str):
                out.append(f"[{role}]\n{content}" if role else content)
                continue
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text" and str(block.get("text", "")).strip():
                    out.append(f"[{role}]\n{block['text']}")
                elif kind == "thinking" and str(block.get("thinking", "")).strip():
                    out.append(f"[{role} thinking]\n{_clip(block['thinking'], THINKING_CHARS)}")
                elif kind == "tool_use":
                    args = block.get("input")
                    try:
                        rendered = json.dumps(args, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        rendered = str(args)
                    out.append(f"[Tool call: {block.get('name', 'unknown')}]\n{_clip(rendered, ARGS_CHARS)}")
                elif kind == "tool_result":
                    name = names.get(str(block.get("tool_use_id")), "unknown")
                    status = " — error" if block.get("is_error") else ""
                    body = _clip(_result_text(block.get("content", "")), RESULT_CHARS)
                    out.append(f"[Tool result: {name}{status}]\n{body}")
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
    return "\n\n========\n\n".join(out) + "\n\n========\n\n" if out else ""


def record_usage(stats_tracker, user_id: str, trace) -> None:
    """Token accounting: one usage row per model used in this turn (never raises)."""
    try:
        from models import estimate_cost

        for model, u in (trace.usage_by_model or {}).items():
            stats_tracker.track_usage(
                user_id,
                model=model,
                api_calls=u["api_calls"],
                input_tokens=u["input_tokens"],
                output_tokens=u["output_tokens"],
                cache_read_tokens=u["cache_read_tokens"],
                cache_write_tokens=u["cache_write_tokens"],
                tool_calls=trace.total,
                duration_s=trace.elapsed,
                cost_usd=estimate_cost(model, u["input_tokens"], u["output_tokens"], u["cache_read_tokens"], u["cache_write_tokens"]),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record token usage for %s: %s", user_id, e)


async def send_reasoning_file(sender: ChatSender, messages: list, settings: UserSettings, caption: str = "Reasoning history.") -> None:
    if not settings.get("reasoning_context", "enabled") or not messages:
        return
    text = reasoning_text(messages)
    if not text:
        return
    try:
        await sender.send_document(text.encode("utf-8"), filename=f"reasoning_{sender.chat_id}_{uuid.uuid4().hex[:8]}.txt", caption=caption)
    except Exception as e:  # noqa: BLE001
        logger.error("Error sending reasoning file: %s", e)


async def run_turn(
    *,
    bot: Bot,
    user_id: str,
    chat_id: int,
    prompt: str,
    thread_id: int | None = None,
    reply_to_message_id: int | None = None,
    ctx: JobContext | None = None,
    limit: int | None = None,
    status: Message | None = None,
    header: str = THINKING,
    speak: bool = False,
    reasoning_caption: str = "Reasoning history.",
) -> TurnResult | None:
    """Run the agent on ``prompt`` and deliver the result. Returns None after reporting an error."""
    from stats import stats_tracker

    settings = UserSettings(user_id)
    settings.save()  # persist any defaults added since the file was written
    user_settings: dict[str, Any] = settings.as_dict()
    rich_enabled = bool(settings.get("rich_messages", "enabled"))
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id, rich=rich_enabled, media_root=user_dir(user_id))

    await sender.typing()
    if status is None:
        status = await sender.send_text(header)

    from agents.trace import ToolTrace

    trace = ToolTrace()
    status_msg = StatusMessage(sender, status, rich=rich_enabled, header=header)
    keep_summary = bool(settings.get("trace", "keep_summary"))

    async def update_status(step: str, details: str, iteration: Any, critique: Any) -> None:
        if step == "saving":
            iteration, critique = "final", "end"
        if ctx is not None:
            ctx.set_progress(f"{step}: {details}"[:120])
        await status_msg.update(step=step, details=details, iteration=iteration, critique=critique, trace=trace, quota=usage_line(user_id, limit))

    on_text_chunk = create_streaming_callback(bot, chat_id, thread_id, rich=rich_enabled)

    try:
        agents = _agents()
        agent = agents.ChainOfThoughtAgent(
            model_type="anthropic",
            model=ANTHROPIC_MODEL,
            user_id=user_id,
            sender=sender,
            user_settings=user_settings,
            message_thread_id=thread_id,
        )
        response, messages = await agent.generate_response(prompt, update_status, on_text_chunk=on_text_chunk, trace=trace)
        stats_tracker.track_message_sent(user_id)
        record_usage(stats_tracker, user_id, trace)

        if speak and response:
            await status_msg.update(step="audio", details="Generating audio", iteration="final", critique="end", trace=trace)
            from voice import VoiceManager

            voice_id = VoiceManager().get_user_voice(user_id)
            audio = await synthesize_speech(response, voice_id)
            if audio:
                try:
                    await sender.send_voice(audio)
                except Exception as e:  # noqa: BLE001
                    logger.error("Error sending voice reply: %s", e)

        if rich_enabled:
            # The answer is a new rich message; the status is shortened into the turn summary
            # (tool calls, thinking, tokens) above it, or deleted when the user turned that off.
            await sender.send_markdown(response)
            await status_msg.finish(trace, keep=keep_summary)
        else:
            await sender.send_markdown(response, edit=status)
        await send_reasoning_file(sender, messages, settings, caption=reasoning_caption)
        return TurnResult(response=response, messages=messages)
    except asyncio.CancelledError:
        reason = ctx.cancel_reason if ctx else None
        await status_msg.set_text("🛑 Stopped." if reason == "user" else "🛑 Cancelled.")
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Turn failed for user %s (trace %s): %s", user_id, trace_id, e)
        await status_msg.set_text(f"❌ An error occurred. Trace ID: {trace_id}")
        return None
