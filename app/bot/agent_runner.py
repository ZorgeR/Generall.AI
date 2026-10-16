"""Run one agent turn for a user and deliver the answer.

This replaces the old ``get_answer`` / ``send_response_to_user`` /
``send_reasoning_file`` trio and the copy of it that lived in the reminder
job. Everything user-facing about a turn (status message, streaming draft,
voice reply, text reply, reasoning file, error and cancel notices) happens
here, so every entry point behaves the same.

A turn that runs as a queue job (``ctx`` given) can be stopped where the user is
looking: the live status carries a Stop button (``queue_stop_<job token>``) and
the streaming drafts use the job id as ``draft_id`` with Telegram's own Stop
button. Both end in ``QueueManager.cancel_current`` for exactly this job; the
answer streamed so far is then kept as a real message, marked as partial. Once
the agent has returned, Stop no longer applies (``ctx.stoppable``), and once the
status has become the answer or the summary, a late /cancel leaves it as it is.

A successful turn ends with buttons (``bot/turn_actions.py``): the turn is saved
as a record, its actions (Continue after the tool budget ran out, Retry, Listen,
Reasoning) go at the end of the rich summary, or under the last answer message
when there is no summary, and the model's ``<follow_ups>`` suggestions become one
button per row under the answer. The reasoning file is sent only when its button
is pressed; if the record cannot be saved it is sent right away, as before, and
the turn gets no buttons.
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

from bot import turn_actions
from bot.limits import usage_line
from bot.media import synthesize_speech, user_dir
from bot.queue import STOP_CALLBACK_PREFIX, JobContext
from bot.sender import ChatSender
from bot.settings import UserSettings
from bot.status import StatusMessage, entity_safe, render_trace, stop_keyboard, trace_summary, usage_text  # noqa: F401 (re-exported)
from bot.streaming import create_streaming_callback
from models import ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

THINKING = "💭 *Thinking...*"
STOPPED_NOTE = "⏹ Stopped — partial answer, not saved to memory."


@dataclass
class TurnResult:
    response: str  # the displayed answer: the follow-ups block is removed
    messages: list
    turn_id: str | None = None  # the saved turn record (bot.turn_actions), None when saving failed


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
                cost_usd=trace.model_cost(model),
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


def _close_drafts(on_text_chunk) -> None:
    """No draft may reappear after the answer (or the stop notice) went out."""
    close = getattr(on_text_chunk, "close", None)
    if close is not None:
        close()


async def save_turn(user_id: str, record: turn_actions.TurnRecord, messages: list, settings: UserSettings) -> turn_actions.TurnRecord | None:
    """Store the turn for its buttons, with the reasoning transcript when that setting is on.

    Returns None (after logging) when it could not be saved: the caller then sends the
    reasoning file right away and shows no buttons, since nothing could answer them.
    """
    reasoning = reasoning_text(messages) if settings.get("reasoning_context", "enabled") and messages else ""
    try:
        await turn_actions.save(user_id, record, reasoning or None)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not save turn %s for %s (no turn buttons): %s", record.id, user_id, e)
        return None
    return record


async def send_partial_answer(sender: ChatSender, on_text_chunk) -> None:
    """After a user stop: keep the answer streamed so far as a real message (the draft vanishes).

    Best-effort and never raises an ``Exception``: the caller is re-raising the
    ``CancelledError``, and a failed send must not replace it.
    """
    partial_text = getattr(on_text_chunk, "partial_text", None)
    if partial_text is None:
        return
    try:
        _close_drafts(on_text_chunk)
        partial = (partial_text() or "").strip()
        if partial:
            await sender.send_markdown(f"{partial}\n\n{STOPPED_NOTE}")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not send the partial answer to %s: %s", sender.chat_id, e)


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
    """Run the agent on ``prompt`` and deliver the result. Returns None after reporting an error.

    With ``ctx`` the status gets a Stop button for ``ctx.job`` and the drafts use
    ``ctx.job.id``; without it (no queue job) there is no Stop button. A successful
    turn is saved (``bot.turn_actions``) and gets its action and follow-up buttons.
    """
    from stats import stats_tracker

    settings = UserSettings(user_id)
    settings.save()  # persist any defaults added since the file was written
    user_settings: dict[str, Any] = settings.as_dict()
    rich_enabled = bool(settings.get("rich_messages", "enabled"))
    sender = ChatSender(bot, chat_id, thread_id, reply_to_message_id, rich=rich_enabled, media_root=user_dir(user_id))

    job = ctx.job if ctx is not None else None
    # The Stop button names this job, so a late press can never stop the next queued turn.
    stop_callback = f"{STOP_CALLBACK_PREFIX}{job.token}" if job is not None else None

    await sender.typing()
    if status is None:
        status = await sender.send_text(header, reply_markup=stop_keyboard(stop_callback) if stop_callback else None)

    from agents.trace import ToolTrace

    trace = ToolTrace()
    status_msg = StatusMessage(sender, status, rich=rich_enabled, header=header, stop_callback=stop_callback)
    keep_summary = bool(settings.get("trace", "keep_summary"))

    async def update_status(step: str, details: str, iteration: Any, critique: Any) -> None:
        if step == "saving":
            iteration, critique = "final", "end"
        if ctx is not None:
            ctx.set_progress(f"{step}: {details}"[:120])
        await status_msg.update(step=step, details=details, iteration=iteration, critique=critique, trace=trace, quota=usage_line(user_id, limit))

    # Drafts carry the job id, so Telegram's Stop button on a draft names the job to cancel.
    on_text_chunk = create_streaming_callback(
        bot, chat_id, thread_id, rich=rich_enabled,
        draft_id=job.id if job is not None else None, can_stop=job is not None,
    )

    agent_done = False
    # Set once the status message is the answer (legacy) or the turn summary (rich): a cancel
    # or error after that (e.g. /cancel while the buttons go on) must not overwrite it.
    status_final = False
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
        _close_drafts(on_text_chunk)
        # The answer is complete (and already in memory): Stop no longer applies, so a late
        # press cannot cut the delivery short or resend the streamed part as "not saved".
        agent_done = True
        status_msg.stop_callback = None
        if ctx is not None:
            ctx.stoppable = False
        stats_tracker.track_message_sent(user_id)
        record_usage(stats_tracker, user_id, trace)

        # A trailing <follow_ups> block becomes buttons; it is never part of the visible answer.
        display, suggestions = turn_actions.parse_follow_ups(response or "")
        if not (user_settings.get("follow_ups") or {}).get("enabled", True):
            suggestions = []
        limit_reached = getattr(getattr(agent, "agent", None), "limit_reached", False) is True
        record = await save_turn(user_id, turn_actions.TurnRecord(
            id=turn_actions.new_turn_id(), prompt=prompt, response=display, speak=speak, thread_id=thread_id,
            limit_reached=limit_reached, follow_ups=suggestions, reasoning_caption=reasoning_caption,
        ), messages, settings)

        if speak and display:
            await status_msg.update(step="audio", details="Generating audio", iteration="final", critique="end", trace=trace)
            from voice import VoiceManager

            voice_id = VoiceManager().get_user_voice(user_id)
            audio = await synthesize_speech(display, voice_id)
            if audio:
                try:
                    await sender.send_voice(audio)
                except Exception as e:  # noqa: BLE001
                    logger.error("Error sending voice reply: %s", e)

        actions = turn_actions.record_actions(record) if record is not None else []
        placed = False
        if rich_enabled:
            # The answer is a new rich message; the status is shortened into the turn summary
            # (tool calls, thinking, tokens, and the action buttons) above it, or deleted when
            # the user turned that off.
            sent = await sender.send_markdown(display)
            placed = await status_msg.finish(trace, keep=keep_summary, actions=actions)
        else:
            sent = await sender.send_markdown(display, edit=status)
        status_final = True
        if record is None:
            await send_reasoning_file(sender, messages, settings, caption=reasoning_caption)
        else:
            # Follow-ups (and the actions, unless the summary carries them) under the last answer message.
            markup = turn_actions.answer_keyboard(record, None if placed else actions)
            if markup is not None and sent:
                await turn_actions.attach_keyboard(bot, chat_id, sent[-1], markup)
        return TurnResult(response=display, messages=messages, turn_id=record.id if record is not None else None)
    except asyncio.CancelledError:
        reason = ctx.cancel_reason if ctx else None
        if reason == "user" and not agent_done:
            await send_partial_answer(sender, on_text_chunk)
        else:
            _close_drafts(on_text_chunk)
        if not status_final:
            await status_msg.set_text("🛑 Stopped." if reason == "user" else "🛑 Cancelled.")
        raise
    except Exception as e:  # noqa: BLE001
        _close_drafts(on_text_chunk)
        trace_id = str(uuid.uuid4())
        logger.exception("Turn failed for user %s (trace %s): %s", user_id, trace_id, e)
        if not status_final:
            await status_msg.set_text(f"❌ An error occurred. Trace ID: {trace_id}")
        return None
