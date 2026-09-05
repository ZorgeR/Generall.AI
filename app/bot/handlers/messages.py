"""Inbound content handlers: text, voice, video, audio, photos/albums, documents.

Handlers validate quickly, enqueue a job on the user's queue and return.
If the user already has a job running they get a busy notice immediately;
the new message is processed as its own turn once the queue reaches it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot import media
from bot.agent_runner import THINKING, run_turn
from bot.queue import Job, JobContext, SubmitResult
from bot.runtime import queue
from bot.sender import ChatSender
from bot.ui import answer_md, delete_quietly
from image_utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

router = Router(name="messages")
control_router = Router(name="queue-control")

MEDIA_GROUP_TIMEOUT = 10.0
CANCEL_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🛑 Stop current task", callback_data="queue_cancel")]])


def _stats():
    from stats import stats_tracker

    return stats_tracker


def _thread_id(message: Message) -> int | None:
    return message.message_thread_id if message.is_topic_message else None


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _md_safe(text: str) -> str:
    """Make free text safe inside legacy-Markdown bold/italic spans."""
    return text.replace("_", "-").replace("*", "").replace("`", "'")


async def notify_busy(message: Message, result: SubmitResult) -> None:
    current = result.current
    if current is not None:
        label = _md_safe(current.label)
        if current.started_at is None:
            text = f"⏳ I'm about to start on your previous request (*{label}*).\nI'll handle this message right after it finishes."
        else:
            step = _md_safe(current.progress or "starting")
            text = (
                f"⏳ I'm still working on your previous request (*{label}*, "
                f"{_format_elapsed(current.elapsed)} elapsed, step: _{step}_).\n"
                "I'll handle this message right after it finishes."
            )
    else:
        text = "⏳ Your previous messages are still queued. I'll handle this one in turn."
    ahead = result.position - 1
    if ahead > 0:
        text += f"\n📬 {ahead} more message(s) are queued ahead of it."
    text += "\nSend /cancel to stop the current task and clear the queue."
    try:
        await message.answer(text, parse_mode="Markdown", reply_markup=CANCEL_KEYBOARD)
    except Exception:  # noqa: BLE001
        await message.answer(text.replace("*", "").replace("_", ""), reply_markup=CANCEL_KEYBOARD)


async def submit(message: Message, job: Job) -> None:
    result = await queue.submit(job)
    if result.was_busy:
        await notify_busy(message, result)


# ---------------------------------------------------------------------------
# /cancel
# ---------------------------------------------------------------------------
async def _cancel_for(user_id: str) -> str:
    cancelled, dropped = await queue.cancel(user_id, reason="user")
    if not cancelled and not dropped:
        return "Nothing is running right now."
    parts = []
    if cancelled:
        parts.append("stopped the current task")
    if dropped:
        parts.append(f"dropped {dropped} queued message(s)")
    return "🛑 " + " and ".join(parts).capitalize() + "."


@control_router.message(F.text, Command("cancel"))
async def cancel_command(message: Message, user_id: str) -> None:
    await message.answer(await _cancel_for(user_id))


@control_router.callback_query(F.data == "queue_cancel")
async def cancel_button(callback: CallbackQuery, user_id: str) -> None:
    text = await _cancel_for(user_id)
    await callback.answer(text, show_alert=False)
    if isinstance(callback.message, Message):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------
async def _run_text(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    await run_turn(
        bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=message.text or "",
        thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
    )


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    logger.info("Text message from %s", user_id)
    _stats().track_message_received(user_id, "text")
    await submit(message, Job(user_id=user_id, label="text message", run=partial(_run_text, bot, message, user_id, limit)))


# ---------------------------------------------------------------------------
# voice
# ---------------------------------------------------------------------------
async def _run_voice(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    media.ensure_temp_dirs()
    temp_ogg = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.oga")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"voice_{uuid.uuid4()}.mp3")
    status = await sender.send_text("🎙️ *Transcribing audio...*")
    try:
        ctx.set_progress("transcribing voice")
        await bot.download(message.voice.file_id, destination=temp_ogg)
        await media.convert_audio(temp_ogg, temp_mp3, "ogg")
        transcription = await media.transcribe_audio(temp_mp3)
        if not transcription:
            await sender.edit_text(status, "❌ Failed to transcribe audio")
            return
        shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
        await sender.edit_text(status, f"🎙️ *Transcription:*\n{shown}")
        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=transcription,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing voice message: %s", e)
        await sender.edit_text(status, "❌ Error processing voice message")
    finally:
        media.remove_files([temp_ogg, temp_mp3])


@router.message(F.voice)
async def on_voice(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "voice")
    await submit(message, Job(user_id=user_id, label="voice message", run=partial(_run_voice, bot, message, user_id, limit)))


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
async def _run_video(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    is_video_note = message.video_note is not None
    video = message.video or message.video_note or message.document
    if not video:
        await sender.send_text("❌ Could not process video message.", markdown=False)
        return
    caption = message.caption or ""
    media.ensure_temp_dirs()
    temp_video = os.path.join(media.TEMP_AUDIO, f"video_{uuid.uuid4()}.mp4")
    temp_mp3 = os.path.join(media.TEMP_AUDIO, f"video_audio_{uuid.uuid4()}.mp3")
    screenshots: list[str] = []
    status = await sender.send_text("🎬 *Processing video...*")
    try:
        ctx.set_progress("downloading video")
        await bot.download(video.file_id, destination=temp_video)
        saved_video = str(media.user_dir(user_id, "videos") / f"video_{uuid.uuid4()}.mp4")
        await asyncio.to_thread(shutil.copy, temp_video, saved_video)

        await sender.edit_text(status, "🎬 *Extracting audio and analyzing frames...*")
        ctx.set_progress("transcribing video")

        async def extract_audio() -> str | None:
            try:
                await media.convert_audio(temp_video, temp_mp3, "mp4")
            except Exception as e:  # noqa: BLE001
                logger.warning("No audio track or conversion failed: %s", e)
                return None
            return await media.transcribe_audio(temp_mp3)

        transcription, screenshots = await asyncio.gather(extract_audio(), media.extract_video_screenshots(temp_video))

        visual = ""
        saved_frames: list[str] = []
        if screenshots:
            images_dir = media.user_dir(user_id, "images")
            for sp in screenshots:
                dest = str(images_dir / f"video_frame_{uuid.uuid4()}.jpg")
                await asyncio.to_thread(shutil.copy, sp, dest)
                saved_frames.append(dest)
            await sender.edit_text(status, "🖼️ *Analyzing video frames...*")
            ctx.set_progress("describing frames")
            visual = await media.describe_video_screenshots(screenshots, transcription=(transcription or "") if is_video_note else "", caption=caption)

        if not (transcription or visual or saved_frames):
            await sender.edit_text(status, "❌ Could not extract any content from video")
            return

        display = "🎬 *Video analysis:*\n"
        if transcription:
            shown = transcription[:512] + "..." if len(transcription) > 1500 else transcription
            display += f"🎙️ *Audio:* {shown}\n"
        if visual:
            display += f"🖼️ *Visual:* {visual[:200]}..."
        if caption:
            display += f"\n📝 *Caption:* {caption}"
        if len(display) > 4000:
            display = display[:2048] + "..."
        await sender.edit_text(status, display)

        parts = [f"User sent a video file. Saved to: {saved_video}"]
        if caption:
            parts.append(f"User caption: {caption}")
        if transcription:
            parts.append(f"Audio transcription from video: {transcription}")
        if visual:
            parts.append(f"Visual description of video: {visual}")
        if saved_frames:
            lines = []
            for i, sp in enumerate(saved_frames):
                label = media.FRAME_LABELS[i] if i < len(media.FRAME_LABELS) else f"frame {i + 1}"
                lines.append(f"  - Frame at {label}: {sp}")
            parts.append("Video screenshots saved:\n" + "\n".join(lines))
        prompt = "\n\n".join(parts)

        await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit, speak=True,
        )
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing video message: %s", e)
        await sender.edit_text(status, "❌ Error processing video message")
    finally:
        media.remove_files([temp_video, temp_mp3, *screenshots])


@router.message(F.video | F.video_note)
async def on_video(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "video")
    await submit(message, Job(user_id=user_id, label="video message", run=partial(_run_video, bot, message, user_id, limit)))


# ---------------------------------------------------------------------------
# audio files
# ---------------------------------------------------------------------------
async def _run_audio(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    audio = message.audio
    if not audio:
        await sender.send_text("❌ Could not process audio file.", markdown=False)
        return
    caption = message.caption or ""
    file_name = os.path.basename(audio.file_name or f"audio_{uuid.uuid4()}.mp3")
    audio_path = str(media.user_dir(user_id, "audio") / file_name)
    status = await sender.send_text("🎵 *Saving audio file...*")
    try:
        ctx.set_progress("saving audio")
        await bot.download(audio.file_id, destination=audio_path)
        duration = f"{audio.duration // 60}m{audio.duration % 60}s" if audio.duration else "unknown"
        prompt = f"User sent an audio file.\nFile: {file_name}\nPath: {audio_path}\nDuration: {duration}"
        if audio.performer:
            prompt += f"\nPerformer: {audio.performer}"
        if audio.title:
            prompt += f"\nTitle: {audio.title}"
        if caption:
            prompt += f"\nUser message: {caption}"
        await sender.edit_text(status, "🎵 *Audio saved, processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🎵 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Error processing audio file: %s", e)
        await sender.edit_text(status, "❌ Error processing audio file")


@router.message(F.audio)
async def on_audio(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "audio")
    await submit(message, Job(user_id=user_id, label="audio file", run=partial(_run_audio, bot, message, user_id, limit)))


# ---------------------------------------------------------------------------
# photos and albums
# ---------------------------------------------------------------------------
async def _run_images(
    bot: Bot, message: Message, user_id: str, limit: int | None, refs: list[Any], caption: str | None,
    status: Message | None, ctx: JobContext,
) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    if caption is None:
        caption = "Describe what is in this image in user language."
        question = caption
    else:
        question = f"Describe what is in this image and answer to this question: {caption}"
    if status is not None:
        await sender.edit_text(status, "🖼️ *Analyzing images...*")
    else:
        status = await sender.send_text("🖼️ *Analyzing images...*")
    media.ensure_temp_dirs()
    temp_files: list[str] = []
    descriptions: list[dict] = []
    try:
        for i, ref in enumerate(refs, 1):
            try:
                ctx.set_progress(f"describing image {i}/{len(refs)}")
                ext = os.path.splitext(getattr(ref, "file_name", "") or "")[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    ext = ".jpg"
                downloaded = os.path.join(media.TEMP_PHOTOS, f"photo_{uuid.uuid4()}{ext}")
                temp_files.append(downloaded)
                await bot.download(ref.file_id, destination=downloaded)
                jpeg = os.path.join(media.TEMP_PHOTOS, f"photo_{uuid.uuid4()}.jpg")
                temp_files.append(jpeg)
                await asyncio.to_thread(media.prepare_downloaded_image_for_vision, downloaded, jpeg)
                permanent = str(media.user_dir(user_id, "images") / f"image_{uuid.uuid4()}.jpg")
                await asyncio.to_thread(shutil.copy, jpeg, permanent)

                suffix = f" for image {i}" if len(refs) > 1 else ""
                await sender.edit_text(status, f"🤖 *Getting Anthropic description{suffix}...*")
                anthropic_desc = await media.describe_image_anthropic(question, jpeg)
                _stats().track_describe_used(user_id, "image_anthropic")
                await sender.edit_text(status, f"🤖 *Getting OpenAI description{suffix}...*")
                openai_desc = await media.describe_image_openai(question, jpeg)
                _stats().track_describe_used(user_id, "image_openai")
                descriptions.append({"anthropic": anthropic_desc, "openai": openai_desc, "path": permanent})
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing photo %d: %s", i, e)
                descriptions.append({"anthropic": f"Error processing image {i}", "openai": f"Error processing image {i}", "path": "error_path"})

        details = "\n\n".join(
            f"Image {i + 1} (path: {d['path']}):\nAnthropic description: {d['anthropic']}\nOpenAI description: {d['openai']}"
            for i, d in enumerate(descriptions)
        )
        prompt = (
            f"{caption}\n\nUser attached {len(descriptions)} image(s) to this message. "
            f"Here are the details about each image from Anthropic and OpenAI:\n\n{details}\n\n"
            "The images are saved in your workspace; to show one inline in your answer, write ![caption](images/<file name>)."
        )
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error analyzing images (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while analyzing the images. Trace ID: {trace_id}", markdown=False)
    finally:
        media.remove_files(temp_files)


@dataclass
class _Album:
    message: Message
    user_id: str
    limit: int | None
    refs: list[Any] = field(default_factory=list)
    caption: str | None = None
    waiting: Message | None = None
    flush_task: asyncio.Task | None = None


_albums: dict[tuple[int, str], _Album] = {}


async def _flush_album(bot: Bot, key: tuple[int, str]) -> None:
    try:
        await asyncio.sleep(MEDIA_GROUP_TIMEOUT)
    except asyncio.CancelledError:
        return
    album = _albums.pop(key, None)
    if album is None or not album.refs:
        return
    _stats().track_media_group_processed(album.user_id, len(album.refs))
    job = Job(
        user_id=album.user_id,
        label=f"photo album ({len(album.refs)} images)",
        run=partial(_run_images, bot, album.message, album.user_id, album.limit, list(album.refs), album.caption, album.waiting),
    )
    await submit(album.message, job)


async def collect_album(message: Message, bot: Bot, user_id: str, limit: int | None, ref: Any, caption: str | None) -> None:
    key = (message.chat.id, str(message.media_group_id))
    album = _albums.get(key)
    if album is None:
        album = _Album(message=message, user_id=user_id, limit=limit)
        _albums[key] = album
        album.waiting = await answer_md(message, "🖼️ *Image media group received... Waiting for images...*")
    if caption and album.caption is None:
        album.caption = caption
    album.refs.append(ref)
    if album.waiting is not None:
        try:
            await album.waiting.edit_text(f"🖼️ *Image {len(album.refs)} received... Waiting for other images...*", parse_mode="Markdown")
        except Exception:  # noqa: BLE001
            pass
    if album.flush_task is not None:
        album.flush_task.cancel()
    album.flush_task = asyncio.create_task(_flush_album(bot, key))


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    _stats().track_message_received(user_id, "photo")
    ref = message.photo[-1]
    if message.media_group_id:
        await collect_album(message, bot, user_id, limit, ref, message.caption)
        return
    await submit(message, Job(
        user_id=user_id, label="photo",
        run=partial(_run_images, bot, message, user_id, limit, [ref], message.caption, None),
    ))


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
async def _run_document(bot: Bot, message: Message, user_id: str, limit: int | None, ctx: JobContext) -> None:
    sender = ChatSender(bot, message.chat.id, _thread_id(message), message.message_id)
    document = message.document
    file_name = os.path.basename((document.file_name or f"document_{uuid.uuid4()}").lower())
    ext = os.path.splitext(file_name)[1].lower()
    doc_type = ext.replace(".", "").upper()
    media.ensure_temp_dirs()
    temp_file = os.path.join(media.TEMP_DOCS, f"doc_{uuid.uuid4()}{ext}")
    status = await sender.send_text(f"📄 *Processing {doc_type} document...*")
    try:
        ctx.set_progress("analyzing document")
        await bot.download(document.file_id, destination=temp_file)
        saved = str(media.user_dir(user_id, "documents") / file_name)
        await asyncio.to_thread(shutil.copy, temp_file, saved)

        if message.caption is None:
            caption = "Analyze this document and describe its contents in detail."
            question = caption
        else:
            caption = message.caption
            question = f"Analyze this document and describe its contents in detail. When you are done, answer the following question: {caption}"

        await sender.edit_text(status, f"🤖 *Analyzing {doc_type} content...*")
        description = await media.describe_document(question, temp_file)
        _stats().track_describe_used(user_id, media.describe_type_for(ext))

        prompt = (
            f"{caption}\n\nUser attached a {doc_type} document to this message.\nFile: {file_name}\nSaved to: {saved}\n\n"
            f"Here is the analysis of document contents:\n\n{description}"
        )
        await sender.edit_text(status, "🤖 *Processing...*")
        result = await run_turn(
            bot=bot, user_id=user_id, chat_id=message.chat.id, prompt=prompt,
            thread_id=_thread_id(message), reply_to_message_id=message.message_id, ctx=ctx, limit=limit,
        )
        if result is not None:
            await sender.edit_text(status, "🤖 *Done!*")
    except asyncio.CancelledError:
        await sender.edit_text(status, "🛑 Stopped.", markdown=False)
        raise
    except Exception as e:  # noqa: BLE001
        trace_id = str(uuid.uuid4())
        logger.exception("Error analyzing document (trace %s): %s", trace_id, e)
        await sender.edit_text(status, f"❌ An error occurred while analyzing the document. Trace ID: {trace_id}", markdown=False)
    finally:
        media.remove_files([temp_file])


@router.message(F.document)
async def on_document(message: Message, bot: Bot, user_id: str, limit: int | None = None) -> None:
    file_name = (message.document.file_name or "").lower()
    ext = os.path.splitext(file_name)[1].lower()
    if ext in media.VIDEO_EXTENSIONS:
        _stats().track_message_received(user_id, "video")
        await submit(message, Job(user_id=user_id, label="video file", run=partial(_run_video, bot, message, user_id, limit)))
        return
    if ext in IMAGE_EXTENSIONS:
        _stats().track_message_received(user_id, "photo")
        if message.media_group_id:
            await collect_album(message, bot, user_id, limit, message.document, message.caption)
            return
        await submit(message, Job(
            user_id=user_id, label="image file",
            run=partial(_run_images, bot, message, user_id, limit, [message.document], message.caption, None),
        ))
        return
    if ext not in media.DOCUMENT_EXTENSIONS:
        supported = ", ".join(e.replace(".", "").upper() for e in media.DOCUMENT_EXTENSIONS)
        images = "/".join(e.replace(".", "").upper() for e in IMAGE_EXTENSIONS)
        await message.answer(f"❌ Only {supported} documents and {images} images are supported.")
        return
    _stats().track_message_received(user_id, "document")
    await submit(message, Job(user_id=user_id, label=f"{ext.replace('.', '').upper()} document", run=partial(_run_document, bot, message, user_id, limit)))
