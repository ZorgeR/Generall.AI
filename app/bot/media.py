"""Media pre-processing: transcription, image/document/video description.

Everything that blocks (pydub, ffmpeg, OpenAI sync client, pandas) is run in
a worker thread so one user's media never stalls the event loop.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from PIL import Image

from bot.clients import anthropic_client, openai_client, whisper_client
from models import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    TTS_MODEL,
    VIDEO_FRAMES_MODEL,
    WHISPER_MODEL,
    anthropic_request_options,
    openai_reasoning_options,
)
from bot.config import config
from image_utils import is_jpeg_image, save_image_as_jpeg

logger = logging.getLogger(__name__)

TEMP_AUDIO = "temp_audio"
TEMP_PHOTOS = "temp_photos"
TEMP_DOCS = "temp_docs"

TEXT_EXTENSIONS = [
    ".txt", ".csv", ".py", ".sh", ".bat", ".md", ".ps1", ".js", ".css", ".html", ".php", ".sql",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".conf", ".log", ".jsonl",
]
DOCUMENT_EXTENSIONS = [".pdf", ".json", ".docx", ".xlsx", ".xls"] + TEXT_EXTENSIONS
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"]
LARGE_TEXT_THRESHOLD = 100_000


def describe_type_for(extension: str) -> str:
    if extension == ".pdf":
        return "pdf"
    if extension == ".json":
        return "json"
    if extension == ".docx":
        return "docx"
    if extension in (".xlsx", ".xls"):
        return "xlsx"
    return "txt"


# ---------------------------------------------------------------------------
# ffmpeg / pydub
# ---------------------------------------------------------------------------
def configure_ffmpeg() -> None:
    """Point pydub at ffmpeg/ffprobe (system binaries in Docker or on PATH)."""
    from pydub import AudioSegment, utils

    in_docker = os.path.exists("/.dockerenv")
    if in_docker or shutil.which("ffmpeg"):
        ffmpeg_bin, prober = "ffmpeg", "ffprobe"
    elif os.name == "posix":
        ffmpeg_bin, prober = "./ffmpeg/ffmpeg/ffmpeg", "./ffmpeg/ffmpeg/ffprobe"
    else:
        base = os.getenv("FFMPEG_DIR", r"C:\ffmpeg\bin")
        ffmpeg_bin, prober = os.path.join(base, "ffmpeg.exe"), os.path.join(base, "ffprobe.exe")
    AudioSegment.converter = ffmpeg_bin
    utils.get_prober_name = lambda: prober  # type: ignore[assignment]


def ensure_temp_dirs() -> None:
    for d in (TEMP_AUDIO, TEMP_PHOTOS, TEMP_DOCS):
        os.makedirs(d, exist_ok=True)


def remove_files(paths) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not remove %s: %s", p, e)


async def convert_audio(source: str, target: str, source_format: str, target_format: str = "mp3") -> None:
    from pydub import AudioSegment

    def _convert() -> None:
        audio = AudioSegment.from_file(source, format=source_format)
        audio.export(target, format=target_format)

    await asyncio.to_thread(_convert)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------
def encode_image(image_path: str) -> str:
    image_bytes = b""
    max_res = config.max_image_resolution_vision
    try:
        with Image.open(image_path) as img:
            needs_resizing = max(img.size) > max_res
            is_jpeg = (img.format or "").upper() in ("JPEG", "JPG")
            if needs_resizing or not is_jpeg:
                if needs_resizing:
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                    img.thumbnail((max_res, max_res), resample)
                buffer = io.BytesIO()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(buffer, format="JPEG", quality=85)
                image_bytes = buffer.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning("Error compressing image: %s. Falling back to original size.", e)
    if not image_bytes:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def prepare_downloaded_image_for_vision(source_path: str, target_path: str) -> None:
    if is_jpeg_image(source_path):
        shutil.copy(source_path, target_path)
        return
    save_image_as_jpeg(source_path, target_path, quality=90)


async def describe_image_anthropic(question: str, image_path: str) -> str:
    base64_image = await asyncio.to_thread(encode_image, image_path)
    message = await anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        **anthropic_request_options(thinking=False),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return message.content[0].text


async def describe_image_openai(question: str, image_path: str) -> str:
    base64_image = await asyncio.to_thread(encode_image, image_path)
    response = await asyncio.to_thread(
        openai_client().chat.completions.create,
        model=OPENAI_MODEL,
        **openai_reasoning_options(OPENAI_MODEL),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
async def _ask_anthropic(system: str, text: str, max_tokens: int = 4096) -> str:
    message = await anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        system=system,
        max_tokens=max_tokens,
        **anthropic_request_options(thinking=False),
    )
    return message.content[0].text


async def describe_document_anthropic(question: str, file_path: str, document_type: str, mime_type: str) -> str:
    try:
        with open(file_path, "rb") as f:
            document_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return "Error: Document file not found"
    message = await anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": document_type, "source": {"type": "base64", "media_type": mime_type, "data": document_b64}},
                {"type": "text", "text": question},
            ],
        }],
        system="You are a very professional document analyze specialist. Analyze the given document in a detailed way, to answer user's question.",
        max_tokens=4096,
        **anthropic_request_options(thinking=False),
    )
    return message.content[0].text


async def process_large_text(content: str, question: str, content_type: str) -> str:
    chunk_size = 50_000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    summaries = []
    for i, chunk in enumerate(chunks):
        summaries.append(await _ask_anthropic(
            "You are a very professional document analyst. Summarize this part of the document concisely.",
            f"Here is part {i + 1} of {len(chunks)} of a {content_type}:\n\n{chunk}\n\nSummarize this part of the document concisely.",
            max_tokens=1000,
        ))
    combined = "\n\n".join(f"Part {i + 1} summary: {s}" for i, s in enumerate(summaries))
    return await _ask_anthropic(
        "You are a very professional document analyst. Based on the provided summaries, answer the user's question thoroughly.",
        f"I have analyzed a large {content_type} in parts. Here are the summaries of each part:\n\n{combined}\n\nBased on these summaries, please answer the following question: {question}",
    )


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


async def describe_txt(question: str, path: str) -> str:
    try:
        content = await asyncio.to_thread(_read_text, path)
    except FileNotFoundError:
        return "Error: TXT file not found"
    except Exception as e:  # noqa: BLE001
        return f"Error reading TXT file: {e}"
    if len(content) > LARGE_TEXT_THRESHOLD:
        return await process_large_text(content, question, "text document")
    return await _ask_anthropic(
        "You are a very professional document analyze specialist. Analyze the given text document in a detailed way, to answer user's question.",
        f"Here is the content of a text document:\n\n<document>\n{content}\n</document>\n\n<user_question>\n{question}\n</user_question>",
    )


async def describe_json(question: str, path: str) -> str:
    try:
        content = await asyncio.to_thread(_read_text, path)
        json.loads(content)
    except FileNotFoundError:
        return "Error: JSON file not found"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:  # noqa: BLE001
        return f"Error reading JSON file: {e}"
    if len(content) > LARGE_TEXT_THRESHOLD:
        return await process_large_text(content, question, "JSON document")
    return await _ask_anthropic(
        "You are a very professional data analyst specializing in JSON. Analyze the given JSON document in a detailed way, to answer user's question. Format your insights clearly.",
        f"Here is the content of a JSON document:\n\n<json_document>\n{content}\n</json_document>\n\n<user_question>\n{question}\n</user_question>",
    )


async def describe_docx(question: str, path: str) -> str:
    try:
        import docx2txt

        content = await asyncio.to_thread(docx2txt.process, path)
    except ImportError:
        return "Error: docx2txt library not installed."
    except FileNotFoundError:
        return "Error: DOCX file not found"
    except Exception as e:  # noqa: BLE001
        return f"Error reading DOCX file: {e}"
    if len(content) > LARGE_TEXT_THRESHOLD:
        return await process_large_text(content, question, "Word document")
    return await _ask_anthropic(
        "You are a very professional document analyst specializing in Word documents. Analyze the given document in a detailed way, to answer user's question.",
        f"Here is the content of a Word document:\n\n{content}\n\n{question}",
    )


def _read_xlsx(path: str) -> str:
    import pandas as pd

    content = ""
    excel_file = pd.ExcelFile(path)
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        content += f"\n\nSheet: {sheet}\n" + df.to_string(index=True) + "\n"
    return content


async def describe_xlsx(question: str, path: str) -> str:
    try:
        content = await asyncio.to_thread(_read_xlsx, path)
    except ImportError:
        return "Error: pandas library not installed."
    except FileNotFoundError:
        return "Error: XLSX file not found"
    except Exception as e:  # noqa: BLE001
        return f"Error reading XLSX file: {e}"
    if len(content) > LARGE_TEXT_THRESHOLD:
        return await process_large_text(content, question, "Excel spreadsheet")
    return await _ask_anthropic(
        "You are a very professional data analyst specializing in Excel spreadsheets. Analyze the given spreadsheet in a detailed way, to answer user's question. Format numeric insights clearly.",
        f"Here is the content of an Excel spreadsheet:\n\n{content}\n\n{question}",
    )


async def describe_document(question: str, file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return await describe_document_anthropic(question, file_path, "document", "application/pdf")
    if ext in TEXT_EXTENSIONS:
        return await describe_txt(question, file_path)
    if ext == ".json":
        return await describe_json(question, file_path)
    if ext == ".docx":
        return await describe_docx(question, file_path)
    if ext in (".xlsx", ".xls"):
        return await describe_xlsx(question, file_path)
    return f"Error: Unsupported file type {ext}"


# ---------------------------------------------------------------------------
# audio / video
# ---------------------------------------------------------------------------
async def transcribe_audio(audio_file_path: str) -> str | None:
    """Transcribe with Whisper; files over 24 MB are split into overlapping chunks."""
    max_size = 24 * 1024 * 1024
    file_size = os.path.getsize(audio_file_path)

    def _transcribe(path: str, prompt: str = "") -> str:
        with open(path, "rb") as audio:
            kwargs = {"model": WHISPER_MODEL, "file": audio, "response_format": "text"}
            if prompt:
                kwargs["prompt"] = prompt
            return whisper_client().audio.transcriptions.create(**kwargs)

    if file_size <= max_size:
        text = await asyncio.to_thread(_transcribe, audio_file_path)
        return text or None

    logger.info("Audio file %.1f MB exceeds limit, splitting into chunks", file_size / 1024 / 1024)
    from pydub import AudioSegment

    def _split() -> list[str]:
        audio = AudioSegment.from_file(audio_file_path)
        chunk_ms, overlap_ms = 10 * 60 * 1000, 10 * 1000
        step = chunk_ms - overlap_ms
        os.makedirs(TEMP_AUDIO, exist_ok=True)
        paths = []
        for start in range(0, len(audio), step):
            end = min(start + chunk_ms, len(audio))
            path = os.path.join(TEMP_AUDIO, f"chunk_{uuid.uuid4()}.mp3")
            audio[start:end].export(path, format="mp3", parameters=["-q:a", "5"])
            paths.append(path)
            if end >= len(audio):
                break
        return paths

    chunk_paths: list[str] = []
    try:
        chunk_paths = await asyncio.to_thread(_split)
        transcriptions: list[str] = []
        for i, path in enumerate(chunk_paths):
            prompt = " ".join(transcriptions[-1].split()[-15:]) if transcriptions else ""
            text = await asyncio.to_thread(_transcribe, path, prompt)
            if text:
                transcriptions.append(text)
            logger.info("Transcribed chunk %d/%d", i + 1, len(chunk_paths))
        return " ".join(transcriptions) if transcriptions else None
    except Exception as e:  # noqa: BLE001
        logger.error("Error splitting and transcribing audio: %s", e)
        return None
    finally:
        remove_files(chunk_paths)


async def extract_video_screenshots(video_path: str) -> list[str]:
    """Grab frames at 10%, 40%, 60%, 85% of the video into temp files."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
    except Exception as e:  # noqa: BLE001
        logger.warning("Error getting video duration: %s", e)
        return []
    os.makedirs(TEMP_AUDIO, exist_ok=True)
    paths: list[str] = []
    for pos in (0.10, 0.40, 0.60, 0.85):
        output = os.path.join(TEMP_AUDIO, f"screenshot_{uuid.uuid4()}.jpg")
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-ss", str(duration * pos), "-i", video_path, "-vframes", "1", "-q:v", "2", "-y", output],
                capture_output=True, timeout=30,
            )
            if os.path.exists(output) and os.path.getsize(output) > 0:
                paths.append(output)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error extracting screenshot at %d%%: %s", int(pos * 100), e)
    return paths


FRAME_LABELS = ["10%", "40%", "60%", "85%"]


async def describe_video_screenshots(screenshot_paths: list[str], transcription: str = "", caption: str = "") -> str:
    if not screenshot_paths:
        return ""
    prompt = (
        "Describe what's happening in this video based on these 4 screenshots taken at different moments "
        "(10%, 40%, 60%, 85% of the video). Give a brief visual summary for each screenshot and then a overall "
        "summary of the video content. If user provided a caption, use it to better understand what's happening "
        "in the video or answer to the user's question."
    )
    if transcription or caption:
        prompt += "\n\nAdditional context from the video:"
        if caption:
            prompt += f"\nUser's caption: {caption}"
        if transcription:
            prompt += f"\nAudio transcription: {transcription}"
        prompt += "\n\nUse this context to better understand what's shown in the screenshots and what the user is asking about."
    content: list[dict] = [{"type": "text", "text": prompt}]
    for i, path in enumerate(screenshot_paths):
        b64 = await asyncio.to_thread(encode_image, path)
        label = FRAME_LABELS[i] if i < len(FRAME_LABELS) else f"frame {i + 1}"
        content.append({"type": "text", "text": f"Frame at {label}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    try:
        response = await asyncio.to_thread(
            openai_client().chat.completions.create,
            model=VIDEO_FRAMES_MODEL,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=8192,  # reasoning tokens count against this cap
            **openai_reasoning_options(VIDEO_FRAMES_MODEL),
        )
        return response.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.error("Error describing video screenshots: %s", e)
        return ""


async def synthesize_speech(text: str, voice_id: str) -> bytes | None:
    """ElevenLabs TTS in a worker thread; returns MP3 bytes or None on failure."""
    if not config.elevenlabs_api_key:
        return None

    def _tts() -> bytes:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=config.elevenlabs_api_key)
        stream = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id=TTS_MODEL)
        return b"".join(stream)

    try:
        return await asyncio.to_thread(_tts)
    except Exception as e:  # noqa: BLE001
        logger.error("Error generating audio: %s", e)
        return None


def user_dir(user_id: str, *parts: str) -> Path:
    path = Path(config.data_dir) / str(user_id)
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path
