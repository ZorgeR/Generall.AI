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

from PIL import Image, ImageOps

from bot.clients import anthropic_client, openai_client, whisper_client
from models import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    TTS_MODEL,
    VIDEO_FRAMES_MODEL,
    WHISPER_MODEL,
    anthropic_request_options,
    anthropic_text,
    estimate_cost,
    openai_reasoning_options,
)
from bot.config import config
from image_utils import flatten_to_rgb, is_jpeg_image, save_image_as_jpeg

logger = logging.getLogger(__name__)

TEMP_AUDIO = "temp_audio"
TEMP_PHOTOS = "temp_photos"
TEMP_DOCS = "temp_docs"

TEXT_EXTENSIONS = [
    ".txt", ".csv", ".py", ".sh", ".bat", ".md", ".ps1", ".js", ".css", ".html", ".php", ".sql",
    ".xml", ".yaml", ".yml", ".toml", ".ini", ".conf", ".log", ".jsonl",
]
OFFICE_EXTENSIONS = [".json", ".docx", ".xlsx", ".xls"]  # extracted to text before the agent sees them
DOCUMENT_EXTENSIONS = [".pdf"] + OFFICE_EXTENSIONS + TEXT_EXTENSIONS
VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"]
LARGE_TEXT_THRESHOLD = 100_000

# A PDF reaches the agent as a real document block when it is within these limits; larger
# ones are extracted to text (docs/design-v3.md, F7).
PDF_MAX_BYTES = 20 * 1024 * 1024
PDF_MAX_PAGES = 100

# Picture formats the Messages API takes as image blocks (PIL format name -> media type);
# anything else (BMP, HEIC, TIFF ...) is re-encoded as JPEG before it is sent.
VISION_MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}
_JPEG_QUALITIES = (90, 80, 70, 60, 50)


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
    """A full-resolution JPEG copy of an uploaded picture (JPEG copied as is; PNG, WebP, GIF, BMP,
    HEIC and anything else PIL opens are converted). This is the permanent workspace file, so it
    is never downscaled: the agent edits and re-sends it later. ``save_vision_copy`` makes the
    smaller copy the model looks at."""
    if is_jpeg_image(source_path):
        shutil.copy(source_path, target_path)
        return
    save_image_as_jpeg(source_path, target_path, quality=90)


def _max_vision_bytes() -> int:
    from agents.content import MAX_IMAGE_BYTES  # no agents import at module level in bot/

    return MAX_IMAGE_BYTES


def _fit_jpeg(img: Image.Image, max_px: int, max_bytes: int) -> tuple[bytes, tuple[int, int]]:
    """JPEG bytes of ``img`` at most ``max_px`` on the long side and under ``max_bytes``: the
    quality drops first, then the picture is halved until it fits. Returns (data, size)."""
    img = ImageOps.exif_transpose(img)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    if max_px > 0 and max(img.size) > max_px:
        img.thumbnail((max_px, max_px), resample)
    img = flatten_to_rgb(img)
    while True:
        data = b""
        for quality in _JPEG_QUALITIES:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= max_bytes:
                return data, img.size
        if max(img.size) <= 256:
            return data, img.size  # cannot shrink any further; the API will say so
        img.thumbnail((max(1, img.size[0] // 2), max(1, img.size[1] // 2)), resample)


def save_vision_copy(source_path: str, target_path: str, max_px: int | None = None) -> tuple[int, int]:
    """Write the copy the model looks at: a JPEG at most ``max_px`` (default
    ``MAX_IMAGE_RESOLUTION_VISION``) on the long side and under the API's size limit. Returns
    the copy's (width, height). Blocking: run in a worker thread."""
    max_px = max_px or config.max_image_resolution_vision
    with Image.open(source_path) as img:
        data, size = _fit_jpeg(img, max_px, _max_vision_bytes())
    with open(target_path, "wb") as f:
        f.write(data)
    return size


def image_for_vision(source_path: str, max_px: int | None = None) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """The bytes of an image as the model should see it, in memory (``view_image``).

    A JPEG/PNG/GIF/WebP file already within ``max_px`` and the size limit is passed through
    unchanged (no second lossy encoding of a screenshot); anything else is re-encoded with
    ``_fit_jpeg``. Returns (data, media_type, original (w, h), sent (w, h)). Blocking.
    """
    max_px = max_px or config.max_image_resolution_vision
    max_bytes = _max_vision_bytes()
    with Image.open(source_path) as img:
        original = ImageOps.exif_transpose(img).size
        media_type = VISION_MEDIA_TYPES.get((img.format or "").upper())
        size = os.path.getsize(source_path)
        # A camera JPEG kept byte for byte may carry an EXIF orientation: pass it through only
        # when it is already upright, otherwise re-encode so the model sees it the right way up.
        upright = _exif_orientation(img) in (None, 1)
        if media_type and upright and max(original) <= max_px and size <= max_bytes and getattr(img, "n_frames", 1) == 1:
            with open(source_path, "rb") as f:
                return f.read(), media_type, original, original
        data, sent = _fit_jpeg(img, max_px, max_bytes)
    return data, "image/jpeg", original, sent


def _exif_orientation(img) -> int | None:
    try:
        return img.getexif().get(0x0112)
    except Exception:  # noqa: BLE001 - a broken EXIF block is not a reason to fail
        return None


async def describe_image_anthropic(question: str, image_path: str) -> str:
    base64_image = await asyncio.to_thread(encode_image, image_path)
    message = await anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,  # adaptive thinking counts against this
        **anthropic_request_options(thinking=False),
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                {"type": "text", "text": question},
            ],
        }],
    )
    return anthropic_text(message)


def _record_openai_usage(user_id: str | None, model: str, response) -> None:
    """One usage_events row for an OpenAI call made outside an agent turn (never raises).

    The prompt tokens (cached share included) are priced as input, completion tokens
    (reasoning included) as output; see models.MODEL_PRICES.
    """
    usage = getattr(response, "usage", None)
    if not user_id or usage is None:
        return
    try:
        from stats import stats_tracker

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        stats_tracker.track_usage(
            user_id,
            model=model,
            api_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record token usage for %s: %s", user_id, e)


async def describe_image_openai(question: str, image_path: str, user_id: str | None = None) -> str:
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
    _record_openai_usage(user_id, OPENAI_MODEL, response)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
async def _ask_anthropic(system: str, text: str, max_tokens: int = 8192) -> str:
    message = await anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}]}],
        system=system,
        max_tokens=max_tokens,
        **anthropic_request_options(thinking=False),
    )
    return anthropic_text(message)


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
        max_tokens=16000,  # adaptive thinking counts against this; still under the non-streaming limit
        **anthropic_request_options(thinking=False),
    )
    return anthropic_text(message)


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
        content = await asyncio.to_thread(_read_json, path)
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


def _read_docx(path: str) -> str:
    import docx2txt

    return docx2txt.process(path)


async def describe_docx(question: str, path: str) -> str:
    try:
        content = await asyncio.to_thread(_read_docx, path)
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


def _read_json(path: str) -> str:
    """The JSON file's text, validated (``json.JSONDecodeError`` when it is not JSON)."""
    content = _read_text(path)
    json.loads(content)
    return content


def _read_pdf_text(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        pages.append(f"--- page {i} ---\n{text}" if text else f"--- page {i} --- (no extractable text)")
    return "\n\n".join(pages)


def pdf_page_count(path: str) -> int:
    """Number of pages (pypdf; raises when the file is not a readable PDF). Blocking."""
    from pypdf import PdfReader

    return len(PdfReader(path).pages)


def pdf_fits_as_block(path: str, pages: int) -> bool:
    """True when the PDF may go to the model as a document block (size and page limits)."""
    return os.path.getsize(path) <= PDF_MAX_BYTES and pages <= PDF_MAX_PAGES


def extract_document_text(path: str) -> str:
    """The plain text of a document the agent should read: txt/code as is, JSON validated,
    DOCX via docx2txt, XLSX/XLS as pandas tables per sheet, PDF via pypdf. The same readers
    the ``describe_*`` functions use. ``ValueError`` for an unsupported extension; the reader's
    own exception when the file cannot be read. Blocking: run in a worker thread."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf_text(path)
    if ext in TEXT_EXTENSIONS:
        return _read_text(path)
    if ext == ".json":
        return _read_json(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in (".xlsx", ".xls"):
        return _read_xlsx(path)
    raise ValueError(f"Unsupported document type {ext}")


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


async def describe_video_screenshots(screenshot_paths: list[str], transcription: str = "", caption: str = "", user_id: str | None = None) -> str:
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
            # no max_completion_tokens: it would cap hidden reasoning + answer together
            **openai_reasoning_options(VIDEO_FRAMES_MODEL),
        )
        _record_openai_usage(user_id, VIDEO_FRAMES_MODEL, response)
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
