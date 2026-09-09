"""Image generation, editing and composition — one tool over several engines.

``generate_image`` is the whole surface: no input images generates, one edits,
several compose. The engine is a purpose word (``auto`` / ``fast`` / ``best`` /
``story``) and the quality a ladder (``auto`` … ``max``); ``models.IMAGE_BACKENDS``
is the only place that knows what those map to and which knobs a provider takes,
so adding an engine never touches this file's dispatch.

``generate_story_images`` keeps the interleaved narrative case, whose *output
shape* (prose between pictures) differs rather than just its parameters.

Every produced image is delivered to the chat exactly once, as a document, with a
caption carrying the parameters it was made with. The result string then reports
the resolved settings and the saved paths, so the transcript records how a picture
was made and the model can embed it inline later if the user asks.

See docs/image-consolidation-design.md; docs/image-pipeline.md describes what this
replaced.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import PIL.Image
import PIL.ImageOps
from google import genai
from google.genai import types

from models import (
    DEFAULT_IMAGE_ENGINE,
    IMAGE_ASPECT_RATIOS,
    IMAGE_BACKENDS,
    IMAGE_MAX_VARIANTS,
    IMAGE_QUALITIES,
    IMAGE_QUALITY_FALLBACK,
    IMAGE_SIZES,
    ImageBackend,
    resolve_image_backend,
)

logger = logging.getLogger(__name__)

# Formats bot/rich.py can embed inline; anything else is delivered but not offered
# for embedding, so the model is never told to write a link that degrades to alt text.
INLINEABLE = ("png", "jpg", "jpeg", "webp")
OUTPUT_FORMATS = ("png", "jpeg", "webp")
# Formats the image APIs accept as input: anything else is converted before upload.
UPLOADABLE_FORMATS = {"PNG", "JPEG", "JPG", "WEBP"}
# Editing is judged by how much of the original survives, so ask for the faithful mode.
INPUT_FIDELITY = "high"
INPUT_FIDELITY_PARAM = "input_fidelity"
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_SIZE = "2K"
# OpenAI image constraints: edges multiples of 16, max edge 3840, total pixels in this window.
GPT_EDGE_STEP, GPT_MAX_EDGE = 16, 3840
GPT_MIN_PIXELS, GPT_MAX_PIXELS = 655_360, 8_294_400


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def openai_client():
    """Built on first use so the module imports without an API key (tests, tooling)."""
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def genai_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def extension_for(data: bytes, fallback: str = "png") -> str:
    """The real format of ``data`` from its magic bytes.

    Derived, never asserted: the old code wrote PNG bytes into ``.jpg`` names and
    hard-coded ``.png`` for edits, so a file's extension could lie about its content.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return fallback


def resolve_pixel_size(size: str, aspect_ratio: str) -> str:
    """``("2K", "16:9") -> "2048x1152"`` for the OpenAI backends.

    Edges are snapped to a multiple of 16 and the result is pulled back into the
    provider's pixel window, which is why 4K rarely keeps a 3840 long edge.
    """
    target = {"1K": 1024, "2K": 2048, "4K": GPT_MAX_EDGE}.get(size, 2048)
    try:
        w_r, h_r = (int(x) for x in str(aspect_ratio).split(":"))
        if w_r <= 0 or h_r <= 0:
            raise ValueError(aspect_ratio)
    except (ValueError, TypeError):
        w_r, h_r = 1, 1

    if w_r >= h_r:
        w, h = target, round(target * h_r / w_r / GPT_EDGE_STEP) * GPT_EDGE_STEP
    else:
        h, w = target, round(target * w_r / h_r / GPT_EDGE_STEP) * GPT_EDGE_STEP
    w, h = max(w, GPT_EDGE_STEP), max(h, GPT_EDGE_STEP)

    if w * h < GPT_MIN_PIXELS:
        scale = math.ceil(math.sqrt(GPT_MIN_PIXELS / (w * h)) * 100) / 100
        w = round(w * scale / GPT_EDGE_STEP) * GPT_EDGE_STEP
        h = round(h * scale / GPT_EDGE_STEP) * GPT_EDGE_STEP
    if w * h > GPT_MAX_PIXELS:
        scale = math.sqrt(GPT_MAX_PIXELS / (w * h))
        w = int(w * scale / GPT_EDGE_STEP) * GPT_EDGE_STEP
        h = int(h * scale / GPT_EDGE_STEP) * GPT_EDGE_STEP
    return f"{max(min(w, GPT_MAX_EDGE), GPT_EDGE_STEP)}x{max(min(h, GPT_MAX_EDGE), GPT_EDGE_STEP)}"


def _one_of(value, allowed: Iterable[str], default: str) -> str:
    """The canonical member of ``allowed`` matching ``value`` case-insensitively, else ``default``.

    Returns the canonical spelling, not the input: the size tiers are uppercase
    (``1K``) while engines and qualities are lowercase, so comparing case-folded
    and handing back the option itself keeps every caller free of that detail.
    """
    text = str(value or "").strip()
    for option in allowed:
        if text.lower() == option.lower():
            return option
    return default


def is_parameter_rejection(error: BaseException, parameter: str) -> bool:
    """Did the provider refuse this one parameter rather than the request as a whole?"""
    return parameter.lower() in str(error).lower()


def is_quality_rejection(error: BaseException) -> bool:
    """Did the provider refuse the quality value rather than the request as a whole?"""
    return is_parameter_rejection(error, "quality")


# Knobs a model turned out not to know, remembered for the life of the process so the
# discovery costs one round trip in total rather than one per image.
_UNSUPPORTED: Dict[str, set] = {}


def knob_allowed(model: str, parameter: str) -> bool:
    return parameter not in _UNSUPPORTED.get(model, set())


def remember_unsupported(model: str, parameter: str) -> None:
    _UNSUPPORTED.setdefault(model, set()).add(parameter)


class ImageTools:
    def __init__(self, user_id: str, sender, settings: Dict[str, Any] | None = None) -> None:
        """
        Args:
            user_id: chat id of the user as a string
            sender: bot.sender.ChatSender bound to the user's chat
            settings: the user's ``image`` settings category (engine/quality/size preferences)
        """
        self.user_id = user_id
        self.sender = sender
        self.settings = dict(settings or {})
        self.base_path = Path("./data") / str(user_id)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.images_path = self.base_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.tools_schema = self._build_schema()

    # ---- user preferences ------------------------------------------------
    @property
    def pref_engine(self) -> str:
        return _one_of(self.settings.get("engine"), IMAGE_BACKENDS, DEFAULT_IMAGE_ENGINE)

    @property
    def pref_quality(self) -> str:
        return _one_of(self.settings.get("quality"), IMAGE_QUALITIES, "auto")

    @property
    def pref_size(self) -> str:
        return _one_of(self.settings.get("size"), IMAGE_SIZES, DEFAULT_SIZE)

    @property
    def max_variants(self) -> int:
        try:
            return max(1, min(IMAGE_MAX_VARIANTS, int(self.settings.get("max_variants", IMAGE_MAX_VARIANTS))))
        except (TypeError, ValueError):
            return IMAGE_MAX_VARIANTS

    @property
    def show_metadata(self) -> bool:
        return self.settings.get("metadata", True) is not False

    # ---- schema ----------------------------------------------------------
    def _build_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "generate_image",
                "description": (
                    "Create or modify an image and deliver it to the user. Pass no `images` to generate from the "
                    "prompt, one to edit it, or two or more to compose them into a single picture.\n"
                    f"The user's saved preference is engine '{self.pref_engine}' at quality '{self.pref_quality}'. "
                    "Follow it unless the request implies otherwise: choose 'best' when the user asks for the best "
                    "result or the image is a final deliverable, and 'fast' only for quick drafts and iteration "
                    "(same price, lower quality). Use 'story' only for illustrated narratives.\n"
                    "The image is sent to the user automatically as a file. Only embed it in your answer with "
                    "Markdown (![caption](images/<name>)) if the user asked to see it inside the message."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to create. Describe subject, composition, lighting and style in one paragraph.",
                        },
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Existing images from the user's workspace, e.g. 'images/photo.jpg'. "
                                "Omit to generate, one to edit that image, several to compose them together."
                            ),
                        },
                        "engine": {
                            "type": "string",
                            "enum": sorted(IMAGE_BACKENDS),
                            "description": "auto = the user's preference (default), best = most capable, fast = quicker and lower quality, story = illustrated narrative.",
                        },
                        "quality": {
                            "type": "string",
                            "enum": list(IMAGE_QUALITIES),
                            "description": "How much effort to spend. 'auto' lets the provider decide from the prompt; raise it when the user asks for quality.",
                        },
                        "aspect_ratio": {"type": "string", "enum": list(IMAGE_ASPECT_RATIOS), "description": f"Shape of the image (default {DEFAULT_ASPECT_RATIO})."},
                        "size": {"type": "string", "enum": list(IMAGE_SIZES), "description": "Resolution tier; 4K is slower and is clamped to the provider's limits."},
                        "variants": {"type": "integer", "description": f"How many alternatives to produce, 1-{IMAGE_MAX_VARIANTS} (default 1)."},
                        "format": {"type": "string", "enum": list(OUTPUT_FORMATS), "description": "File format of the result (default png)."},
                        "caption": {"type": "string", "description": "Caption shown with the delivered file."},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "generate_story_images",
                "description": (
                    "Tell an illustrated story: alternating passages of text and pictures, delivered to the user as "
                    "they are produced. Use only when the user wants a narrative with images, not for a single picture."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "What the story is about."},
                        "style": {"type": "string", "description": "Visual style of the illustrations, e.g. '3d digital art'."},
                    },
                    "required": ["prompt"],
                },
            },
        ]

    # ---- dispatch --------------------------------------------------------
    _GENERATE_KEYS = ("prompt", "images", "engine", "quality", "aspect_ratio", "size", "variants", "format", "caption")
    _STORY_KEYS = ("prompt", "style")

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        args = tool_args if isinstance(tool_args, dict) else {}
        if tool_name == "generate_image":
            return await self._generate_image(**{k: v for k, v in args.items() if k in self._GENERATE_KEYS})
        if tool_name == "generate_story_images":
            return await self._generate_story(**{k: v for k, v in args.items() if k in self._STORY_KEYS})
        return f"Unknown tool: {tool_name}"

    # ---- workspace helpers ----------------------------------------------
    def _resolve_input(self, image_path: str) -> Path | None:
        """An existing image inside the user's workspace, or None."""
        from agents.paths import resolve_under

        resolved = resolve_under(self.base_path, image_path, fallback_dirs=("images", "downloads", "documents"))
        return resolved if resolved is not None and resolved.is_file() else None

    def _prepare_input(self, image_path: Path, temp_paths: List[Path]) -> Path:
        """The file to upload: the original when the provider takes it, a converted copy otherwise.

        An edit is judged on how much of the input survives, so a picture that is already an
        accepted format and size is sent untouched instead of being re-encoded, and one that
        does need converting keeps its transparency as PNG rather than being flattened.
        """
        limit = _env_int("MAX_IMAGE_RESOLUTION_EDIT", 4096)
        try:
            with PIL.Image.open(image_path) as img:
                needs_resize = limit > 0 and max(img.size) > limit
                needs_convert = (img.format or "").upper() not in UPLOADABLE_FORMATS
                if not (needs_resize or needs_convert):
                    return image_path
                img = PIL.ImageOps.exif_transpose(img)
                if needs_resize:
                    img.thumbnail((limit, limit), getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS)
                keeps_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
                suffix, fmt, options = ("png", "PNG", {}) if keeps_alpha else ("jpg", "JPEG", {"quality": 95})
                temp_path = self.images_path / f"edit_input_{uuid.uuid4().hex[:8]}.{suffix}"
                temp_paths.append(temp_path)
                img.convert("RGBA" if keeps_alpha else "RGB").save(temp_path, format=fmt, optimize=True, **options)
                return temp_path
        except Exception as e:  # noqa: BLE001 - an unreadable input is the provider's problem to report
            logger.warning("Could not prepare %s for editing, sending the original: %s", image_path, e)
            return image_path

    def _cleanup(self, temp_paths: List[Path]) -> None:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not remove temporary image %s: %s", path, e)

    async def _deliver(self, data: bytes, *, prefix: str, fmt: str, caption: str) -> Path:
        """Write the bytes with a truthful extension and send them once, as a document."""
        extension = extension_for(data, fallback=fmt)
        path = self.images_path / f"{prefix}_{uuid.uuid4().hex[:8]}.{extension}"
        await asyncio.to_thread(path.write_bytes, data)
        await self.sender.send_document(str(path), caption=caption)
        return path

    def _caption(self, base: str, backend: ImageBackend, quality: str, size: str, index: int, total: int) -> str:
        """Caption plus the parameters the picture was made with, so the chat records them."""
        text = f"{base} (variant {index}/{total})" if total > 1 else base
        if not self.show_metadata:
            return text
        bits = [backend.model, size]
        if backend.supports_quality:
            bits.insert(1, f"quality {quality}")
        return f"{text}\n{' · '.join(bits)}"

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except ValueError:
            return f"images/{path.name}"

    def _result(self, *, backend: ImageBackend, operation: str, saved: List[Path], quality: str, size: str, seconds: float, note: str = "") -> str:
        """What the model reads: the resolved settings, the paths, and how to embed one."""
        if not saved:
            return f"No image was produced ({operation} with {backend.model}). {note}".strip()
        lines = [
            f"{operation.capitalize()} {len(saved)} image(s) with {backend.model}"
            + (f" at quality {quality}" if backend.supports_quality else "")
            + f", {size}, in {seconds:.0f}s. Delivered to the user."
        ]
        if note:
            lines.append(note)
        lines.append("Saved: " + ", ".join(self._rel(p) for p in saved))
        inlineable = [p for p in saved if p.suffix.lstrip(".").lower() in INLINEABLE]
        if inlineable:
            lines.append(
                "The user already has the file(s). Only if they asked to see the image inside your answer, "
                f"embed it with ![caption]({self._rel(inlineable[0])})"
            )
        return "\n".join(lines)

    # ---- the tool --------------------------------------------------------
    async def _generate_image(
        self,
        prompt: str,
        images: List[str] | None = None,
        engine: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        size: str | None = None,
        variants: int = 1,
        format: str | None = None,  # noqa: A002 - the tool-facing name
        caption: str | None = None,
    ) -> str:
        if not str(prompt or "").strip():
            return "Error: 'prompt' is required."

        # The tool call wins, then the user's preference, then the default.
        engine = _one_of(engine, IMAGE_BACKENDS, "") or self.pref_engine
        quality = _one_of(quality, IMAGE_QUALITIES, "") or self.pref_quality
        aspect_ratio = _one_of(aspect_ratio, IMAGE_ASPECT_RATIOS, DEFAULT_ASPECT_RATIO)
        size = _one_of(size, IMAGE_SIZES, "") or self.pref_size
        fmt = _one_of(format, OUTPUT_FORMATS, "png")
        caption = str(caption or "Here is your image")
        try:
            variants = max(1, min(self.max_variants, int(variants or 1)))
        except (TypeError, ValueError):
            variants = 1

        backend = resolve_image_backend(engine, quality)
        requested = [p for p in (images or []) if str(p).strip()] if isinstance(images, (list, tuple)) else []

        resolved: List[Path] = []
        for candidate in requested:
            path = self._resolve_input(str(candidate))
            if path is None:
                return f"Error: '{candidate}' is not an image in your workspace. List the files first if you are unsure of the name."
            resolved.append(path)

        operation = "generated" if not resolved else ("edited" if len(resolved) == 1 else "composed")
        if resolved and backend.max_input_images == 0:
            return f"Error: engine '{backend.engine}' can only generate new images, not edit existing ones."
        if len(resolved) > backend.max_input_images:
            return (
                f"Error: engine '{backend.engine}' accepts at most {backend.max_input_images} input images, "
                f"{len(resolved)} were given. Use a different engine or fewer images."
            )

        started = time.monotonic()
        temp_paths: List[Path] = []
        try:
            if backend.provider == "openai":
                saved, note = await self._openai_images(
                    backend, prompt=prompt, inputs=resolved, quality=quality, aspect_ratio=aspect_ratio,
                    size=size, variants=variants, fmt=fmt, caption=caption, temp_paths=temp_paths,
                )
            else:
                saved, note = await self._gemini_images(
                    backend, prompt=prompt, inputs=resolved, quality=quality, aspect_ratio=aspect_ratio,
                    size=size, variants=variants, fmt=fmt, caption=caption, temp_paths=temp_paths,
                )
        except Exception as e:  # noqa: BLE001 - one failed image must not end the turn
            logger.exception("Image %s failed (engine=%s model=%s): %s", operation, backend.engine, backend.model, e)
            return f"Error: could not {operation[:-1] if operation.endswith('ed') else operation} the image with {backend.model}: {e}"
        finally:
            self._cleanup(temp_paths)

        return self._result(
            backend=backend, operation=operation, saved=saved, quality=quality,
            size=size, seconds=time.monotonic() - started, note=note,
        )

    # ---- providers -------------------------------------------------------
    async def _openai_images(self, backend, *, prompt, inputs, quality, aspect_ratio, size, variants, fmt, caption, temp_paths):
        """One call for every variant: images.generate, or images.edit when inputs are given."""
        pixel_size = resolve_pixel_size(size, aspect_ratio)
        prepared = [await asyncio.to_thread(self._prepare_input, p, temp_paths) for p in inputs]
        # Only edits have an input to stay faithful to, and only while this model accepts the knob.
        fidelity = INPUT_FIDELITY if prepared and knob_allowed(backend.model, INPUT_FIDELITY_PARAM) else None

        def _call(effective_quality: str, effective_fidelity: str | None):
            kwargs: Dict[str, Any] = {
                "model": backend.model, "prompt": prompt, "size": pixel_size,
                "n": variants, "quality": effective_quality, "output_format": fmt,
            }
            if not prepared:
                return openai_client().images.generate(**kwargs)
            if effective_fidelity:
                kwargs[INPUT_FIDELITY_PARAM] = effective_fidelity
            handles = [open(p, "rb") for p in prepared]
            try:
                kwargs["image"] = handles[0] if len(handles) == 1 else handles
                return openai_client().images.edit(**kwargs)
            finally:
                for handle in handles:
                    handle.close()

        # These model ids are newer than any published SDK, so a knob it does not know is a
        # possibility rather than a bug: drop the knob and try again instead of failing the turn.
        note = ""
        used_quality = quality
        result = None
        failure: BaseException | None = None
        for _ in range(3):
            try:
                result = await asyncio.to_thread(_call, used_quality, fidelity)
                failure = None
                break
            except Exception as e:  # noqa: BLE001
                failure = e
                if fidelity and is_parameter_rejection(e, INPUT_FIDELITY_PARAM):
                    logger.warning("%s does not accept %s, retrying without it: %s", backend.model, INPUT_FIDELITY_PARAM, e)
                    remember_unsupported(backend.model, INPUT_FIDELITY_PARAM)
                    fidelity = None
                    continue
                if used_quality in ("xhigh", "max") and is_quality_rejection(e):
                    logger.warning("Quality %r refused by %s, retrying at %s: %s", used_quality, backend.model, IMAGE_QUALITY_FALLBACK, e)
                    note = f"Quality '{quality}' is not supported by {backend.model}; used '{IMAGE_QUALITY_FALLBACK}' instead."
                    used_quality = IMAGE_QUALITY_FALLBACK
                    continue
                raise
        if failure is not None:
            raise failure

        saved: List[Path] = []
        items = list(getattr(result, "data", None) or [])
        for index, item in enumerate(items, start=1):
            payload = getattr(item, "b64_json", None)
            if not payload:
                continue
            saved.append(await self._deliver(
                base64.b64decode(payload), prefix=backend.prefix, fmt=fmt,
                caption=self._caption(caption, backend, used_quality, pixel_size, index, len(items)),
            ))
        return saved, note

    async def _gemini_images(self, backend, *, prompt, inputs, quality, aspect_ratio, size, variants, fmt, caption, temp_paths):
        """Gemini has no quality knob and returns parts, so variants are separate calls."""
        config = types.GenerateContentConfig(
            response_modalities=["Image"] if not backend.interleaved_text else ["Text", "Image"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=size),
        )
        opened: List[PIL.Image.Image] = []
        saved: List[Path] = []
        try:
            for path in inputs:
                prepared = await asyncio.to_thread(self._prepare_input, path, temp_paths)
                opened.append(await asyncio.to_thread(PIL.Image.open, prepared))
            contents = [*opened, prompt] if opened else [prompt]
            for index in range(1, variants + 1):
                response = await asyncio.to_thread(
                    genai_client().models.generate_content, model=backend.model, contents=contents, config=config
                )
                saved.extend(await self._collect_parts(
                    response, backend=backend, fmt=fmt,
                    caption=self._caption(caption, backend, quality, size, index, variants),
                ))
        finally:
            for image in opened:
                image.close()
        return saved, ""

    async def _collect_parts(self, response, *, backend, fmt, caption, send_text: bool = False) -> List[Path]:
        """Deliver the images (and optionally the prose) of one Gemini response."""
        candidates = list(getattr(response, "candidates", None) or [])
        if not candidates:
            return []
        saved: List[Path] = []
        for part in list(getattr(candidates[0].content, "parts", None) or []):
            fields = getattr(part, "model_fields_set", ())
            if "inline_data" in fields and getattr(part.inline_data, "data", None):
                saved.append(await self._deliver(part.inline_data.data, prefix=backend.prefix, fmt=fmt, caption=caption))
            elif send_text and "text" in fields and (part.text or "").strip():
                await self.sender.send_markdown(part.text)
        return saved

    # ---- story -----------------------------------------------------------
    async def _generate_story(self, prompt: str, style: str | None = None) -> str:
        if not str(prompt or "").strip():
            return "Error: 'prompt' is required."
        backend = resolve_image_backend("story")
        style = str(style or "3d digital art")
        started = time.monotonic()
        try:
            response = await asyncio.to_thread(
                genai_client().models.generate_content,
                model=backend.model,
                contents=f"Generate a story about {prompt} in a {style} style. For each scene, generate an image.",
                config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
            )
            saved = await self._collect_parts(
                response, backend=backend, fmt="png", caption="Story illustration", send_text=True
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Story generation failed (%s): %s", backend.model, e)
            return f"Error: could not generate the story with {backend.model}: {e}"
        return self._result(
            backend=backend, operation="told", saved=saved, quality="auto",
            size=DEFAULT_SIZE, seconds=time.monotonic() - started,
            note="The story text and its illustrations were sent to the user as they were produced.",
        )
