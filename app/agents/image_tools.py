"""Image generation, editing and composition — one tool over several engines.

``generate_image`` is the whole surface: no input images generates, one edits,
several compose. The engine is a purpose word (``auto`` / ``fast`` / ``best`` /
``story``) and the quality a ladder (``auto`` … ``max``); ``models.IMAGE_BACKENDS``
is the only place that knows what those map to and which knobs a provider takes,
so adding an engine never touches this file's dispatch.

``generate_story_images`` keeps the interleaved narrative case, whose *output
shape* (prose between pictures) differs rather than just its parameters, and
``generate_animation`` / ``edit_animation`` the animated GIF case (GPT Image 2.6): a
GIF has frames, timing and its own size budget, and it is delivered as a Telegram
animation that plays in the chat. ``edit_animation`` changes how a GIF plays locally
(agents/animation.py) and only calls the model to redraw one frame.

Every produced image is delivered to the chat exactly once, as a document (a GIF as an
animation), with a caption carrying the parameters it was made with. The result string
then reports the resolved settings and the saved paths, so the transcript records how a
picture was made and the model can embed it inline later if the user asks.

Every OpenAI request goes through ``_openai_request``: these model ids and knobs are
newer than some deployed SDKs and accounts, so an optional knob the provider refuses is
dropped (and remembered), an unverified quality tier falls back once and a size the
model refuses is replaced once by a standard one, instead of failing the turn.

See docs/image-consolidation-design.md and docs/image-2.6-animations.md;
docs/image-pipeline.md describes what the consolidation replaced.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageOps
from google import genai
from google.genai import types

from agents import animation as anim
from models import (
    DEFAULT_IMAGE_ENGINE,
    GIF_FRAME_MS_DEFAULT,
    GIF_FRAMES_DEFAULT,
    GIF_FRAMES_MAX,
    GIF_FRAMES_MIN,
    GIF_SHORT_EDGE,
    IMAGE_ASPECT_RATIOS,
    IMAGE_BACKENDS,
    IMAGE_COMPLEXITIES,
    IMAGE_MAX_VARIANTS,
    IMAGE_QUALITIES,
    IMAGE_QUALITY_FALLBACK,
    IMAGE_SIZES,
    VIDEO_TOOLS_ENABLED,
    ImageBackend,
    estimate_image_cost,
    resolve_animation_backend,
    resolve_image_backend,
)

logger = logging.getLogger(__name__)

# Formats bot/rich.py can embed inline; anything else is delivered but not offered
# for embedding, so the model is never told to write a link that degrades to alt text.
INLINEABLE = ("png", "jpg", "jpeg", "webp")
OUTPUT_FORMATS = ("png", "jpeg", "webp", "svg")
# Formats the image APIs accept as input: anything else is converted before upload.
UPLOADABLE_FORMATS = {"PNG", "JPEG", "JPG", "WEBP"}
# Editing is judged by how much of the original survives, so ask for the faithful mode
# (models before gpt-image-2 only: models.IMAGE_MODEL_FEATURES).
INPUT_FIDELITY = "high"
INPUT_FIDELITY_PARAM = "input_fidelity"
SAFETY_IDENTIFIER_PARAM = "safety_identifier"
COMPLEXITY_PARAM = "complexity"
MARKUP_PARAM = "markup"
# Knobs whose loss changes the result, so the agent is told when the provider refused one.
REPORTED_KNOBS = (COMPLEXITY_PARAM, MARKUP_PARAM)
# Words a provider uses for "I do not know that parameter" (as opposed to "wrong value"):
# only then is a refused knob remembered for the rest of the process.
UNSUPPORTED_WORDS = ("unknown", "unsupported", "not supported", "unrecognized", "unexpected keyword", "not allowed", "extra inputs")
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_SIZE = "2K"
# OpenAI image constraints: edges multiples of 16, max edge 3840, total pixels in this window.
GPT_EDGE_STEP, GPT_MAX_EDGE = 16, 3840
GPT_MIN_PIXELS, GPT_MAX_PIXELS = 655_360, 8_294_400
# The sizes every GPT image model takes; a refused custom size falls back to the closest.
STANDARD_SIZES = ("1024x1024", "1536x1024", "1024x1536")
# GIF frames keep a 512 short edge; the long edge follows the ratio on a 64 grid (16:9 -> 896x512).
GIF_EDGE_STEP = 64
GIF_SIZE_FALLBACK = "auto"
# Markup overlays: the outline drawn around a region, and what counts as "drawn on" when a
# marked copy of the photo is compared with the original (per channel, 0-255; JPEG noise
# stays below it).
MARKUP_COLOR = (255, 0, 0, 255)
MARKUP_DIFF_THRESHOLD = 60


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
    head = data[:1024].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        return "svg"
    return fallback


def gif_pixel_size(aspect_ratio: str) -> str:
    """``"16:9" -> "896x512"``: GIF frames share one image budget, so they stay small."""
    try:
        w_r, h_r = (int(x) for x in str(aspect_ratio).split(":"))
        if w_r <= 0 or h_r <= 0:
            raise ValueError(aspect_ratio)
    except (ValueError, TypeError):
        w_r, h_r = 1, 1
    long_edge = max(GIF_SHORT_EDGE, round(GIF_SHORT_EDGE * max(w_r, h_r) / min(w_r, h_r) / GIF_EDGE_STEP) * GIF_EDGE_STEP)
    return f"{long_edge}x{GIF_SHORT_EDGE}" if w_r >= h_r else f"{GIF_SHORT_EDGE}x{long_edge}"


def standard_size_for(pixel_size: str) -> str:
    """The standard size closest in shape to ``pixel_size`` (what every GPT image model takes)."""
    try:
        w, h = (int(x) for x in pixel_size.split("x"))
        ratio = w / h
    except (ValueError, ZeroDivisionError, AttributeError):
        return STANDARD_SIZES[0]
    if ratio > 1.2:
        return "1536x1024"
    if ratio < 1 / 1.2:
        return "1024x1536"
    return "1024x1024"


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


def says_unsupported(error: BaseException) -> bool:
    """Is this "the provider does not know the parameter" rather than "a bad value"?"""
    text = str(error).lower()
    return isinstance(error, TypeError) or any(word in text for word in UNSUPPORTED_WORDS)


def _short(error: BaseException, limit: int = 160) -> str:
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sdk_hint(error: BaseException) -> str:
    """What to do when the installed openai package does not know a parameter we must send."""
    if isinstance(error, TypeError) and "unexpected keyword" in str(error):
        return " (the installed openai package is too old for this option: the bot needs openai>=3.46.0, rebuild its image)"
    return ""


def _has_transparency(img: PIL.Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        return img.getchannel("A").getextrema()[0] < 255
    return img.mode == "P" and "transparency" in img.info


REGION_SHAPES = ("box", "circle")


def parse_regions(regions) -> List[tuple[float, float, float, float, str]]:
    """Marks ``{x, y, width, height, shape}`` for the markup overlay, as given (fractions or pixels).

    ``shape`` is ``box`` (default) or ``circle`` (an ellipse inside that box). A mark only
    points at an area; the prompt says whether to change it or to keep it.
    """
    if regions in (None, "", []):
        return []
    if not isinstance(regions, (list, tuple)):
        raise ValueError("regions must be a list of {x, y, width, height} marks")
    marks = []
    for item in regions:
        if not isinstance(item, dict):
            raise ValueError("each region must be an object with x, y, width and height")
        try:
            box = tuple(float(item[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            raise ValueError("each region needs numeric x, y, width and height") from None
        if box[2] <= 0 or box[3] <= 0 or min(box) < 0:
            raise ValueError("region width and height must be positive and x, y not negative")
        shape = _one_of(item.get("shape"), REGION_SHAPES, "") or (
            "circle" if str(item.get("shape") or "").strip().lower() in ("ellipse", "oval", "round") else "box"
        )
        marks.append((*box, shape))
    return marks


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
        self.trace = None  # agents.trace.ToolTrace of the current turn, set by ChainOfThoughtAgent
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

    @property
    def safety_identifier(self) -> str:
        """A stable, non-identifying id of this user for OpenAI's abuse monitoring."""
        return hashlib.sha256(f"generall:{self.user_id}".encode()).hexdigest()[:32]

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
                    "Markdown (![caption](images/<name>)) if the user asked to see it inside the message.\n"
                    "format 'svg' makes a vector drawing (logos, icons, flat illustrations): new images only, one at a time. "
                    "For a precise edit, point at the area with `regions` (look at the image first) or pass the user's "
                    "`markup` picture, and say in the prompt what the marks mean: change the marked area (\"replace the "
                    "circled cup with a glass\"), or keep it exactly while changing the rest (\"keep the circled face as "
                    "it is, change the background\"). For animated GIFs use generate_animation."
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
                        "format": {"type": "string", "enum": list(OUTPUT_FORMATS), "description": "File format of the result (default png; svg = vector, see above)."},
                        "complexity": {
                            "type": "string",
                            "enum": list(IMAGE_COMPLEXITIES),
                            "description": "Planning effort before drawing, separate from quality. Omit for the model's default (high); 'low' suits quick, simple pictures.",
                        },
                        "regions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    **{k: {"type": "number"} for k in ("x", "y", "width", "height")},
                                    "shape": {"type": "string", "enum": list(REGION_SHAPES), "description": "box (default) or circle"},
                                },
                                "required": ["x", "y", "width", "height"],
                            },
                            "description": (
                                "Edits only: marks on the first image, as fractions 0-1 of its width and height (x, y = top-left "
                                "corner of the box; a circle fills that box). Drawn as red outlines on a transparent overlay; "
                                "be generous rather than tight. The prompt says whether to change or to keep what is marked."
                            ),
                        },
                        "markup": {
                            "type": "string",
                            "description": (
                                "Edits only: a workspace picture with the user's own marks on the first image: a transparent PNG "
                                "overlay, or a copy of the same photo the user drew on in Telegram's editor (the drawing is "
                                "extracted by comparing it with the first image, which must be the clean original)."
                            ),
                        },
                        "caption": {"type": "string", "description": "Caption shown with the delivered file."},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "generate_animation",
                "description": (
                    f"Make an animation: a short animated GIF ({GIF_FRAMES_MIN}-{GIF_FRAMES_MAX} frames, about {GIF_SHORT_EDGE} px) "
                    "that plays and loops in the chat. This is the tool whenever the user asks for an animation, a GIF, a "
                    "sticker or a loop, or to animate / bring to life a picture (pass it in `images`); "
                    + (
                        "use the video tools only when the user explicitly asks for a video (an MP4 clip with camera "
                        "moves, realism or sound).\n"
                        if VIDEO_TOOLS_ENABLED else
                        "video generation is not available right now, so when the user asks for a video, say so and offer "
                        "an animation instead.\n"
                    )
                    +
                    "Describe the subject AND the motion, and say whether it should loop seamlessly. Total length = frames x "
                    f"frame_duration_ms (default {GIF_FRAMES_DEFAULT} x {GIF_FRAME_MS_DEFAULT} ms). Speed, direction, frame "
                    "selection and single-frame fixes are cheaper with edit_animation than a new generation."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Subject, style and the motion across the frames, e.g. 'a cat waving its paw, seamless loop'."},
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Workspace pictures to animate or to take the character/style from, e.g. 'images/photo.jpg'.",
                        },
                        "engine": {"type": "string", "enum": ["auto", "best", "fast"], "description": "auto/best = most capable (default), fast = quicker drafts."},
                        "quality": {"type": "string", "enum": list(IMAGE_QUALITIES), "description": "Rendering effort; more frames and higher quality cost more."},
                        "aspect_ratio": {"type": "string", "enum": list(IMAGE_ASPECT_RATIOS), "description": f"Shape of the frames (default {DEFAULT_ASPECT_RATIO}; 16:9 gives 896x512)."},
                        "frames": {"type": "integer", "description": f"Number of frames, {GIF_FRAMES_MIN}-{GIF_FRAMES_MAX} (default {GIF_FRAMES_DEFAULT})."},
                        "frame_duration_ms": {"type": "integer", "description": f"Delay per frame in ms, 20-2000 (default {GIF_FRAME_MS_DEFAULT})."},
                        "complexity": {"type": "string", "enum": list(IMAGE_COMPLEXITIES), "description": "Planning effort; omit for the model's default."},
                        "caption": {"type": "string", "description": "Caption shown with the animation."},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "edit_animation",
                "description": (
                    "Change an existing GIF locally, without generating it again: speed, a fixed frame delay, reverse, "
                    "boomerang (forward then back), keep some frames, loop count, or redraw ONE frame from a prompt "
                    "(the only option that calls the image model). Also turns a video from the workspace into a GIF, or a "
                    "GIF into an MP4. The result is a new file, delivered to the user; the original is kept.\n"
                    "Frame numbers are 1-based, as in the frame sheet generate_animation saved. Changes apply in this order: "
                    "edit_frame, frames, reverse, boomerang, timing, loop."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The GIF (or a video to convert), e.g. 'images/sunburst_anim_1a2b3c4d.gif'."},
                        "speed": {"type": "number", "description": "Playback speed factor: 2 = twice as fast, 0.5 = half speed."},
                        "frame_duration_ms": {"type": "integer", "description": "Set every frame to this delay (20-2000 ms); overrides speed."},
                        "reverse": {"type": "boolean", "description": "Play backwards."},
                        "boomerang": {"type": "boolean", "description": "Play forward, then backward, for a seamless back-and-forth loop."},
                        "frames": {"type": "string", "description": "Frames to keep, e.g. '1-6' or '1,3,5,7'."},
                        "loop": {"type": "integer", "description": "How many times the GIF file plays: 0 = forever (default), 1 = once. Telegram always loops animations in the chat."},
                        "edit_frame": {"type": "integer", "description": "Number of the frame to redraw with edit_prompt."},
                        "edit_prompt": {"type": "string", "description": "What to change in that frame; everything else is kept as is."},
                        "output": {"type": "string", "enum": ["gif", "mp4"], "description": "gif (default) or an mp4 video copy."},
                        "caption": {"type": "string", "description": "Caption shown with the result."},
                    },
                    "required": ["path"],
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
    _GENERATE_KEYS = (
        "prompt", "images", "engine", "quality", "aspect_ratio", "size", "variants", "format",
        "complexity", "regions", "markup", "caption",
    )
    _STORY_KEYS = ("prompt", "style")
    _ANIMATION_KEYS = ("prompt", "images", "engine", "quality", "aspect_ratio", "frames", "frame_duration_ms", "complexity", "caption")
    _EDIT_ANIMATION_KEYS = (
        "path", "speed", "frame_duration_ms", "reverse", "boomerang", "frames", "loop",
        "edit_frame", "edit_prompt", "output", "caption",
    )

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        args = tool_args if isinstance(tool_args, dict) else {}
        if tool_name == "generate_image":
            return await self._generate_image(**{k: v for k, v in args.items() if k in self._GENERATE_KEYS})
        if tool_name == "generate_story_images":
            return await self._generate_story(**{k: v for k, v in args.items() if k in self._STORY_KEYS})
        if tool_name == "generate_animation":
            return await self._generate_animation(**{k: v for k, v in args.items() if k in self._ANIMATION_KEYS})
        if tool_name == "edit_animation":
            return await self._edit_animation(**{k: v for k, v in args.items() if k in self._EDIT_ANIMATION_KEYS})
        return f"Unknown tool: {tool_name}"

    # ---- workspace helpers ----------------------------------------------
    def _resolve_input(self, image_path: str, fallback_dirs=("images", "downloads", "documents")) -> Path | None:
        """An existing file inside the user's workspace, or None."""
        from agents.paths import resolve_under

        resolved = resolve_under(self.base_path, image_path, fallback_dirs=fallback_dirs)
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
        complexity: str | None = None,
        regions: List[Dict[str, Any]] | None = None,
        markup: str | None = None,
        caption: str | None = None,
    ) -> str:
        if not str(prompt or "").strip():
            return "Error: 'prompt' is required."
        if str(format or "").strip().lower() == "gif":
            return "Error: animated GIFs are made with generate_animation, not generate_image."

        # The tool call wins, then the user's preference, then the default.
        engine = _one_of(engine, IMAGE_BACKENDS, "") or self.pref_engine
        quality = _one_of(quality, IMAGE_QUALITIES, "") or self.pref_quality
        aspect_ratio = _one_of(aspect_ratio, IMAGE_ASPECT_RATIOS, DEFAULT_ASPECT_RATIO)
        size = _one_of(size, IMAGE_SIZES, "") or self.pref_size
        fmt = _one_of(format, OUTPUT_FORMATS, "png")
        complexity = _one_of(complexity, IMAGE_COMPLEXITIES, "") or None
        caption = str(caption or "Here is your image")
        try:
            variants = max(1, min(self.max_variants, int(variants or 1)))
        except (TypeError, ValueError):
            variants = 1
        try:
            boxes = parse_regions(regions)
        except ValueError as e:
            return f"Error: {e}."

        backend = resolve_image_backend(engine, quality)
        requested = [p for p in (images or []) if str(p).strip()] if isinstance(images, (list, tuple)) else []

        resolved: List[Path] = []
        for candidate in requested:
            path = self._resolve_input(str(candidate))
            if path is None:
                return f"Error: '{candidate}' is not an image in your workspace. List the files first if you are unsure of the name."
            resolved.append(path)

        marked: Path | None = None
        if str(markup or "").strip():
            marked = self._resolve_input(str(markup))
            if marked is None:
                return f"Error: markup '{markup}' is not an image in your workspace."

        operation = "generated" if not resolved else ("edited" if len(resolved) == 1 else "composed")
        if resolved and backend.max_input_images == 0:
            return f"Error: engine '{backend.engine}' can only generate new images, not edit existing ones."
        if len(resolved) > backend.max_input_images:
            return (
                f"Error: engine '{backend.engine}' accepts at most {backend.max_input_images} input images, "
                f"{len(resolved)} were given. Use a different engine or fewer images."
            )
        if (boxes or marked) and not resolved:
            return "Error: regions and markup mark what to change in an input image; pass that image in `images`."
        if (boxes or marked) and not backend.supports("markup"):
            return f"Error: {backend.model} does not take markup; describe the area in the prompt or use engine 'best'."
        notes: List[str] = []
        if fmt == "svg":
            if not backend.supports("svg"):
                return f"Error: {backend.model} cannot make SVG files; SVG output needs GPT Image 2.6 (engine 'best' or 'fast')."
            if resolved:
                return "Error: SVG output only makes new drawings; existing pictures cannot be edited or vectorized as SVG."
            if variants > 1:
                notes.append("SVG output makes one drawing per call, so only one was made.")
                variants = 1
        if complexity and not backend.supports("complexity"):
            notes.append(f"{backend.model} has no complexity setting; it was left out.")
            complexity = None

        started = time.monotonic()
        temp_paths: List[Path] = []
        try:
            if backend.provider == "openai":
                saved, note = await self._openai_images(
                    backend, prompt=prompt, inputs=resolved, quality=quality, aspect_ratio=aspect_ratio,
                    size=size, variants=variants, fmt=fmt, caption=caption, temp_paths=temp_paths,
                    complexity=complexity, boxes=boxes, marked=marked,
                )
            else:
                saved, note = await self._gemini_images(
                    backend, prompt=prompt, inputs=resolved, quality=quality, aspect_ratio=aspect_ratio,
                    size=size, variants=variants, fmt=fmt, caption=caption, temp_paths=temp_paths,
                )
        except Exception as e:  # noqa: BLE001 - one failed image must not end the turn
            logger.exception("Image %s failed (engine=%s model=%s): %s", operation, backend.engine, backend.model, e)
            verb = operation[:-1] if operation.endswith("ed") else operation
            return f"Error: could not {verb} the image with {backend.model}: {e}{_sdk_hint(e)}"
        finally:
            self._cleanup(temp_paths)

        if note:
            notes.append(note)
        if fmt == "svg" and saved:
            notes.append("The SVG is a vector file: Telegram shows it as a file without a preview, and you cannot view it as an image.")
        return self._result(
            backend=backend, operation=operation, saved=saved, quality=quality,
            size=size, seconds=time.monotonic() - started, note=" ".join(notes),
        )

    # ---- providers -------------------------------------------------------
    def _optional_knobs(self, backend: ImageBackend, *, editing: bool, complexity: str | None = None) -> Dict[str, Any]:
        """The droppable knobs this model takes, minus any it refused earlier in this process."""
        knobs: Dict[str, Any] = {SAFETY_IDENTIFIER_PARAM: self.safety_identifier}
        # Only edits have an input to stay faithful to, and only models before gpt-image-2 take it.
        if editing and backend.supports(INPUT_FIDELITY_PARAM):
            knobs[INPUT_FIDELITY_PARAM] = INPUT_FIDELITY
        if complexity and backend.supports(COMPLEXITY_PARAM):
            knobs[COMPLEXITY_PARAM] = complexity
        return {k: v for k, v in knobs.items() if knob_allowed(backend.model, k)}

    async def _openai_request(
        self,
        backend: ImageBackend,
        *,
        prompt: str,
        prepared: List[Path],
        quality: str,
        size: str,
        n: int,
        fmt: str,
        fixed: Dict[str, Any] | None = None,
        optional: Dict[str, Any] | None = None,
        size_fallback: str | None = None,
    ):
        """One images API call: ``images.generate``, or ``images.edit`` when inputs are given.

        ``fixed`` knobs are part of the request's meaning (GIF frames): a refusal is an error.
        ``optional`` ones are dropped when the provider refuses them, and remembered for the
        process when it says it does not know them (``markup`` is a path, uploaded as a file).
        An unverified quality tier falls back once to IMAGE_QUALITY_FALLBACK, and a size the
        model refuses is replaced once by ``size_fallback`` (remembered once that worked).
        These model ids and knobs are newer than some SDKs and accounts, so being wrong about
        one costs a round trip rather than the turn.

        Returns ``(result, used_quality, used_size, notes)``.
        """
        model = backend.model
        opts = dict(optional or {})
        notes: List[str] = []
        used_quality, used_size = quality, size
        if size_fallback and not knob_allowed(model, f"size={size}"):
            used_size = size_fallback
        refused_size: str | None = None

        def _call(q: str, sz: str, knobs: Dict[str, Any]):
            kwargs: Dict[str, Any] = {
                "model": model, "prompt": prompt, "size": sz, "n": n, "quality": q, "output_format": fmt,
                **(fixed or {}), **{k: v for k, v in knobs.items() if k != MARKUP_PARAM},
            }
            if not prepared:
                return openai_client().images.generate(**kwargs)
            opened = [open(p, "rb") for p in prepared]
            try:
                kwargs["image"] = opened[0] if len(opened) == 1 else list(opened)
                if knobs.get(MARKUP_PARAM):
                    opened.append(open(knobs[MARKUP_PARAM], "rb"))
                    kwargs[MARKUP_PARAM] = opened[-1]
                return openai_client().images.edit(**kwargs)
            finally:
                for handle in opened:
                    handle.close()

        failure: BaseException | None = None
        for _ in range(len(opts) + 3):
            try:
                result = await asyncio.to_thread(_call, used_quality, used_size, dict(opts))
                break
            except Exception as e:  # noqa: BLE001 - classified below, re-raised when it is not a knob
                failure = e
                dropped = next((k for k in opts if is_parameter_rejection(e, k)), None)
                if dropped:
                    logger.warning("%s refused %s, retrying without it: %s", model, dropped, e)
                    opts.pop(dropped)
                    if says_unsupported(e):
                        remember_unsupported(model, dropped)
                    if dropped in REPORTED_KNOBS:
                        notes.append(f"{model} refused '{dropped}' ({_short(e)}); the request ran without it.")
                    continue
                if used_quality in ("xhigh", "max") and is_quality_rejection(e):
                    logger.warning("Quality %r refused by %s, retrying at %s: %s", used_quality, model, IMAGE_QUALITY_FALLBACK, e)
                    notes.append(f"Quality '{quality}' is not supported by {model}; used '{IMAGE_QUALITY_FALLBACK}' instead.")
                    used_quality = IMAGE_QUALITY_FALLBACK
                    continue
                if size_fallback and refused_size is None and used_size != size_fallback and is_parameter_rejection(e, "size"):
                    logger.warning("Size %s refused by %s, retrying at %s: %s", used_size, model, size_fallback, e)
                    notes.append(f"Size {used_size} is not accepted by {model}; used {size_fallback} instead.")
                    refused_size, used_size = used_size, size_fallback
                    continue
                raise
        else:
            raise failure if failure is not None else RuntimeError("the image request was not sent")
        if refused_size:
            remember_unsupported(model, f"size={refused_size}")
        self._record_usage(model, getattr(result, "usage", None))
        self._record_svg_usage(getattr(result, "svg_usage", None))
        return result, used_quality, used_size, notes

    async def _openai_images(
        self, backend, *, prompt, inputs, quality, aspect_ratio, size, variants, fmt, caption, temp_paths,
        complexity=None, boxes=(), marked=None,
    ):
        """One call for every variant: images.generate, or images.edit when inputs are given."""
        pixel_size = resolve_pixel_size(size, aspect_ratio)
        prepared = [await asyncio.to_thread(self._prepare_input, p, temp_paths) for p in inputs]
        optional = self._optional_knobs(backend, editing=bool(prepared), complexity=complexity)
        if prepared and (boxes or marked):
            optional[MARKUP_PARAM] = str(await asyncio.to_thread(self._build_markup, prepared[0], marked, list(boxes), temp_paths))

        result, used_quality, used_size, notes = await self._openai_request(
            backend, prompt=prompt, prepared=prepared, quality=quality, size=pixel_size, n=variants, fmt=fmt,
            optional=optional, size_fallback=None if pixel_size in STANDARD_SIZES else standard_size_for(pixel_size),
        )

        saved: List[Path] = []
        items = list(getattr(result, "data", None) or [])
        for index, item in enumerate(items, start=1):
            payload = getattr(item, "b64_json", None)
            if not payload:
                continue
            saved.append(await self._deliver(
                base64.b64decode(payload), prefix=backend.prefix, fmt=fmt,
                caption=self._caption(caption, backend, used_quality, used_size, index, len(items)),
            ))
        return saved, " ".join(notes)

    def _build_markup(self, target: Path, marked: Path | None, boxes: List[tuple], temp_paths: List[Path]) -> Path:
        """A transparent PNG the size of ``target`` (as oriented) carrying the marks to follow.

        ``marked`` is used as the overlay when it has transparency; an opaque picture is taken
        to be a copy of the photo the user drew on, and the drawing is what differs from
        ``target``. ``boxes`` are outlined on top (fractions 0-1 of the image, or pixels).
        """
        resampling = getattr(PIL.Image, "Resampling", PIL.Image)
        with PIL.Image.open(target) as img:
            base = PIL.ImageOps.exif_transpose(img).convert("RGB")
        width, height = base.size
        overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))

        if marked is not None:
            with PIL.Image.open(marked) as source:
                drawn = PIL.ImageOps.exif_transpose(source)
                drawn.load()
            if _has_transparency(drawn):
                overlay.alpha_composite(drawn.convert("RGBA").resize(base.size, resampling.NEAREST))
            else:
                drawn = drawn.convert("RGB").resize(base.size, resampling.LANCZOS)
                r, g, b = PIL.ImageChops.difference(base, drawn).split()
                strongest = PIL.ImageChops.lighter(r, PIL.ImageChops.lighter(g, b))
                mask = strongest.point(lambda v: 255 if v > MARKUP_DIFF_THRESHOLD else 0)
                # Opening: specks of JPEG noise go, strokes stay (and get a little bolder).
                mask = mask.filter(PIL.ImageFilter.MinFilter(3)).filter(PIL.ImageFilter.MaxFilter(5))
                if mask.getbbox() is None:
                    raise ValueError("the markup picture does not differ from the image being edited; nothing is marked")
                layer = drawn.convert("RGBA")
                layer.putalpha(mask)
                overlay.alpha_composite(layer)

        if boxes:
            draw = PIL.ImageDraw.Draw(overlay)
            stroke = max(3, round(min(width, height) / 150))
            for x, y, w, h, shape in boxes:
                if max(x, y, w, h) <= 1.0:
                    x, y, w, h = x * width, y * height, w * width, h * height
                left, top = max(0, min(width - 1, x)), max(0, min(height - 1, y))
                right, bottom = max(left + 1, min(width - 1, x + w)), max(top + 1, min(height - 1, y + h))
                outline = draw.ellipse if shape == "circle" else draw.rectangle
                outline((left, top, right, bottom), outline=MARKUP_COLOR, width=stroke)

        if overlay.getchannel("A").getextrema()[0] == 255:
            raise ValueError("the markup covers the whole image; mark only the area to change")
        path = self.images_path / f"markup_{uuid.uuid4().hex[:8]}.png"
        temp_paths.append(path)
        overlay.save(path, format="PNG")
        return path

    def _record_usage(self, model: str, usage) -> None:
        """Add one images API call's tokens and estimated cost to the turn's trace (never raises)."""
        if usage is None or self.trace is None:
            return
        try:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            details = getattr(usage, "input_tokens_details", None)
            text_in = int(getattr(details, "text_tokens", 0) or 0) if details else input_tokens
            image_in = int(getattr(details, "image_tokens", 0) or 0) if details else 0
            out_details = getattr(usage, "output_tokens_details", None)
            image_out = int(getattr(out_details, "image_tokens", 0) or 0) if out_details else output_tokens
            self.trace.add_usage(
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
                model=model,
                cost_usd=estimate_image_cost(model, text_in, image_in, image_out),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record image usage for %s: %s", model, e)

    def _record_svg_usage(self, svg_usage) -> None:
        """SVG output adds the SVG author model's tokens, priced via MODEL_PRICES when listed."""
        if svg_usage is None or self.trace is None:
            return
        try:
            self.trace.add_usage(
                {
                    "input_tokens": int(getattr(svg_usage, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(svg_usage, "output_tokens", 0) or 0),
                },
                model=str(getattr(svg_usage, "model", "") or "svg-author"),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record SVG usage: %s", e)

    def _record_gemini_usage(self, model: str, response) -> None:
        """Add one Gemini call's tokens and estimated cost to the turn's trace (never raises).

        Thinking tokens are billed as output, so they are counted with the candidates.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is None or self.trace is None:
            return
        try:
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0)
            self.trace.add_usage(
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
                model=model,
                cost_usd=estimate_image_cost(model, input_tokens, 0, output_tokens),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record image usage for %s: %s", model, e)

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
                self._record_gemini_usage(backend.model, response)
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
            self._record_gemini_usage(backend.model, response)
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

    # ---- animations (GIF) -------------------------------------------------
    async def _send_animation(self, path: Path, caption: str) -> None:
        """A GIF as a Telegram animation (it plays and loops in the chat)."""
        send = getattr(self.sender, "send_animation", None)
        if send is None:  # a sender without animations still gets the file
            await self.sender.send_document(str(path), caption=caption)
        else:
            await send(str(path), caption=caption)

    def _describe_animation(self, path: Path) -> tuple[anim.Animation | None, Path | None]:
        """Read back what was written (frames, timing) and save the frame sheet next to it."""
        try:
            info = anim.read_gif(path)
            return info, anim.frame_sheet(info, path.with_name(f"{path.stem}_frames.png"))
        except Exception as e:  # noqa: BLE001 - the GIF is still delivered without a sheet
            logger.warning("Could not read back the animation %s: %s", path, e)
            return None, None

    def _animation_caption(self, base: str, backend: ImageBackend | None, quality: str | None, info: anim.Animation | None) -> str:
        if not self.show_metadata or info is None:
            return base
        w, h = info.size
        delays = sorted(set(info.durations))
        timing = f"{delays[0]} ms" if len(delays) == 1 else f"{delays[0]}-{delays[-1]} ms"
        bits = [backend.model] if backend else []
        if backend and quality:
            bits.append(f"quality {quality}")
        bits += [f"{len(info.frames)} frames", timing, f"{w}x{h}"]
        return f"{base}\n{' · '.join(bits)}"

    def _animation_result(self, headline: str, *, path: Path, info, sheet, notes: List[str], delivered: str) -> str:
        lines = [headline + (f": {info.describe()}" if info else "") + f". {delivered}."]
        if notes:
            lines.append(" ".join(notes))
        lines.append(f"Saved: {self._rel(path)}")
        if sheet is not None:
            lines.append(
                f"Frame sheet: {self._rel(sheet)} (view_image it to check the motion; "
                "a GIF itself shows you only its first frame)."
            )
        lines.append(
            "Speed, direction, frame selection, loops, one redrawn frame or an MP4 copy: edit_animation, "
            "no new generation needed. If the user wants the .gif file itself, send it with "
            "send_file_path_to_user_via_telegram."
        )
        return "\n".join(lines)

    @staticmethod
    def _animation_error(error: BaseException, backend: ImageBackend, has_inputs: bool) -> str:
        text = str(error)
        low = text.lower()
        hint = _sdk_hint(error)
        if not hint and any(w in low for w in ("preview", "access", "not enabled", "permission")):
            hint = " GIF output needs the GIF preview to be enabled for this model on the OpenAI account."
        elif not hint and has_inputs and any(w in low for w in ("gif", "frame", "output_format")):
            hint = (
                " Animating an existing picture was refused. Describe the picture in the prompt and call "
                "generate_animation without images; a video already in the workspace can be turned into a GIF with "
                "edit_animation."
            )
        return f"Error: could not make the animation with {backend.model}: {text}{hint}"

    async def _generate_animation(
        self,
        prompt: str,
        images: List[str] | None = None,
        engine: str | None = None,
        quality: str | None = None,
        aspect_ratio: str | None = None,
        frames: int | None = None,
        frame_duration_ms: int | None = None,
        complexity: str | None = None,
        caption: str | None = None,
    ) -> str:
        if not str(prompt or "").strip():
            return "Error: 'prompt' is required."
        backend = resolve_animation_backend(_one_of(engine, ("auto", "best", "fast"), "") or self.pref_engine)
        if not backend.supports("gif"):
            return (
                f"Error: {backend.model} cannot make animated GIFs; GPT_ANIMATION_MODEL / GPT_ANIMATION_MODEL_FAST "
                "must name a GPT Image 2.6 model."
            )
        quality = _one_of(quality, IMAGE_QUALITIES, "") or self.pref_quality
        aspect_ratio = _one_of(aspect_ratio, IMAGE_ASPECT_RATIOS, "")
        complexity = _one_of(complexity, IMAGE_COMPLEXITIES, "") or None
        caption = str(caption or "Here is your animation")
        frame_count = _clamp_int(frames, GIF_FRAMES_MIN, GIF_FRAMES_MAX, GIF_FRAMES_DEFAULT)
        try:
            frame_ms = anim.clamp_duration(GIF_FRAME_MS_DEFAULT if frame_duration_ms in (None, "") else frame_duration_ms)
        except anim.AnimationError as e:
            return f"Error: {e}."

        requested = [p for p in (images or []) if str(p).strip()] if isinstance(images, (list, tuple)) else []
        resolved: List[Path] = []
        for candidate in requested:
            path = self._resolve_input(str(candidate))
            if path is None:
                return f"Error: '{candidate}' is not an image in your workspace. List the files first if you are unsure of the name."
            resolved.append(path)
        if len(resolved) > backend.max_input_images:
            return f"Error: at most {backend.max_input_images} input images, {len(resolved)} were given."
        if not aspect_ratio:  # an animated photo keeps the photo's shape
            aspect_ratio = await asyncio.to_thread(_closest_ratio, resolved[0]) if resolved else DEFAULT_ASPECT_RATIO

        started = time.monotonic()
        temp_paths: List[Path] = []
        try:
            prepared = [await asyncio.to_thread(self._prepare_input, p, temp_paths) for p in resolved]
            result, used_quality, _used_size, notes = await self._openai_request(
                backend, prompt=prompt, prepared=prepared, quality=quality, size=gif_pixel_size(aspect_ratio),
                n=1, fmt="gif", fixed={"frame_count": frame_count, "frame_duration_ms": frame_ms},
                optional=self._optional_knobs(backend, editing=False, complexity=complexity),
                size_fallback=GIF_SIZE_FALLBACK,
            )
        except Exception as e:  # noqa: BLE001 - one failed animation must not end the turn
            logger.exception("Animation failed (model=%s): %s", backend.model, e)
            return self._animation_error(e, backend, bool(resolved))
        finally:
            self._cleanup(temp_paths)

        items = [i for i in (getattr(result, "data", None) or []) if getattr(i, "b64_json", None)]
        if not items:
            return f"No animation was produced ({backend.model}). {' '.join(notes)}".strip()
        data = base64.b64decode(items[0].b64_json)
        if not anim.is_gif(data):
            path = await self._deliver(data, prefix=backend.prefix, fmt="png", caption=caption)
            return (
                f"{backend.model} returned a still {path.suffix.lstrip('.')} image instead of a GIF (GIF output may not be "
                f"enabled for this account yet). It was delivered as a file. Saved: {self._rel(path)}"
            )
        path = self.images_path / f"{backend.prefix}_{uuid.uuid4().hex[:8]}.gif"
        await asyncio.to_thread(path.write_bytes, data)
        info, sheet = await asyncio.to_thread(self._describe_animation, path)
        await self._send_animation(path, self._animation_caption(caption, backend, used_quality, info))
        operation = "Animated the given picture" if resolved else "Made an animated GIF"
        return self._animation_result(
            f"{operation} with {backend.model} at quality {used_quality} in {time.monotonic() - started:.0f}s",
            path=path, info=info, sheet=sheet, notes=notes,
            delivered="Delivered to the user as an animation that plays in the chat",
        )

    async def _redraw_frame(self, animation: anim.Animation, index: int, prompt: str, temp_paths: List[Path], notes: List[str]) -> anim.Animation:
        """One frame through the image model (an edit), fitted back into the animation."""
        backend = resolve_image_backend(self.pref_engine)
        if backend.provider != "openai":
            backend = resolve_image_backend(DEFAULT_IMAGE_ENGINE)
        frame_path = self.images_path / f"frame_input_{uuid.uuid4().hex[:8]}.png"
        temp_paths.append(frame_path)
        await asyncio.to_thread(animation.frames[index].save, frame_path, "PNG")
        w, h = animation.size
        pixel_size = resolve_pixel_size("1K", f"{w}:{h}")
        full_prompt = (
            f"{prompt}\n\nThis picture is frame {index + 1} of {len(animation.frames)} of an animated GIF. Change only what "
            "is asked; keep the style, framing, colours, lighting and everything else exactly as they are, so the frame "
            "still matches the frames around it."
        )
        result, _quality, _size, request_notes = await self._openai_request(
            backend, prompt=full_prompt, prepared=[frame_path], quality=self.pref_quality, size=pixel_size, n=1, fmt="png",
            optional=self._optional_knobs(backend, editing=True),
            size_fallback=None if pixel_size in STANDARD_SIZES else standard_size_for(pixel_size),
        )
        notes.extend(request_notes)
        items = [i for i in (getattr(result, "data", None) or []) if getattr(i, "b64_json", None)]
        if not items:
            raise anim.AnimationError(f"{backend.model} returned no picture for frame {index + 1}")

        def _fit() -> anim.Animation:
            import io

            with PIL.Image.open(io.BytesIO(base64.b64decode(items[0].b64_json))) as img:
                img.load()
                return anim.replace_frame(animation, index, img)

        notes.append(f"Frame {index + 1} was redrawn with {backend.model}.")
        return await asyncio.to_thread(_fit)

    async def _edit_animation(
        self,
        path: str,
        speed: float | None = None,
        frame_duration_ms: int | None = None,
        reverse: bool = False,
        boomerang: bool = False,
        frames: str | None = None,
        loop: int | None = None,
        edit_frame: int | None = None,
        edit_prompt: str | None = None,
        output: str | None = None,
        caption: str | None = None,
    ) -> str:
        if not str(path or "").strip():
            return "Error: 'path' is required."
        source = self._resolve_input(str(path), fallback_dirs=("images", "videos", "downloads", "documents"))
        if source is None:
            return f"Error: '{path}' is not a file in your workspace. List the files first if you are unsure of the name."
        output = _one_of(output, ("gif", "mp4"), "gif")
        reverse, boomerang = _flag(reverse), _flag(boomerang)
        given = lambda value: value not in (None, "")  # noqa: E731
        if given(edit_frame) and not str(edit_prompt or "").strip():
            return "Error: edit_frame needs edit_prompt (what to change in that frame)."
        is_video = source.suffix.lower() in anim.VIDEO_EXTENSIONS
        if not is_video:
            head = await asyncio.to_thread(_read_head, source)
            if not anim.is_gif(head):
                return f"Error: {source.name} is neither a GIF nor a video."
        changes = [given(speed), given(frame_duration_ms), reverse, boomerang, given(frames), given(loop), given(edit_frame)]
        if not any(changes) and not is_video and output == "gif":
            return "Error: nothing to change; pass speed, frame_duration_ms, reverse, boomerang, frames, loop, edit_frame or output 'mp4'."

        started = time.monotonic()
        temp_paths: List[Path] = []
        notes: List[str] = []
        applied: List[str] = []
        try:
            gif_source = source
            if is_video:
                gif_source = self.images_path / f".convert_{uuid.uuid4().hex[:8]}.gif"
                temp_paths.append(gif_source)
                await asyncio.to_thread(anim.video_to_gif, source, gif_source)
                applied.append(f"made from the first {anim.VIDEO_GIF_MAX_SECONDS} s of the video at {anim.VIDEO_GIF_FPS} fps")
            animation = await asyncio.to_thread(anim.read_gif, gif_source)
            notes.extend(animation.notes)

            if given(edit_frame):
                number = _clamp_int(edit_frame, 0, 10**6, 0)
                if not 1 <= number <= len(animation.frames):
                    raise anim.AnimationError(f"edit_frame must be between 1 and {len(animation.frames)}")
                animation = await self._redraw_frame(animation, number - 1, str(edit_prompt), temp_paths, notes)
                applied.append(f"redrew frame {number}")
            if given(frames):
                animation = anim.select_frames(animation, anim.parse_frame_selection(frames, len(animation.frames)))
                applied.append(f"kept frames {str(frames).replace(' ', '')}")
            if reverse:
                animation = anim.reverse(animation)
                applied.append("reversed")
            if boomerang:
                animation = anim.boomerang(animation)
                applied.append("boomerang")
            if given(frame_duration_ms) or given(speed):
                animation = anim.retime(
                    animation, speed=speed if given(speed) else None,
                    frame_duration_ms=frame_duration_ms if given(frame_duration_ms) else None,
                )
                notes.extend(animation.notes)
                applied.append(f"{animation.durations[0]} ms per frame" if given(frame_duration_ms) else f"speed x{float(speed):g}")
            if given(loop):
                animation = anim.set_loop(animation, loop)
                applied.append("loops forever" if animation.loop == 0 else "loop count changed")

            stem = re.sub(r"(_edit_[0-9a-f]{8})+$", "", source.stem).lstrip(".")[:40] or "animation"
            if any(changes) or is_video:
                target = self.images_path / f"{stem}_edit_{uuid.uuid4().hex[:8]}.gif"
                await asyncio.to_thread(anim.write_gif, animation, target)
            else:  # only an MP4 copy was asked for: convert the GIF as it is
                target = source
            info, sheet = await asyncio.to_thread(self._describe_animation, target)
            caption = str(caption or "Here is your animation")
            if output == "mp4":
                videos = self.base_path / "videos"
                videos.mkdir(parents=True, exist_ok=True)
                mp4 = videos / f"{stem}_{uuid.uuid4().hex[:8]}.mp4"
                await asyncio.to_thread(anim.gif_to_mp4, target, mp4)
                await self.sender.send_video(str(mp4), caption=caption)
                applied.append("MP4 copy")
                delivered = f"Delivered to the user as an MP4 video ({self._rel(mp4)}; the GIF is kept too)"
            else:
                await self._send_animation(target, self._animation_caption(caption, None, None, info))
                delivered = "Delivered to the user as an animation that plays in the chat"
        except anim.AnimationError as e:
            return f"Error: {e}."
        except Exception as e:  # noqa: BLE001 - one failed change must not end the turn
            logger.exception("Animation edit failed (%s): %s", source, e)
            return f"Error: could not change the animation: {e}{_sdk_hint(e)}"
        finally:
            self._cleanup(temp_paths)

        summary = ", ".join(applied) if applied else "re-encoded"
        return self._animation_result(
            f"Changed the animation ({summary}) in {time.monotonic() - started:.0f}s",
            path=target, info=info, sheet=sheet, notes=notes, delivered=delivered,
        )


def _clamp_int(value, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _closest_ratio(path: Path) -> str:
    """The IMAGE_ASPECT_RATIOS entry closest to a picture's shape (as displayed), 1:1 if unreadable."""
    try:
        with PIL.Image.open(path) as img:
            width, height = PIL.ImageOps.exif_transpose(img).size
    except Exception:  # noqa: BLE001
        return DEFAULT_ASPECT_RATIO

    def distance(ratio: str) -> float:
        w, h = (int(x) for x in ratio.split(":"))
        return abs(math.log((width / height) / (w / h)))

    return min(IMAGE_ASPECT_RATIOS, key=distance)


def _read_head(path: Path, size: int = 16) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(size)
