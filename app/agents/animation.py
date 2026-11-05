"""Animated GIF helpers for the image tools: read, change, write, preview, convert.

The GIF itself is drawn by the image model (``generate_animation`` in
agents/image_tools.py). Everything a user asks for afterwards that only changes how it
plays (speed, direction, which frames, how often it loops) is done here, locally and for
free; ``edit_animation`` calls these. One frame redrawn by the model goes through the
image API and comes back here to be put in place.

Pure functions over files and PIL images, synchronous: callers run them in
``asyncio.to_thread``. ffmpeg (already in the bot image for audio and video) converts
between GIF and video; it is given a hard timeout.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import PIL.Image
import PIL.ImageDraw
import PIL.ImageSequence

from models import GIF_FRAME_MS_MAX, GIF_FRAME_MS_MIN

GIF_MAGIC = (b"GIF87a", b"GIF89a")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_FRAMES = 150  # a converted video (10 s at 10 fps) or a hand-made GIF, not just the model's 16
# Pixels of all frames together while they are held in memory (RGBA: 4 bytes each, so 160 MB):
# a big uploaded GIF is scaled down while it is read rather than filling the bot's memory.
MAX_TOTAL_PIXELS = 40_000_000
FFMPEG_TIMEOUT = 120
# A video turned into a GIF: frames per second, longest side and length kept.
VIDEO_GIF_FPS, VIDEO_GIF_MAX_SIDE, VIDEO_GIF_MAX_SECONDS = 10, 512, 10


class AnimationError(ValueError):
    """A request this module cannot carry out; the message is meant for the agent."""


def is_gif(data: bytes) -> bool:
    return data[:6] in GIF_MAGIC


def clamp_duration(ms) -> int:
    """A frame delay GIF can store: whole centiseconds within the API's 20..2000 ms."""
    try:
        value = int(round(float(ms) / 10.0)) * 10
    except (TypeError, ValueError):
        raise AnimationError(f"frame duration must be a number of milliseconds, not {ms!r}") from None
    return max(GIF_FRAME_MS_MIN, min(GIF_FRAME_MS_MAX, value))


@dataclass
class Animation:
    frames: List[PIL.Image.Image]  # RGBA, all the same size
    durations: List[int]  # ms per frame
    loop: int | None = 0  # 0 = forever, n = n extra repeats as GIF stores it, None = play once
    notes: List[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.frames[0].size

    @property
    def total_ms(self) -> int:
        return sum(self.durations)

    def describe(self) -> str:
        w, h = self.size
        unique = sorted(set(self.durations))
        timing = f"{unique[0]} ms per frame" if len(unique) == 1 else f"{min(unique)}-{max(unique)} ms per frame"
        looping = "loops forever" if self.loop == 0 else ("plays once" if self.loop is None else f"plays {self.loop + 1} times")
        return f"{len(self.frames)} frames, {w}x{h}, {timing}, {self.total_ms / 1000:.1f} s per loop, {looping}"


# ---- reading and writing ----------------------------------------------------
def read_gif(path: str | Path) -> Animation:
    """Every frame of a GIF, composited (what a viewer shows), with its delay."""
    try:
        with PIL.Image.open(path) as img:
            if (img.format or "").upper() != "GIF":
                raise AnimationError(f"{Path(path).name} is not a GIF")
            loop = img.info.get("loop")
            count = min(int(getattr(img, "n_frames", 1) or 1), MAX_FRAMES)
            width, height = img.size
            scale = min(1.0, math.sqrt(MAX_TOTAL_PIXELS / max(1, count * width * height)))
            fit = (max(1, int(width * scale)), max(1, int(height * scale)))
            frames, durations = [], []
            for frame in PIL.ImageSequence.Iterator(img):
                if len(frames) >= MAX_FRAMES:
                    break
                durations.append(int(frame.info.get("duration") or img.info.get("duration") or 100))
                rgba = frame.convert("RGBA")
                frames.append(rgba.resize(fit, getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS) if scale < 1 else rgba.copy())
    except AnimationError:
        raise
    except Exception as e:  # noqa: BLE001 - a broken file is the caller's message, not a crash
        raise AnimationError(f"could not read {Path(path).name} as a GIF: {e}") from e
    if not frames:
        raise AnimationError(f"{Path(path).name} has no frames")
    result = Animation(frames, durations, loop if isinstance(loop, int) else None)
    if scale < 1:
        result.notes.append(f"The frames were scaled down from {width}x{height} to {fit[0]}x{fit[1]} to fit in memory.")
    if count == MAX_FRAMES and len(frames) == MAX_FRAMES:
        result.notes.append(f"Only the first {MAX_FRAMES} frames were kept.")
    return result


def write_gif(animation: Animation, path: str | Path) -> Path:
    path = Path(path)
    frames = [f.convert("RGBA") for f in animation.frames]
    for frame in frames:
        # A frame read from a GIF carries that file's loop/duration/transparency, which
        # Pillow would write back over what is asked here (a "play once" kept looping).
        frame.info = {}
    options = {"save_all": True, "append_images": frames[1:], "duration": list(animation.durations), "disposal": 2}
    if animation.loop is not None:
        options["loop"] = int(animation.loop)
    frames[0].save(path, format="GIF", **options)
    return path


def frame_sheet(animation: Animation, path: str | Path, max_side: int = 1024) -> Path:
    """All frames side by side in a numbered grid (PNG), so the agent can see the motion.

    The vision models look at the first frame of a GIF only; a sheet is how the agent
    checks that the frames really change and stay consistent.
    """
    count = len(animation.frames)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    w, h = animation.size
    scale = min(1.0, max_side / (cols * w), max_side / (rows * h))
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    gap = 4
    sheet = PIL.Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap), (40, 40, 40))
    draw = PIL.ImageDraw.Draw(sheet)
    for index, frame in enumerate(animation.frames):
        cell = PIL.Image.new("RGBA", frame.size, (255, 255, 255, 255))
        cell.alpha_composite(frame)
        cell = cell.convert("RGB").resize((cw, ch), getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS)
        x = gap + (index % cols) * (cw + gap)
        y = gap + (index // cols) * (ch + gap)
        sheet.paste(cell, (x, y))
        label = str(index + 1)
        draw.rectangle((x, y, x + 8 + 7 * len(label), y + 14), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 255))
    sheet.save(path, format="PNG", optimize=True)
    return Path(path)


# ---- changes ------------------------------------------------------------------
def parse_frame_selection(spec, count: int) -> List[int]:
    """``"1-6"``, ``"1,3,5"``, ``"4-"``, ``"-3"`` (1-based, inclusive) -> 0-based indices."""
    text = str(spec or "").replace(" ", "")
    if not text:
        raise AnimationError("frames is empty; use e.g. '1-6' or '1,3,5'")
    picked: List[int] = []
    for part in text.split(","):
        if not part:
            continue
        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text) if start_text else 1
                end = int(end_text) if end_text else count
            else:
                start = end = int(part)
        except ValueError:
            raise AnimationError(f"cannot read frame range {part!r}; use e.g. '1-6' or '1,3,5'") from None
        if start < 1 or end > count or start > end:
            raise AnimationError(f"frame range {part!r} is outside 1-{count}")
        picked.extend(range(start - 1, end))
    if len(picked) < 2:
        raise AnimationError("an animation needs at least 2 frames")
    return picked


def select_frames(animation: Animation, indices: List[int]) -> Animation:
    return Animation([animation.frames[i] for i in indices], [animation.durations[i] for i in indices], animation.loop)


def reverse(animation: Animation) -> Animation:
    return Animation(animation.frames[::-1], animation.durations[::-1], animation.loop)


def boomerang(animation: Animation) -> Animation:
    """Forward, then back, without showing the first and last frame twice in a row."""
    back = list(range(len(animation.frames) - 2, 0, -1))
    order = list(range(len(animation.frames))) + back
    return Animation([animation.frames[i] for i in order], [animation.durations[i] for i in order], animation.loop)


def retime(animation: Animation, *, speed=None, frame_duration_ms=None) -> Animation:
    """A fixed delay for every frame, or every delay divided by ``speed`` (2 = twice as fast)."""
    if frame_duration_ms is not None:
        durations = [clamp_duration(frame_duration_ms)] * len(animation.frames)
    elif speed is not None:
        try:
            factor = float(speed)
        except (TypeError, ValueError):
            raise AnimationError(f"speed must be a number such as 2 or 0.5, not {speed!r}") from None
        if not (0.05 <= factor <= 20):
            raise AnimationError("speed must be between 0.05 and 20")
        durations = [clamp_duration(d / factor) for d in animation.durations]
    else:
        return animation
    result = Animation(animation.frames, durations, animation.loop)
    if any(d in (GIF_FRAME_MS_MIN, GIF_FRAME_MS_MAX) for d in durations):
        result.notes.append(f"Some delays hit the {GIF_FRAME_MS_MIN}-{GIF_FRAME_MS_MAX} ms limit.")
    return result


def set_loop(animation: Animation, plays) -> Animation:
    """``plays`` = how many times it plays: 0 = forever, 1 = once, n = n times."""
    try:
        plays = int(plays)
    except (TypeError, ValueError):
        raise AnimationError(f"loop must be a whole number (0 = forever), not {plays!r}") from None
    if plays < 0:
        raise AnimationError("loop cannot be negative (0 = forever)")
    loop = 0 if plays == 0 else (None if plays == 1 else plays - 1)
    return Animation(animation.frames, animation.durations, loop)


def replace_frame(animation: Animation, index: int, image: PIL.Image.Image) -> Animation:
    """Put a redrawn frame in place, fitted to the animation's size."""
    frames = list(animation.frames)
    fitted = image.convert("RGBA")
    if fitted.size != animation.size:
        fitted = fitted.resize(animation.size, getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS)
    frames[index] = fitted
    return Animation(frames, list(animation.durations), animation.loop)


# ---- video ----------------------------------------------------------------------
def ffmpeg_binary() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _run_ffmpeg(args: List[str]) -> None:
    try:
        done = subprocess.run(
            [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args],
            capture_output=True, timeout=FFMPEG_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise AnimationError("ffmpeg is not installed on the bot host") from None
    except subprocess.TimeoutExpired:
        raise AnimationError(f"ffmpeg took longer than {FFMPEG_TIMEOUT} s") from None
    if done.returncode != 0:
        detail = (done.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise AnimationError("ffmpeg failed: " + (detail[-1] if detail else f"exit code {done.returncode}"))


def video_to_gif(source: str | Path, target: str | Path, *, fps: int = VIDEO_GIF_FPS,
                 max_side: int = VIDEO_GIF_MAX_SIDE, max_seconds: int = VIDEO_GIF_MAX_SECONDS) -> Path:
    """The first ``max_seconds`` of a video as a looping GIF with its own palette."""
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))':flags=lanczos"
    graph = f"fps={fps},{scale},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    _run_ffmpeg(["-t", str(max_seconds), "-i", str(source), "-filter_complex", graph, "-loop", "0", str(target)])
    return Path(target)


def gif_to_mp4(source: str | Path, target: str | Path) -> Path:
    """An MP4 Telegram and every phone plays (even edges, yuv420p, fast start)."""
    _run_ffmpeg([
        "-i", str(source), "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-an", str(target),
    ])
    return Path(target)
