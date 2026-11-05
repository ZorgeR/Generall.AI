"""Animated GIFs (GPT Image 2.6): generate_animation, edit_animation and agents/animation.py."""
import base64
import io
import os
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import PIL.Image
import pytest

import models
from agents import animation as anim
from agents.image_tools import ImageTools, gif_pixel_size

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def gif_bytes(count=9, size=(512, 512), duration=200, loop=0):
    frames = [PIL.Image.new("RGB", size, (20 * i % 256, 100, 255 - 20 * i % 256)) for i in range(count)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=duration, loop=loop)
    return buf.getvalue()


def png_bytes(size=(1024, 1024), color=(0, 200, 0)):
    buf = io.BytesIO()
    PIL.Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeSender:
    def __init__(self):
        self.documents, self.animations, self.videos = [], [], []

    async def send_document(self, path, caption=None, **kw):
        self.documents.append((path, caption))

    async def send_animation(self, path, caption=None):
        self.animations.append((path, caption))

    async def send_video(self, path, caption=None):
        self.videos.append((path, caption))


class FakeImages:
    def __init__(self, payload=None, error=None, edit_payload=None):
        self.payload = payload if payload is not None else gif_bytes()
        self.edit_payload = edit_payload
        self.error = error
        self.generate_calls, self.edit_calls = [], []

    def _result(self, payload):
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(payload).decode())])

    def generate(self, **kw):
        self.generate_calls.append(kw)
        if self.error:
            raise self.error(kw)
        return self._result(self.payload)

    def edit(self, **kw):
        kw = dict(kw)
        image = kw.pop("image")
        kw["image_bytes"] = image.read() if not isinstance(image, list) else [i.read() for i in image]
        self.edit_calls.append(kw)
        if self.error:
            raise self.error(kw)
        return self._result(self.edit_payload if self.edit_payload is not None else self.payload)


@pytest.fixture(autouse=True)
def forget_unsupported_knobs():
    from agents import image_tools

    image_tools._UNSUPPORTED.clear()
    yield
    image_tools._UNSUPPORTED.clear()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "7" / "images").mkdir(parents=True)
    return tmp_path


def make_tools(monkeypatch, fake_images, settings=None):
    sender = FakeSender()
    tools = ImageTools("7", sender, settings)
    monkeypatch.setattr("agents.image_tools.openai_client", lambda: SimpleNamespace(images=fake_images))
    return tools, sender


def images_dir(workspace):
    return workspace / "data" / "7" / "images"


# ---- sizes and limits -------------------------------------------------------------
def test_gif_frames_keep_a_512_short_edge_and_follow_the_ratio():
    assert gif_pixel_size("1:1") == "512x512"
    assert gif_pixel_size("16:9") == "896x512"  # the size the API returns for 16:9
    assert gif_pixel_size("9:16") == "512x896"
    assert gif_pixel_size("nonsense") == "512x512"
    for ratio in models.IMAGE_ASPECT_RATIOS:
        w, h = (int(x) for x in gif_pixel_size(ratio).split("x"))
        assert min(w, h) == 512 and w % 64 == 0 and h % 64 == 0, ratio


def test_animation_models_are_their_own_pair_and_make_gifs():
    assert models.GPT_ANIMATION_MODEL == "gpt-image-2.6-sunburst"
    assert models.GPT_ANIMATION_MODEL_FAST == "gpt-image-2.6-flare"
    assert models.resolve_animation_backend("fast").model == models.GPT_ANIMATION_MODEL_FAST
    assert models.resolve_animation_backend("story").model == models.GPT_ANIMATION_MODEL  # no Gemini GIFs
    assert all(b.supports("gif") for b in models.ANIMATION_BACKENDS.values())
    assert (models.GIF_FRAMES_MIN, models.GIF_FRAMES_MAX, models.GIF_FRAMES_DEFAULT) == (2, 16, 9)


# ---- generate_animation -------------------------------------------------------------
async def test_a_gif_is_requested_with_frames_and_timing_and_plays_in_the_chat(workspace, monkeypatch):
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {
        "prompt": "a cat waving, seamless loop", "aspect_ratio": "16:9", "frames": 12, "frame_duration_ms": 123,
    })

    call = images.generate_calls[0]
    assert call["model"] == models.GPT_ANIMATION_MODEL and call["output_format"] == "gif" and call["n"] == 1
    assert call["frame_count"] == 12 and call["frame_duration_ms"] == 120  # GIF stores centiseconds
    assert call["size"] == "896x512" and "input_fidelity" not in call and "complexity" not in call
    assert not images.edit_calls

    # one animation, no document; the caption says what came back (the fake returns 9 frames)
    assert len(sender.animations) == 1 and not sender.documents
    path, caption = sender.animations[0]
    assert path.endswith(".gif") and "sunburst_anim_" in path
    assert "9 frames · 200 ms · 512x512" in caption and models.GPT_ANIMATION_MODEL in caption
    assert "9 frames, 512x512, 200 ms per frame, 1.8 s per loop, loops forever" in result
    assert "Frame sheet: images/" in result and "edit_animation" in result

    sheets = list(images_dir(workspace).glob("*_frames.png"))
    assert len(sheets) == 1
    with PIL.Image.open(sheets[0]) as sheet:
        assert sheet.width > 512 and sheet.height > 512  # a 3x3 grid of the frames


async def test_defaults_and_clamps(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images, {"engine": "story", "quality": "high"})
    await tools.execute_tool("generate_animation", {"prompt": "x"})
    await tools.execute_tool("generate_animation", {"prompt": "x", "frames": 99, "frame_duration_ms": 5, "engine": "fast", "complexity": "low"})
    await tools.execute_tool("generate_animation", {"prompt": "x", "frames": 1, "frame_duration_ms": 99999})
    first, second, third = images.generate_calls
    assert (first["frame_count"], first["frame_duration_ms"], first["size"]) == (9, 200, "512x512")
    assert first["model"] == models.GPT_ANIMATION_MODEL and first["quality"] == "high"  # a story preference means the default
    assert (second["frame_count"], second["frame_duration_ms"]) == (16, 20)
    assert second["model"] == models.GPT_ANIMATION_MODEL_FAST and second["complexity"] == "low"
    assert (third["frame_count"], third["frame_duration_ms"]) == (2, 2000)
    assert (await tools.execute_tool("generate_animation", {"prompt": " "})).startswith("Error:")


async def test_animating_a_photo_uses_edits_and_keeps_its_shape(workspace, monkeypatch):
    PIL.Image.new("RGB", (800, 600), (90, 90, 90)).save(images_dir(workspace) / "photo.jpg", format="JPEG")
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "make it wave", "images": ["images/photo.jpg"]})

    assert not images.generate_calls and len(images.edit_calls) == 1
    call = images.edit_calls[0]
    assert call["output_format"] == "gif" and call["frame_count"] == 9
    assert call["size"] == "704x512"  # 4:3 like the photo, not a square
    assert result.startswith("Animated the given picture") and len(sender.animations) == 1

    missing = await tools.execute_tool("generate_animation", {"prompt": "x", "images": ["images/nope.jpg"]})
    assert missing.startswith("Error:") and "workspace" in missing


async def test_a_refused_photo_animation_points_to_the_alternatives(workspace, monkeypatch):
    PIL.Image.new("RGB", (64, 64)).save(images_dir(workspace) / "photo.png")
    images = FakeImages(error=lambda kw: ValueError("output_format 'gif' is not supported for image edits"))
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "x", "images": ["images/photo.png"]})
    assert result.startswith("Error:") and "without images" in result and "edit_animation" in result
    assert "image_to_video_generator" not in result  # never steer an animation request to the video tools
    assert len(images.edit_calls) == 1 and not sender.animations


@pytest.mark.parametrize("error, needle", [
    (lambda kw: PermissionError("GIF output requires access to the GIF preview for this model."), "GIF preview to be enabled"),
    (lambda kw: TypeError("Images.generate() got an unexpected keyword argument 'frame_count'"), "openai>=3.46.0"),
])
async def test_account_and_sdk_problems_are_explained(workspace, monkeypatch, error, needle):
    images = FakeImages(error=error)
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "x"})
    assert result.startswith("Error:") and needle in result
    assert len(images.generate_calls) == 1  # frame_count is the request's meaning, never dropped


async def test_a_refused_gif_size_falls_back_to_auto_once(workspace, monkeypatch):
    def error(kw):
        return ValueError("The requested size must satisfy the model's GIF size limits.") if kw["size"] != "auto" else None

    class Picky(FakeImages):
        def generate(self, **kw):
            self.generate_calls.append(kw)
            if error(kw):
                raise error(kw)
            return self._result(self.payload)

    images = Picky()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "x", "aspect_ratio": "21:9"})
    assert [c["size"] for c in images.generate_calls] == ["1216x512", "auto"]
    assert "used auto instead" in result and len(sender.animations) == 1


async def test_a_still_image_instead_of_a_gif_is_delivered_as_a_file(workspace, monkeypatch):
    images = FakeImages(payload=png_bytes())
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "x"})
    assert "instead of a GIF" in result and len(sender.documents) == 1 and not sender.animations


async def test_a_model_without_gif_output_is_refused_before_any_call(workspace, monkeypatch):
    monkeypatch.setitem(
        models.ANIMATION_BACKENDS, "auto", replace(models.ANIMATION_BACKENDS["auto"], model="gpt-image-2.5-sunburst")
    )
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_animation", {"prompt": "x"})
    assert result.startswith("Error:") and "GPT_ANIMATION_MODEL" in result and not images.generate_calls


async def test_animation_usage_reaches_the_trace(workspace, monkeypatch):
    from agents.trace import ToolTrace

    class Priced(FakeImages):
        def _result(self, payload):
            result = super()._result(payload)
            result.usage = SimpleNamespace(
                input_tokens=50, output_tokens=1835,
                input_tokens_details=SimpleNamespace(text_tokens=50, image_tokens=0),
                output_tokens_details=SimpleNamespace(image_tokens=1835, text_tokens=0),
            )
            return result

    tools, _ = make_tools(monkeypatch, Priced())
    tools.trace = ToolTrace()
    await tools.execute_tool("generate_animation", {"prompt": "x"})
    expected = (50 * 5 + 1835 * 30) / 1_000_000  # ~$0.055, what a 10-frame GIF cost at launch
    assert abs(tools.trace.model_cost(models.GPT_ANIMATION_MODEL) - expected) < 1e-9


# ---- edit_animation ------------------------------------------------------------------
def write_gif(workspace, name="loop.gif", **kw):
    path = images_dir(workspace) / name
    path.write_bytes(gif_bytes(**kw))
    return f"images/{name}"


async def test_local_changes_make_a_new_gif_and_keep_the_original(workspace, monkeypatch):
    source = write_gif(workspace, count=8, duration=200)
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("edit_animation", {"path": source, "frames": "1-5", "boomerang": True, "speed": 2})

    assert not images.generate_calls and not images.edit_calls  # free: no model call
    assert "kept frames 1-5, boomerang, speed x2" in result
    assert "8 frames" in result and "100 ms per frame" in result  # 5 forward + 3 back, twice as fast
    out = sender.animations[0][0]
    assert out.endswith(".gif") and "loop_edit_" in out
    assert (images_dir(workspace) / "loop.gif").read_bytes() == gif_bytes(count=8, duration=200)
    info = anim.read_gif(out)
    assert len(info.frames) == 8 and set(info.durations) == {100}

    # an edit of an edit is named after the original, not loop_edit_x_edit_y
    rel = "images/" + out.rsplit("/", 1)[-1]
    await tools.execute_tool("edit_animation", {"path": rel, "reverse": True, "frame_duration_ms": 300, "loop": 1})
    second = sender.animations[1][0].rsplit("/", 1)[-1]
    assert second.startswith("loop_edit_") and second.count("_edit_") == 1
    again = anim.read_gif(sender.animations[1][0])
    assert set(again.durations) == {300} and again.loop is None  # plays once


async def test_bad_requests_are_refused_with_a_reason(workspace, monkeypatch):
    source = write_gif(workspace, count=4)
    (images_dir(workspace) / "still.png").write_bytes(png_bytes((8, 8)))
    tools, sender = make_tools(monkeypatch, FakeImages())
    for args, needle in (
        ({"path": source}, "nothing to change"),
        ({"path": "images/nope.gif", "reverse": True}, "not a file"),
        ({"path": "images/still.png", "reverse": True}, "neither a GIF nor a video"),
        ({"path": source, "frames": "2-9"}, "outside 1-4"),
        ({"path": source, "frames": "3"}, "at least 2 frames"),
        ({"path": source, "speed": "fast"}, "speed must be a number"),
        ({"path": source, "loop": -1}, "cannot be negative"),
        ({"path": source, "edit_frame": 2}, "needs edit_prompt"),
        ({"path": source, "edit_frame": 7, "edit_prompt": "x"}, "between 1 and 4"),
    ):
        result = await tools.execute_tool("edit_animation", args)
        assert result.startswith("Error:") and needle in result, (args, result)
    assert not sender.animations


async def test_one_frame_is_redrawn_by_the_model_and_fitted_back(workspace, monkeypatch):
    source = write_gif(workspace, count=4, size=(512, 512))
    images = FakeImages(edit_payload=png_bytes((1024, 1024), color=(255, 0, 0)))
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("edit_animation", {"path": source, "edit_frame": 3, "edit_prompt": "make the sky red"})

    call = images.edit_calls[0]
    assert call["model"] == models.GPT_IMAGE_MODEL and call["output_format"] == "png" and call["size"] == "1024x1024"
    assert "frame 3 of 4" in call["prompt"] and call["prompt"].startswith("make the sky red")
    with PIL.Image.open(io.BytesIO(call["image_bytes"])) as sent:
        assert sent.size == (512, 512) and sent.format == "PNG"
    assert "redrew frame 3" in result and "Frame 3 was redrawn" in result

    info = anim.read_gif(sender.animations[0][0])
    assert info.size == (512, 512) and len(info.frames) == 4
    red = info.frames[2].convert("RGB").getpixel((256, 256))
    assert red[0] > 200 and red[1] < 60  # the redrawn frame is in place
    assert info.frames[1].convert("RGB").getpixel((256, 256))[0] < 100  # its neighbours are untouched
    assert not list(images_dir(workspace).glob("frame_input_*"))


@ffmpeg
async def test_mp4_copy_and_video_to_gif(workspace, monkeypatch):
    source = write_gif(workspace, count=6, size=(320, 240), duration=100)
    tools, sender = make_tools(monkeypatch, FakeImages())
    result = await tools.execute_tool("edit_animation", {"path": source, "output": "mp4"})
    assert len(sender.videos) == 1 and sender.videos[0][0].endswith(".mp4") and "MP4" in result
    assert not list(images_dir(workspace).glob("loop_edit_*.gif"))  # nothing changed, so no second GIF
    video = sender.videos[0][0]
    assert (workspace / "data" / "7" / "videos").is_dir()

    result = await tools.execute_tool("edit_animation", {"path": "videos/" + video.rsplit("/", 1)[-1]})
    assert not result.startswith("Error"), result
    assert "made from the first 10 s of the video" in result and len(sender.animations) == 1
    info = anim.read_gif(sender.animations[0][0])
    assert len(info.frames) >= 2 and max(info.size) <= 512
    assert not list(images_dir(workspace).glob(".convert_*"))


# ---- agents/animation.py ----------------------------------------------------------------
def test_frame_selection_parsing():
    assert anim.parse_frame_selection("1-3", 9) == [0, 1, 2]
    assert anim.parse_frame_selection(" 1, 3 ,5 ", 9) == [0, 2, 4]
    assert anim.parse_frame_selection("7-", 9) == [6, 7, 8]
    assert anim.parse_frame_selection("-2", 9) == [0, 1]
    for bad in ("", "0-3", "5-2", "a-b", "10", "4"):
        with pytest.raises(anim.AnimationError):
            anim.parse_frame_selection(bad, 9)


def test_timing_loop_and_order_operations(tmp_path):
    path = tmp_path / "a.gif"
    path.write_bytes(gif_bytes(count=5, duration=200))
    a = anim.read_gif(path)
    assert (len(a.frames), a.durations, a.loop) == (5, [200] * 5, 0)
    assert anim.clamp_duration(123) == 120 and anim.clamp_duration(1) == 20 and anim.clamp_duration(10**6) == 2000

    slow = anim.retime(a, speed=0.5)
    assert slow.durations == [400] * 5
    capped = anim.retime(a, speed=20)
    assert capped.durations == [20] * 5 and capped.notes  # the limit is reported
    assert anim.retime(a, frame_duration_ms=70, speed=3).durations == [70] * 5  # a fixed delay wins

    back = anim.boomerang(a)
    assert len(back.frames) == 8  # 0 1 2 3 4 3 2 1: no frame twice in a row, loops cleanly
    colors = [f.getpixel((0, 0)) for f in back.frames]
    assert all(colors[i] != colors[i + 1] for i in range(len(colors) - 1)) and colors[-1] != colors[0]
    assert [f.getpixel((0, 0)) for f in anim.reverse(a).frames] == [f.getpixel((0, 0)) for f in a.frames][::-1]

    assert anim.set_loop(a, 0).loop == 0 and anim.set_loop(a, 1).loop is None and anim.set_loop(a, 3).loop == 2
    out = anim.write_gif(anim.set_loop(slow, 3), tmp_path / "b.gif")
    again = anim.read_gif(out)
    assert again.durations == [400] * 5 and again.loop == 2
    assert "plays 3 times" in again.describe()


def test_frame_sheet_numbers_every_frame(tmp_path):
    path = tmp_path / "a.gif"
    path.write_bytes(gif_bytes(count=16, size=(512, 512)))
    sheet = anim.frame_sheet(anim.read_gif(path), tmp_path / "sheet.png")
    with PIL.Image.open(sheet) as img:
        assert img.format == "PNG" and max(img.size) <= 1024 + 5 * 4  # a 4x4 grid within the cap


def test_a_huge_gif_is_scaled_down_while_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(anim, "MAX_TOTAL_PIXELS", 4 * 100 * 100)
    path = tmp_path / "big.gif"
    path.write_bytes(gif_bytes(count=4, size=(400, 300)))
    a = anim.read_gif(path)
    assert len(a.frames) == 4 and a.size[0] * a.size[1] * 4 <= 4 * 100 * 100
    assert a.size[0] / a.size[1] == pytest.approx(4 / 3, rel=0.05) and "scaled down" in a.notes[0]


def test_not_a_gif_is_a_readable_error(tmp_path):
    path = tmp_path / "x.gif"
    path.write_bytes(png_bytes((4, 4)))
    with pytest.raises(anim.AnimationError, match="not a GIF"):
        anim.read_gif(path)


# ---- delivery ------------------------------------------------------------------------------
async def test_chat_sender_sends_animations_and_can_keep_a_gif_a_gif():
    from bot.sender import ChatSender

    class Bot:
        def __init__(self):
            self.calls = []

        async def send_animation(self, **kw):
            self.calls.append(("send_animation", kw))

        async def send_document(self, **kw):
            self.calls.append(("send_document", kw))

    bot = Bot()
    sender = ChatSender(bot, 5, thread_id=9)
    await sender.send_animation(b"GIF89a", filename="a.gif", caption="hi")
    await sender.send_document(b"GIF89a", filename="a.gif", keep_format=True)
    await sender.send_document(b"x", filename="a.txt")
    (name1, kw1), (name2, kw2), (name3, kw3) = bot.calls
    assert name1 == "send_animation" and kw1["caption"] == "hi" and kw1["message_thread_id"] == 9
    assert kw2["disable_content_type_detection"] is True
    assert "disable_content_type_detection" not in kw3


async def test_sending_a_gif_file_keeps_it_a_gif(tmp_path, monkeypatch):
    from agents.file_ops import FileOperations

    monkeypatch.chdir(tmp_path)
    sent = []

    class Sender:
        async def send_document(self, path, caption=None, **kw):
            sent.append((path, kw))

    ops = FileOperations("7", Sender())
    (tmp_path / "data" / "7" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "7" / "images" / "a.gif").write_bytes(gif_bytes(count=2))
    (tmp_path / "data" / "7" / "images" / "a.png").write_bytes(png_bytes((4, 4)))
    for name in ("images/a.gif", "images/a.png"):
        result = await ops.execute_tool("send_file_path_to_user_via_telegram", {"file_path": name})
        assert "Successfully" in result, result
    assert sent[0][1] == {"keep_format": True} and sent[1][1] == {}


def _animation_description(monkeypatch, video_on):
    monkeypatch.setattr("agents.image_tools.VIDEO_TOOLS_ENABLED", video_on)
    tools = ImageTools("7", FakeSender())
    return next(t for t in tools.tools_schema if t["name"] == "generate_animation")["description"]


def test_animation_requests_go_to_generate_animation_not_the_video_tools(workspace, monkeypatch):
    """"Make an animation of this picture" picked image_to_video_generator first: both claimed to animate pictures."""
    monkeypatch.setenv("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY") or "test-key")  # built at import time
    from agents.video_tools import VideoTools

    animation = _animation_description(monkeypatch, True)
    assert "whenever the user asks for an animation" in animation and "explicitly asks for a video" in animation

    video = {t["name"]: t["description"] for t in VideoTools("7", FakeSender()).all_tools_schema}
    for name in ("video_generator", "image_to_video_generator"):
        assert "generate_animation" in video[name] and "Use only when the user" in video[name], name
    assert "animates it" not in video["image_to_video_generator"]


async def test_dead_video_tools_are_hidden_and_video_requests_get_an_honest_answer(workspace, monkeypatch):
    """Veo is gone from the Gemini API (404 on every call): listed, the agent still "tried the video route
    first" even after a 404 in the same chat, and every try sent two chat messages."""
    monkeypatch.setenv("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY") or "test-key")
    from agents import video_tools

    assert models.VIDEO_TOOLS_ENABLED is False  # the default while no video model answers
    tools = video_tools.VideoTools("7", FakeSender())
    assert tools.tools_schema == [] and len(tools.all_tools_schema) == 5
    result = await tools.execute_tool("image_to_video_generator", {"prompt": "x", "image_path": "images/a.jpg"})
    assert result == video_tools.VIDEO_UNAVAILABLE and "generate_animation" in result

    off = _animation_description(monkeypatch, False)
    assert "video generation is not available" in off and "video tools" not in off

    monkeypatch.setattr(video_tools, "VIDEO_TOOLS_ENABLED", True)
    assert [t["name"] for t in video_tools.VideoTools("7", FakeSender()).tools_schema][:2] == [
        "video_generator", "image_to_video_generator",
    ]
