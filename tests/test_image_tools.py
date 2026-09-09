"""The consolidated image tool: engine resolution, provider calls, delivery and results."""
import base64
from types import SimpleNamespace

import pytest

import models
from agents.image_tools import ImageTools, extension_for, is_quality_rejection, resolve_pixel_size

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
JPG = b"\xff\xd8\xff\xe0" + b"0" * 32
WEBP = b"RIFF" + b"0000" + b"WEBP" + b"0" * 32


class FakeSender:
    def __init__(self):
        self.documents = []
        self.texts = []

    async def send_document(self, path, caption=None):
        self.documents.append((path, caption))

    async def send_markdown(self, text):
        self.texts.append(text)


class FakeImages:
    """Stands in for client.images: records calls, returns canned base64 payloads."""

    def __init__(self, payloads=(PNG,), error=None):
        self.payloads = list(payloads)
        self.error = error
        self.generate_calls = []
        self.edit_calls = []

    def _result(self):
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(p).decode()) for p in self.payloads])

    def generate(self, **kw):
        self.generate_calls.append(kw)
        if self.error:
            raise self.error(kw)
        return self._result()

    def edit(self, **kw):
        kw = dict(kw)
        kw["image_count"] = 1 if not isinstance(kw.get("image"), list) else len(kw["image"])
        kw.pop("image", None)
        self.edit_calls.append(kw)
        if self.error:
            raise self.error(kw)
        return self._result()


@pytest.fixture(autouse=True)
def forget_unsupported_knobs():
    """The "this model refuses that knob" memo is process-wide; no test may leak into the next."""
    from agents import image_tools

    image_tools._UNSUPPORTED.clear()
    yield
    image_tools._UNSUPPORTED.clear()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    images = tmp_path / "data" / "7" / "images"
    images.mkdir(parents=True)
    (images / "cat.png").write_bytes(PNG)
    (tmp_path / "data" / "8").mkdir(parents=True)
    (tmp_path / "data" / "8" / "secret.png").write_bytes(PNG)
    return tmp_path


def make_tools(monkeypatch, fake_images, settings=None):
    sender = FakeSender()
    tools = ImageTools("7", sender, settings)
    monkeypatch.setattr("agents.image_tools.openai_client", lambda: SimpleNamespace(images=fake_images))
    return tools, sender


# ---- pure helpers ------------------------------------------------------------
def test_extension_is_derived_from_the_bytes():
    assert extension_for(PNG) == "png"
    assert extension_for(JPG) == "jpg"
    assert extension_for(WEBP) == "webp"
    assert extension_for(b"nonsense", fallback="jpeg") == "jpeg"


def test_resolve_pixel_size_snaps_and_clamps():
    assert resolve_pixel_size("2K", "16:9") == "2048x1152"
    assert resolve_pixel_size("1K", "1:1") == "1024x1024"
    for size in ("1K", "2K", "4K"):
        for ratio in models.IMAGE_ASPECT_RATIOS:
            w, h = (int(x) for x in resolve_pixel_size(size, ratio).split("x"))
            assert w % 16 == 0 and h % 16 == 0
            assert max(w, h) <= 3840
            assert 655_360 <= w * h <= 8_294_400, (size, ratio, w, h)
    # a malformed ratio degrades to square instead of raising
    assert resolve_pixel_size("2K", "1:0") == resolve_pixel_size("2K", "1:1")
    assert resolve_pixel_size("2K", "nonsense") == resolve_pixel_size("2K", "1:1")


def test_backend_resolution_and_gemini_quality_escalation():
    assert models.resolve_image_backend("auto").model == models.GPT_IMAGE_MODEL
    assert models.resolve_image_backend("best").model == models.GPT_IMAGE_MODEL
    assert models.resolve_image_backend("fast").model == models.GPT_IMAGE_MODEL_FAST
    assert models.resolve_image_backend("FAST").model == models.GPT_IMAGE_MODEL_FAST
    assert models.resolve_image_backend("nonsense").engine == "auto"  # unknown falls back
    story = models.resolve_image_backend("story")
    assert story.provider == "gemini" and story.model == models.GEMINI_IMAGE_MODEL_FLASH
    # Gemini has no quality knob, so effort above "high" buys the Pro model instead
    assert models.resolve_image_backend("story", "max").model == models.GEMINI_IMAGE_MODEL_PRO
    assert models.resolve_image_backend("story", "high").model == models.GEMINI_IMAGE_MODEL_FLASH


def test_quality_rejection_detection():
    assert is_quality_rejection(ValueError("Invalid value for 'quality': 'max'"))
    assert not is_quality_rejection(ValueError("model not found"))


# ---- generation --------------------------------------------------------------
async def test_generate_sends_once_and_reports_resolved_settings(workspace, monkeypatch):
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)

    result = await tools.execute_tool("generate_image", {"prompt": "a cat", "quality": "high", "aspect_ratio": "16:9"})

    call = images.generate_calls[0]
    assert call["model"] == models.GPT_IMAGE_MODEL and call["quality"] == "high"
    assert call["size"] == "2048x1152" and call["n"] == 1 and call["output_format"] == "png"
    assert not images.edit_calls
    # delivered exactly once, as a document, with the parameters in the caption
    assert len(sender.documents) == 1
    path, caption = sender.documents[0]
    assert path.endswith(".png") and "sunburst_" in path
    assert models.GPT_IMAGE_MODEL in caption and "quality high" in caption and "2048x1152" in caption
    # the result records how it was made and offers the inline embed conditionally
    assert "Generated 1 image(s)" in result and models.GPT_IMAGE_MODEL in result
    assert "Saved: images/" in result
    assert "Only if they asked" in result and "![caption](images/" in result


async def test_tool_call_overrides_the_user_preference(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images, {"engine": "fast", "quality": "low", "size": "1K"})

    await tools.execute_tool("generate_image", {"prompt": "x"})
    assert images.generate_calls[0]["model"] == models.GPT_IMAGE_MODEL_FAST  # preference honoured
    assert images.generate_calls[0]["quality"] == "low"
    assert images.generate_calls[0]["size"] == "1024x1024"

    await tools.execute_tool("generate_image", {"prompt": "x", "engine": "best", "quality": "max"})
    assert images.generate_calls[1]["model"] == models.GPT_IMAGE_MODEL  # the agent may override it
    assert images.generate_calls[1]["quality"] == "max"


async def test_variants_are_one_call_and_clamped_by_the_user_cap(workspace, monkeypatch):
    images = FakeImages(payloads=(PNG, JPG))
    tools, sender = make_tools(monkeypatch, images, {"max_variants": 2})

    await tools.execute_tool("generate_image", {"prompt": "x", "variants": 9})
    assert len(images.generate_calls) == 1 and images.generate_calls[0]["n"] == 2
    assert [c for _, c in sender.documents] == [
        c for _, c in sender.documents if "variant" in c
    ]  # every caption is numbered when there is more than one
    assert sender.documents[0][0].endswith(".png") and sender.documents[1][0].endswith(".jpg")  # extension per bytes

    for bad in (0, -3, None, "x"):
        images.generate_calls.clear()
        await tools.execute_tool("generate_image", {"prompt": "x", "variants": bad})
        assert images.generate_calls[0]["n"] == 1


async def test_unverified_quality_falls_back_once(workspace, monkeypatch):
    seen = []

    def error(kw):
        seen.append(kw["quality"])
        return ValueError("Invalid value for 'quality'") if kw["quality"] == "max" else None

    class Flaky(FakeImages):
        def generate(self, **kw):
            self.generate_calls.append(kw)
            err = error(kw)
            if err:
                raise err
            return self._result()

    images = Flaky()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "x", "quality": "max"})

    assert seen == ["max", models.IMAGE_QUALITY_FALLBACK]  # retried once, at the known-good tier
    assert "not supported" in result and models.IMAGE_QUALITY_FALLBACK in result
    assert len(sender.documents) == 1


async def test_a_real_error_is_reported_not_retried(workspace, monkeypatch):
    images = FakeImages(error=lambda kw: RuntimeError("content policy violation"))
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "x", "quality": "max"})
    assert len(images.generate_calls) == 1  # not a quality problem, so no retry
    assert result.startswith("Error:") and "content policy" in result
    assert not sender.documents


# ---- editing and composition -------------------------------------------------
async def test_editing_uses_the_edit_endpoint_with_the_input(workspace, monkeypatch):
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "make it blue", "images": ["images/cat.png"]})
    assert not images.generate_calls and len(images.edit_calls) == 1
    assert images.edit_calls[0]["image_count"] == 1 and images.edit_calls[0]["output_format"] == "png"
    assert "Edited 1 image(s)" in result and len(sender.documents) == 1


async def test_composition_passes_every_input_in_one_call(workspace, monkeypatch):
    (workspace / "data" / "7" / "images" / "dog.png").write_bytes(PNG)
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool(
        "generate_image", {"prompt": "together", "images": ["images/cat.png", "images/dog.png"], "variants": 3}
    )
    assert len(images.edit_calls) == 1  # one call for all variants, inputs uploaded once
    assert images.edit_calls[0]["image_count"] == 2 and images.edit_calls[0]["n"] == 3
    assert "Composed" in result


async def test_missing_and_escaping_inputs_are_refused(workspace, monkeypatch):
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images)
    for bad in ("images/nope.png", "../8/secret.png", "/etc/passwd"):
        result = await tools.execute_tool("generate_image", {"prompt": "x", "images": [bad]})
        assert result.startswith("Error:") and "workspace" in result, bad
    assert not images.edit_calls and not sender.documents


async def test_too_many_inputs_for_the_engine_is_refused_before_any_call(workspace, monkeypatch):
    for i in range(4):
        (workspace / "data" / "7" / "images" / f"p{i}.png").write_bytes(PNG)
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool(
        "generate_image", {"prompt": "x", "engine": "story", "images": [f"images/p{i}.png" for i in range(4)]}
    )
    assert result.startswith("Error:") and "at most 3" in result
    assert not images.edit_calls


# ---- schema and dispatch -----------------------------------------------------
def test_schema_is_two_tools_and_advertises_the_preference(workspace):
    tools = ImageTools("7", FakeSender(), {"engine": "fast", "quality": "low"})
    names = [t["name"] for t in tools.tools_schema]
    assert names == ["generate_image", "generate_story_images"]
    schema = tools.tools_schema[0]
    assert schema["input_schema"]["required"] == ["prompt"]
    assert set(schema["input_schema"]["properties"]) == {
        "prompt", "images", "engine", "quality", "aspect_ratio", "size", "variants", "format", "caption",
    }
    assert "'fast'" in schema["description"] and "'low'" in schema["description"]
    assert "only embed" in schema["description"].lower()


async def test_unknown_arguments_are_ignored_and_unknown_tool_reported(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    # an argument the signature does not accept used to raise TypeError before any handler ran
    result = await tools.execute_tool("generate_image", {"prompt": "x", "style": "vivid", "nonsense": 1})
    assert not result.startswith("Error:") and images.generate_calls
    assert await tools.execute_tool("nope", {}) == "Unknown tool: nope"
    assert (await tools.execute_tool("generate_image", {"prompt": "  "})).startswith("Error:")


async def test_metadata_can_be_switched_off(workspace, monkeypatch):
    images = FakeImages()
    tools, sender = make_tools(monkeypatch, images, {"metadata": False})
    await tools.execute_tool("generate_image", {"prompt": "x", "caption": "My picture"})
    assert sender.documents[0][1] == "My picture"


def test_the_sandbox_patcher_does_not_rebind_image_tool_methods():
    """Image tools need API access, so they run in the bot process and must not be monkey-patched.

    A patcher that replaced a method here would silently impose its own signature: that is how
    ``_generate_image`` once lost its ``images`` argument and every edit became a fresh generation.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    assert "ImageTools" not in source and "image_tools" not in source, (
        "tool_integrator.py touches the image tools again; a patched method must keep the real signature"
    )


# ---- input fidelity ----------------------------------------------------------
async def test_edits_ask_the_provider_to_stay_faithful_to_the_input(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)

    await tools.execute_tool("generate_image", {"prompt": "add a hat", "images": ["images/cat.png"]})
    assert images.edit_calls[0]["input_fidelity"] == "high"

    # there is nothing to be faithful to when generating from scratch
    await tools.execute_tool("generate_image", {"prompt": "a hat"})
    assert "input_fidelity" not in images.generate_calls[0]


async def test_a_model_that_refuses_input_fidelity_is_retried_and_remembered(workspace, monkeypatch):
    class Picky(FakeImages):
        def edit(self, **kw):
            super().edit(**kw)
            if "input_fidelity" in kw:
                raise ValueError("Unknown parameter: 'input_fidelity'")
            return self._result()

    images = Picky()
    tools, sender = make_tools(monkeypatch, images)

    result = await tools.execute_tool("generate_image", {"prompt": "x", "images": ["images/cat.png"]})
    assert [("input_fidelity" in c) for c in images.edit_calls] == [True, False]
    assert "Edited 1 image(s)" in result and len(sender.documents) == 1

    # the discovery costs one round trip in total, not one per image
    images.edit_calls.clear()
    await tools.execute_tool("generate_image", {"prompt": "y", "images": ["images/cat.png"]})
    assert [("input_fidelity" in c) for c in images.edit_calls] == [False]


# ---- preparing the input -----------------------------------------------------
def _write(path, mode, size, fmt):
    import PIL.Image

    PIL.Image.new(mode, size, (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)).save(path, format=fmt)
    return path


def test_an_acceptable_input_is_uploaded_untouched(workspace, monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_RESOLUTION_EDIT", "256")
    tools = ImageTools("7", FakeSender())
    temp = []
    for name, mode, fmt in (("real.png", "RGB", "PNG"), ("real.jpg", "RGB", "JPEG"), ("real.webp", "RGB", "WEBP")):
        source = _write(workspace / "data" / "7" / "images" / name, mode, (64, 64), fmt)
        assert tools._prepare_input(source, temp) == source  # no re-encode, no quality lost
    assert temp == []


def test_an_oversized_input_is_downscaled_and_transparency_survives_conversion(workspace, monkeypatch):
    import PIL.Image

    monkeypatch.setenv("MAX_IMAGE_RESOLUTION_EDIT", "64")
    tools = ImageTools("7", FakeSender())

    temp = []
    big = _write(workspace / "data" / "7" / "images" / "big.jpg", "RGB", (200, 100), "JPEG")
    prepared = tools._prepare_input(big, temp)
    assert prepared != big and temp == [prepared] and prepared.suffix == ".jpg"
    with PIL.Image.open(prepared) as img:
        assert max(img.size) <= 64

    temp = []
    logo = _write(workspace / "data" / "7" / "images" / "logo.tiff", "RGBA", (32, 32), "TIFF")
    prepared = tools._prepare_input(logo, temp)
    assert prepared.suffix == ".png"  # flattening a logo onto white would ruin a composition
    with PIL.Image.open(prepared) as img:
        assert img.mode == "RGBA"
