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
        self.animations = []
        self.videos = []
        self.texts = []

    async def send_document(self, path, caption=None, **kw):
        self.documents.append((path, caption))

    async def send_animation(self, path, caption=None):
        self.animations.append((path, caption))

    async def send_video(self, path, caption=None):
        self.videos.append((path, caption))

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
        if "markup" in kw:  # the upload is closed after the call: keep what was sent
            kw["markup"] = kw["markup"].read()
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


def use_model(monkeypatch, engine, model):
    """Point one engine at another model id (the capabilities follow the id's family)."""
    from dataclasses import replace

    monkeypatch.setitem(models.IMAGE_BACKENDS, engine, replace(models.IMAGE_BACKENDS[engine], model=model))


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
def test_schema_is_four_tools_and_advertises_the_preference(workspace):
    tools = ImageTools("7", FakeSender(), {"engine": "fast", "quality": "low"})
    names = [t["name"] for t in tools.tools_schema]
    assert names == ["generate_image", "generate_animation", "edit_animation", "generate_story_images"]
    schema = tools.tools_schema[0]
    assert schema["input_schema"]["required"] == ["prompt"]
    assert set(schema["input_schema"]["properties"]) == {
        "prompt", "images", "engine", "quality", "aspect_ratio", "size", "variants", "format",
        "complexity", "regions", "markup", "caption",
    }
    assert "svg" in schema["input_schema"]["properties"]["format"]["enum"]
    assert "gif" not in schema["input_schema"]["properties"]["format"]["enum"]  # GIFs are generate_animation's
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
    use_model(monkeypatch, "auto", "gpt-image-2.5-sunburst")  # the knob belongs to the models before 2.6
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)

    await tools.execute_tool("generate_image", {"prompt": "add a hat", "images": ["images/cat.png"]})
    assert images.edit_calls[0]["input_fidelity"] == "high"

    # there is nothing to be faithful to when generating from scratch
    await tools.execute_tool("generate_image", {"prompt": "a hat"})
    assert "input_fidelity" not in images.generate_calls[0]


async def test_gpt_image_2_6_edits_omit_input_fidelity(workspace, monkeypatch):
    """"For gpt-image-2, omit this parameter": a 2.6 edit never sends it, not even once."""
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    await tools.execute_tool("generate_image", {"prompt": "add a hat", "images": ["images/cat.png"]})
    assert images.edit_calls[0]["model"] == models.GPT_IMAGE_MODEL
    assert "input_fidelity" not in images.edit_calls[0]


async def test_a_model_that_refuses_input_fidelity_is_retried_and_remembered(workspace, monkeypatch):
    class Picky(FakeImages):
        def edit(self, **kw):
            super().edit(**kw)
            if "input_fidelity" in kw:
                raise ValueError("Unknown parameter: 'input_fidelity'")
            return self._result()

    use_model(monkeypatch, "auto", "gpt-image-2.5-sunburst")
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


async def test_image_usage_and_cost_reach_the_turn_trace(workspace, monkeypatch):
    from agents.trace import ToolTrace

    class PricedImages(FakeImages):
        def _result(self):
            result = super()._result()
            result.usage = SimpleNamespace(
                input_tokens=1_000_100, output_tokens=1_000_000,
                input_tokens_details=SimpleNamespace(text_tokens=100, image_tokens=1_000_000),
                output_tokens_details=SimpleNamespace(image_tokens=1_000_000, text_tokens=0),
            )
            return result

    tools, _ = make_tools(monkeypatch, PricedImages())
    tools.trace = ToolTrace()
    await tools.execute_tool("generate_image", {"prompt": "a cat"})

    bucket = tools.trace.usage_by_model[models.GPT_IMAGE_MODEL]
    assert bucket["api_calls"] == 1 and bucket["input_tokens"] == 1_000_100 and bucket["output_tokens"] == 1_000_000
    # 100 text tokens at $5, 1M image input at $8, 1M image output at $30 per 1M
    expected = (100 * 5 + 1_000_000 * 8 + 1_000_000 * 30) / 1_000_000
    assert abs(tools.trace.model_cost(models.GPT_IMAGE_MODEL) - expected) < 1e-9
    assert abs(tools.trace.cost_usd - expected) < 1e-9


async def test_no_trace_or_no_usage_is_not_an_error(workspace, monkeypatch):
    tools, sender = make_tools(monkeypatch, FakeImages())
    assert tools.trace is None
    await tools.execute_tool("generate_image", {"prompt": "a cat"})
    assert len(sender.documents) == 1


async def test_gemini_usage_and_cost_reach_the_turn_trace(workspace, monkeypatch):
    from agents.trace import ToolTrace

    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=PNG, mime_type="image/png"), text=None)]))],
        usage_metadata=SimpleNamespace(prompt_token_count=1_000_000, candidates_token_count=900_000, thoughts_token_count=100_000),
    )
    calls = []

    def generate_content(**kw):
        calls.append(kw)
        return response

    tools, _ = make_tools(monkeypatch, FakeImages())
    monkeypatch.setattr("agents.image_tools.genai_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    tools.trace = ToolTrace()
    await tools.execute_tool("generate_image", {"prompt": "a dragon", "engine": "story", "variants": 2})

    bucket = tools.trace.usage_by_model[models.GEMINI_IMAGE_MODEL_FLASH]
    assert len(calls) == 2 and bucket["api_calls"] == 2
    assert bucket["input_tokens"] == 2_000_000 and bucket["output_tokens"] == 2_000_000  # thinking counted as output
    assert abs(tools.trace.model_cost(models.GEMINI_IMAGE_MODEL_FLASH) - 2 * (0.5 + 60.0)) < 1e-9


# ---- GPT Image 2.6: capabilities, SVG, complexity, knobs, sizes, markup -------
def test_2_6_is_the_default_and_capabilities_follow_the_model_family():
    assert models.GPT_IMAGE_MODEL == "gpt-image-2.6-sunburst"
    assert models.GPT_IMAGE_MODEL_FAST == "gpt-image-2.6-flare"
    features = models.image_model_features
    assert features("openai", "gpt-image-2.6-flare") >= {"gif", "svg", "complexity", "markup"}
    assert "input_fidelity" not in features("openai", "gpt-image-2.6-sunburst")
    assert features("openai", "gpt-image-2.6-sunburst-2026-11-04") == features("openai", "gpt-image-2.6-sunburst")  # snapshots
    assert features("openai", "gpt-image-2.5-sunburst") == {"markup", "input_fidelity"}
    assert features("openai", "gpt-image-1.5") == {"input_fidelity"}  # unknown family: the old behaviour
    assert features("gemini", models.GEMINI_IMAGE_MODEL_FLASH) == frozenset()
    assert models.resolve_image_backend("best").supports("gif")
    assert not models.resolve_image_backend("story").supports("markup")


def test_extension_recognises_svg_and_gif():
    assert extension_for(b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>') == "svg"
    assert extension_for(b"  <svg viewBox='0 0 1 1'></svg>") == "svg"
    assert extension_for(b"<?xml version='1.0'?><note/>", fallback="png") == "png"
    assert extension_for(b"GIF89a" + b"0" * 10) == "gif"


SVG = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'


async def test_svg_is_one_new_vector_drawing_delivered_as_a_file(workspace, monkeypatch):
    images = FakeImages(payloads=(SVG,))
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "a fox logo", "format": "svg", "variants": 3})

    call = images.generate_calls[0]
    assert call["output_format"] == "svg" and call["n"] == 1
    assert sender.documents[0][0].endswith(".svg")
    assert "only one was made" in result and "without a preview" in result
    assert "![caption]" not in result  # a vector file is never offered for inline embedding

    for args, needle in (
        ({"prompt": "x", "format": "svg", "images": ["images/cat.png"]}, "only makes new drawings"),
        ({"prompt": "x", "format": "svg", "engine": "story"}, "cannot make SVG"),
        ({"prompt": "x", "format": "gif"}, "generate_animation"),
    ):
        result = await tools.execute_tool("generate_image", args)
        assert result.startswith("Error:") and needle in result, args
    assert len(images.generate_calls) == 1 and not images.edit_calls


async def test_svg_author_usage_reaches_the_trace(workspace, monkeypatch):
    from agents.trace import ToolTrace

    class SvgImages(FakeImages):
        def _result(self):
            result = super()._result()
            result.svg_usage = SimpleNamespace(model="gpt-6.1-sol", input_tokens=1_000_000, output_tokens=1_000_000, reasoning_tokens=0)
            return result

    tools, _ = make_tools(monkeypatch, SvgImages(payloads=(SVG,)))
    tools.trace = ToolTrace()
    await tools.execute_tool("generate_image", {"prompt": "logo", "format": "svg"})
    assert tools.trace.usage_by_model["gpt-6.1-sol"]["output_tokens"] == 1_000_000
    assert tools.trace.model_cost("gpt-6.1-sol") == models.estimate_cost("gpt-6.1-sol", 1_000_000, 1_000_000)


async def test_complexity_is_sent_only_when_asked_and_supported(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    await tools.execute_tool("generate_image", {"prompt": "x"})
    await tools.execute_tool("generate_image", {"prompt": "x", "complexity": "LOW"})
    assert "complexity" not in images.generate_calls[0]  # the model's own default
    assert images.generate_calls[1]["complexity"] == "low"

    use_model(monkeypatch, "auto", "gpt-image-2.5-sunburst")
    result = await tools.execute_tool("generate_image", {"prompt": "x", "complexity": "low"})
    assert "complexity" not in images.generate_calls[2] and "no complexity setting" in result


async def test_safety_identifier_is_a_stable_hash_and_dropped_when_refused(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    await tools.execute_tool("generate_image", {"prompt": "x"})
    sent = images.generate_calls[0]["safety_identifier"]
    assert sent == tools.safety_identifier == ImageTools("7", FakeSender()).safety_identifier
    assert "7" != sent and len(sent) == 32 and sent != ImageTools("8", FakeSender()).safety_identifier

    class Refuses(FakeImages):
        def generate(self, **kw):
            self.generate_calls.append(kw)
            if "safety_identifier" in kw:
                raise ValueError("Unknown parameter: 'safety_identifier'.")
            return self._result()

    images = Refuses()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "x"})
    assert [("safety_identifier" in c) for c in images.generate_calls] == [True, False]
    assert not result.startswith("Error") and "refused" not in result  # not worth telling the agent
    await tools.execute_tool("generate_image", {"prompt": "y"})
    assert "safety_identifier" not in images.generate_calls[2]  # remembered


async def test_a_refused_custom_size_falls_back_to_the_closest_standard_size_once(workspace, monkeypatch):
    class StandardOnly(FakeImages):
        def generate(self, **kw):
            self.generate_calls.append(kw)
            if kw["size"] not in ("1024x1024", "1536x1024", "1024x1536"):
                raise ValueError(f"Invalid size '{kw['size']}'. Supported sizes are 1024x1024, 1536x1024, 1024x1536.")
            return self._result()

    images = StandardOnly()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {"prompt": "x", "aspect_ratio": "16:9"})
    assert [c["size"] for c in images.generate_calls] == ["2048x1152", "1536x1024"]
    assert "used 1536x1024 instead" in result and "1536x1024" in sender.documents[0][1]

    images.generate_calls.clear()
    await tools.execute_tool("generate_image", {"prompt": "x", "aspect_ratio": "16:9"})
    assert [c["size"] for c in images.generate_calls] == ["1536x1024"]  # no second wasted round trip

    images.generate_calls.clear()
    result = await tools.execute_tool("generate_image", {"prompt": "x", "size": "1K"})  # 1024x1024 is standard
    assert [c["size"] for c in images.generate_calls] == ["1024x1024"] and not result.startswith("Error")


def _photo(path, size=(400, 300), color=(200, 200, 200)):
    import PIL.Image

    PIL.Image.new("RGB", size, color).save(path, format="JPEG", quality=95)
    return path


async def test_regions_become_a_transparent_overlay_the_size_of_the_input(workspace, monkeypatch):
    import io

    import PIL.Image

    _photo(workspace / "data" / "7" / "images" / "room.jpg")
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {
        "prompt": "make the lamp red", "images": ["images/room.jpg"],
        "regions": [{"x": 0.25, "y": 0.5, "width": 0.25, "height": 0.25}],
    })
    assert not result.startswith("Error"), result
    with PIL.Image.open(io.BytesIO(images.edit_calls[0]["markup"])) as overlay:
        assert overlay.format == "PNG" and overlay.size == (400, 300) and overlay.mode == "RGBA"
        alpha = overlay.getchannel("A")
        assert alpha.getpixel((0, 0)) == 0  # transparent outside the marks
        assert alpha.getpixel((100, 150)) == 255 and overlay.getpixel((100, 150))[:3] == (255, 0, 0)  # the outline
        assert alpha.getpixel((125, 187)) == 0  # inside the box stays visible
    assert not list((workspace / "data" / "7" / "images").glob("markup_*"))  # the overlay is temporary


async def test_circles_and_the_meaning_of_marks(workspace, monkeypatch):
    import io

    import PIL.Image

    from agents.image_tools import parse_regions

    assert parse_regions([{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}])[0][4] == "box"
    assert parse_regions([{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2, "shape": "Oval"}])[0][4] == "circle"

    _photo(workspace / "data" / "7" / "images" / "room.jpg")
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    await tools.execute_tool("generate_image", {
        "prompt": "keep the circled face exactly as it is, make the background a beach", "images": ["images/room.jpg"],
        "regions": [{"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5, "shape": "circle"}],
    })
    with PIL.Image.open(io.BytesIO(images.edit_calls[0]["markup"])) as overlay:
        alpha = overlay.getchannel("A")
        assert alpha.getpixel((100, 75)) == 0  # the box corner: a circle leaves it clear
        assert alpha.getpixel((200, 75)) == 255  # the top of the circle
        assert alpha.getpixel((200, 150)) == 0  # the marked area itself is not painted over

    description = tools.tools_schema[0]["description"]
    assert "keep it exactly while changing the rest" in description  # marks point; the prompt says what they mean


async def test_a_photo_the_user_drew_on_becomes_the_markup(workspace, monkeypatch):
    import io

    import PIL.Image
    import PIL.ImageDraw

    folder = workspace / "data" / "7" / "images"
    _photo(folder / "room.jpg")
    drawn = PIL.Image.open(folder / "room.jpg").convert("RGB")
    PIL.ImageDraw.Draw(drawn).ellipse((300, 50, 380, 130), outline=(0, 0, 255), width=6)
    drawn.save(folder / "room_marked.jpg", format="JPEG", quality=90)

    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {
        "prompt": "remove the circled object", "images": ["images/room.jpg"], "markup": "images/room_marked.jpg",
    })
    assert not result.startswith("Error"), result
    with PIL.Image.open(io.BytesIO(images.edit_calls[0]["markup"])) as overlay:
        alpha = overlay.getchannel("A")
        box = alpha.getbbox()
        assert box is not None and 290 <= box[0] <= 302 and 378 <= box[2] <= 392  # only the drawing
        assert alpha.getpixel((10, 10)) == 0 and alpha.getpixel((200, 250)) == 0

    # the same photo unchanged marks nothing: refused before any call
    images.edit_calls.clear()
    result = await tools.execute_tool("generate_image", {
        "prompt": "x", "images": ["images/room.jpg"], "markup": "images/room.jpg",
    })
    assert result.startswith("Error") and "nothing is marked" in result and not images.edit_calls


async def test_markup_needs_an_input_image_and_a_model_that_takes_it(workspace, monkeypatch):
    images = FakeImages()
    tools, _ = make_tools(monkeypatch, images)
    region = [{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}]
    for args, needle in (
        ({"prompt": "x", "regions": region}, "pass that image"),
        ({"prompt": "x", "images": ["images/cat.png"], "regions": region, "engine": "story"}, "does not take markup"),
        ({"prompt": "x", "images": ["images/cat.png"], "regions": [{"x": "a"}]}, "numeric"),
        ({"prompt": "x", "images": ["images/cat.png"], "markup": "images/nope.png"}, "not an image"),
    ):
        result = await tools.execute_tool("generate_image", args)
        assert result.startswith("Error:") and needle in result, args
    assert not images.edit_calls and not images.generate_calls


async def test_a_refused_markup_is_dropped_and_reported(workspace, monkeypatch):
    _photo(workspace / "data" / "7" / "images" / "room.jpg")

    class NoMarkup(FakeImages):
        def edit(self, **kw):
            super().edit(**kw)
            if "markup" in kw:
                raise ValueError("Unsupported parameter: 'markup' is not supported with this model.")
            return self._result()

    images = NoMarkup()
    tools, sender = make_tools(monkeypatch, images)
    result = await tools.execute_tool("generate_image", {
        "prompt": "x", "images": ["images/room.jpg"], "regions": [{"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}],
    })
    assert [("markup" in c) for c in images.edit_calls] == [True, False]
    assert "refused 'markup'" in result and len(sender.documents) == 1


def test_the_sandbox_patcher_does_not_touch_the_animation_helpers():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    assert "animation" not in source.lower()
