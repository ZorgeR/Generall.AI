"""VisionTools (F7): view_image returns the picture as a block, describe_image asks another
model, read_document attaches a PDF or extracts text; paths stay inside the workspace; the
sandbox patcher never touches the class."""
import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agents import content
from agents.vision_tools import VisionTools

USER = "7"


def png(path: Path, width: int, height: int, mode="RGB") -> None:
    Image.new(mode, (width, height), (10, 200, 10) if mode == "RGB" else (10, 200, 10, 0)).save(path, format="PNG")


def pdf(path: Path, pages: int) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "data" / USER
    (base / "images").mkdir(parents=True)
    (base / "documents").mkdir()
    png(base / "images" / "cat.png", 2000, 1000)
    png(base / "images" / "small.png", 100, 50)
    png(base / "images" / "glass.png", 300, 300, mode="RGBA")
    pdf(base / "documents" / "r.pdf", 2)
    (base / "documents" / "notes.txt").write_text("alpha beta gamma delta")
    (tmp_path / "data" / "8").mkdir()
    png(tmp_path / "data" / "8" / "secret.png", 10, 10)
    return base


def decoded(block: dict) -> Image.Image:
    assert block["type"] == "image" and block["source"]["type"] == "base64"
    return Image.open(io.BytesIO(base64.standard_b64decode(block["source"]["data"])))


# ---- view_image -------------------------------------------------------------------
async def test_view_image_downscales_and_reports_the_real_dimensions(workspace):
    tools = VisionTools(USER)
    result = await tools.execute_tool("view_image", {"path": "images/cat.png"})
    assert isinstance(result, list) and len(result) == 2
    with decoded(result[0]) as img:
        assert img.format == "JPEG" and img.size == (1024, 512)
    assert result[0]["source"]["media_type"] == "image/jpeg"
    assert result[1] == {"type": "text", "text": "images/cat.png · 2000x1000 px (shown at 1024x512)"}
    assert len(base64.standard_b64decode(result[0]["source"]["data"])) <= content.MAX_IMAGE_BYTES


async def test_view_image_passes_a_small_png_through_unchanged(workspace):
    result = await VisionTools(USER).view_image("images/small.png")
    assert result[0]["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(result[0]["source"]["data"]) == (workspace / "images" / "small.png").read_bytes()
    assert result[1]["text"] == "images/small.png · 100x50 px"


async def test_view_image_max_px_and_bare_names_and_transparency(workspace):
    tools = VisionTools(USER)
    result = await tools.execute_tool("view_image", {"path": "cat.png", "max_px": 200})  # looked up in images/
    with decoded(result[0]) as img:
        assert max(img.size) == 200
    result = await tools.view_image("/home/runner/workspace/images/cat.png", max_px=5)  # clamped to the minimum
    with decoded(result[0]) as img:
        assert max(img.size) == 128
    result = await tools.view_image("images/glass.png", max_px=100)  # transparent PNG becomes a flat JPEG
    with decoded(result[0]) as img:
        assert img.format == "JPEG" and img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)


async def test_view_image_refuses_paths_outside_the_workspace(workspace):
    tools = VisionTools(USER)
    for bad in ("../8/secret.png", "/etc/passwd", "data/8/secret.png", str((workspace.parent / "8" / "secret.png").resolve())):
        result = await tools.execute_tool("view_image", {"path": bad})
        assert isinstance(result, str) and result.startswith("Error:"), bad
    assert (await tools.view_image("images/nope.png")).startswith("Error: 'images/nope.png' is not a file")
    assert (await tools.view_image("")).startswith("Error:")
    not_image = await tools.view_image("documents/notes.txt")
    assert not_image.startswith("Error: 'documents/notes.txt' could not be read as an image")


# ---- describe_image ---------------------------------------------------------------
async def test_describe_image_prepares_a_small_jpeg_and_asks_the_chosen_model(workspace, monkeypatch):
    from bot import media

    seen: list[tuple] = []

    async def anthropic(question, path):
        with Image.open(path) as img:
            seen.append(("anthropic", question, img.format, img.size, path))
        return "a cat"

    async def openai(question, path, user_id=None):
        seen.append(("openai", question, user_id, path))
        return "a tabby cat"

    monkeypatch.setattr(media, "describe_image_anthropic", anthropic)
    monkeypatch.setattr(media, "describe_image_openai", openai)
    tools = VisionTools(USER)

    assert await tools.execute_tool("describe_image", {"path": "images/cat.png"}) == "a cat"
    kind, question, fmt, size, path = seen[0]
    assert question == "Describe what is in this image in detail." and fmt == "JPEG" and size == (1024, 512)
    assert not Path(path).exists()  # the temporary copy is removed

    assert await tools.execute_tool("describe_image", {"path": "cat.png", "question": "breed?", "model": "openai"}) == "a tabby cat"
    assert seen[1][:3] == ("openai", "breed?", USER)
    assert (await tools.describe_image("images/cat.png", model="gemini")).startswith("Error: unknown model")
    assert (await tools.describe_image("images/none.png")).startswith("Error:")


async def test_describe_image_reports_model_failures_as_errors(workspace, monkeypatch):
    from bot import media

    monkeypatch.setattr(media, "describe_image_anthropic", AsyncMock(side_effect=RuntimeError("quota")))
    result = await VisionTools(USER).execute_tool("describe_image", {"path": "images/cat.png"})
    assert result == "Error: quota"


# ---- read_document ------------------------------------------------------------------
async def test_read_document_small_pdf_is_a_document_block(workspace):
    result = await VisionTools(USER).execute_tool("read_document", {"path": "documents/r.pdf"})
    assert isinstance(result, list) and len(result) == 2
    assert result[0]["type"] == "document" and result[0]["title"] == "r.pdf"
    assert result[0]["source"]["media_type"] == content.PDF_MEDIA_TYPE
    assert base64.standard_b64decode(result[0]["source"]["data"]) == (workspace / "documents" / "r.pdf").read_bytes()
    assert result[1] == {"type": "text", "text": "r.pdf · 2 pages"}


async def test_read_document_large_pdf_and_text_files_come_back_as_text(workspace, monkeypatch):
    from bot import media
    from agents import vision_tools

    tools = VisionTools(USER)
    monkeypatch.setattr(media, "PDF_MAX_PAGES", 1)
    result = await tools.read_document("r.pdf")  # bare name: documents/ is a fallback dir
    assert isinstance(result, str) and result.startswith("documents/r.pdf · 2 pages (too large to attach")
    assert "--- page 1 ---" in result and "--- page 2 ---" in result

    result = await tools.execute_tool("read_document", {"path": "documents/notes.txt"})
    assert result == "documents/notes.txt\n\nalpha beta gamma delta"
    monkeypatch.setattr(vision_tools, "EXTRACTED_TEXT_CAP", 5)
    result = await tools.read_document("documents/notes.txt")
    assert result.startswith("documents/notes.txt\n\nalpha\n…(truncated")


async def test_read_document_unknown_types_go_to_describe_document(workspace, monkeypatch):
    from bot import media

    (workspace / "documents" / "data.bin").write_bytes(b"\x00\x01")
    describe = AsyncMock(return_value="binary stuff")
    monkeypatch.setattr(media, "describe_document", describe)
    assert await VisionTools(USER).read_document("documents/data.bin", question="what?") == "binary stuff"
    assert describe.await_args.args == ("what?", str((workspace / "documents" / "data.bin").resolve()))
    assert (await VisionTools(USER).read_document("../8/secret.png")).startswith("Error:")


# ---- contract -------------------------------------------------------------------------
def test_schema_notes_and_unknown_tool():
    tools = VisionTools(USER, sender=None)
    assert [t["name"] for t in tools.tools_schema] == ["view_image", "describe_image", "read_document"]
    for t in tools.tools_schema:
        assert t["input_schema"]["required"] == ["path"] and t["description"]
    for name in ("view_image", "describe_image", "read_document"):
        assert name in VisionTools.prompt_notes
    assert "image/document blocks" in VisionTools.prompt_notes


async def test_unknown_tool_and_errors_are_strings(workspace):
    tools = VisionTools(USER)
    assert await tools.execute_tool("nothing", {}) == "Unknown tool: nothing"
    assert (await tools.execute_tool("view_image", {})).startswith("Error:")


async def test_registered_provider_returns_blocks_through_the_agent(workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    import agents.main as m

    monkeypatch.setattr(m.stats_tracker, "track_tool_used", lambda *a, **k: None)
    agent = m.AgentAnthropic(user_id=USER)
    agent.register_provider(VisionTools(USER))
    assert [t["name"] for t in agent.get_tools_schema()] == ["view_image", "describe_image", "read_document"]
    result = await agent.execute_tool("view_image", {"path": "images/small.png"})
    assert isinstance(result, list) and result[0]["type"] == "image"


def test_the_sandbox_patcher_never_names_the_vision_tools():
    """VisionTools reads files and calls models from the bot process; a patcher that rebound a
    method would impose its own signature (see the image tools history in CLAUDE.md)."""
    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    for name in ("VisionTools", "vision_tools", "view_image", "describe_image", "read_document"):
        assert name not in source, f"tool_integrator.py names {name}"
