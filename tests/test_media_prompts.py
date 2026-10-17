"""F7: photos and documents reach the agent as content blocks, not as model descriptions.

``_run_images`` builds [text, image_ref...] with a downscaled copy for the model and the
full-resolution file named in the text; ``_run_document`` attaches a PDF as a document_ref
within the limits, puts text / extracted office text in the prompt (capped) and falls back
to the description path otherwise; ``on_document`` routes PNG/WebP/GIF/BMP to the image job.
"""
import io
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message
from PIL import Image

from bot import media
from bot.handlers import messages
from bot.queue import Job, JobContext
from image_utils import IMAGE_EXTENSIONS

USER = "42"
CHAT = {"id": 42, "type": "private"}
ALICE = {"id": 42, "is_bot": False, "first_name": "Alice"}
PHOTO = [{"file_id": "p1", "file_unique_id": "pu", "width": 10, "height": 10}]


def msg(**fields) -> Message:
    return Message.model_validate({"message_id": 10, "date": 1760700000, "chat": CHAT, "from": ALICE, **fields})


def document(file_id: str, name: str) -> dict:
    return {"file_id": file_id, "file_unique_id": f"{file_id}u", "file_name": name}


def png_bytes(width: int, height: int, color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def pdf_bytes(pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def docx_bytes(text: str) -> bytes:
    import docx

    doc = docx.Document()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


class FakeBot:
    """Records sends/edits; ``download`` writes the payload registered for the file id."""

    id = 777

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[tuple[str, dict]] = []
        self._id = 100

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_chat_action(self, **kw):
        return True

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return self._msg()

    async def download(self, file, destination=None, **kw):
        with open(destination, "wb") as f:
            f.write(self.payloads[file])

    def texts(self) -> list[str]:
        return [kw.get("text", "") for _, kw in self.calls]


class _Stats:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    stats = ModuleType("stats")
    stats.stats_tracker = _Stats()
    monkeypatch.setitem(sys.modules, "stats", stats)
    return stats.stats_tracker


@pytest.fixture
def run_turn(monkeypatch) -> AsyncMock:
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr(messages, "run_turn", mock)
    return mock


@pytest.fixture
def no_describe(monkeypatch) -> dict[str, AsyncMock]:
    """Every description function mocked, so a call shows up as an awaited mock."""
    mocks = {
        "describe_image_anthropic": AsyncMock(return_value="a cat"),
        "describe_image_openai": AsyncMock(return_value="a tabby cat"),
        "describe_document": AsyncMock(return_value="quarterly numbers"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(media, name, mock)
    return mocks


def ctx() -> JobContext:
    return JobContext(Job(user_id=USER, label="test", run=None))


def prompt_of(run_turn: AsyncMock):
    run_turn.assert_awaited_once()
    return run_turn.await_args.kwargs["prompt"]


def describe_calls(stats: _Stats) -> list:
    return [c for c in stats.calls if c[0] == "track_describe_used"]


# ---- photos ------------------------------------------------------------------------
async def test_photo_job_sends_image_blocks_and_describes_nothing(run_turn, no_describe, env, tmp_path):
    bot = FakeBot({"p1": png_bytes(1600, 900)})
    m = msg(photo=PHOTO, caption="which breed?")
    await messages._run_images(bot, m, USER, None, [m.photo[-1]], m.caption, None, ctx())

    prompt = prompt_of(run_turn)
    assert isinstance(prompt, list) and [b["type"] for b in prompt] == ["text", "media_ref"]
    text = prompt[0]["text"]
    assert text.startswith("which breed?\n\nUser attached 1 image(s) to this message.")
    assert "Image 1: data/42/images/image_" in text and messages.IMAGE_EMBED_HINT in text
    ref = prompt[1]
    assert ref["kind"] == "image" and ref["path"].endswith("_v.jpg")

    # the model's copy is downscaled; the file named in the text is the full-resolution JPEG
    full_path = text.split("Image 1: ")[1].split("\n")[0]
    assert full_path.endswith(".jpg") and ref["path"] == full_path[:-4] + "_v.jpg"
    with Image.open(tmp_path / full_path) as full, Image.open(tmp_path / ref["path"]) as small:
        assert full.format == "JPEG" and full.size == (1600, 900)
        assert small.format == "JPEG" and max(small.size) == 1024

    for mock in no_describe.values():
        mock.assert_not_awaited()
    assert describe_calls(env) == []
    assert any("Preparing images" in t for t in bot.texts())
    assert run_turn.await_args.kwargs["ctx"] is not None


async def test_png_document_is_converted_to_jpeg_and_a_lone_album_has_no_caption(run_turn, no_describe, tmp_path):
    bot = FakeBot({"d1": png_bytes(300, 200), "d2": png_bytes(50, 50)})
    m = msg(document=document("d1", "Shot.PNG"))
    refs = [m.document, SimpleNamespace(file_id="d2", file_name="other.webp")]
    await messages._run_images(bot, m, USER, None, refs, None, None, ctx())

    prompt = prompt_of(run_turn)
    assert prompt[0]["text"].startswith("User sent 2 photo(s). They are shown below")
    assert "Image 1: " in prompt[0]["text"] and "Image 2: " in prompt[0]["text"]
    assert len(prompt) == 3 and all(b["type"] == "media_ref" for b in prompt[1:])
    for ref in prompt[1:]:
        with Image.open(tmp_path / ref["path"]) as img:
            assert img.format == "JPEG"
    assert ".png" in IMAGE_EXTENSIONS and ".webp" in IMAGE_EXTENSIONS and ".gif" in IMAGE_EXTENSIONS and ".bmp" in IMAGE_EXTENSIONS


async def test_a_broken_picture_is_reported_and_the_rest_still_go_through(run_turn, no_describe):
    bot = FakeBot({"bad": b"not an image", "p1": png_bytes(40, 40)})
    m = msg(photo=PHOTO)
    await messages._run_images(bot, m, USER, None, [SimpleNamespace(file_id="bad"), m.photo[-1]], "look", None, ctx())
    prompt = prompt_of(run_turn)
    assert "Image 1: could not be processed" in prompt[0]["text"] and "Image 2: data/42/images/" in prompt[0]["text"]
    assert len(prompt) == 2  # one image block for the picture that worked


async def test_no_usable_picture_stops_before_the_agent(run_turn, no_describe):
    bot = FakeBot({"bad": b"not an image"})
    await messages._run_images(bot, msg(photo=PHOTO), USER, None, [SimpleNamespace(file_id="bad")], None, None, ctx())
    run_turn.assert_not_awaited()
    assert any("Could not process" in t for t in bot.texts())


# ---- documents ---------------------------------------------------------------------
async def test_pdf_within_limits_becomes_a_document_ref(run_turn, no_describe, env, tmp_path):
    bot = FakeBot({"d": pdf_bytes(2)})
    m = msg(document=document("d", "Report.PDF"), caption="summarize")
    await messages._run_document(bot, m, USER, None, ctx())

    prompt = prompt_of(run_turn)
    assert isinstance(prompt, list) and len(prompt) == 2
    text = prompt[0]["text"]
    assert text.startswith("User sent a PDF: report.pdf (2 pages), saved at data/42/documents/report.pdf.")
    assert "User message: summarize" in text and "Read it from the attached document block" in text
    assert prompt[1] == {"type": "media_ref", "kind": "document", "path": "data/42/documents/report.pdf", "title": "report.pdf"}
    assert (tmp_path / "data" / "42" / "documents" / "report.pdf").read_bytes() == pdf_bytes(2)
    no_describe["describe_document"].assert_not_awaited()
    assert describe_calls(env) == []


async def test_pdf_over_the_limits_falls_back_to_the_description(run_turn, no_describe, env, monkeypatch):
    monkeypatch.setattr(media, "PDF_MAX_PAGES", 1)
    bot = FakeBot({"d": pdf_bytes(2)})
    m = msg(document=document("d", "big.pdf"), caption="summarize")
    await messages._run_document(bot, m, USER, None, ctx())

    prompt = prompt_of(run_turn)
    assert isinstance(prompt, str)
    assert prompt.startswith("summarize\n\nUser attached a PDF document to this message.\nFile: big.pdf")
    assert "too large to attach directly (2 pages" in prompt and "quarterly numbers" in prompt
    no_describe["describe_document"].assert_awaited_once()
    assert describe_calls(env) == [("track_describe_used", (USER, "pdf"))]


async def test_unreadable_pdf_falls_back_too(run_turn, no_describe):
    bot = FakeBot({"d": b"not a pdf"})
    await messages._run_document(bot, msg(document=document("d", "odd.pdf")), USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert isinstance(prompt, str) and "could not be parsed as a PDF" in prompt and "quarterly numbers" in prompt


async def test_text_document_content_is_in_the_prompt_and_capped(run_turn, no_describe, env, monkeypatch):
    monkeypatch.setattr(messages, "TEXT_PROMPT_CAP", 100)
    body = "".join(f"line {i}\n" for i in range(60))  # > 100 characters
    bot = FakeBot({"d": body.encode()})
    m = msg(document=document("d", "notes.txt"), caption="what is this?")
    await messages._run_document(bot, m, USER, None, ctx())

    prompt = prompt_of(run_turn)
    assert isinstance(prompt, str)
    assert prompt.startswith(f"User sent a TXT file: notes.txt ({len(body)} characters), saved at data/42/documents/notes.txt.")
    assert "User message: what is this?" in prompt
    assert "<file_content>\n" + body[:100] + "\n…(truncated: use read_file for the rest)\n</file_content>" in prompt
    no_describe["describe_document"].assert_not_awaited()
    assert describe_calls(env) == []


async def test_docx_text_is_extracted_and_a_long_one_falls_back(run_turn, no_describe, env, monkeypatch):
    bot = FakeBot({"d": docx_bytes("hello from docx")})
    await messages._run_document(bot, msg(document=document("d", "memo.docx")), USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert isinstance(prompt, str) and prompt.startswith("User sent a DOCX document: memo.docx, saved at data/42/documents/memo.docx.")
    assert "<document_text>\nhello from docx" in prompt
    no_describe["describe_document"].assert_not_awaited()

    run_turn.reset_mock()
    monkeypatch.setattr(messages, "EXTRACTED_TEXT_CAP", 5)
    await messages._run_document(bot, msg(document=document("d", "memo.docx"), caption="gist?"), USER, None, ctx())
    prompt = prompt_of(run_turn)
    assert prompt.startswith("gist?\n\nUser attached a DOCX document") and "too long to include" in prompt
    assert "quarterly numbers" in prompt
    no_describe["describe_document"].assert_awaited_once()
    assert describe_calls(env) == [("track_describe_used", (USER, "docx"))]


async def test_extract_document_text_covers_json_and_xlsx(tmp_path):
    (tmp_path / "a.json").write_text('{"a": 1}')
    assert media.extract_document_text(str(tmp_path / "a.json")) == '{"a": 1}'
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(ValueError):  # json.JSONDecodeError
        media.extract_document_text(str(tmp_path / "bad.json"))
    import pandas as pd

    pd.DataFrame({"x": [1, 2]}).to_excel(tmp_path / "s.xlsx", index=False)
    text = media.extract_document_text(str(tmp_path / "s.xlsx"))
    assert "Sheet: Sheet1" in text and "x" in text
    with pytest.raises(ValueError):
        media.extract_document_text(str(tmp_path / "x.exe"))


# ---- routing -----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["shot.png", "pic.WEBP", "anim.gif", "old.bmp", "photo.jpg", "img.heic"])
async def test_on_document_routes_pictures_to_the_image_job(monkeypatch, name):
    submit = AsyncMock()
    monkeypatch.setattr(messages, "submit", submit)
    m = SimpleNamespace(document=SimpleNamespace(file_name=name, file_id="d"), media_group_id=None, caption=None, answer=AsyncMock())
    await messages.on_document(m, FakeBot({}), USER, None)
    job = submit.await_args.args[1]
    assert job.label == "image file" and job.run.func is messages._run_images
    m.answer.assert_not_awaited()


async def test_on_document_rejects_unknown_types_naming_the_pictures(monkeypatch):
    submit = AsyncMock()
    monkeypatch.setattr(messages, "submit", submit)
    m = SimpleNamespace(document=SimpleNamespace(file_name="setup.exe", file_id="d"), media_group_id=None, caption=None, answer=AsyncMock())
    await messages.on_document(m, FakeBot({}), USER, None)
    submit.assert_not_awaited()
    text = m.answer.await_args.args[0]
    assert text.startswith("❌ Only") and "PNG" in text and "WEBP" in text and "PDF" in text


async def test_on_document_queues_documents_and_videos(monkeypatch):
    submit = AsyncMock()
    monkeypatch.setattr(messages, "submit", submit)
    m = SimpleNamespace(document=SimpleNamespace(file_name="r.pdf", file_id="d"), media_group_id=None, caption=None, answer=AsyncMock())
    await messages.on_document(m, FakeBot({}), USER, None)
    assert submit.await_args.args[1].run.func is messages._run_document
    m = SimpleNamespace(document=SimpleNamespace(file_name="clip.mp4", file_id="v"), media_group_id=None, caption=None, answer=AsyncMock())
    await messages.on_document(m, FakeBot({}), USER, None)
    assert submit.await_args.args[1].run.func is messages._run_video
