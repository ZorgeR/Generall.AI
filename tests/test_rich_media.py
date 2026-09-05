"""Inline images: Markdown ![]() in an answer becomes rich-message media or separate photos."""
from types import SimpleNamespace

import pytest
from aiogram.exceptions import TelegramNotFound
from aiogram.methods import SendMessage
from aiogram.types import FSInputFile

from bot import rich
from bot.sender import ChatSender


class FakeBot:
    def __init__(self, rich_error=None):
        self.calls = []
        self.rich_error = rich_error
        self._id = 0

    def _msg(self):
        self._id += 1
        return SimpleNamespace(message_id=self._id)

    async def send_rich_message(self, **kw):
        self.calls.append(("send_rich_message", kw))
        if self.rich_error:
            raise self.rich_error
        return self._msg()

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return self._msg()

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))
        return self._msg()

    async def delete_message(self, **kw):
        self.calls.append(("delete_message", kw))
        return True

    async def send_photo(self, **kw):
        self.calls.append(("send_photo", kw))
        return self._msg()

    async def send_video(self, **kw):
        self.calls.append(("send_video", kw))
        return self._msg()

    def names(self):
        return [n for n, _ in self.calls]


@pytest.fixture(autouse=True)
def _reset():
    rich.reset()
    yield
    rich.reset()


@pytest.fixture
def workspace(tmp_path):
    base = tmp_path / "data" / "42"
    (base / "images").mkdir(parents=True)
    (base / "images" / "cat.jpg").write_bytes(b"\xff\xd8 jpeg")
    (base / "videos").mkdir()
    (base / "videos" / "clip.mp4").write_bytes(b"mp4")
    (tmp_path / "data" / "41").mkdir()
    (tmp_path / "data" / "41" / "private.jpg").write_bytes(b"secret")
    return base


TEXT = (
    "Here is the cat:\n\n![a cat](images/cat.jpg)\n\nand a clip ![clip](clip.mp4), "
    "a web picture ![web](https://example.com/p.png), a missing one ![gone](images/gone.jpg), "
    "someone else's ![leak](../41/private.jpg) and a text file ![notes](notes.txt)."
)


def test_extract_media_resolves_only_safe_existing_media(workspace):
    ex = rich.extract_media(TEXT, workspace)
    assert [(i.id, i.kind) for i in ex.items] == [("m1", "photo"), ("m2", "video"), ("m3", "photo")]
    assert ex.items[0].source == (workspace / "images" / "cat.jpg").resolve()
    assert ex.items[1].source == (workspace / "videos" / "clip.mp4").resolve()  # bare name found in videos/
    assert ex.items[2].source == "https://example.com/p.png"
    assert "![a cat](tg://photo?id=m1)" in ex.rich_text and "![clip](tg://video?id=m2)" in ex.rich_text
    for stripped in ("gone.jpg", "private.jpg", "notes.txt", "images/cat.jpg"):
        assert stripped not in ex.rich_text
    assert "gone" in ex.rich_text and "leak" in ex.rich_text and "notes" in ex.rich_text  # captions survive
    assert "tg://" not in ex.plain_text and "a cat" in ex.plain_text
    media = ex.input_media()
    assert [m.id for m in media] == ["m1", "m2", "m3"]
    assert isinstance(media[0].media.media, FSInputFile) and media[2].media.media == "https://example.com/p.png"


def test_no_media_root_means_only_urls(workspace):
    ex = rich.extract_media(TEXT, None)
    assert [i.kind for i in ex.items] == ["photo"] and ex.items[0].source.startswith("https://")


def test_rich_send_carries_media_inline(workspace):
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=True, media_root=workspace)
    import asyncio

    asyncio.run(sender.send_markdown("Look ![a cat](images/cat.jpg) now"))
    assert bot.names() == ["send_rich_message"]
    rm = bot.calls[0][1]["rich_message"]
    assert rm.markdown == "Look ![a cat](tg://photo?id=m1) now"
    assert rm.media[0].id == "m1" and isinstance(rm.media[0].media.media, FSInputFile)


async def test_fallback_tiers_send_pictures_separately(workspace):
    bot = FakeBot(rich_error=TelegramNotFound(method=SendMessage(chat_id=1, text="x"), message="Not Found"))
    sender = ChatSender(bot, 42, rich=True, media_root=workspace)
    await sender.send_markdown("Look ![a cat](images/cat.jpg) now", edit=SimpleNamespace(message_id=9))
    assert bot.names() == ["send_rich_message", "send_message", "send_photo", "delete_message"]
    assert bot.calls[1][1]["parse_mode"] == "MarkdownV2" and "tg://" not in bot.calls[1][1]["text"]
    photo = bot.calls[2][1]
    assert isinstance(photo["photo"], FSInputFile) and photo["caption"] == "a cat" and photo["chat_id"] == 42


async def test_legacy_mode_sends_text_then_photo(workspace):
    bot = FakeBot()
    sender = ChatSender(bot, 42, rich=False, media_root=workspace)
    await sender.send_markdown("Look ![a cat](images/cat.jpg) now", edit=SimpleNamespace(message_id=9))
    assert bot.names() == ["edit_message_text", "send_photo"]
    assert bot.calls[0][1]["text"] == "Look a cat now"


async def test_media_send_failure_does_not_break_the_answer(workspace):
    class Flaky(FakeBot):
        async def send_photo(self, **kw):
            raise RuntimeError("photo too large")

    bot = Flaky()
    sender = ChatSender(bot, 42, rich=False, media_root=workspace)
    sent = await sender.send_markdown("![a cat](images/cat.jpg)")
    assert len(sent) == 1 and bot.names() == ["send_message"]
