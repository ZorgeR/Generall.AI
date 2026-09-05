from aiogram.exceptions import TelegramBadRequest, TelegramNotFound
from aiogram.methods import SendMessage

from bot import rich

_M = SendMessage(chat_id=1, text="x")


def test_unsupported_error_detection():
    assert rich.is_unsupported_error(TelegramNotFound(method=_M, message="Not Found"))
    assert rich.is_unsupported_error(TelegramBadRequest(method=_M, message="Bad Request: method not found"))
    assert not rich.is_unsupported_error(TelegramBadRequest(method=_M, message="Bad Request: can't parse entities"))
    assert not rich.is_unsupported_error(ValueError("x"))


def test_split_markdown_respects_byte_limit_and_keeps_all_text():
    text = "\n\n".join(f"P{i} " + "слово " * 500 for i in range(30))  # multibyte
    chunks = rich.split_markdown(text)
    assert len(chunks) > 1
    assert all(len(c.encode("utf-8")) <= rich.RICH_BYTE_LIMIT for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks).replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_short_text_is_not_split():
    assert rich.split_markdown("hello") == ["hello"]


def test_halve_markdown_splits_at_paragraph():
    text = "a\n\nb\n\nc\n\nd"
    halves = rich.halve_markdown(text)
    assert len(halves) == 2 and "".join(halves).replace("\n", "") == "abcd"
    assert rich.halve_markdown("single") == ["single"]


def test_html_message_converts_tables_and_headings():
    msg = rich.html_message("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```py\nprint(1)\n```")
    assert msg is not None and msg.html
    assert "<table>" in msg.html and "<pre>" in msg.html


def test_markdown_v2_chunks_escape_and_split():
    pieces = rich.markdown_v2_chunks("snake_case and a.dot\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert len(pieces) == 1
    assert "snake\\_case" in pieces[0] and "a\\.dot" in pieces[0]
    long = "\n\n".join("line_with_underscores " * 100 for _ in range(30))
    pieces = rich.markdown_v2_chunks(long)
    assert len(pieces) > 1 and all(len(p) <= rich.MDV2_MAX for p in pieces)


def test_unescape_markdown_v2_roundtrip():
    assert rich.unescape_markdown_v2("a\\_b\\.c\\*d") == "a_b.c*d"


def test_drafts_are_bounded():
    big = "x" * 100_000
    assert len(rich.text_draft(big).markdown.encode()) <= rich.DRAFT_BYTE_LIMIT
    t = rich.thinking_draft("<think> & more")
    assert t.html.startswith("<tg-thinking>") and "&lt;think&gt; &amp;" in t.html
