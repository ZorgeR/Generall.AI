from bot.sender import MAX_CAPTION, _caption, split_text_intelligently


def test_short_text_is_one_chunk():
    assert split_text_intelligently("hello", 10) == ["hello"]


def test_chunks_respect_limit_and_reassemble():
    paragraphs = ["para " * 300 for _ in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = split_text_intelligently(text, 1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == text
    assert len(chunks) > 1


def test_prefers_paragraph_then_line_then_space():
    text = "a" * 10 + "\n\n" + "b" * 10 + "\n" + "c" * 10 + " " + "d" * 10
    chunks = split_text_intelligently(text, 25)
    assert chunks[0] == "a" * 10 + "\n\n"
    assert "".join(chunks) == text


def test_hard_cut_when_no_boundary():
    text = "x" * 50
    chunks = split_text_intelligently(text, 20)
    assert chunks == ["x" * 20, "x" * 20, "x" * 10]


def test_caption_truncation():
    assert _caption(None) is None
    assert _caption("short") == "short"
    long = "y" * (MAX_CAPTION + 50)
    assert len(_caption(long)) == MAX_CAPTION
    assert _caption(long).endswith("…")
