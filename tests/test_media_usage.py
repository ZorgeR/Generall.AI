"""OpenAI photo / video-frame descriptions record their token usage and estimated cost."""
import sys
from types import SimpleNamespace

import pytest

from bot import media


class FakeTracker:
    def __init__(self):
        self.rows = []

    def track_usage(self, user_id, **kwargs):
        self.rows.append((user_id, kwargs))


@pytest.fixture
def tracker(monkeypatch):
    fake = FakeTracker()
    monkeypatch.setitem(sys.modules, "stats", SimpleNamespace(stats_tracker=fake))
    return fake


def _client(usage):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="a cat"))],
        usage=usage,
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))


async def test_photo_description_records_usage(tracker, monkeypatch, tmp_path):
    image = tmp_path / "x.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(media, "openai_client", lambda: _client(SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=200_000)))

    assert await media.describe_image_openai("what is it?", str(image), user_id="42") == "a cat"

    [(user_id, row)] = tracker.rows
    assert user_id == "42" and row["model"] == media.OPENAI_MODEL and row["api_calls"] == 1
    assert row["input_tokens"] == 1_000_000 and row["output_tokens"] == 200_000
    assert abs(row["cost_usd"] - (2.0 + 2.0)) < 1e-9  # gpt-6.1-sol: $2 in, $10 out per 1M


async def test_video_frames_record_usage(tracker, monkeypatch, tmp_path):
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(media, "openai_client", lambda: _client(SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000)))

    assert await media.describe_video_screenshots([str(frame)], user_id="7") == "a cat"

    [(user_id, row)] = tracker.rows
    assert user_id == "7" and row["model"] == media.VIDEO_FRAMES_MODEL
    assert abs(row["cost_usd"] - (0.1 + 0.5)) < 1e-9  # gpt-6-luna: $0.1 in, $0.5 out per 1M


async def test_no_user_or_no_usage_records_nothing(tracker, monkeypatch, tmp_path):
    image = tmp_path / "x.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(media, "openai_client", lambda: _client(SimpleNamespace(prompt_tokens=10, completion_tokens=5)))
    await media.describe_image_openai("q", str(image))
    monkeypatch.setattr(media, "openai_client", lambda: _client(None))
    await media.describe_image_openai("q", str(image), user_id="42")
    assert tracker.rows == []
