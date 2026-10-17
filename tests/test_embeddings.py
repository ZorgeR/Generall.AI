"""agents/embeddings.py: cosine index, model tag, stale detection and the background rebuild.

The OpenAI client is a fake returning deterministic bag-of-words vectors (sha1 of each word
picks a dimension), so related texts share dimensions and rank above unrelated ones.
"""
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from agents import embeddings
from agents.embeddings import ConversationEmbeddings

DIM = 1536
USER = "7"


def fake_vector(text: str, dim: int = DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.sha1(word.encode()).hexdigest(), 16) % dim] += 1.0
    if not v.any():
        v[0] = 1.0
    return v * 3.0  # deliberately not unit length: the module must normalize


class FakeEmbeddingsClient:
    """``client.embeddings.create(model=, input=[...])`` with the OpenAI response shape."""

    def __init__(self, *, gate: threading.Event | None = None, fail: bool = False):
        self.calls: list[list[str]] = []
        self.models: list[str] = []
        self.gate = gate
        self.fail = fail
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, *, model, input):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("embeddings API down")
        self.calls.append(list(input))
        self.models.append(model)
        # reversed order on purpose: the module must sort by .index
        data = [SimpleNamespace(index=i, embedding=fake_vector(t).tolist()) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def write_conversation(root: Path, uid: str, ts: str, topic: str, question: str, response: str, summary: str = "") -> Path:
    folder = root / uid / "conversations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"conversation_{ts}_{topic}.json"
    path.write_text(json.dumps({"timestamp": ts, "thread_id": None, "topic": topic, "summary": summary,
                                "question": question, "response": response, "full_history": []}), encoding="utf-8")
    return path


def write_legacy_index(root: Path, uid: str, n: int = 2) -> None:
    """What the pre-v3 code wrote: an L2 index next to a bare list of items."""
    folder = root / uid / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(DIM)
    index.add(np.stack([fake_vector(f"legacy {i}") for i in range(n)]))
    faiss.write_index(index, str(folder / "faiss_index.bin"))
    (folder / "metadata.json").write_text(json.dumps(
        [{"question": f"legacy {i}", "answer": "old", "timestamp": "t", "index": i} for i in range(n)]))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fresh process-wide state, automatic rebuilds off, instances rooted in tmp_path."""
    embeddings.reset()
    monkeypatch.setattr(embeddings, "AUTO_REINDEX", False)
    root = tmp_path / "data"

    def make(uid: str = USER, client=None, model: str | None = None) -> ConversationEmbeddings:
        return ConversationEmbeddings(uid, client=client or FakeEmbeddingsClient(), data_root=root, model=model)

    yield SimpleNamespace(root=root, make=make)
    embeddings.reset()


def test_constructor_does_no_api_call_and_no_io(env):
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    assert client.calls == []
    assert not env.root.exists()
    assert emb.index is None and emb.stale is False
    assert emb.model == embeddings.models.EMBEDDING_MODEL


def test_add_and_search_ranks_the_right_item(env):
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    emb.add_conversation("How do I bake sourdough bread at home?", "Feed the starter, autolyse, fold, proof, bake hot.", "20260101_100000")
    emb.add_conversation("Explain the asyncio event loop", "It schedules coroutines and callbacks.", "20260102_100000")
    emb.add_conversation("Weather in Paris tomorrow", "Rainy, 14 degrees.", "20260103_100000")

    hits = emb.search_conversations("sourdough bread baking tips", k=10)  # k larger than the index
    assert [h["question"] for h in hits][0] == "How do I bake sourdough bread at home?"
    assert len(hits) == 3  # no FAISS -1 padding leaking through
    assert set(hits[0]) == {"question", "answer", "timestamp", "topic", "score"}
    assert hits[0]["timestamp"] == "20260101_100000"
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    assert hits[0]["score"] > hits[-1]["score"]
    assert client.models[-1] == embeddings.models.EMBEDDING_MODEL
    assert client.calls[0] == ["Question: How do I bake sourdough bread at home?\nAnswer: Feed the starter, autolyse, fold, proof, bake hot."]


def test_vectors_are_normalized_and_layout_is_v2(env):
    emb = env.make()
    emb.add_conversation("Q alpha", "A beta", "t1")
    assert isinstance(emb.index, faiss.IndexFlatIP)
    assert abs(float(np.linalg.norm(emb.index.reconstruct(0))) - 1.0) < 1e-5
    # the embedded text itself as the query is a perfect cosine match (score is clamped cosine)
    assert emb.search_conversations("Question: Q alpha\nAnswer: A beta", k=1)[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < emb.search_conversations("Q alpha", k=1)[0]["score"] < 1.0

    meta = json.loads(emb.metadata_path.read_text(encoding="utf-8"))
    assert meta["version"] == 2 and meta["model"] == emb.model and meta["dimension"] == DIM
    assert [set(i) for i in meta["items"]] == [{"question", "answer", "timestamp", "topic", "text_hash"}]

    fresh = env.make()  # a new instance (next turn) loads the files and is not stale
    assert fresh.search_conversations("Q alpha", k=1)[0]["answer"] == "A beta"
    assert fresh.stale is False


def test_legacy_metadata_is_stale_and_rebuilds_from_conversations(env):
    write_legacy_index(env.root, USER)
    write_conversation(env.root, USER, "20260101_090000", "bread", "How to bake bread?", "Knead and bake.", "Bread baking basics.")
    write_conversation(env.root, USER, "20260102_090000", "python", "What is asyncio?", "An event loop.", "Python asyncio.")
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)

    assert emb.search_conversations("bread", k=3) == []
    assert emb.stale is True
    assert client.calls == []  # a stale index is never queried

    outcome = emb.reindex(blocking=True)
    assert outcome["ok"] is True and outcome["items"] == 2
    assert client.calls == [["Topic: bread\nQuestion: How to bake bread?\nAnswer: Knead and bake.\nSummary: Bread baking basics.",
                             "Topic: python\nQuestion: What is asyncio?\nAnswer: An event loop.\nSummary: Python asyncio."]]
    hits = emb.search_conversations("bake bread", k=3)
    assert hits[0]["question"] == "How to bake bread?" and hits[0]["topic"] == "bread"
    assert emb.stale is False
    meta = json.loads(emb.metadata_path.read_text(encoding="utf-8"))
    assert meta["version"] == 2 and meta["model"] == emb.model and len(meta["items"]) == 2
    assert faiss.read_index(str(emb.index_path)).metric_type == faiss.METRIC_INNER_PRODUCT
    assert embeddings.reindex_status(USER)["last"]["ok"] is True


def test_dimension_mismatch_is_stale(env):
    emb = env.make()
    emb.add_conversation("Q", "A", "t")
    other = ConversationEmbeddings(USER, client=FakeEmbeddingsClient(), data_root=env.root, dimension=3072)
    assert other.search_conversations("Q", k=1) == [] and other.stale is True


def test_stale_index_schedules_one_background_rebuild(env, monkeypatch):
    monkeypatch.setattr(embeddings, "AUTO_REINDEX", True)
    write_legacy_index(env.root, USER)
    write_conversation(env.root, USER, "20260101_090000", "bread", "How to bake bread?", "Knead and bake.")
    emb = env.make()

    assert emb.search_conversations("bread", k=3) == []
    thread = embeddings.rebuild_thread(USER)
    assert thread is not None
    other = env.make()  # a second turn while the first rebuild runs or ran: no second thread
    other.search_conversations("bread", k=3)
    assert embeddings.rebuild_thread(USER) is thread
    thread.join(10)
    assert not thread.is_alive()

    # both instances see the rebuilt files: the scheduling one by the swap, the other by reloading
    assert emb.search_conversations("bake bread", k=3)[0]["question"] == "How to bake bread?"
    assert other.search_conversations("bake bread", k=3)[0]["question"] == "How to bake bread?"
    assert embeddings.reindex_status(USER) == {"running": False, "scheduled": True, "last": embeddings.reindex_status(USER)["last"]}


def test_model_change_triggers_rebuild(env):
    write_conversation(env.root, USER, "20260101_090000", "bread", "How to bake bread?", "Knead and bake.")
    first = env.make(model="model-a")
    assert first.reindex(blocking=True)["ok"] is True
    assert json.loads(first.metadata_path.read_text())["model"] == "model-a"
    assert first.search_conversations("bread", k=1)[0]["question"] == "How to bake bread?"

    client = FakeEmbeddingsClient()
    second = env.make(client=client, model="model-b")
    assert second.search_conversations("bread", k=1) == [] and second.stale is True
    assert second.reindex(blocking=True)["ok"] is True
    assert client.models == ["model-b", "model-b"] or client.models[0] == "model-b"
    assert json.loads(second.metadata_path.read_text())["model"] == "model-b"
    assert second.search_conversations("bread", k=1)[0]["question"] == "How to bake bread?"


def test_files_are_written_atomically(env, monkeypatch):
    emb = env.make()
    emb.add_conversation("Q one", "A one", "t1")
    assert sorted(p.name for p in emb.embeddings_dir.iterdir()) == ["faiss_index.bin", "metadata.json"]
    before_index, before_meta = emb.index_path.read_bytes(), emb.metadata_path.read_text()

    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith("metadata.json"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(embeddings.os, "replace", failing_replace)
    emb.add_conversation("Q two", "A two", "t2")  # fails inside _save_state, must not raise
    monkeypatch.undo()
    assert emb.metadata_path.read_text() == before_meta  # the old metadata is intact
    fresh = env.make()
    fresh.search_conversations("Q", k=1)
    assert fresh.stale is True  # index (new) and metadata (old) disagree: a rebuild, not garbage
    assert emb.index_path.read_bytes() != before_index


def test_duplicate_questions_collapse(env):
    emb = env.make()
    emb.add_conversation("What is the capital of France?", "Paris.", "t1")
    emb.add_conversation("Message received time in UTC+0: 2026-01-02 10:00:00\n\nwhat is the capital of France", "Paris, still.", "t2")
    emb.add_conversation("Unrelated asyncio question", "Loop.", "t3")
    hits = emb.search_conversations("capital of France", k=5)
    assert [h["question"] for h in hits] == ["What is the capital of France?", "Unrelated asyncio question"] or \
        [h["question"] for h in hits] == ["what is the capital of France", "Unrelated asyncio question"]
    assert len(hits) == 2


def test_add_during_rebuild_is_queued_and_appended_after_the_swap(env):
    write_conversation(env.root, USER, "20260101_090000", "bread", "How to bake bread?", "Knead and bake.")
    gate = threading.Event()
    client = FakeEmbeddingsClient(gate=gate)
    emb = env.make(client=client)
    thread = emb.reindex(blocking=False)
    assert isinstance(thread, threading.Thread)
    for _ in range(100):  # wait until the rebuild is inside the (blocked) API call
        if embeddings.reindex_status(USER)["running"]:
            break
        threading.Event().wait(0.02)
    assert embeddings.reindex_status(USER)["running"] is True

    emb.add_conversation("What is asyncio?", "An event loop.", "20260102_090000")  # queued, returns at once
    assert client.calls == []  # the add did not call the API itself
    gate.set()
    thread.join(10)
    outcome = embeddings.reindex_status(USER)["last"]
    assert outcome["ok"] is True and outcome["items"] == 2 and outcome["queued"] == 1
    assert emb.search_conversations("asyncio", k=1)[0]["question"] == "What is asyncio?"
    assert len(json.loads(emb.metadata_path.read_text())["items"]) == 2


def test_queued_add_already_in_a_conversation_file_is_not_duplicated(env):
    write_conversation(env.root, USER, "20260101_090000", "bread", "How to bake bread?", "Knead and bake.")
    gate = threading.Event()
    emb = env.make(client=FakeEmbeddingsClient(gate=gate))
    thread = emb.reindex(blocking=False)
    for _ in range(100):
        if embeddings.reindex_status(USER)["running"]:
            break
        threading.Event().wait(0.02)
    emb.add_conversation("How to bake bread?", "Knead and bake.", "20260101_090000")  # the same turn, queued
    gate.set()
    thread.join(10)
    assert embeddings.reindex_status(USER)["last"]["items"] == 1
    assert len(emb.search_conversations("bread", k=5)) == 1


def test_search_and_add_never_raise(env):
    emb = env.make(client=FakeEmbeddingsClient(fail=True))
    assert emb.add_conversation("Q", "A", "t") is None
    assert emb.search_conversations("Q", k=3) == []
    assert emb.search_conversations("Q", k=0) == []
    ok = env.make()
    ok.add_conversation("Q", "A", "t")
    assert ok.search_conversations("Q", k=1)[0]["answer"] == "A"
    broken = env.make(client=FakeEmbeddingsClient(fail=True))
    assert broken.search_conversations("Q", k=1) == []  # the query embedding failed: [] not an exception


def test_time_prefix_and_follow_ups_are_stripped(env):
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    answer = "Paris.\n\n<follow_ups>\n- Population?\n- Weather?\n</follow_ups>"
    emb.add_conversation("Message received time in UTC+0: 2026-01-02 10:00:00\n\nCapital of France?", answer, "t")
    assert client.calls[0] == ["Question: Capital of France?\nAnswer: Paris."]
    hit = emb.search_conversations("Message received time in UTC+0: 2026-01-02 10:00:00\n\ncapital", k=1)[0]
    assert hit["question"] == "Capital of France?" and hit["answer"] == "Paris."
    assert client.calls[-1] == ["capital"]


def test_long_texts_are_capped(env):
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    emb.add_conversation("q" * 5000, "a" * 5000, "t")
    assert len(client.calls[0][0]) <= embeddings.EMBED_TEXT_CHARS
    item = emb.items[0]
    assert len(item["question"]) == embeddings.STORED_CHARS and len(item["answer"]) == embeddings.STORED_CHARS


def test_rebuild_batches_up_to_64_texts(env):
    for i in range(70):
        write_conversation(env.root, USER, f"20260101_{i:06d}", f"t{i}", f"question number {i}", f"answer {i}")
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    outcome = emb.reindex(blocking=True)
    assert outcome["items"] == 70
    assert [len(c) for c in client.calls] == [64, 6]
    assert emb.index.ntotal == 70 and len(emb.items) == 70


def test_clear_writes_a_fresh_cosine_index(env):
    emb = env.make()
    emb.add_conversation("Q", "A", "t")
    emb.clear()
    assert emb.index.ntotal == 0 and emb.items == []
    fresh = env.make()
    assert fresh.search_conversations("Q", k=1) == [] and fresh.stale is False


# ---- inputs the API refuses: byte caps, byte-budgeted batches, per-item fallback -------------------------
class RefusedInput(Exception):
    """What openai.BadRequestError looks like to the module: a 400."""
    status_code = 400


class RefusingClient(FakeEmbeddingsClient):
    """Refuses (400) a whole request when any input holds ``POISON``, like the API's 8191-token check."""

    def create(self, *, model, input):
        self.calls.append(list(input))
        if any("POISON" in text for text in input):
            raise RefusedInput("This model's maximum context length is 8191 tokens")
        self.calls.pop()
        return super().create(model=model, input=input)


TELUGU = "తెలుగు భాషలో ఒక పొడవైన ప్రశ్న " * 200  # 3 bytes per character: far more tokens than characters


def test_embedded_texts_and_queries_are_capped_by_bytes_at_a_character_boundary(env):
    text = embeddings._embed_text(TELUGU, TELUGU, "topic", TELUGU)
    assert len(text.encode("utf-8")) <= embeddings.EMBED_TEXT_BYTES
    assert text.startswith("Topic: topic\nQuestion: తెలుగు") and "�" not in text
    assert embeddings._embed_text("Q", "A") == "Question: Q\nAnswer: A"  # short ASCII texts unchanged
    client = FakeEmbeddingsClient()
    emb = env.make(client=client)
    emb.add_conversation("Q", "A", "t")
    emb.search_conversations(TELUGU * 5, k=1)
    assert len(client.calls[-1][0].encode("utf-8")) <= embeddings.EMBED_TEXT_BYTES


def test_rebuild_batches_stay_under_the_byte_budget():
    texts = ["x" * embeddings.EMBED_TEXT_BYTES] * 70
    spans = embeddings._batches(texts)
    assert spans[0] == (0, embeddings.BATCH_BYTES // embeddings.EMBED_TEXT_BYTES) and spans[-1][1] == 70
    for start, end in spans:
        assert 0 < end - start <= embeddings.BATCH_SIZE
        assert sum(len(t) for t in texts[start:end]) <= embeddings.BATCH_BYTES
    assert embeddings._batches(["short"] * 70) == [(0, 64), (64, 70)]
    assert embeddings._batches(["y" * (embeddings.BATCH_BYTES + 1), "z"]) == [(0, 1), (1, 2)]  # never an empty batch
    assert embeddings._batches([]) == []


def test_one_refused_conversation_is_skipped_and_the_rest_is_indexed(env):
    """A single input the API refuses must not keep the whole index stale (and semantic memory off)."""
    for i in range(5):
        write_conversation(env.root, USER, f"20260101_00000{i}", f"t{i}", f"question about topic {i}", f"answer {i}")
    write_conversation(env.root, USER, "20260101_000009", "bad", "POISON question", "answer")
    write_legacy_index(env.root, USER)
    client = RefusingClient()
    emb = env.make(client=client)
    emb._ensure_loaded()
    assert emb.stale

    outcome = emb.reindex(blocking=True)
    assert outcome["ok"] is True and outcome["items"] == 5
    assert [len(c) for c in client.calls] == [6, 1, 1, 1, 1, 1, 1]  # the batch, then one text at a time
    fresh = env.make(client=FakeEmbeddingsClient())
    fresh._ensure_loaded()  # what the next turn reads from disk
    assert fresh.stale is False and len(fresh.items) == 5 and fresh.index.ntotal == 5
    assert all("POISON" not in item["question"] for item in fresh.items)
    assert fresh.search_conversations("question about topic 3", k=1)[0]["answer"] == "answer 3"


def test_a_rebuild_failing_for_another_reason_still_fails_without_per_item_retries(env):
    for i in range(3):
        write_conversation(env.root, USER, f"20260101_00000{i}", f"t{i}", f"q{i}", f"a{i}")
    client = FakeEmbeddingsClient(fail=True)  # network / quota: retrying item by item would only skip everything
    outcome = env.make(client=client).reindex(blocking=True)
    assert outcome["ok"] is False and "embeddings API down" in outcome["error"]


def test_rebuild_never_reads_a_linked_conversation_file(env, tmp_path):
    """conversations/ lies in the sandbox's workspace: a link to another user's conversation is not embedded."""
    victim = write_conversation(env.root, "999", "20260101_000000", "secret", "my bank PIN is 4711", "noted")
    write_conversation(env.root, USER, "20260101_000001", "mine", "my own question", "my answer")
    os.symlink(victim, env.root / USER / "conversations" / "conversation_20260101_000002_x.json")
    emb = env.make()
    outcome = emb.reindex(blocking=True)
    assert outcome["items"] == 1 and [item["question"] for item in emb.items] == ["my own question"]
