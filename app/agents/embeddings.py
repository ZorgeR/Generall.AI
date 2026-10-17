"""Semantic memory over past conversations: one cosine FAISS index per user.

The index is a derived cache of ``data/<uid>/conversations/*.json`` (the record every turn
writes), so it can always be rebuilt from them and is distrusted rather than migrated:

* ``metadata.json`` records the embedding model and the layout version. Vectors of two
  models are not comparable, so a different model, the legacy layout (a bare list next to
  an L2 index), a dimension mismatch or files that disagree mark the index *stale*: searches
  answer ``[]`` and one background rebuild per user and process is scheduled. The rebuild
  embeds every conversation in batches and replaces both files atomically (tmp + replace).
* ``ConversationEmbeddings`` is constructed on the event loop once per turn, so the
  constructor does nothing. The files are read on the first call, and every call that
  touches the API or the disk (``add_conversation``, ``search_conversations``, ``reindex``)
  is meant to run in a worker thread, as the agent does (``asyncio.to_thread``).
* Several instances of one user can be alive at once (this turn's instance and the thread
  rebuilding for an earlier one). The per-user ``RLock`` serialises their disk writes and
  index swaps; a generation counter tells an instance that the files changed under it, so it
  reloads instead of appending to a copy of the old index. A conversation added while a
  rebuild is embedding is queued and appended after the swap (deduplicated by ``text_hash``,
  because the rebuild may already have read its conversation file).

Searches and adds never raise: semantic memory is optional for the agent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

import models
from agents.paths import regular_files

logger = logging.getLogger(__name__)

METADATA_VERSION = 2
BATCH_SIZE = 64  # texts per embeddings.create call during a rebuild
EMBED_TEXT_CHARS = 6000  # the embedded text (the model takes 8191 tokens)
# Characters do not bound tokens (Telugu, Gujarati, dense CJK: > 8191 tokens in 6000 chars). A
# byte-level BPE token covers at least one UTF-8 byte, so bytes do: every input stays under
# EMBED_TEXT_BYTES (the search query too) and a rebuild call under BATCH_BYTES (the API caps one
# request at 300k tokens over all its inputs).
EMBED_TEXT_BYTES = 8000
BATCH_BYTES = 250_000
PART_CHARS = {"topic": 100, "question": 2500, "answer": 2500, "summary": 800}
STORED_CHARS = 2000  # question / answer kept in metadata items
AUTO_REINDEX = True  # a stale index schedules its own rebuild (tests switch it off)

_TIME_PREFIX = re.compile(r"^\s*Message received time in UTC\+0: [^\n]*\n*")

# ---------------------------------------------------------------------------
# Process-wide state: one lock, one generation counter and at most one automatic
# rebuild per user, shared by every instance of that user.
# ---------------------------------------------------------------------------
_registry_lock = threading.Lock()
_user_locks: dict[str, threading.RLock] = {}
_generations: dict[str, int] = {}  # bumped on every write of a user's files
_auto_rebuilds: dict[str, threading.Thread] = {}  # the one automatic rebuild per user per process
_active: set[str] = set()  # users whose rebuild is running right now
_pending: dict[str, list[dict]] = {}  # adds queued while a rebuild runs
_last_rebuild: dict[str, dict] = {}  # outcome of the latest rebuild per user


def _user_lock(user_id: str) -> threading.RLock:
    with _registry_lock:
        return _user_locks.setdefault(user_id, threading.RLock())


def _generation(user_id: str) -> int:
    with _registry_lock:
        return _generations.get(user_id, 0)


def _bump_generation(user_id: str) -> int:
    with _registry_lock:
        _generations[user_id] = _generations.get(user_id, 0) + 1
        return _generations[user_id]


def rebuild_thread(user_id: str) -> threading.Thread | None:
    """The automatic background rebuild started for ``user_id`` in this process, if any."""
    with _registry_lock:
        return _auto_rebuilds.get(str(user_id))


def reindex_status(user_id: str) -> dict[str, Any]:
    """``{"running", "scheduled", "last"}`` for an admin view: is a rebuild running, was one started, how did the last one end."""
    uid = str(user_id)
    with _registry_lock:
        return {"running": uid in _active, "scheduled": uid in _auto_rebuilds, "last": dict(_last_rebuild.get(uid) or {})}


def reset() -> None:
    """Forget every per-user flag (tests)."""
    with _registry_lock:
        _user_locks.clear()
        _generations.clear()
        _auto_rebuilds.clear()
        _active.clear()
        _pending.clear()
        _last_rebuild.clear()


# ---------------------------------------------------------------------------
# Text helpers: what is embedded and what is stored
# ---------------------------------------------------------------------------
def _clean_question(text: Any) -> str:
    """The user's text without the ``Message received time`` prefix every prompt carries."""
    return _TIME_PREFIX.sub("", str(text or ""), count=1).strip()


def _clean_answer(text: Any) -> str:
    """The answer without its (last) ``<follow_ups>`` block, which is UI, not content."""
    text = str(text or "")
    start = text.rfind("<follow_ups>")
    if start >= 0:
        end = text.find("</follow_ups>", start)
        if end >= 0:
            text = text[:start] + text[end + len("</follow_ups>"):]
    return text.strip()


def _squash(text: str) -> str:
    return " ".join(text.split())


def _dedupe_key(question: str) -> str:
    """Two hits with the same key are the same question asked again."""
    return _squash(_clean_question(question)).lower().rstrip(" ?.!")


def _text_hash(question: str, answer: str) -> str:
    """Identity of a conversation (question + answer), the same whichever path indexed it."""
    raw = f"{_squash(question)}\n{_squash(answer)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _make_item(question: Any, answer: Any, timestamp: Any, topic: Any) -> dict[str, Any]:
    q, a = _clean_question(question), _clean_answer(answer)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "question": q[:STORED_CHARS],
        "answer": a[:STORED_CHARS],
        "timestamp": str(timestamp),
        "topic": str(topic or ""),
        "text_hash": _text_hash(q, a),
    }


def _embed_text(question: Any, answer: Any, topic: Any = None, summary: Any = None) -> str:
    q, a = _clean_question(question), _clean_answer(answer)
    parts = []
    if topic:
        parts.append(f"Topic: {str(topic).replace('_', ' ')[:PART_CHARS['topic']]}")
    parts.append(f"Question: {q[:PART_CHARS['question']]}")
    parts.append(f"Answer: {a[:PART_CHARS['answer']]}")
    if summary:
        parts.append(f"Summary: {str(summary)[:PART_CHARS['summary']]}")
    return _cap_bytes("\n".join(parts)[:EMBED_TEXT_CHARS])


def _cap_bytes(text: str, limit: int = EMBED_TEXT_BYTES) -> str:
    """``text`` cut to at most ``limit`` UTF-8 bytes, at a character boundary."""
    data = text.encode("utf-8")
    return text if len(data) <= limit else data[:limit].decode("utf-8", errors="ignore")


def _batches(texts: list[str]) -> list[tuple[int, int]]:
    """``(start, end)`` slices of ``texts`` with at most BATCH_SIZE texts and BATCH_BYTES bytes each."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(texts):
        end, size = start, 0
        while end < len(texts) and end - start < BATCH_SIZE:
            size += len(texts[end].encode("utf-8"))
            if end > start and size > BATCH_BYTES:
                break
            end += 1
        spans.append((start, end))
        start = end
    return spans


def _is_refused_input(error: Exception) -> bool:
    """The API refused the request's content (a 400), as opposed to a network, quota or config failure."""
    return getattr(error, "status_code", None) == 400


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so inner product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _atomic_write_index(index: faiss.Index, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _atomic_write_json(data: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class ConversationEmbeddings:
    """Per-user semantic index of past conversations (see the module docstring)."""

    def __init__(self, user_id: str, *, client: Any = None, data_root: str | Path | None = None,
                 model: str | None = None, dimension: int | None = None):
        self.user_id = str(user_id)
        self.model = model or models.EMBEDDING_MODEL
        self.dimension = int(dimension or models.EMBEDDING_DIMENSION)
        root = Path(data_root) if data_root is not None else Path("./data")
        self.embeddings_dir = root / self.user_id / "embeddings"
        self.conversations_dir = root / self.user_id / "conversations"
        self.index_path = self.embeddings_dir / "faiss_index.bin"
        self.metadata_path = self.embeddings_dir / "metadata.json"
        self._client = client
        self._lock = _user_lock(self.user_id)
        self.index: faiss.Index | None = None  # loaded on first use
        self.items: list[dict[str, Any]] = []
        self.stale = False
        self._loaded_generation: int | None = None

    # -- client ---------------------------------------------------------------
    @property
    def openai_client(self) -> Any:
        """Built on first use (never in the constructor); tests inject a fake."""
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    @openai_client.setter
    def openai_client(self, value: Any) -> None:
        self._client = value

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors for ``texts``, one API call (the caller batches)."""
        response = self.openai_client.embeddings.create(model=self.model, input=texts)
        data = sorted(response.data, key=lambda d: getattr(d, "index", 0) or 0)
        if len(data) != len(texts):
            raise ValueError(f"{self.model} returned {len(data)} vectors for {len(texts)} texts")
        matrix = np.asarray([d.embedding for d in data], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"{self.model} returned {matrix.shape[-1]}-dim vectors but EMBEDDING_DIMENSION is {self.dimension}")
        return _normalize(matrix)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray | None]:
        """One vector (1×dim) per text for a rebuild; ``None`` for a text the API refuses.

        A refused batch (400) is retried one text at a time and the texts still refused are
        skipped with a log line, so one conversation cannot keep the whole index stale. Any
        other failure (network, quota, wrong dimension) propagates and fails the rebuild.
        """
        try:
            matrix = self._embed(texts)
            return [matrix[i:i + 1] for i in range(len(texts))]
        except Exception as e:  # noqa: BLE001 - only a refused input is handled here
            if not _is_refused_input(e):
                raise
            if len(texts) == 1:
                logger.warning("Embeddings of user %s: skipping a conversation the API refuses: %s", self.user_id, e)
                return [None]
        vectors: list[np.ndarray | None] = []
        for text in texts:
            vectors.extend(self._embed_batch([text]))
        return vectors

    # -- loading --------------------------------------------------------------
    def _empty_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self.dimension)

    def _ensure_loaded(self) -> None:
        """Read the files once per instance, and again whenever another instance wrote them."""
        with self._lock:
            generation = _generation(self.user_id)
            if self.index is not None and self._loaded_generation == generation:
                return
            self.index, self.items, self.stale = self._read_files()
            self._loaded_generation = generation
            if self.stale and AUTO_REINDEX:
                self.reindex(blocking=False)

    def _read_files(self) -> tuple[faiss.Index, list[dict[str, Any]], bool]:
        """``(index, items, stale)``: no files is a fresh empty index, anything untrusted is stale."""
        if not self.index_path.exists() and not self.metadata_path.exists():
            return self._empty_index(), [], False
        try:
            index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:  # noqa: BLE001 - a missing or corrupt file: rebuild
            logger.warning("Embeddings of user %s cannot be read (%s); scheduling a rebuild", self.user_id, e)
            return self._empty_index(), [], True
        reason = self._stale_reason(index, meta)
        if reason:
            logger.info("Embeddings of user %s need a rebuild: %s", self.user_id, reason)
            return self._empty_index(), [], True
        return index, list(meta["items"]), False

    def _stale_reason(self, index: faiss.Index, meta: Any) -> str | None:
        if not isinstance(meta, dict) or meta.get("version") != METADATA_VERSION:
            return "legacy layout"
        if meta.get("model") != self.model:
            return f"built with {meta.get('model')!r}, configured {self.model!r}"
        if int(meta.get("dimension") or 0) != self.dimension or index.d != self.dimension:
            return f"dimension {meta.get('dimension')}/{index.d}, configured {self.dimension}"
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            return "not a cosine index"
        items = meta.get("items")
        if not isinstance(items, list) or index.ntotal != len(items):
            return "index and metadata disagree"
        return None

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": METADATA_VERSION,
            "model": self.model,
            "dimension": self.dimension,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "items": self.items,
        }

    def _save_state(self) -> None:
        """Write both files atomically (the lock is held) and tell other instances."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_index(self.index, self.index_path)
        _atomic_write_json(self._metadata(), self.metadata_path)
        self._loaded_generation = _bump_generation(self.user_id)

    # -- the public contract (wrapped by name in tool_integrator.patch_embeddings) --
    def add_conversation(self, question: str, answer: str, timestamp: str | None = None,
                         topic: str | None = None, summary: str | None = None) -> None:
        """Index one finished turn. Queued while a rebuild runs; dropped (logged) while the index is stale."""
        try:
            item = _make_item(question, answer, timestamp, topic)
            if not (item["question"] or item["answer"]):
                return
            text = _embed_text(question, answer, topic, summary)
            self._ensure_loaded()
            if self._queue_if_rebuilding(item, text):
                return
            if self.stale:
                logger.info("Embeddings of user %s are stale; the conversation waits for the rebuild", self.user_id)
                return
            vector = self._embed([text])
            with self._lock:
                self._ensure_loaded()  # the files may have changed while we embedded
                if self._queue_if_rebuilding(item, text) or self.stale:
                    return
                self.index.add(vector)
                self.items.append(item)
                self._save_state()
        except Exception as e:  # noqa: BLE001 - memory is optional, the answer is already on its way
            logger.warning("Could not index the conversation of user %s: %s", self.user_id, e)

    def _queue_if_rebuilding(self, item: dict[str, Any], text: str) -> bool:
        with _registry_lock:
            if self.user_id not in _active:
                return False
            _pending.setdefault(self.user_id, []).append({"item": item, "text": text})
        logger.info("Embeddings of user %s are being rebuilt; conversation queued", self.user_id)
        return True

    def search_conversations(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Top ``k`` past conversations by cosine similarity, one per distinct question.

        Each hit is ``{question, answer, timestamp, topic, score}`` with ``score`` in 0..1.
        A stale index answers ``[]`` (its vectors come from another model).
        """
        try:
            k = int(k)
            if k <= 0:
                return []
            self._ensure_loaded()
            if self.stale or not self.items:
                return []
            vector = self._embed([_cap_bytes(_clean_question(query) or str(query))])
            with self._lock:
                self._ensure_loaded()
                if self.stale or not self.items:
                    return []
                candidates = min(self.index.ntotal, max(k * 3, k + 5))  # room for collapsed duplicates
                scores, ids = self.index.search(vector, candidates)
                items = self.items
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(items):  # FAISS pads with -1 when the index is small
                    continue
                item = items[int(idx)]
                key = _dedupe_key(item.get("question", ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "timestamp": item.get("timestamp", ""),
                    "topic": item.get("topic", ""),
                    "score": round(max(0.0, min(1.0, float(score))), 4),
                })
                if len(results) >= k:
                    break
            return results
        except Exception as e:  # noqa: BLE001
            logger.warning("Semantic search failed for user %s: %s", self.user_id, e)
            return []

    def clear(self) -> None:
        """Drop every vector; the next rebuild (from conversations/) refills the index."""
        try:
            with self._lock:
                self.index, self.items, self.stale = self._empty_index(), [], False
                self._save_state()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not clear the embeddings of user %s: %s", self.user_id, e)

    # -- rebuilding -----------------------------------------------------------
    def reindex(self, blocking: bool = True) -> dict[str, Any] | threading.Thread | None:
        """Rebuild the index from ``conversations/*.json``.

        ``blocking=True`` rebuilds in the calling thread (or waits for the automatic rebuild
        already running) and returns the outcome ``{"ok", "items", "queued", "model",
        "seconds"[, "error"]}``. ``blocking=False`` starts the one automatic background rebuild
        of this user for this process and returns its thread, or ``None`` when it was started
        earlier (whatever its outcome).
        """
        if not blocking:
            return self._start_background()
        running = rebuild_thread(self.user_id)
        if running is not None and running.is_alive() and running is not threading.current_thread():
            running.join()
            self._ensure_loaded()
            return dict(_last_rebuild.get(self.user_id) or {})
        return self._rebuild()

    def _start_background(self) -> threading.Thread | None:
        with _registry_lock:
            if self.user_id in _auto_rebuilds:
                return None
            thread = threading.Thread(target=self._rebuild, name=f"embeddings-reindex-{self.user_id}", daemon=True)
            _auto_rebuilds[self.user_id] = thread
        logger.info("Scheduling an embeddings rebuild for user %s", self.user_id)
        thread.start()
        return thread

    def _conversation_records(self) -> list[tuple[dict[str, Any], str]]:
        """``(item, text to embed)`` per conversation file, oldest first, duplicates skipped."""
        records: list[tuple[dict[str, Any], str]] = []
        seen: set[str] = set()
        # links and non-regular files skipped: conversations/ lies in the sandbox's workspace
        for file, raw in regular_files(self.conversations_dir, "conversation_*.json"):
            try:
                data = json.loads(raw)
            except ValueError as e:
                logger.warning("Skipping unreadable conversation %s: %s", file.name, e)
                continue
            if not isinstance(data, dict):
                continue
            question = data.get("question") or ""
            answer = data.get("response") or data.get("answer") or ""
            item = _make_item(question, answer, data.get("timestamp"), data.get("topic"))
            if not (item["question"] or item["answer"]) or item["text_hash"] in seen:
                continue
            seen.add(item["text_hash"])
            records.append((item, _embed_text(question, answer, data.get("topic"), data.get("summary"))))
        return records

    def _rebuild(self) -> dict[str, Any]:
        """Embed every conversation, then swap and save under the lock. Never raises."""
        uid, started = self.user_id, time.monotonic()
        outcome: dict[str, Any] = {"ok": False, "model": self.model}
        with _registry_lock:
            _active.add(uid)
            _pending.pop(uid, None)  # leftovers of an earlier failed rebuild are on disk anyway
        try:
            records = self._conversation_records()
            total = len(records)
            logger.info("Rebuilding the embeddings of user %s: %d conversations with %s", uid, total, self.model)
            texts = [text for _, text in records]
            vectors: list[np.ndarray | None] = []
            for start, end in _batches(texts):
                vectors.extend(self._embed_batch(texts[start:end]))
                logger.info("Embeddings rebuild of user %s: %d/%d", uid, end, total)
            kept = [(item, vector) for (item, _), vector in zip(records, vectors) if vector is not None]
            index = self._empty_index()
            if kept:
                index.add(np.vstack([vector for _, vector in kept]))
            items = [item for item, _ in kept]
            with self._lock:
                with _registry_lock:
                    pending = _pending.pop(uid, [])
                    _active.discard(uid)  # from here on adds go to the new index
                known = {item["text_hash"] for item in items}
                fresh = [p for p in pending if p["item"]["text_hash"] not in known]
                if fresh:
                    try:
                        index.add(self._embed([p["text"] for p in fresh]))
                        items.extend(p["item"] for p in fresh)
                    except Exception as e:  # noqa: BLE001 - they are in conversations/ for the next rebuild
                        logger.warning("Could not append %d queued conversations of user %s: %s", len(fresh), uid, e)
                        fresh = []
                self.index, self.items, self.stale = index, items, False
                self._save_state()
            outcome.update(ok=True, items=len(items), queued=len(fresh))
            logger.info("Embeddings of user %s rebuilt: %d conversations", uid, len(items))
        except Exception as e:  # noqa: BLE001
            outcome["error"] = f"{type(e).__name__}: {e}"
            logger.warning("Embeddings rebuild of user %s failed: %s", uid, outcome["error"])
        finally:
            outcome["seconds"] = round(time.monotonic() - started, 2)
            with _registry_lock:
                _active.discard(uid)
                _pending.pop(uid, None)
                _last_rebuild[uid] = outcome
        return outcome
