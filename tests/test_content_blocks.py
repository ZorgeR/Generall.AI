"""Prompt / tool-result content blocks (agents/content.py) and the hubs that take them:
the provider registry, list tool results, block prompts through run_turn, media-aware
token estimates and turn records that keep the prompt blocks for Retry."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from agents import content
from agents import transcript as tr

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_normalize_and_text_view(tmp_path):
    assert content.normalize("hi") == [{"type": "text", "text": "hi"}]
    blocks = content.normalize(["a", content.image_ref(tmp_path / "x.jpg"), {"type": "text", "text": "b"}])
    assert [b["type"] for b in blocks] == ["text", content.MEDIA_REF, "text"]
    text = content.prompt_text(blocks)
    assert text.startswith("a\n\n[image: ") and text.endswith("\n\nb")
    assert content.prompt_text("plain") == "plain"


def test_with_prefix_goes_on_first_text_block_or_a_new_one():
    out = content.with_prefix([content.image_ref("p.jpg"), content.text_block("q")], "T\n\n")
    assert out[0] == {"type": "text", "text": "T"} and out[2]["text"] == "q"
    out = content.with_prefix([content.text_block("q")], "T\n\n")
    assert out == [{"type": "text", "text": "T\n\nq"}]


def test_materialize_image_and_pdf(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(JPEG)
    sniffed = tmp_path / "noext"
    sniffed.write_bytes(PNG)
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    missing = tmp_path / "gone.jpg"
    out = content.materialize([
        content.text_block("look"), content.image_ref(img), content.image_ref(sniffed),
        content.document_ref(pdf, title="Report"), content.image_ref(missing),
    ])
    assert out[0] == {"type": "text", "text": "look"}
    assert out[1]["type"] == "image" and out[1]["source"]["media_type"] == "image/jpeg"
    assert base64.standard_b64decode(out[1]["source"]["data"]) == JPEG
    assert out[2]["source"]["media_type"] == "image/png"
    assert out[3]["type"] == "document" and out[3]["title"] == "Report"
    assert out[3]["source"]["media_type"] == content.PDF_MEDIA_TYPE
    assert out[4]["type"] == "text" and "not found" in out[4]["text"]


def test_materialize_refuses_non_pdf_documents_and_huge_images(tmp_path, monkeypatch):
    doc = tmp_path / "a.docx"
    doc.write_bytes(b"x")
    big = tmp_path / "big.jpg"
    big.write_bytes(JPEG)
    monkeypatch.setattr(content, "MAX_IMAGE_BYTES", 10)
    out = content.materialize([content.document_ref(doc), content.image_ref(big)])
    assert out[0]["type"] == "text" and "read_file" in out[0]["text"]
    assert out[1]["type"] == "text" and "too large" in out[1]["text"]


def test_estimate_tokens_counts_media_by_kind():
    data = base64.standard_b64encode(b"\x00" * 300_000).decode()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "x" * 320},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
    ]}]
    tokens = tr.estimate_tokens(msgs)
    assert content.IMAGE_TOKENS <= tokens < content.IMAGE_TOKENS + 200  # not the 90k the base64 would be as text
    doc = [{"role": "user", "content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}]}]
    assert content.DOCUMENT_MIN_TOKENS <= tr.estimate_tokens(doc) < 5000
    nested = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, {"type": "text", "text": "ok"}]}]}]
    assert content.IMAGE_TOKENS <= tr.estimate_tokens(nested) < content.IMAGE_TOKENS + 100
    assert tr.estimate_tokens([{"role": "user", "content": "plain text"}]) > 0


def test_result_preview_cap_blocks_and_strip_media():
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}, {"type": "text", "text": "a" * 50}]
    assert content.result_preview(blocks) == "[image]\n" + "a" * 50
    assert content.result_preview("plain") == "plain"
    capped = content.cap_blocks(blocks, 10)
    assert capped[0]["type"] == "image" and capped[1]["text"].startswith("a" * 10 + "\n…[truncated 40")
    assert content.cap_blocks(blocks, None) == blocks
    msgs = [{"role": "user", "content": [blocks[0], {"type": "tool_result", "tool_use_id": "1", "content": [blocks[0]]}]}]
    stripped = content.strip_media(msgs)
    assert stripped[0]["content"][0] == {"type": "text", "text": "[image]"}
    assert stripped[0]["content"][1]["content"] == [{"type": "text", "text": "[image]"}]
    assert msgs[0]["content"][0]["type"] == "image"  # the original is untouched


# ---- the hubs ------------------------------------------------------------------
class Provider:
    def __init__(self, name, result):
        self.name, self.result, self.calls = name, result, []

    @property
    def tools_schema(self):
        return [{"name": self.name, "description": "t", "input_schema": {"type": "object", "properties": {}}}]

    async def execute_tool(self, tool_name, tool_args):
        self.calls.append((tool_name, tool_args))
        return self.result


class SyncProvider(Provider):
    def execute_tool(self, tool_name, tool_args):
        self.calls.append((tool_name, tool_args))
        return self.result


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    import agents.main as m

    monkeypatch.setattr(m.stats_tracker, "track_tool_used", lambda *a, **k: None)
    a = m.AgentAnthropic(user_id="1")
    a.file_ops = Provider("list_files", "files")
    return a


async def test_registry_schema_order_and_dispatch(agent):
    blocks_provider = Provider("view_image", [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}, {"type": "text", "text": "1x1"}])
    sync_provider = SyncProvider("browse_webpage", "page text")
    agent.register_provider(blocks_provider)
    agent.register_provider(sync_provider, blocking=True)
    names = [t["name"] for t in agent.get_tools_schema()]
    assert names == ["list_files", "view_image", "browse_webpage"]
    assert await agent.execute_tool("browse_webpage", {"url": "u"}) == "page text"
    assert sync_provider.calls == [("browse_webpage", {"url": "u"})]
    assert await agent.execute_tool("nothing", {}) == "Unknown tool: nothing"

    agent.allowed_tools = {"view_image"}
    assert [t["name"] for t in agent.get_tools_schema()] == ["view_image"]


async def test_tool_batch_keeps_block_results_and_caps_text_only(agent):
    long_text = "y" * 100
    agent.register_provider(Provider("view_image", [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}, {"type": "text", "text": long_text}]))
    agent.max_tool_result_chars = 20
    calls = [SimpleNamespace(id="a", name="view_image", input={}), SimpleNamespace(id="b", name="list_files", input={})]
    results = await agent.run_tool_batch(calls)
    assert results[0]["tool_use_id"] == "a" and isinstance(results[0]["content"], list)
    assert results[0]["content"][0]["type"] == "image"
    assert results[0]["content"][1]["text"].startswith("y" * 20 + "\n…[truncated 80")
    assert results[1] == {"type": "tool_result", "tool_use_id": "b", "content": "files"}


def test_subagent_child_inherits_registered_providers(agent, monkeypatch):
    from agents.subagent import SubagentTools
    from agents.trace import TurnBudget

    agent.register_provider(Provider("view_image", "x"))
    sub = SubagentTools(agent, {})
    assert "view_image" in sub._available_tool_names()
    child = sub._child(agent.model, None, TurnBudget(3))
    assert [t["name"] for t in child.get_tools_schema()] == ["list_files", "view_image"]


def test_turn_record_keeps_prompt_blocks_round_trip():
    from bot.turn_actions import TurnRecord

    blocks = [content.text_block("see"), content.image_ref("data/1/images/x.jpg")]
    record = TurnRecord(id="abcdef012345", prompt="see\n\n[image: data/1/images/x.jpg]", response="ok", prompt_blocks=blocks)
    loaded = TurnRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert loaded is not None and loaded.prompt_blocks == blocks
    assert TurnRecord.from_dict({"id": "abcdef012345", "prompt": "p", "response": "r"}).prompt_blocks is None
    assert TurnRecord.from_dict({"id": "abcdef012345", "prompt": "p", "response": "r", "prompt_blocks": []}).prompt_blocks is None


async def test_build_user_message_materializes_and_media_skips_quick_path(agent, monkeypatch, tmp_path):
    import agents.main as m

    img = tmp_path / "p.jpg"
    img.write_bytes(JPEG)
    cot = m.ChainOfThoughtAgent.__new__(m.ChainOfThoughtAgent)
    msg = await cot._build_user_message("<context>\nt\n</context>", [content.text_block("q"), content.image_ref(img)])
    assert msg["role"] == "user"
    assert [b["type"] for b in msg["content"]] == ["text", "text", "image"]

    async def classify(question, history):
        return "simple"

    cot._classify_complexity = classify
    assert await cot._decide_complexity("q", [], has_media=True) == "complex"
    assert await cot._decide_complexity("q", [], has_media=False) == "simple"
