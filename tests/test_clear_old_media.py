"""transcript.clear_old_media (F7): old image / document blocks become text markers that name
the way back to the file; the last N turns are never touched; prune reports it."""
import base64
import json

from agents import content
from agents import transcript as T

DATA = base64.standard_b64encode(b"\x00" * 3000).decode()


def image():
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": DATA}}


def document(title="r.pdf"):
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": DATA}, "title": title}


def user(i, *media):
    return {"role": "user", "content": [{"type": "text", "text": f"<context>\nt\n</context>"}, {"type": "text", "text": f"look {i}"}, *media]}


def assistant(text, tool=None):
    blocks = [{"type": "thinking", "thinking": "hmm", "signature": "sig"}, {"type": "text", "text": text}]
    if tool:
        blocks.append({"type": "tool_use", "id": f"tu-{tool}", "name": "view_image", "input": {"path": "x.png"}})
    return {"role": "assistant", "content": blocks}


def tool_result(tool, *blocks):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu-{tool}", "content": list(blocks)}]}


def turn(i):
    """A user turn with a photo, a view_image call whose result carries a picture, and the answer."""
    return [
        user(i, image()),
        assistant("let me look", tool=f"t{i}"),
        tool_result(f"t{i}", image(), {"type": "text", "text": "images/x.png · 10x10 px"}),
        assistant(f"answer {i}"),
    ]


def media_blocks(messages):
    found = []
    for m in messages:
        for b in m["content"]:
            if content.is_media_block(b):
                found.append(b)
            elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                found += [x for x in b["content"] if content.is_media_block(x)]
    return found


def test_clear_old_media_strips_older_turns_including_tool_results_and_keeps_the_last_n():
    msgs = turn(1) + turn(2) + turn(3) + turn(4)
    assert len(media_blocks(msgs)) == 8
    assert T.clear_old_media(msgs, keep_turns=2) == 4
    assert len(media_blocks(msgs)) == 4 and media_blocks(msgs) == media_blocks(turn(3) + turn(4))
    # the user message of turn 1: context, text, marker (the block count is unchanged)
    assert msgs[0]["content"][2] == {"type": "text", "text": T.IMAGE_CLEARED_MARKER}
    assert "view_image" in T.IMAGE_CLEARED_MARKER
    # inside the tool_result list the picture is gone, the text next to it stays
    inner = msgs[2]["content"][0]["content"]
    assert inner[0] == {"type": "text", "text": T.IMAGE_CLEARED_MARKER} and inner[1]["text"].startswith("images/x.png")
    # the recent turns are byte-identical
    assert json.dumps(msgs[-8:]) == json.dumps(turn(3) + turn(4))
    assert T.clear_old_media(msgs, keep_turns=2) == 0  # idempotent
    # tool_use / tool_result pairing is untouched
    assert [b["tool_use_id"] for m in msgs if T.is_tool_result_message(m) for b in m["content"]] == ["tu-t1", "tu-t2", "tu-t3", "tu-t4"]


def test_documents_get_their_own_marker_and_few_turns_are_left_alone():
    msgs = [user(1, document("Report.pdf")), assistant("ok"), user(2, image()), assistant("ok")]
    assert T.clear_old_media(msgs, keep_turns=3) == 0 and len(media_blocks(msgs)) == 2
    assert T.clear_old_media(msgs, keep_turns=1) == 1
    assert msgs[0]["content"][2] == {"type": "text", "text": T.DOCUMENT_CLEARED_MARKER.format(title="Report.pdf")}
    assert "Report.pdf" in msgs[0]["content"][2]["text"] and "read_document" in msgs[0]["content"][2]["text"]
    assert msgs[2]["content"][2]["type"] == "image"
    assert T.clear_old_media(msgs, keep_turns=0) == 1 and media_blocks(msgs) == []


def test_string_content_and_assistant_messages_are_skipped():
    msgs = [{"role": "user", "content": "plain"}, {"role": "assistant", "content": [image()]}, user(2), assistant("x"), user(3), assistant("y")]
    assert T.clear_old_media(msgs, keep_turns=1) == 0
    assert msgs[1]["content"][0]["type"] == "image"  # an assistant message never carries user media; left alone


async def test_prune_reports_cleared_media_and_strips_thinking():
    msgs = [m for i in range(6) for m in turn(i)]
    before = T.estimate_tokens(msgs)
    assert before > 6 * 2 * content.IMAGE_TOKENS  # pictures dominate the estimate
    stats = await T.prune(msgs, max_context_tokens=before - 1, keep_tool_results_turns=2, summarize=None)
    # 4 old turns: their tool results (pictures included) are cleared wholesale first, so
    # clear_old_media finds the 4 photos of the user messages
    assert stats["cleared"] == 4 and stats["cleared_media"] == 4
    assert stats["thinking_stripped"] > 0 and not any(b.get("type") == "thinking" for m in msgs for b in m["content"])
    assert len(media_blocks(msgs)) == 4 and T.estimate_tokens(msgs) < before - 8 * content.IMAGE_TOKENS + 400


async def test_prune_below_the_budget_touches_nothing():
    msgs = turn(1) + turn(2) + turn(3)
    before = json.dumps(msgs)
    stats = await T.prune(msgs, max_context_tokens=10 ** 6, keep_tool_results_turns=1, summarize=None)
    assert stats["cleared_media"] == 0 and json.dumps(msgs) == before
