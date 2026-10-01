import json

import pytest

from agents import transcript as T


def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant(text, tool=None):
    blocks = [{"type": "thinking", "thinking": "hmm", "signature": "sig"}, {"type": "text", "text": text}]
    if tool:
        blocks.append({"type": "tool_use", "id": f"tu-{tool}", "name": tool, "input": {"q": "x"}})
    return {"role": "assistant", "content": blocks}


def tool_result(tool, text):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu-{tool}", "content": text}]}


def turn(i, result_size=100):
    return [user(f"question {i}"), assistant("let me look", tool=f"t{i}"), tool_result(f"t{i}", "r" * result_size), assistant(f"answer {i}")]


def test_message_classification_and_text_view():
    msgs = turn(1)
    assert T.is_user_turn(msgs[0]) and not T.is_user_turn(msgs[2])
    assert T.is_tool_result_message(msgs[2]) and not T.is_tool_result_message(msgs[0])
    view = T.text_only_view(msgs)
    assert view == [
        {"role": "user", "content": "question 1"},
        {"role": "assistant", "content": "let me look\n\nanswer 1"},
    ]
    assert T.text_only_view([assistant("orphan"), user("q")]) == [{"role": "user", "content": "q"}]


def test_cap_text_and_clear_tool_results():
    capped = T.cap_text("r" * 5000, 1000)
    assert capped.startswith("r" * 1000) and capped.endswith("[truncated 4000 characters]")
    assert T.cap_text("short", 1000) == "short" and T.cap_text("x" * 50, None) == "x" * 50
    msgs = turn(1, 5000) + turn(2, 5000) + turn(3, 5000) + turn(4, 5000)
    assert T.clear_old_tool_results(msgs, keep_turns=2) == 2
    results = [m["content"][0]["content"] for m in msgs if T.is_tool_result_message(m)]
    assert results[0] == T.CLEARED_MARKER and results[1] == T.CLEARED_MARKER
    assert results[2] != T.CLEARED_MARKER and results[3] != T.CLEARED_MARKER
    assert T.clear_old_tool_results(msgs, keep_turns=2) == 0  # idempotent
    # tool_use / tool_result pairing untouched
    assert [b["tool_use_id"] for m in msgs if T.is_tool_result_message(m) for b in m["content"]] == ["tu-t1", "tu-t2", "tu-t3", "tu-t4"]


async def test_prune_summarizes_oldest_half_when_still_too_big():
    msgs = []
    for i in range(8):
        msgs += turn(i, 4000)
    calls = []

    async def summarize(old):
        calls.append(len(old))
        return "summary of the beginning"

    stats = await T.prune(msgs, max_context_tokens=900, keep_tool_results_turns=1, summarize=summarize)
    assert stats["cleared"] == 7 and stats["summarized"] > 0
    # an edited history cannot keep its thinking blocks: they were bound to the old one
    assert stats["thinking_stripped"] > 0
    assert not any(b.get("type") == "thinking" for m in msgs for b in m["content"])
    assert msgs[0]["role"] == "user" and T.SUMMARY_TAG in msgs[0]["content"][0]["text"]
    assert msgs[1]["role"] == "user"  # consecutive user turns are fine: the API merges them
    assert calls and all(n > 0 for n in calls)
    assert T.estimate_tokens(msgs) < T.estimate_tokens([m for i in range(8) for m in turn(i, 4000)])


async def test_prune_leaves_small_transcripts_alone_and_survives_summarizer_failure():
    msgs = turn(1) + turn(2)
    before = json.dumps(msgs)
    stats = await T.prune(msgs, max_context_tokens=100000, keep_tool_results_turns=1, summarize=None)
    assert stats == {"cleared": 0, "summarized": 0, "thinking_stripped": 0} and json.dumps(msgs) == before

    async def boom(old):
        raise RuntimeError("no model")

    big = [m for i in range(6) for m in turn(i, 3000)]
    stats = await T.prune(big, max_context_tokens=1000, keep_tool_results_turns=0, summarize=boom)
    assert stats["summarized"] == 0 and len(big) == 24  # nothing lost


def test_store_round_trip_and_seed(tmp_path):
    store = T.TranscriptStore(tmp_path)
    assert not store.exists("42", None)
    t = store.load("42", None)
    assert t.messages == [] and t.user_id == "42"
    t.messages.extend(turn(1))
    t.model = "claude-sonnet-5-5"
    store.save(t)
    again = store.load("42", None)
    assert again.messages == turn(1) and again.model == "claude-sonnet-5-5" and again.created and again.updated
    assert store.path("42", 77).name == "topic_77_transcript.json" and not store.exists("42", 77)

    (tmp_path / "42" / "transcripts" / "transcript.json").write_text("{not json")
    assert store.load("42", None).messages == []  # corrupt file → fresh transcript, no crash

    seeded = store.seed_from_dialog_history("42", None, [
        {"role": "assistant", "content": "stray"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ])
    assert [m["role"] for m in seeded.messages] == ["user", "assistant"] and seeded.seeded_from == "dialog_history"


def test_strip_private_keys():
    msgs = [{"role": "user", "content": "x", "_ephemeral": True}, {"role": "assistant", "content": "y"}]
    assert T.strip_private_keys(msgs) == [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]


def test_sanitize_turn_repairs_incomplete_turns():
    tool_use = {"type": "tool_use", "id": "tu-1", "name": "t", "input": {}}
    turn_msgs = [
        user("q"),
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},              # empty draft → dropped
        {"role": "assistant", "content": [{"type": "text", "text": "Let me check"}, tool_use]},  # budget ran out: no result
    ]
    out = T.sanitize_turn(turn_msgs, "final answer")
    assert [m["role"] for m in out] == ["user", "assistant", "assistant"]
    assert out[1]["content"] == [{"type": "text", "text": "Let me check"}]  # dangling tool_use removed
    assert out[2]["content"] == [{"type": "text", "text": "final answer"}]
    assert T.sanitize_turn([user("q")], "") == [user("q"), {"role": "assistant", "content": [{"type": "text", "text": "(no answer)"}]}]
    led = T.sanitize_turn([{"role": "assistant", "content": "stray"}, user("q")], "y")
    assert [m["role"] for m in led] == ["user", "assistant"] and T.message_text(led[1]) == "y"
    complete = turn(1)
    assert T.sanitize_turn(list(complete), "answer 1") == complete


def test_context_blocks_are_stored_but_not_part_of_the_text():
    context = {"type": "text", "text": T.CONTEXT_OPEN + "Current time\n<memory>\n• [d1] a\n• [d2] b\n</memory>\n</context>"}
    msgs = [{"role": "user", "content": [context, {"type": "text", "text": "question"}]}, assistant("answer")]
    assert T.is_context_block(context) and not T.is_context_block(msgs[0]["content"][1])
    assert T.message_text(msgs[0]) == "question"
    assert T.text_only_view(msgs)[0] == {"role": "user", "content": "question"}
    assert T.shown_memory_lines(msgs) == {"• [d1] a", "• [d2] b"}
    assert T.shown_memory_lines(turn(1)) == set()


def test_strip_thinking_keeps_everything_else():
    msgs = turn(1) + [
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "xx"}]},
        user("next"),
    ]
    assert T.strip_thinking(msgs) == 3  # two thinking blocks from turn(1), one redacted
    assert not any(b.get("type") in T.THINKING_TYPES for m in msgs for b in m["content"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]  # the empty one is gone
    assert msgs[1]["content"][-1]["type"] == "tool_use"
    assert T.strip_thinking(msgs) == 0


def test_prefix_fingerprint():
    tools = [{"name": "b", "description": "x"}, {"name": "a", "description": "y"}]
    fp = T.prefix_fingerprint("m", "system", tools)
    assert fp == T.prefix_fingerprint("m", "system", list(reversed(tools)))  # bound as a set
    assert fp != T.prefix_fingerprint("m2", "system", tools)
    assert fp != T.prefix_fingerprint("m", "system!", tools)
    assert fp != T.prefix_fingerprint("m", "system", [{"name": "b", "description": "x"}, {"name": "a", "description": "z"}])


def test_fingerprint_round_trips(tmp_path):
    store = T.TranscriptStore(tmp_path)
    t = store.load("7", None)
    t.messages, t.fingerprint = turn(1), "abc"
    store.save(t)
    assert store.load("7", None).fingerprint == "abc"
