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


def test_cap_and_clear_tool_results():
    msgs = turn(1, 5000) + turn(2, 5000) + turn(3, 5000) + turn(4, 5000)
    assert T.cap_tool_results(msgs, 1000) == 4
    assert all(len(m["content"][0]["content"]) < 1100 for m in msgs if T.is_tool_result_message(m))
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

    stats = await T.prune(msgs, max_context_tokens=900, keep_tool_results_turns=1, max_tool_result_chars=2000, summarize=summarize)
    assert stats["truncated"] == 8 and stats["cleared"] == 7 and stats["summarized"] > 0
    assert msgs[0]["role"] == "user" and T.SUMMARY_TAG in msgs[0]["content"][0]["text"]
    assert msgs[1]["role"] == "user"  # consecutive user turns are fine: the API merges them
    assert calls and all(n > 0 for n in calls)
    assert T.estimate_tokens(msgs) < T.estimate_tokens([m for i in range(8) for m in turn(i, 4000)])


async def test_prune_leaves_small_transcripts_alone_and_survives_summarizer_failure():
    msgs = turn(1) + turn(2)
    before = json.dumps(msgs)
    stats = await T.prune(msgs, max_context_tokens=100000, keep_tool_results_turns=1, max_tool_result_chars=20000, summarize=None)
    assert stats == {"truncated": 0, "cleared": 0, "summarized": 0} and json.dumps(msgs) == before

    async def boom(old):
        raise RuntimeError("no model")

    big = [m for i in range(6) for m in turn(i, 3000)]
    stats = await T.prune(big, max_context_tokens=1000, keep_tool_results_turns=0, max_tool_result_chars=100, summarize=boom)
    assert stats["summarized"] == 0 and len(big) == 24  # nothing lost


def test_store_round_trip_and_seed(tmp_path):
    store = T.TranscriptStore(tmp_path)
    assert not store.exists("42", None)
    t = store.load("42", None)
    assert t.messages == [] and t.user_id == "42"
    t.messages.extend(turn(1))
    t.model = "claude-sonnet-5"
    store.save(t)
    again = store.load("42", None)
    assert again.messages == turn(1) and again.model == "claude-sonnet-5" and again.created and again.updated
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
