"""memory_search runs on the host over the user's real long-term memory:
conversation files, the profile and the transcript text view, newest first; the sandbox patch is gone."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.search_tools import MEMORY_SEARCH_MAX_RESULTS, SearchTools
from secure_container.secure_tool_wrapper import SecureToolWrapper
from secure_container.tool_integrator import ToolIntegrator

USER = "42"
TOOL_ONLY = "zqxtoolresultonly"  # appears in a tool result only: never a hit
SUMMARY = "<earlier_conversation_summary>\nEarlier the user asked about apiary hives and got frame sizes.\n</earlier_conversation_summary>"


def conversation(folder: Path, ts: str, topic: str, **fields) -> None:
    data = {"timestamp": ts, "thread_id": None, "topic": topic, "summary": "", "question": "", "response": "", "full_history": []}
    data.update(fields)
    (folder / f"conversation_{ts}_{topic}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """data/42 under tmp_path with two conversations, a profile and a transcript."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    root = tmp_path / "data" / USER
    conv = root / "conversations"
    conv.mkdir(parents=True)
    conversation(conv, "20261001_120000", "lisbon_trip",
                 summary="The user plans a trip to Lisbon in May and asked about flights.",
                 question="Message received time in UTC+0: 2026-10-01 12:00:00\n\nWhat is the cheapest way to fly to Lisbon?",
                 response="Flying to Lisbon is cheapest midweek. " + "More details. " * 40)
    conversation(conv, "20260915_090000", "beekeeping_talk",
                 summary="The user prepares a talk on urban beekeeping.",
                 question="Help me outline a beekeeping talk", response="Sure, here is an outline for the beekeeping talk.")
    (conv / "conversation_20260901_000000_broken.json").write_text("{not json", encoding="utf-8")
    (root / "profile.md").write_text("## About\n- Lives in Lisbon\n\n## Notes\n- [2026-10-10] Talk on beekeeping in November\n", encoding="utf-8")
    edited = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(root / "profile.md", (edited, edited))
    transcripts = root / "transcripts"
    transcripts.mkdir()
    transcript = {
        "version": 1, "user_id": USER, "thread_id": None,
        "created": "2026-10-14T07:00:00+00:00", "updated": "2026-10-16T08:00:05+00:00",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": SUMMARY}]},  # what prune leaves on top
            {"role": "user", "content": [
                {"type": "text", "text": "<context>\nCurrent time in UTC+0: 2026-10-16 08:00:00\n<memory>\n• [x] Lisbon context line\n</memory>\n</context>"},
                {"type": "text", "text": "Remind me about the Lisbon flights tomorrow"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "TGlzYm9u" * 50}},
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Lisbon flights... " + TOOL_ONLY, "signature": "s"},
                {"type": "text", "text": "Sure, I will remind you about the Lisbon flights."},
                {"type": "tool_use", "id": "t1", "name": "schedule_reminder", "input": {"text": "Lisbon flights"}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": f"Reminder set: Lisbon flights {TOOL_ONLY}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done, reminder set."}]},
        ],
    }
    (transcripts / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    (transcripts / "topic_7_transcript.json").write_text(json.dumps({
        "version": 1, "user_id": USER, "thread_id": 7,
        "created": "2026-10-11T09:30:00+00:00", "updated": "2026-10-12T10:00:00+00:00",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "In this topic we talk about beekeeping only"}]}],
    }), encoding="utf-8")
    (transcripts / "topic_9_transcript.json").write_text(json.dumps({  # an old file: no created, seeded plain strings
        "version": 1, "user_id": USER, "thread_id": 9, "updated": "2026-10-16T09:00:00+00:00",
        "messages": [{"role": "user", "content": "Seeded legacy question about an apiary"}, {"role": "assistant", "content": "ok"}],
    }), encoding="utf-8")
    return root


@pytest.fixture
def tools(workspace):
    return SearchTools(USER)


def hits(result: str) -> list[dict]:
    data = json.loads(result)
    assert isinstance(data, list)
    for hit in data:
        assert set(hit) == {"source", "timestamp", "topic", "field", "snippet"}
    return data


def test_finds_hits_in_conversations_profile_and_transcript_newest_first(tools):
    found = hits(tools.memory_search("lisbon"))
    assert {h["source"] for h in found} == {"conversation", "profile", "transcript"}
    stamps = [h["timestamp"] for h in found]
    assert stamps == sorted(stamps, reverse=True) and stamps[0] == "2026-10-16 08:00:00"  # the transcript turn's time
    conversation_hits = [h for h in found if h["source"] == "conversation"]
    assert {h["field"] for h in conversation_hits} == {"summary", "question", "response", "topic"}
    assert all(h["timestamp"] == "2026-10-01 12:00:00" and h["topic"] == "lisbon_trip" for h in conversation_hits)
    (profile_hit,) = [h for h in found if h["source"] == "profile"]
    assert profile_hit["field"] == "About" and profile_hit["snippet"] == "- Lives in Lisbon"
    assert profile_hit["timestamp"] == "2026-10-05 12:00:00"  # the file's last edit
    transcript_hits = [h for h in found if h["source"] == "transcript"]
    assert [h["field"] for h in transcript_hits] == ["user", "assistant"]
    assert all(h["topic"] == "current conversation" for h in transcript_hits)


def test_skips_context_blocks_tool_results_thinking_and_media(tools):
    assert tools.memory_search(TOOL_ONLY).startswith("No matches for")
    assert tools.memory_search("Lisbon context line").startswith("No matches for")
    assert tools.memory_search("TGlzYm9u").startswith("No matches for")  # the base64 is not text
    assert not any("Message received time" in h["snippet"] and h["source"] == "transcript" for h in hits(tools.memory_search("Lisbon")))


def test_profile_notes_and_topic_transcripts_are_searched(tools):
    found = hits(tools.memory_search("beekeeping"))
    (note_hit,) = [h for h in found if h["source"] == "profile"]
    assert note_hit["field"] == "Notes" and note_hit["timestamp"] == "2026-10-10 00:00:00"  # a dated note is as old as its date
    (topic_hit,) = [h for h in found if h["source"] == "transcript"]
    # no context block before it: the oldest content of that file, stamped with its creation time, not its last update
    assert topic_hit["topic"] == "topic 7" and topic_hit["timestamp"] == "2026-10-11 09:30:00"
    assert all(h["timestamp"] <= "2026-10-16 08:00:00" for h in found)


def test_messages_before_the_first_context_block_are_the_oldest(tools):
    """The summary prune leaves on top and a seeded transcript's plain strings precede every
    context block: they take the file's `created` time ("" without one: sorted last), never
    `updated`, which would put the oldest content first."""
    summary_hit, legacy_hit = hits(tools.memory_search("apiary"))
    assert summary_hit["topic"] == "current conversation" and summary_hit["timestamp"] == "2026-10-14 07:00:00"
    assert summary_hit["snippet"].startswith("<earlier_conversation_summary>")
    assert legacy_hit["topic"] == "topic 9" and legacy_hit["timestamp"] == "" and legacy_hit["field"] == "user"
    mixed = hits(tools.memory_search("apiary | Lisbon flights"))
    assert [h["timestamp"] for h in mixed][:3] == ["2026-10-16 08:00:00", "2026-10-16 08:00:00", "2026-10-14 07:00:00"]
    assert mixed[-1] is not None and mixed[-1]["timestamp"] == ""  # the undated legacy message comes last


def test_alternatives_case_and_the_cap(tools):
    both = hits(tools.memory_search("beekeeping | lisbon"))
    assert {h["source"] for h in both} >= {"conversation", "profile", "transcript"}
    assert len(both) > len(hits(tools.memory_search("beekeeping")))
    assert len(hits(tools.memory_search("lisbon", max_results=2))) == 2
    assert len(hits(tools.memory_search("lisbon", max_results=0))) <= MEMORY_SEARCH_MAX_RESULTS
    assert len(hits(tools.memory_search("lisbon", max_results="garbage"))) <= MEMORY_SEARCH_MAX_RESULTS
    assert tools.memory_search("LISBON", case_sensitive=True).startswith("No matches for LISBON")
    assert tools.memory_search("Lisbon", case_sensitive=True)
    assert tools.memory_search("   ").startswith("Error:")
    assert tools.memory_search("nothing like this anywhere") == "No matches for nothing like this anywhere in your memory (conversations, profile, transcript)."


def test_snippets_are_cut_around_the_match(tools):
    (response_hit,) = [h for h in hits(tools.memory_search("midweek")) if h["field"] == "response"]
    snippet = response_hit["snippet"]
    assert snippet.startswith("Flying to Lisbon is cheapest midweek.") and snippet.endswith("…")
    assert len(snippet) <= len("midweek") + 2 * 120 + 2


def test_a_user_without_memory_gets_no_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert SearchTools("nobody").memory_search("x").startswith("No matches for x")


def test_execute_tool_routes_with_the_new_parameters(tools):
    result = tools.execute_tool("memory_search", {"query": "lisbon", "max_results": 1})
    assert len(hits(result)) == 1
    (schema,) = [t for t in tools.tools_schema if t["name"] == "memory_search"]
    assert "profile" in schema["description"] and "transcript" in schema["description"]
    assert "do not remember" in schema["description"]
    assert set(schema["input_schema"]["properties"]) == {"query", "case_sensitive", "max_results"}


# ---- the sandbox patch is gone --------------------------------------------------
def test_tool_integrator_no_longer_names_memory_search():
    source = (Path(__file__).resolve().parents[1] / "app" / "secure_container" / "tool_integrator.py").read_text()
    assert "memory_search" not in source
    assert "memory_search" not in SecureToolWrapper.SECURE_TOOL_METHODS


def test_patch_search_tools_leaves_the_method_alone():
    original = SearchTools.memory_search
    ToolIntegrator.patch_search_tools(SimpleNamespace(secure_wrapper=None), SearchTools)
    assert SearchTools.memory_search is original and not hasattr(SearchTools, "secure_wrapper")
