"""Per-turn record of tool calls, rendered into the status message while the agent works.

The agent loop records every tool call here (``ToolTrace.start`` / ``ToolCall.done``);
``bot/agent_runner.py`` renders the trace under the "Thinking..." status and keeps a
one-line summary above the answer afterwards. Pure data, no Telegram or API imports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

PREVIEW_CHARS = 60
# Argument keys worth showing, most informative first.
_ARG_PRIORITY = ("query", "command", "script", "code", "url", "prompt", "text", "message", "path", "filename", "file_path")


def describe_args(args: dict | None, limit: int = PREVIEW_CHARS) -> str:
    """One short line describing a tool call's arguments (first string argument, truncated)."""
    if not isinstance(args, dict) or not args:
        return ""
    value = None
    for key in _ARG_PRIORITY:
        if isinstance(args.get(key), str) and args[key].strip():
            value = args[key]
            break
    if value is None:
        for candidate in args.values():
            if isinstance(candidate, str) and candidate.strip():
                value = candidate
                break
    if value is None:
        return ""
    line = " ".join(value.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _first_line(text: str, limit: int = PREVIEW_CHARS) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    ok: bool | None = None
    preview: str = ""
    depth: int = 0  # >0 for calls made by a subagent (rendered indented)

    @property
    def running(self) -> bool:
        return self.finished is None

    @property
    def duration(self) -> float:
        return (self.finished if self.finished is not None else time.monotonic()) - self.started

    def done(self, result: str, ok: bool) -> None:
        self.finished = time.monotonic()
        self.ok = ok
        self.preview = _first_line(result)


@dataclass
class TurnBudget:
    """Tool-call allowance of one turn, shared by the main agent and its subagents."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def take(self, n: int = 1) -> None:
        self.used += n


@dataclass
class ToolTrace:
    calls: list[ToolCall] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add_usage(self, usage) -> None:
        """Accumulate a Messages API ``usage`` object (attributes or dict)."""
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else (lambda k, d=None: getattr(usage, k, d))
        self.api_calls += 1
        self.input_tokens += int(get("input_tokens") or 0)
        self.output_tokens += int(get("output_tokens") or 0)
        self.cache_read_tokens += int(get("cache_read_input_tokens") or 0)
        self.cache_write_tokens += int(get("cache_creation_input_tokens") or 0)

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens served from cache (0..1)."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    def start(self, name: str, args: dict | None = None, depth: int = 0) -> ToolCall:
        call = ToolCall(name=name, args=dict(args or {}), depth=depth)
        self.calls.append(call)
        return call

    @property
    def total(self) -> int:
        return len(self.calls)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.calls if c.ok is False)

    @property
    def running(self) -> int:
        return sum(1 for c in self.calls if c.running)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def counts_by_name(self) -> list[tuple[str, int]]:
        """Tool names with their call counts, in first-seen order."""
        order: list[str] = []
        counts: dict[str, int] = {}
        for call in self.calls:
            if call.name not in counts:
                order.append(call.name)
            counts[call.name] = counts.get(call.name, 0) + 1
        return [(name, counts[name]) for name in order]
