"""Per-turn record of tool calls, rendered into the status message while the agent works.

The agent loop records every tool call here (``ToolTrace.start`` / ``ToolCall.done``);
``bot/agent_runner.py`` renders the trace under the "Thinking..." status and keeps a
one-line summary above the answer afterwards. Pure data, no Telegram or API imports.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

PREVIEW_CHARS = 60
ARGS_CHARS = 500  # JSON of a call's arguments kept for the expandable summary
RESULT_CHARS = 800  # beginning of a call's result kept for the expandable summary
THINKING_CHARS = 6000  # of the model's (summarized) thinking kept per turn
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
    result_excerpt: str = ""
    depth: int = 0  # >0 for calls made by a subagent (rendered indented)

    @property
    def args_text(self) -> str:
        """Arguments as compact JSON for the expandable summary (truncated)."""
        if not self.args:
            return ""
        try:
            text = json.dumps(self.args, ensure_ascii=False, indent=1)
        except (TypeError, ValueError):
            text = str(self.args)
        return text if len(text) <= ARGS_CHARS else text[: ARGS_CHARS - 1] + "…"

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
        text = (result or "").strip()
        self.result_excerpt = text if len(text) <= RESULT_CHARS else text[: RESULT_CHARS - 1] + "…"


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
    usage_by_model: dict = field(default_factory=dict)  # model -> the same five counters
    thinking: list[str] = field(default_factory=list)  # summarized thinking of the turn, in order

    def add_usage(self, usage, model: str | None = None) -> None:
        """Accumulate a Messages API ``usage`` object (attributes or dict), per model too."""
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else (lambda k, d=None: getattr(usage, k, d))
        counts = {
            "api_calls": 1,
            "input_tokens": int(get("input_tokens") or 0),
            "output_tokens": int(get("output_tokens") or 0),
            "cache_read_tokens": int(get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(get("cache_creation_input_tokens") or 0),
        }
        for key, value in counts.items():
            setattr(self, key, getattr(self, key) + value)
        bucket = self.usage_by_model.setdefault(model or "unknown", {k: 0 for k in counts})
        for key, value in counts.items():
            bucket[key] += value

    def add_thinking(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.thinking.append(text)

    @property
    def thinking_text(self) -> str:
        text = "\n\n".join(self.thinking)
        return text if len(text) <= THINKING_CHARS else text[: THINKING_CHARS - 1] + "…"

    @property
    def cost_usd(self) -> float | None:
        """Estimated cost of the turn (models with a known price only), None when nothing is known."""
        from models import estimate_cost

        total, known = 0.0, False
        for model, u in self.usage_by_model.items():
            cost = estimate_cost(model, u["input_tokens"], u["output_tokens"], u["cache_read_tokens"], u["cache_write_tokens"])
            if cost is not None:
                total += cost
                known = True
        return total if known else None

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
