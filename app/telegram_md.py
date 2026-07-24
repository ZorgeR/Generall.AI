"""Rich Markdown formatting for Telegram messages.

The Telegram Bot API renders two rich flavours - ``MarkdownV2`` and ``HTML`` -
and both expose the same entity set: bold, italic, underline, strikethrough,
spoiler, inline code, pre-formatted blocks with syntax highlighting, inline
links and blockquotes (regular and expandable).

The model writes ordinary (CommonMark-ish) Markdown: ``##`` headings,
``**bold**``, tables, nested lists, task lists, ``~~strikethrough~~``. Telegram
understands none of that - the legacy ``Markdown`` parse mode shows ``##`` and
``**`` verbatim and rejects the whole message the moment an unbalanced ``*`` or
a stray ``_`` shows up in a word like ``file_name``.

This module bridges the two: it parses the source Markdown into a small AST and
renders it into real Telegram entities, escaping everything that is not markup.
Long answers are split at block boundaries *before* rendering, so an entity is
never cut in half, and fenced code blocks are re-opened in the next message.

Typical usage::

    from telegram_md import reply_rich, edit_rich, send_rich

    await reply_rich(update.message, answer)
    await edit_rich(status_message, "*Done!*")
    await send_rich(context.bot, chat_id, answer, message_thread_id=thread_id)

Every sender falls back on its own: MarkdownV2 -> HTML -> plain text, per
message chunk, so a malformed fragment degrades to readable text instead of
raising.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MARKDOWN_V2 = "MarkdownV2"
HTML = "HTML"
PLAIN = "Plain"

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_MAX_CAPTION_LENGTH = 1024

#: Quotes longer than this are sent as expandable blockquotes.
EXPANDABLE_QUOTE_LINES = 8
EXPANDABLE_QUOTE_CHARS = 420

#: Tables wider than this are rendered as records instead of a monospace grid,
#: because a wide grid forces horizontal scrolling on phones.
MAX_TABLE_WIDTH = 58

_MAX_INLINE_DEPTH = 8

# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #


@dataclass
class Text:
    value: str


@dataclass
class Style:
    """An inline entity: bold, italic, underline, strike or spoiler."""

    kind: str
    children: list


@dataclass
class Code:
    value: str


@dataclass
class Link:
    href: str
    children: list


@dataclass
class Paragraph:
    children: list


@dataclass
class Heading:
    level: int
    children: list


@dataclass
class CodeBlock:
    language: str
    value: str


@dataclass
class Quote:
    blocks: list
    expandable: bool = False


@dataclass
class ListItem:
    marker: str
    depth: int
    children: list


@dataclass
class ListBlock:
    items: list


@dataclass
class Table:
    header: list
    rows: list


@dataclass
class Rule:
    pass


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #

_MDV2_SPECIAL = set("_*[]()~`>#+-=|{}.!\\")
_MD_PUNCTUATION = set("\\`*_{}[]()#+-.!|~>$")


def escape_markdown_v2(text: str) -> str:
    """Escape text so Telegram treats it as literal characters."""
    return "".join("\\" + ch if ch in _MDV2_SPECIAL else ch for ch in text)


def escape_markdown_v2_code(text: str) -> str:
    """Inside code entities only the backslash and the backtick are special."""
    return text.replace("\\", "\\\\").replace("`", "\\`")


def escape_markdown_v2_url(text: str) -> str:
    """Inside a link target only ``)`` and the backslash are special."""
    return text.replace("\\", "\\\\").replace(")", "\\)")


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Inline parsing
# --------------------------------------------------------------------------- #

_INLINE_CODE_RE = re.compile(r"(?P<fence>`+)(?P<code>[^\n]*?)(?P=fence)")
_LINK_RE = re.compile(r"\[(?P<label>(?:[^\[\]\\]|\\.|\[[^\]]*\])*)\]\((?P<href>[^()\s]*(?:\([^()]*\))?[^()\s]*)(?:\s+\"[^\"]*\")?\)")
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<href>[^()\s]*)(?:\s+\"[^\"]*\")?\)")
_AUTOLINK_RE = re.compile(r"<(?P<href>(?:https?|tg)://[^>\s]+)>")
_MATH_RE = re.compile(r"\$\$(?P<math>[^\n$]+?)\$\$")

# Longest markers first: ``**`` must win over ``*``.
_MARKERS = (
    ("||", "spoiler"),
    ("**", "bold"),
    ("__", "underline"),
    ("~~", "strike"),
    ("*", "bold_or_italic"),
    ("_", "italic"),
    ("~", "strike"),
)

_WORDISH = re.compile(r"[\w\d]")


def _is_wordish(text: str, index: int) -> bool:
    return 0 <= index < len(text) and bool(_WORDISH.match(text[index]))


def _find_closing(text: str, marker: str, start: int) -> int:
    """Find the closing ``marker`` for an entity opened just before ``start``."""
    pos = start
    while True:
        pos = text.find(marker, pos)
        if pos == -1:
            return -1
        if text[pos - 1] == "\\" and (pos < 2 or text[pos - 2] != "\\"):
            pos += len(marker)
            continue
        # ``*`` inside ``**`` (or ``_`` inside ``__``) - keep looking for the pair.
        if len(marker) == 1 and text[pos : pos + 2] == marker * 2:
            pos += 2
            continue
        return pos


def parse_inline(text: str, depth: int = 0) -> list:
    """Parse inline Markdown into a list of AST nodes."""
    nodes: list = []
    buffer: list = []
    index = 0
    length = len(text)

    def flush() -> None:
        if buffer:
            nodes.append(Text("".join(buffer)))
            buffer.clear()

    while index < length:
        char = text[index]

        if char == "\\" and index + 1 < length and text[index + 1] in _MD_PUNCTUATION:
            buffer.append(text[index + 1])
            index += 2
            continue

        if char == "`":
            match = _INLINE_CODE_RE.match(text, index)
            if match and match.group("code").strip():
                flush()
                nodes.append(Code(match.group("code").strip()))
                index = match.end()
                continue

        if char == "$" and text.startswith("$$", index):
            match = _MATH_RE.match(text, index)
            if match:
                flush()
                nodes.append(Code(match.group("math").strip()))
                index = match.end()
                continue

        if char == "!" and text.startswith("![", index):
            match = _IMAGE_RE.match(text, index)
            if match:
                flush()
                alt = match.group("alt").strip() or "image"
                nodes.append(Link(match.group("href"), [Text(f"🖼 {alt}")]))
                index = match.end()
                continue

        if char == "[":
            match = _LINK_RE.match(text, index)
            if match:
                flush()
                label = match.group("label")
                children = parse_inline(label, depth + 1) if depth < _MAX_INLINE_DEPTH else [Text(label)]
                nodes.append(Link(match.group("href"), children or [Text(match.group("href"))]))
                index = match.end()
                continue

        if char == "<":
            match = _AUTOLINK_RE.match(text, index)
            if match:
                flush()
                href = match.group("href")
                nodes.append(Link(href, [Text(href)]))
                index = match.end()
                continue

        if depth < _MAX_INLINE_DEPTH:
            node, consumed = _try_style(text, index, depth)
            if node is not None:
                flush()
                nodes.append(node)
                index += consumed
                continue

        buffer.append(char)
        index += 1

    flush()
    return nodes


def _try_style(text: str, index: int, depth: int):
    """Try to read a styled span starting at ``index``.

    Returns ``(node, consumed)`` or ``(None, 0)``.
    """
    for marker, kind in _MARKERS:
        if not text.startswith(marker, index):
            continue

        # ``_`` and ``__`` must not fire inside identifiers such as ``file_name``.
        if marker[0] == "_" and _is_wordish(text, index - 1):
            continue

        content_start = index + len(marker)
        close = content_start
        for _ in range(16):
            close = _find_closing(text, marker, close)
            if close == -1:
                break

            content = text[content_start:close]
            if not content or content != content.strip():
                close += len(marker)
                continue
            # ``_step_name_`` closes at the *last* underscore, not the first.
            if marker[0] == "_" and _is_wordish(text, close + len(marker)):
                close += len(marker)
                continue

            if kind == "bold_or_italic":
                # Single ``*`` is italic in CommonMark, but the model uses it
                # for bold far more often; ``**`` already covers bold, so
                # ``*x*`` maps to italic to keep both available.
                kind = "italic"

            children = parse_inline(content, depth + 1)
            if not children:
                break
            return Style(kind, children), (close + len(marker)) - index

    return None, 0


# --------------------------------------------------------------------------- #
# Block parsing
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"^\s{0,3}(?P<fence>```|~~~)\s*(?P<lang>[\w+#.\-]*)\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*#*\s*$")
_RULE_RE = re.compile(r"^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(?P<content>.*)$")
_LIST_RE = re.compile(r"^(?P<indent>\s*)(?P<marker>[-*+]|\d{1,3}[.)])\s+(?P<content>.*)$")
_TASK_RE = re.compile(r"^\[(?P<state>[ xX])\]\s+(?P<content>.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")
_LANG_SAFE_RE = re.compile(r"[^\w+#.\-]")

_BULLETS = ("•", "◦", "▪", "·")


def _looks_like_table_row(line: str) -> bool:
    return line.count("|") >= 2


def _split_table_row(line: str) -> list:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def parse_blocks(text: str) -> list:
    """Parse Markdown source into a list of block nodes."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list = []
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]

        if not line.strip():
            index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            language = _LANG_SAFE_RE.sub("", fence.group("lang") or "")
            closing = fence.group("fence")
            index += 1
            body: list = []
            while index < total:
                candidate = lines[index]
                if candidate.strip().startswith(closing):
                    index += 1
                    break
                body.append(candidate)
                index += 1
            blocks.append(CodeBlock(language, "\n".join(body).strip("\n")))
            continue

        if _RULE_RE.match(line):
            blocks.append(Rule())
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            title = heading.group("title").strip()
            if title:
                blocks.append(Heading(len(heading.group("hashes")), parse_inline(title)))
            index += 1
            continue

        if _QUOTE_RE.match(line):
            quoted: list = []
            while index < total and (_QUOTE_RE.match(lines[index]) or (quoted and lines[index].strip() and not _is_block_start(lines[index]))):
                match = _QUOTE_RE.match(lines[index])
                quoted.append(match.group("content") if match else lines[index].strip())
                index += 1
            inner = "\n".join(quoted).strip("\n")
            expandable = len(quoted) > EXPANDABLE_QUOTE_LINES or len(inner) > EXPANDABLE_QUOTE_CHARS
            blocks.append(Quote(parse_blocks(inner), expandable))
            continue

        if _looks_like_table_row(line) and index + 1 < total and _TABLE_SEP_RE.match(lines[index + 1]):
            header = _split_table_row(line)
            index += 2
            rows = []
            while index < total and _looks_like_table_row(lines[index]) and lines[index].strip():
                rows.append(_split_table_row(lines[index]))
                index += 1
            blocks.append(Table(header, rows))
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            items, index = _parse_list(lines, index)
            blocks.append(ListBlock(items))
            continue

        paragraph: list = []
        while index < total and lines[index].strip() and not _is_block_start(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        if paragraph:
            blocks.append(Paragraph(parse_inline("\n".join(paragraph))))
        else:  # pragma: no cover - defensive, keeps the loop moving
            index += 1

    return blocks


def _is_block_start(line: str) -> bool:
    return bool(
        _FENCE_RE.match(line)
        or _HEADING_RE.match(line)
        or _RULE_RE.match(line)
        or _QUOTE_RE.match(line)
        or _LIST_RE.match(line)
    )


def _parse_list(lines: list, index: int):
    """Read a (possibly nested) list starting at ``index``."""
    items: list = []
    total = len(lines)
    indents: list = []

    while index < total:
        line = lines[index]
        if not line.strip():
            # A blank line ends the list unless another item follows directly.
            if index + 1 < total and _LIST_RE.match(lines[index + 1]):
                index += 1
                continue
            break

        match = _LIST_RE.match(line)
        if not match:
            if items and not _is_block_start(line):
                # Lazy continuation line - append to the previous item.
                items[-1].children.extend(parse_inline(" " + line.strip()))
                index += 1
                continue
            break

        indent = len(match.group("indent").replace("\t", "    "))
        while indents and indents[-1] > indent:
            indents.pop()
        if not indents or indents[-1] < indent:
            indents.append(indent)
        depth = max(0, len(indents) - 1)

        raw_marker = match.group("marker")
        content = match.group("content")

        task = _TASK_RE.match(content)
        if task:
            marker = "☑" if task.group("state").lower() == "x" else "☐"
            content = task.group("content")
        elif raw_marker[-1] in ".)":
            marker = raw_marker
        else:
            marker = _BULLETS[min(depth, len(_BULLETS) - 1)]

        items.append(ListItem(marker, depth, parse_inline(content.strip())))
        index += 1

    return items, index


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class _Renderer:
    """Base renderer - subclasses provide the entity syntax."""

    flavour = PLAIN

    def escape(self, text: str) -> str:
        return text

    def wrap(self, kind: str, content: str) -> str:
        return content

    def code(self, text: str) -> str:
        return text

    def pre(self, text: str, language: str) -> str:
        return text

    def link(self, href: str, label: str) -> str:
        return label

    def quote(self, content: str, expandable: bool) -> str:
        return "\n".join("| " + line for line in content.split("\n"))

    # -- shared logic ----------------------------------------------------- #

    def join(self, parts: list) -> str:
        return "".join(parts)

    def render_inline(self, nodes: list, in_link: bool = False) -> str:
        out = []
        for node in nodes:
            if isinstance(node, Text):
                out.append(self.escape(node.value))
            elif isinstance(node, Code):
                out.append(self.code(node.value))
            elif isinstance(node, Style):
                out.append(self.wrap(node.kind, self.render_inline(node.children, in_link)))
            elif isinstance(node, Link):
                label = self.render_inline(node.children, True)
                # Telegram has no nested links - keep the label, drop the
                # inner target.
                out.append(label if in_link else self.link(node.href, label))
        return self.join(out)

    def render_blocks(self, blocks: list, in_quote: bool = False) -> str:
        parts = []
        for block in blocks:
            rendered = self.render_block(block, in_quote=in_quote)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)

    def render_block(self, block, in_quote: bool = False) -> str:
        if isinstance(block, Paragraph):
            return self.render_inline(block.children)

        if isinstance(block, Heading):
            content = self.render_inline(block.children)
            if block.level <= 2:
                return self.wrap("bold", self.wrap("underline", content))
            return self.wrap("bold", content)

        if isinstance(block, CodeBlock):
            if in_quote:
                # Telegram forbids pre entities inside blockquotes, so keep the
                # code readable as one monospace span per line.
                return "\n".join(self.code(line) if line.strip() else "" for line in block.value.split("\n"))
            return self.pre(block.value, block.language)

        if isinstance(block, Quote):
            inner = self.render_blocks(block.blocks, in_quote=True)
            if in_quote:
                # Telegram has no nested blockquotes - flatten instead of
                # emitting a second quote marker, which would be rejected.
                return inner
            return self.quote(inner, block.expandable)

        if isinstance(block, ListBlock):
            lines = []
            for item in block.items:
                indent = "  " * item.depth
                marker = self.wrap("bold", self.escape(item.marker)) if item.marker[-1] in ".)" else self.escape(item.marker)
                lines.append(f"{indent}{marker} {self.render_inline(item.children)}")
            return "\n".join(lines)

        if isinstance(block, Table):
            return self.render_table(block, in_quote=in_quote)

        if isinstance(block, Rule):
            return self.escape("──────────")

        return ""

    def render_table(self, table: Table, in_quote: bool = False) -> str:
        grid = _table_grid(table)
        if grid is not None and not in_quote:
            return self.pre(grid, "")
        return self.render_table_records(table)

    def render_table_records(self, table: Table) -> str:
        blocks = []
        for row in table.rows:
            lines = []
            title = row[0] if row else ""
            if title:
                lines.append(self.wrap("bold", self.render_inline(parse_inline(title))))
            for header, cell in zip(table.header[1:], row[1:]):
                if not cell:
                    continue
                label = self.wrap("italic", self.escape(header)) if header else ""
                value = self.render_inline(parse_inline(cell))
                lines.append(f"{label}: {value}" if label else value)
            if lines:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _table_grid(table: Table) -> Optional[str]:
    """Render a table as an aligned monospace grid, or ``None`` if too wide."""
    rows = [table.header] + list(table.rows)
    columns = max((len(row) for row in rows), default=0)
    if not columns:
        return None

    normalised = [[_strip_inline_markup(row[i]) if i < len(row) else "" for i in range(columns)] for row in rows]
    widths = [max(len(row[i]) for row in normalised) for i in range(columns)]
    total = sum(widths) + 3 * (columns - 1)
    if total > MAX_TABLE_WIDTH:
        return None

    def line(cells: list) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(normalised[0]), "─" * min(total, MAX_TABLE_WIDTH)]
    out.extend(line(row) for row in normalised[1:])
    return "\n".join(out)


class MarkdownV2Renderer(_Renderer):
    flavour = MARKDOWN_V2

    def escape(self, text: str) -> str:
        return escape_markdown_v2(text)

    _MARKERS = {
        "bold": "*",
        "italic": "_",
        "underline": "__",
        "strike": "~",
        "spoiler": "||",
    }

    @staticmethod
    def _ends_unescaped(content: str, char: str) -> bool:
        if not content.endswith(char):
            return False
        trailing = 0
        index = len(content) - len(char) - 1
        while index >= 0 and content[index] == "\\":
            trailing += 1
            index -= 1
        return trailing % 2 == 0

    def join(self, parts: list) -> str:
        """Concatenate siblings, keeping their markers apart.

        ``_italic___underline__`` is ambiguous - Telegram reads the run of
        underscores greedily and ends up with crossing entities - so a
        carriage return is inserted between two touching markers.
        """
        out = ""
        for part in parts:
            if out and part and part[0] in "_*~|" and self._ends_unescaped(out, part[0]):
                out += "\r"
            out += part
        return out

    def wrap(self, kind: str, content: str) -> str:
        if not content:
            return ""
        marker = self._MARKERS.get(kind)
        if marker is None:
            return content
        # A marker touching an identical character is ambiguous for Telegram
        # (``__`` is always read greedily as underline). A carriage return
        # between them disambiguates - the documented MarkdownV2 trick.
        if content.startswith(marker[-1]):
            content = "\r" + content
        if self._ends_unescaped(content, marker[0]):
            content += "\r"
        return f"{marker}{content}{marker}"

    def code(self, text: str) -> str:
        return f"`{escape_markdown_v2_code(text)}`"

    def pre(self, text: str, language: str) -> str:
        return f"```{language}\n{escape_markdown_v2_code(text)}\n```"

    def link(self, href: str, label: str) -> str:
        if not href:
            return label
        return f"[{label}]({escape_markdown_v2_url(href)})"

    def quote(self, content: str, expandable: bool) -> str:
        lines = content.split("\n")
        quoted = "\n".join(">" + line for line in lines)
        if expandable:
            quoted = "**" + quoted + "||"
        return quoted


class HtmlRenderer(_Renderer):
    flavour = HTML

    _TAGS = {
        "bold": ("<b>", "</b>"),
        "italic": ("<i>", "</i>"),
        "underline": ("<u>", "</u>"),
        "strike": ("<s>", "</s>"),
        "spoiler": ('<span class="tg-spoiler">', "</span>"),
    }

    def escape(self, text: str) -> str:
        return escape_html(text)

    def wrap(self, kind: str, content: str) -> str:
        if not content:
            return ""
        open_tag, close_tag = self._TAGS.get(kind, ("", ""))
        return f"{open_tag}{content}{close_tag}"

    def code(self, text: str) -> str:
        return f"<code>{escape_html(text)}</code>"

    def pre(self, text: str, language: str) -> str:
        body = escape_html(text)
        if language:
            return f'<pre><code class="language-{language}">{body}</code></pre>'
        return f"<pre>{body}</pre>"

    def link(self, href: str, label: str) -> str:
        if not href:
            return label
        return f'<a href="{escape_html(href)}">{label}</a>'

    def quote(self, content: str, expandable: bool) -> str:
        tag = "<blockquote expandable>" if expandable else "<blockquote>"
        return f"{tag}{content}</blockquote>"


class PlainRenderer(_Renderer):
    flavour = PLAIN

    def link(self, href: str, label: str) -> str:
        # Without entities the target would be lost, so keep it visible.
        if not href or href == label:
            return label or href
        return f"{label} ({href})"


_RENDERERS = {
    MARKDOWN_V2: MarkdownV2Renderer(),
    HTML: HtmlRenderer(),
    PLAIN: PlainRenderer(),
}


def _strip_inline_markup(text: str) -> str:
    return _RENDERERS[PLAIN].render_inline(parse_inline(text))


def render(text: str, flavour: str = MARKDOWN_V2) -> str:
    """Convert Markdown ``text`` into Telegram markup of the given flavour."""
    if text is None:
        return ""
    renderer = _RENDERERS.get(flavour, _RENDERERS[PLAIN])
    rendered = renderer.render_blocks(parse_blocks(str(text)))
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


def render_plain(text: str) -> str:
    """Strip all markup, keeping the text readable."""
    return render(text, PLAIN)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def _wrap_long_lines(lines: list, budget: int) -> list:
    out = []
    for line in lines:
        while len(line) > budget:
            cut = line.rfind(" ", 0, budget)
            if cut <= 0:
                cut = budget
            out.append(line[:cut])
            line = line[cut:].lstrip()
        out.append(line)
    return out


def split_source(text: str, budget: int) -> list:
    """Split Markdown *source* into pieces of roughly ``budget`` characters.

    Splits happen at block boundaries; a piece that ends inside a fenced code
    block is closed and the next piece re-opens the fence with the same
    language, so every piece is valid Markdown on its own.
    """
    lines = _wrap_long_lines(text.split("\n"), max(budget, 32))

    # ``states[i]`` is the fence language open *before* line ``i`` (None if the
    # line is outside a code block).
    states: list = []
    open_lang: Optional[str] = None
    for line in lines:
        states.append(open_lang)
        fence = _FENCE_RE.match(line)
        if fence:
            open_lang = None if open_lang is not None else (fence.group("lang") or "")
    states.append(open_lang)

    pieces: list = []
    index = 0
    total = len(lines)

    while index < total:
        end = index
        size = 0
        last_blank = None
        while end < total:
            addition = len(lines[end]) + 1
            if size + addition > budget and end > index:
                break
            if not lines[end].strip() and states[end] is None:
                last_blank = end
            size += addition
            end += 1

        if end < total and last_blank is not None and last_blank > index:
            end = last_blank

        prefix = "```" + (states[index] or "") + "\n" if states[index] is not None else ""
        suffix = "\n```" if states[end] is not None else ""
        body = "\n".join(lines[index:end]).strip("\n")
        if body:
            pieces.append(prefix + body + suffix)

        index = end
        while index < total and not lines[index].strip() and states[index] is None:
            index += 1

    return pieces or ([text] if text.strip() else [])


def split_markdown(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list:
    """Split Markdown source into pieces that fit ``limit`` once rendered.

    The check is made against every rich flavour, so the same pieces can be
    rendered as MarkdownV2 or HTML without exceeding Telegram's limit.
    """
    text = (text or "").strip()
    if not text:
        return []

    if all(len(render(text, flavour)) <= limit for flavour in (MARKDOWN_V2, HTML)):
        return [text]

    budget = limit
    for _ in range(8):
        pieces = split_source(text, budget)
        if all(
            len(render(piece, flavour)) <= limit
            for piece in pieces
            for flavour in (MARKDOWN_V2, HTML)
        ):
            return pieces
        budget = max(200, int(budget * 0.7))

    # Nothing fit - fall back to hard slices of the plain text.
    plain = render_plain(text)
    return [plain[i : i + limit] for i in range(0, len(plain), limit)] or [text]


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - telegram is unavailable in unit tests
    from telegram.error import BadRequest, TelegramError
except Exception:  # pragma: no cover
    class BadRequest(Exception):
        pass

    class TelegramError(Exception):
        pass


_PARSE_ERROR_HINTS = (
    "can't parse entities",
    "can't find end",
    "unsupported start tag",
    "unclosed start tag",
    "entity",
    "tag",
    "reserved",
)


def _is_formatting_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(hint in message for hint in _PARSE_ERROR_HINTS)


def _is_not_modified(error: Exception) -> bool:
    return "not modified" in str(error).lower()


async def _send_with_fallback(send: Callable, source: str, **kwargs) -> Any:
    """Send one piece, degrading MarkdownV2 -> HTML -> plain text."""
    last_error: Optional[Exception] = None
    for flavour in (MARKDOWN_V2, HTML, PLAIN):
        body = render(source, flavour)
        if not body:
            return None
        try:
            if flavour is PLAIN:
                return await send(text=body, **kwargs)
            return await send(text=body, parse_mode=flavour, **kwargs)
        except BadRequest as error:
            if _is_not_modified(error):
                return None
            if not _is_formatting_error(error):
                raise
            logger.warning("Telegram rejected %s formatting (%s), falling back", flavour, error)
            last_error = error
        except TelegramError as error:
            raise error
    if last_error:  # pragma: no cover - plain text virtually never fails
        raise last_error
    return None


async def deliver_rich(send: Callable, text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH, **kwargs) -> list:
    """Render ``text`` and push it through ``send`` in as many parts as needed."""
    pieces = split_markdown(text, limit)
    sent = []
    for piece in pieces:
        result = await _send_with_fallback(send, piece, **kwargs)
        if result is not None:
            sent.append(result)
    return sent


async def reply_rich(message, text: str, **kwargs) -> list:
    """Reply to ``message`` with richly formatted ``text``."""
    return await deliver_rich(message.reply_text, text, **kwargs)


async def send_rich(bot, chat_id, text: str, message_thread_id: Optional[int] = None, **kwargs) -> list:
    """Send richly formatted ``text`` to ``chat_id``."""
    if message_thread_id:
        kwargs["message_thread_id"] = message_thread_id
    return await deliver_rich(
        lambda **payload: bot.send_message(chat_id=chat_id, **payload), text, **kwargs
    )


async def edit_rich(message, text: str, reply_to=None, **kwargs) -> list:
    """Replace ``message`` with richly formatted ``text``.

    Overflow is delivered as follow-up messages, so an answer that no longer
    fits a single message is never truncated.
    """
    pieces = split_markdown(text, TELEGRAM_MAX_MESSAGE_LENGTH)
    if not pieces:
        return []

    sent = []
    first = await _send_with_fallback(message.edit_text, pieces[0], **kwargs)
    if first is not None:
        sent.append(first)

    target = reply_to or message
    for piece in pieces[1:]:
        follow_up = await _send_with_fallback(target.reply_text, piece)
        if follow_up is not None:
            sent.append(follow_up)
    return sent


async def reply_status(message, text: str, **kwargs):
    """Send a short status message and return it, so it can be edited later."""
    return await _send_with_fallback(message.reply_text, text, **kwargs)


async def edit_status(message, text: str, **kwargs) -> None:
    """Best-effort status update - never raises, so it cannot break a handler."""
    try:
        await _send_with_fallback(message.edit_text, text, **kwargs)
    except Exception as error:  # pragma: no cover - status updates are cosmetic
        logger.warning("Failed to update status message: %s", error)


def format_caption(text: str) -> str:
    """Render ``text`` as a MarkdownV2 caption, trimmed to Telegram's limit."""
    pieces = split_markdown(text, TELEGRAM_MAX_CAPTION_LENGTH)
    return render(pieces[0], MARKDOWN_V2) if pieces else ""
