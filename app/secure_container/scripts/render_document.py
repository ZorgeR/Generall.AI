#!/usr/bin/env python3
"""Render a document inside the sandbox: HTML -> PDF (Chromium), Markdown -> DOCX / XLSX.

Run by ``ContainerManager.run_script`` on behalf of ``agents.document_tools.DocumentTools``,
which renders md/html on the host and delegates only the formats that need a browser or a
heavy library here. The last line of stdout is ONE JSON object ``{"output", "bytes", "error"}``;
every failure sets ``error`` and exits 0 so the host gets the reason instead of a lost log.

The Markdown understood here is the subset the agent is told to write: ``#``..``###``
headings, paragraphs, ``-``/``*`` bullets and ``1.`` lists, ``**bold**`` / ``*italic*`` /
`` `code` `` runs, fenced code blocks (a ``csv`` fence becomes a sheet in xlsx) and pipe
tables. A tiny parser is kept on purpose: it is one file with no shared imports, and it
produces the same block list for both builders so a table lands as a Word table and as a
sheet from one reading.

Usage: render_document.py --format pdf|docx|xlsx --input PATH --output PATH [--title TEXT]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
from typing import Any

DEADLINE_SECONDS = 75
PDF_MARGIN = {"top": "15mm", "right": "15mm", "bottom": "15mm", "left": "15mm"}

_FENCE_RE = re.compile(r"^\s*```\s*([\w+-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")  # GFM: any number of dashes
# One capturing group: re.split() then alternates plain text (even indices) and matches (odd).
# ``*italic*`` must stand alone (no word character on either side) so ``2*3*4`` stays as
# written; ``_text_`` is deliberately NOT a style, snake_case identifiers are common in reports.
_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|(?<!\w)\*[^*\s][^*]*\*(?!\w))")
_NUMBER_CELL_RE = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")

_output_lock = threading.Lock()
_printed = False


def emit(output: str | None, nbytes: int = 0, error: str | None = None) -> None:
    global _printed
    with _output_lock:
        if _printed:
            return
        _printed = True
        sys.stdout.flush()
        # ensure_ascii (the default) keeps U+2028/U+2029 escaped: the host splits the logs on "\n"
        print(json.dumps({"output": output, "bytes": nbytes, "error": error}))
        sys.stdout.flush()


# ---- Markdown subset parser ------------------------------------------------------------
def split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def is_table_start(lines: list[str], i: int) -> bool:
    """A header line with pipes followed by a delimiter row of the same width (GFM)."""
    if i + 1 >= len(lines) or "|" not in lines[i] or not lines[i].strip():
        return False
    if _TABLE_SEP_RE.match(lines[i + 1]) is None:
        return False
    return len(split_table_row(lines[i + 1])) == len(split_table_row(lines[i]))


def parse_markdown(text: str) -> list[dict[str, Any]]:
    """Blocks: heading(level, text) | paragraph(text) | list(ordered, items) | code(lang, text)
    | table(header, rows). Blank lines separate paragraphs; everything unknown is a paragraph."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(s.strip() for s in paragraph)})
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_RE.match(line)
        if fence:
            flush_paragraph()
            lang, body = fence.group(1).lower(), []
            i += 1
            while i < len(lines) and not _FENCE_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            blocks.append({"type": "code", "lang": lang, "text": "\n".join(body)})
            i += 1
            continue
        if not line.strip():
            flush_paragraph()
            i += 1
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            blocks.append({"type": "heading", "level": len(heading.group(1)), "text": heading.group(2)})
            i += 1
            continue
        if is_table_start(lines, i):
            flush_paragraph()
            header = split_table_row(lines[i])
            rows = []
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                row = split_table_row(lines[i])
                row = (row + [""] * len(header))[: len(header)]
                rows.append(row)
                i += 1
            blocks.append({"type": "table", "header": header, "rows": rows})
            continue
        bullet, number = _BULLET_RE.match(line), _NUMBER_RE.match(line)
        if bullet or number:
            flush_paragraph()
            ordered = number is not None
            items = []
            while i < len(lines):
                m = (_NUMBER_RE if ordered else _BULLET_RE).match(lines[i])
                if not m:
                    break
                items.append(m.group(1).strip())
                i += 1
            blocks.append({"type": "list", "ordered": ordered, "items": items})
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return blocks


def inline_runs(text: str) -> list[tuple[str, str]]:
    """Split a line into (text, style) runs; style is "" | "bold" | "italic" | "code".

    Markers are stripped only from a part the regex matched; text between matches is kept
    verbatim, so nothing that merely looks like a marker (``_a_``, ``2*3*4``) loses characters."""
    runs: list[tuple[str, str]] = []
    for index, part in enumerate(_INLINE_RE.split(text)):
        if not part:
            continue
        if index % 2 == 0:
            runs.append((part, ""))
        elif part.startswith("**"):
            runs.append((part[2:-2], "bold"))
        elif part.startswith("`"):
            runs.append((part[1:-1], "code"))
        else:
            runs.append((part[1:-1], "italic"))
    return runs


def plain_text(text: str) -> str:
    return "".join(t for t, _ in inline_runs(text))


def cell_value(text: str) -> Any:
    """Numeric-looking cells become numbers so the sheet can be summed; the rest stays text."""
    raw = plain_text(text).strip()
    if _NUMBER_CELL_RE.match(raw):
        try:
            return int(raw) if re.fullmatch(r"[-+]?\d+", raw) else float(raw)
        except ValueError:
            return raw
    return raw


# ---- DOCX -------------------------------------------------------------------------------
def build_docx(markdown_text: str, title: str = ""):
    from docx import Document
    from docx.shared import Pt

    blocks = parse_markdown(markdown_text)
    doc = Document()
    if title:
        doc.core_properties.title = title
        if not any(b["type"] == "heading" and b["level"] == 1 for b in blocks):
            doc.add_heading(title, level=0)

    def add_runs(paragraph, text):
        for run_text, style in inline_runs(text):
            run = paragraph.add_run(run_text)
            if style == "bold":
                run.bold = True
            elif style == "italic":
                run.italic = True
            elif style == "code":
                run.font.name = "Courier New"
                run.font.size = Pt(10)

    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            doc.add_heading(plain_text(block["text"]), level=min(block["level"], 9))
        elif kind == "paragraph":
            add_runs(doc.add_paragraph(), block["text"])
        elif kind == "list":
            style = "List Number" if block["ordered"] else "List Bullet"
            for item in block["items"]:
                add_runs(doc.add_paragraph(style=style), item)
        elif kind == "code":
            for line in block["text"].split("\n") or [""]:
                run = doc.add_paragraph().add_run(line)
                run.font.name = "Courier New"
                run.font.size = Pt(9)
        elif kind == "table":
            header, rows = block["header"], block["rows"]
            table = doc.add_table(rows=1 + len(rows), cols=len(header))
            table.style = "Table Grid"
            for col, text in enumerate(header):  # a new cell holds one empty paragraph: add runs to it
                table.rows[0].cells[col].paragraphs[0].add_run(plain_text(text)).bold = True
            for r, row in enumerate(rows, start=1):
                for col, text in enumerate(row):
                    add_runs(table.rows[r].cells[col].paragraphs[0], text)
    return doc


# ---- XLSX -------------------------------------------------------------------------------
def sheet_name(candidate: str, taken: set[str]) -> str:
    base = re.sub(r"\s+", " ", _INVALID_SHEET_CHARS.sub(" ", plain_text(candidate))).strip()[:31] or "Sheet"
    name, n = base, 2
    while name.lower() in taken:
        suffix = f" ({n})"
        name = base[: 31 - len(suffix)] + suffix
        n += 1
    taken.add(name.lower())
    return name


def build_xlsx(markdown_text: str, title: str = ""):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    blocks = parse_markdown(markdown_text)
    wb = Workbook()
    wb.remove(wb.active)
    if title:
        wb.properties.title = title
    taken: set[str] = set()
    heading = ""
    tables = 0

    def add_sheet(name_hint: str, header: list[str] | None, rows: list[list[Any]]):
        ws = wb.create_sheet(sheet_name(name_hint, taken))
        if header:
            ws.append([plain_text(h) for h in header])
            for cell in ws[1]:
                cell.font = Font(bold=True)
        for row in rows:
            ws.append(row)
        for row_cells in ws.iter_rows():
            for cell in row_cells:  # openpyxl stores "=..." strings as formulas; these are table text
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cell.data_type = "s"
        for column in ws.columns:
            width = max((len(str(c.value)) for c in column if c.value is not None), default=8)
            ws.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 60)

    for block in blocks:
        if block["type"] == "heading":
            heading = block["text"]
        elif block["type"] == "table":
            tables += 1
            add_sheet(heading or f"Table {tables}", block["header"], [[cell_value(c) for c in r] for r in block["rows"]])
        elif block["type"] == "code" and block["lang"] == "csv":
            tables += 1
            parsed = list(csv.reader(io.StringIO(block["text"])))
            parsed = [r for r in parsed if any(c.strip() for c in r)]
            if parsed:
                add_sheet(heading or f"Table {tables}", parsed[0], [[cell_value(c) for c in r] for r in parsed[1:]])
    if not wb.sheetnames:
        ws = wb.create_sheet(sheet_name(title or "Document", taken))
        for line in markdown_text.replace("\r\n", "\n").split("\n"):
            ws.append([plain_text(line)])
    return wb


# ---- PDF --------------------------------------------------------------------------------
def render_pdf(input_path: str, output_path: str) -> None:
    from playwright.sync_api import sync_playwright

    with open(input_path, "r", encoding="utf-8") as f:
        html = f.read()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(path=output_path, format="A4", print_background=True, margin=PDF_MARGIN)
        finally:
            browser.close()


# ---- entry point ------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a document inside the sandbox.")
    parser.add_argument("--format", required=True, choices=("pdf", "docx", "xlsx"))
    parser.add_argument("--input", required=True, help="HTML file for pdf, Markdown file for docx/xlsx")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args(argv)


def render(fmt: str, input_path: str, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if fmt == "pdf":
        render_pdf(input_path, output_path)
        return
    with open(input_path, "r", encoding="utf-8") as f:
        source = f.read()
    if fmt == "docx":
        build_docx(source, title).save(output_path)
    elif fmt == "xlsx":
        build_xlsx(source, title).save(output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    def watchdog():
        emit(None, 0, f"Timed out after {DEADLINE_SECONDS} s while rendering the {args.format}")
        os._exit(0)

    timer = threading.Timer(DEADLINE_SECONDS, watchdog)
    timer.daemon = True
    timer.start()
    try:
        render(args.format, args.input, args.output, args.title)
        emit(args.output, os.path.getsize(args.output))
    except Exception as e:  # noqa: BLE001 - reported as data, never a traceback
        emit(None, 0, f"{type(e).__name__}: {e}")
    finally:
        timer.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
