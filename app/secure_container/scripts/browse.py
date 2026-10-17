#!/usr/bin/env python3
"""Read a web page with headless Chromium inside the sandbox and print ONE JSON line.

Runs where ``ContainerManager.run_script`` puts it: the user's workspace, as the non-root
``runner`` user, in a throwaway container with network on. Everything the host needs comes
back as the last line of stdout::

    {"title", "final_url", "text", "links": [[text, href], ...], "screenshot", "error"}

``text`` is trafilatura's reading of the rendered HTML (tables kept, link markup dropped) or,
when trafilatura finds nothing, the page's visible text. Any failure, including a page that
never settles, ends in the same JSON shape with ``error`` set and exit code 0: the host tool
decides what to tell the model. A watchdog thread enforces the overall deadline because a
hung browser process would otherwise only die with the container timeout and lose the logs.

Usage: browse.py URL [--screenshot PATH] [--wait-ms 1500] [--max-chars 60000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from urllib.parse import urlparse

DEADLINE_SECONDS = 60
NETWORK_IDLE_CAP_MS = 15_000
MAX_LINKS = 50
VIEWPORT = {"width": 1280, "height": 900}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
CHROMIUM_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]

_output_lock = threading.Lock()
_printed = False


def emit(result: dict) -> None:
    """Print the single JSON result line exactly once (the watchdog and main thread race)."""
    global _printed
    with _output_lock:
        if _printed:
            return
        _printed = True
        sys.stdout.flush()
        # ensure_ascii (the default) keeps U+2028/U+2029 escaped: the host splits the logs on "\n"
        print(json.dumps(result))
        sys.stdout.flush()


def result(title=None, final_url=None, text="", links=None, screenshot=None, error=None) -> dict:
    return {
        "title": title,
        "final_url": final_url,
        "text": text or "",
        "links": links or [],
        "screenshot": screenshot,
        "error": error,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a page with headless Chromium and print JSON.")
    parser.add_argument("url")
    parser.add_argument("--screenshot", default="", help="workspace-relative PNG path, or empty for none")
    parser.add_argument("--wait-ms", type=int, default=1500, help="extra settle time after load")
    parser.add_argument("--max-chars", type=int, default=60_000, help="cap on the extracted text")
    return parser.parse_args(argv)


def validate_url(url: str) -> str | None:
    """Only http(s) URLs with a host; the browser must not open file:// or chrome:// pages."""
    try:
        parsed = urlparse(url.strip())
    except Exception as e:  # noqa: BLE001
        return f"Invalid URL: {e}"
    if parsed.scheme not in ("http", "https"):
        return "Only http and https URLs can be opened"
    if not parsed.netloc:
        return "The URL has no host"
    return None


def extract_text(html: str, fallback: str) -> str:
    """trafilatura's main-content extraction, else the page's visible text."""
    try:
        import trafilatura  # type: ignore

        text = trafilatura.extract(html, include_tables=True, include_links=False)
    except Exception:  # noqa: BLE001 - missing or broken extractor: use the browser's text
        text = None
    return (text or fallback or "").strip()


def collect_links(page, limit: int = MAX_LINKS) -> list[list[str]]:
    """Up to ``limit`` (text, absolute href) pairs for http(s) anchors, duplicates dropped."""
    try:
        raw = page.evaluate(
            """(limit) => Array.from(document.querySelectorAll('a[href]'))
                 .map(a => [(a.innerText || a.textContent || '').trim().slice(0, 120), a.href])
                 .filter(([t, h]) => h && /^https?:/i.test(h))
                 .slice(0, limit * 3)"""
            , limit * 3,
        )
    except Exception:  # noqa: BLE001
        return []
    links, seen = [], set()
    for text, href in raw or []:
        if href in seen:
            continue
        seen.add(href)
        links.append([text, href])
        if len(links) >= limit:
            break
    return links


def browse(url: str, screenshot: str, wait_ms: int, max_chars: int) -> dict:
    from playwright.sync_api import sync_playwright  # imported late: the import itself may fail

    started = time.monotonic()

    def remaining_ms(cap: int | None = None) -> float:
        left = max(1.0, (DEADLINE_SECONDS - (time.monotonic() - started)) * 1000 - 500)
        return min(left, cap) if cap else left

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        try:
            context = browser.new_context(user_agent=USER_AGENT, viewport=VIEWPORT, ignore_https_errors=True)
            page = context.new_page()
            page.on("dialog", lambda dialog: dialog.dismiss())
            page.goto(url, wait_until="domcontentloaded", timeout=remaining_ms())
            try:
                page.wait_for_load_state("networkidle", timeout=remaining_ms(NETWORK_IDLE_CAP_MS))
            except Exception:  # noqa: BLE001 - busy pages never go idle; read what is there
                pass
            if wait_ms > 0:
                page.wait_for_timeout(min(wait_ms, remaining_ms()))
            html = page.content()
            try:
                body_text = page.inner_text("body", timeout=remaining_ms(5000))
            except Exception:  # noqa: BLE001
                body_text = ""
            text = extract_text(html, body_text)
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + "\n[... truncated]"
            links = collect_links(page)
            shot = None
            if screenshot:
                os.makedirs(os.path.dirname(screenshot) or ".", exist_ok=True)
                page.screenshot(path=screenshot, full_page=False, timeout=remaining_ms(10_000))
                shot = screenshot
            return result(title=page.title(), final_url=page.url, text=text, links=links, screenshot=shot)
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    error = validate_url(args.url)
    if error:
        emit(result(final_url=args.url, error=error))
        return 0

    def watchdog():
        emit(result(final_url=args.url, error=f"Timed out after {DEADLINE_SECONDS} s while loading the page"))
        os._exit(0)

    timer = threading.Timer(DEADLINE_SECONDS, watchdog)
    timer.daemon = True
    timer.start()
    try:
        emit(browse(args.url, args.screenshot, max(0, args.wait_ms), max(0, args.max_chars)))
    except Exception as e:  # noqa: BLE001 - every failure is reported as data, never a traceback
        emit(result(final_url=args.url, error=f"{type(e).__name__}: {e}"))
    finally:
        timer.cancel()
    return 0


if __name__ == "__main__":
    sys.exit(main())
