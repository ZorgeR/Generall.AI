#!/usr/bin/env python3
"""Render the production ``.env`` from the GitHub ``PROD`` environment.

Runs on the GitHub Actions runner (never on the server), called by
``.github/workflows/deploy.yml``::

    SECRETS_JSON='${{ toJSON(secrets) }}' VARS_JSON='${{ toJSON(vars) }}' \\
        python3 deploy/render_env.py "$RUNNER_TEMP/prod.env"

Rules, all implemented in :func:`render`:

* only keys that look like environment variables are kept: ``^[A-Z][A-Z0-9_]*$``;
* deploy-only keys never reach the server: ``SERVER_*`` (the SSH connection
  settings) and ``github_token`` / ``GITHUB_TOKEN``;
* a secret wins over a variable with the same name;
* values matching ``^[A-Za-z0-9_./:@+=,\\-]*$`` are written bare, anything else
  is double-quoted with ``\\`` and ``"`` backslash-escaped and ``$`` doubled
  (Docker Compose interpolates ``$VAR`` inside double quotes and reads ``$$``
  as a literal ``$``; an unescaped ``$`` can even make ``docker compose up``
  fail with "invalid interpolation format");
* values containing a newline are skipped and reported, Compose would read
  the file as broken.

The file only has to be understood by Docker Compose's ``env_file`` parser:
the bot image is built from ``app/`` and never contains the ``.env`` itself.

Stdlib only. Nothing here prints a value: stdout gets the sorted key names and
their count, stderr gets the skipped keys.
"""
from __future__ import annotations

import json
import os
import re
import sys

KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
BARE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@+=,\-]*$")
DEPLOY_ONLY_PREFIX = "SERVER_"
DEPLOY_ONLY_KEYS = {"GITHUB_TOKEN"}


def is_app_key(key: str) -> bool:
    """True for keys that belong in the application's ``.env``."""
    if not KEY_RE.match(key):
        return False
    if key.startswith(DEPLOY_ONLY_PREFIX):
        return False
    return key.upper() not in DEPLOY_ONLY_KEYS


def format_value(value: str) -> str:
    """Quote ``value`` the way Docker Compose's env_file parser expects."""
    if BARE_VALUE_RE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$")
    return f'"{escaped}"'


def _as_text(value: object) -> str:
    """GitHub sends strings; be tolerant of JSON scalars anyway (``true``, ``3``)."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value)


def render(
    secrets: dict | None,
    variables: dict | None,
    *,
    skipped: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Turn the two GitHub context objects into ``.env`` text.

    Returns ``(text, keys)`` where ``keys`` is the sorted list of keys that
    made it into ``text``. Keys whose value contains a newline are left out and
    appended to ``skipped`` when a list is given. Pure: no I/O.
    """
    merged: dict[str, str] = {}
    for source in (variables, secrets):  # secrets are applied last, so they win
        for key, value in (source or {}).items():
            if isinstance(key, str) and is_app_key(key):
                merged[key] = _as_text(value)

    lines: list[str] = []
    keys: list[str] = []
    for key in sorted(merged):
        value = merged[key]
        if "\n" in value or "\r" in value:
            if skipped is not None:
                skipped.append(key)
            continue
        lines.append(f"{key}={format_value(value)}")
        keys.append(key)
    return "".join(line + "\n" for line in lines), keys


def _load_json_object(env_name: str) -> dict:
    raw = os.environ.get(env_name, "").strip()
    if not raw or raw == "null":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{env_name} is not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise SystemExit(f"{env_name} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _write_private(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "usage: render_env.py OUTPUT_PATH   (reads SECRETS_JSON and VARS_JSON from the environment)",
            file=sys.stderr,
        )
        return 2
    output_path = argv[0]

    skipped: list[str] = []
    text, keys = render(_load_json_object("SECRETS_JSON"), _load_json_object("VARS_JSON"), skipped=skipped)
    _write_private(output_path, text)

    for key in skipped:
        print(f"skipped {key}: its value contains a newline", file=sys.stderr)
    if not keys:
        print("::error::No application keys found in the PROD environment; refusing to render an empty .env", file=sys.stderr)
        return 1
    print(f"Rendered {len(keys)} keys into {output_path}:")
    for key in keys:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
