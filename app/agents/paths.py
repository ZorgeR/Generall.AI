"""Resolve model-supplied paths safely inside a user's workspace.

Every tool joins the model's path under ``data/<uid>/``. A plain ``Path.joinpath``
lets an absolute path or ``..`` escape that directory, so all lookups go through
:func:`resolve_under`, which also understands the two other spellings the model
sees for the same files: the sandbox mount (``/home/runner/workspace/...``) and
the host-relative form printed by tools (``data/<uid>/...``).
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable

SANDBOX_WORKSPACE = PurePosixPath("/home/runner/workspace")


def _strip_known_prefixes(user_path: str, base: Path) -> str:
    """Map the sandbox mount and the host-relative spelling of the workspace onto ``base``."""
    text = user_path.strip().replace("\\", "/")
    posix = PurePosixPath(text)
    if posix.is_absolute():
        try:
            return str(posix.relative_to(SANDBOX_WORKSPACE))
        except ValueError:
            return text  # some other absolute path: judged by resolve() below
    if text.startswith("./"):
        text = text[2:]
        posix = PurePosixPath(text)
    base_posix = PurePosixPath(base.as_posix().removeprefix("./"))
    prefixes = [base_posix]
    if len(base_posix.parts) >= 2:  # the host spelling "data/<uid>/..." even when base is absolute
        prefixes.append(PurePosixPath(*base_posix.parts[-2:]))
    for prefix in prefixes:
        try:
            return str(posix.relative_to(prefix))
        except ValueError:
            continue
    return text


def resolve_under(
    base: Path,
    user_path: str,
    *,
    must_exist: bool = True,
    fallback_dirs: Iterable[str] = (),
) -> Path | None:
    """Return the path ``user_path`` names inside ``base``, or None if it escapes it.

    With ``must_exist`` the first existing candidate is returned: the path
    itself, then ``base/<dir>/<basename>`` for each of ``fallback_dirs`` (tools
    accept a bare file name for images/videos). Without it the primary
    candidate is returned whether or not it exists (for files about to be
    created). Symlinks are resolved before the containment check.
    """
    if not user_path or not str(user_path).strip():
        return None
    base_resolved = base.resolve()
    rel = _strip_known_prefixes(str(user_path), base)
    primary = Path(rel) if Path(rel).is_absolute() else base / rel

    def _inside(candidate: Path) -> Path | None:
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            return None
        return resolved if resolved == base_resolved or resolved.is_relative_to(base_resolved) else None

    safe_primary = _inside(primary)
    if not must_exist:
        return safe_primary
    if safe_primary is not None and safe_primary.exists():
        return safe_primary
    name = Path(rel).name
    if name:
        for sub in fallback_dirs:
            candidate = _inside(base / sub / name)
            if candidate is not None and candidate.exists():
                return candidate
    return None
