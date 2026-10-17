"""Resolve model-supplied paths safely inside a user's workspace.

Every tool joins the model's path under ``data/<uid>/``. A plain ``Path.joinpath``
lets an absolute path or ``..`` escape that directory, so all lookups go through
:func:`resolve_under`, which also understands the two other spellings the model
sees for the same files: the sandbox mount (``/home/runner/workspace/...``) and
the host-relative form printed by tools (``data/<uid>/...``).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

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


def regular_files(folder: Path, pattern: str) -> Iterator[tuple[Path, str]]:
    """``(path, UTF-8 text)`` of each regular file directly in ``folder`` whose name matches ``pattern``.

    For host code that reads the user's workspace (``data/<uid>`` is the sandbox's mount, so
    anything in it may have been replaced by a symbolic link to another user's files or to
    ``/proc/self/environ``): a ``folder`` that is itself a link yields nothing, a file that is a
    link, a FIFO or anything but a regular file is skipped, and so is a file that is not UTF-8.
    The directory is opened once and every file relative to it with ``O_NOFOLLOW``, so a link
    swapped in between the listing and the read is not followed either. Never raises ``OSError``.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow or os.open not in os.supports_dir_fd or os.listdir not in os.supports_fd:
        yield from _regular_files_by_path(folder, pattern)  # no O_NOFOLLOW / dir_fd on this platform
        return
    try:
        dir_fd = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow)
    except FileNotFoundError:
        return
    except OSError as e:  # a link (ELOOP) or not a directory
        logger.warning("Not reading %s: %s", folder, e)
        return
    try:
        try:
            names = sorted(name for name in os.listdir(dir_fd) if fnmatch.fnmatchcase(name, pattern))
        except OSError as e:
            logger.warning("Cannot list %s: %s", folder, e)
            return
        for name in names:
            try:
                fd = os.open(name, os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0), dir_fd=dir_fd)
                with os.fdopen(fd, "rb") as f:
                    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                        continue
                    data = f.read()
            except OSError:
                continue  # a link, gone, or unreadable
            try:
                yield folder / name, data.decode("utf-8")
            except UnicodeDecodeError:
                continue
    finally:
        os.close(dir_fd)


def _regular_files_by_path(folder: Path, pattern: str) -> Iterator[tuple[Path, str]]:
    """:func:`regular_files` where the OS has no ``O_NOFOLLOW`` or ``dir_fd`` (checks, then reads)."""
    try:
        if folder.is_symlink() or not folder.is_dir():
            return
        paths = sorted(folder.glob(pattern))
    except OSError:
        return
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            yield path, path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
