"""Git-backed source freeze checks that permit approved runtime output."""

from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_FREEZE_TAG = "sigmetrics-2027-execution-v1"
APPROVED_RUNTIME_PREFIXES = ("outputs/sigmetrics/",)


def validate_execution_freeze(
    root: Path,
    *,
    tag: str = DEFAULT_FREEZE_TAG,
    expected_commit: str | None = None,
) -> dict[str, object]:
    """Require an annotated tag at HEAD and an unchanged tracked frozen tree."""
    commit = _git(root, "rev-parse", "HEAD")
    tag_commit = _git(root, "rev-list", "-n", "1", tag)
    tag_type = _git(root, "cat-file", "-t", tag)
    if tag_type != "tag" or tag_commit != commit:
        raise RuntimeError(f"Final execution requires annotated freeze tag {tag!r} at HEAD.")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("Current HEAD differs from the frozen execution commit recorded by the plan.")

    tracked = sorted(
        set(_git_lines(root, "diff", "--name-only", "HEAD", "--"))
        | set(_git_lines(root, "diff", "--cached", "--name-only", "HEAD", "--"))
    )
    if tracked:
        raise RuntimeError(f"Frozen tracked files changed after tagging: {tracked}")

    untracked = _git_lines(root, "ls-files", "--others", "--exclude-standard")
    unexpected = sorted(path for path in untracked if not _approved_runtime_path(path))
    if unexpected:
        raise RuntimeError(f"Unexpected untracked files exist outside approved runtime outputs: {unexpected}")
    return {
        "valid": True,
        "commit": commit,
        "tag": tag,
        "tag_type": tag_type,
        "tracked_changes": tracked,
        "unexpected_untracked": unexpected,
        "approved_runtime_prefixes": list(APPROVED_RUNTIME_PREFIXES),
    }


def inspect_execution_freeze(
    root: Path,
    *,
    tag: str = DEFAULT_FREEZE_TAG,
    expected_commit: str | None = None,
) -> dict[str, object]:
    """Return freeze status without weakening the fail-closed validator."""
    try:
        return validate_execution_freeze(root, tag=tag, expected_commit=expected_commit)
    except RuntimeError as exc:
        return {
            "valid": False,
            "commit": _git_optional(root, "rev-parse", "HEAD"),
            "tag": tag,
            "error": str(exc),
            "approved_runtime_prefixes": list(APPROVED_RUNTIME_PREFIXES),
        }


def _approved_runtime_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return any(normalized.startswith(prefix) for prefix in APPROVED_RUNTIME_PREFIXES)


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Git freeze inspection failed: git {' '.join(args)}") from exc


def _git_optional(root: Path, *args: str) -> str | None:
    try:
        return _git(root, *args)
    except RuntimeError:
        return None


def _git_lines(root: Path, *args: str) -> list[str]:
    output = _git(root, *args)
    return [line for line in output.splitlines() if line]
