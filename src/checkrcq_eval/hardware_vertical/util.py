"""Small deterministic and credential-safe utilities for the hardware campaign."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SECRET_MARKERS = (
    "token",
    "api_key",
    "apikey",
    "password",
    "authorization",
    "credential",
    "secret",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def stable_hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a JSON file and fsync its containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def sanitized(value: Any) -> Any:
    """Recursively remove secret-bearing fields before persistence or printing."""
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if any(marker in name.lower() for marker in SECRET_MARKERS):
                cleaned[name] = "<redacted>"
            else:
                cleaned[name] = sanitized(item)
        return cleaned
    if isinstance(value, (list, tuple, set)):
        return [sanitized(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return repr(value)


def assert_no_secrets(value: object) -> None:
    """Fail if a persisted structure still has a secret-like key or known token."""
    text = stable_json(value)
    lower = text.lower()
    for marker in SECRET_MARKERS:
        if f'"{marker}"' in lower and "<redacted>" not in lower:
            raise ValueError(f"Secret-like field escaped redaction: {marker}")
    for environment_name in ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN"):
        token = os.getenv(environment_name)
        if token and token in text:
            raise ValueError(f"Environment credential {environment_name} appeared in output.")


def software_environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("qiskit", "qiskit-ibm-runtime", "numpy", "scipy", "pandas", "checkrcq-eval"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "git_commit": git_commit(),
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def slug(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in value)
    return cleaned.strip("-") or stable_hash(value)[7:23]


def _json_default(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "__dict__"):
        return sanitized(vars(value))
    return repr(value)

