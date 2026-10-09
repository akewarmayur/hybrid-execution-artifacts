"""Deterministic seed helpers."""

from __future__ import annotations

import hashlib


def stable_int_seed(*parts: object, modulus: int = 2**32 - 1) -> int:
    """Hash arbitrary inputs into a deterministic integer seed."""
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus
