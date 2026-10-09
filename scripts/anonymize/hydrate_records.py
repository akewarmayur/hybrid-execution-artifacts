#!/usr/bin/env python3
"""Restore packaged `.jsonl.gz` files for legacy analysis entry points."""

from __future__ import annotations

import csv
import gzip
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    manifest = ROOT / "manifests/compressed_sources.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = ROOT / row["source_path"]
        archive = ROOT / row["release_path"]
        if source.exists():
            if digest(source) != row["original_sha256"]:
                raise RuntimeError(f"Existing hydrated file has wrong hash: {source}")
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive, "rb") as src, source.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if digest(source) != row["original_sha256"]:
            source.unlink(missing_ok=True)
            raise RuntimeError(f"Hydration hash mismatch: {source}")
    print(f"Hydrated {len(rows)} JSONL record streams.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
