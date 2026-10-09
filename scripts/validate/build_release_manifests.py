#!/usr/bin/env python3
"""Build deterministic release inventory and SHA-256 checksum manifests."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests/artifact_inventory.csv"
CHECKSUMS = ROOT / "manifests/sha256_checksums.txt"
EXCLUDED = {INVENTORY.resolve(), CHECKSUMS.resolve()}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def category(relative: Path) -> str:
    first = relative.parts[0]
    if first == "src":
        return "implementation"
    if first == "tests":
        return "tests"
    if first == "configs":
        return "configuration"
    if first in {"outputs", "data", "experiments"}:
        return "experimental_evidence"
    if first in {"Results", "paper_assets_final", "plots"}:
        return "paper_assets"
    if first in {"analysis", "rq4_final_methodology_audit"}:
        return "analysis"
    if first in {"docs", "manifests", "validation"} or relative.suffix.lower() == ".md":
        return "documentation_provenance"
    if first == "scripts" or relative.suffix.lower() == ".py":
        return "runner_tooling"
    return "release_metadata"


def main() -> int:
    files = [
        path for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.resolve() not in EXCLUDED and ".git" not in path.parts
    ]
    rows: list[dict[str, object]] = []
    hashes: list[tuple[str, str]] = []
    for path in files:
        relative = path.relative_to(ROOT)
        digest = sha256(path)
        rows.append(
            {
                "path": relative.as_posix(),
                "category": category(relative),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "status": "included",
            }
        )
        hashes.append((digest, relative.as_posix()))
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "category", "bytes", "sha256", "status"))
        writer.writeheader()
        writer.writerows(rows)
    CHECKSUMS.write_text("".join(f"{digest}  {path}\n" for digest, path in hashes), encoding="utf-8")
    print(f"Inventory: {len(rows)} files, {sum(int(row['bytes']) for row in rows)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
