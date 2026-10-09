#!/usr/bin/env python3
"""Losslessly compress records and pseudonymize release-only metadata.

This utility is for artifact maintainers. It never edits raw provider result
records and records the pre/post hashes of every transformed file.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}
CRN = re.compile(r"crn:v1:" + r"bluemix:public:quantum-computing:" + r"[^\s\"']+")
PROJECT_PATH = re.compile(r"/Users/[^/]+/Desktop/CheckRCQ")
HOME_PATH = re.compile(r"/Users/[^/]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compress_jsonl() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    hardware_root = ROOT / "experiments/hardware_lih_final_evidence"
    simulation_root = ROOT / "outputs/sigmetrics"
    for source in sorted(simulation_root.rglob("*.jsonl")):
        if ".git" in source.parts or hardware_root in source.parents:
            continue
        target = source.with_suffix(source.suffix + ".gz")
        original_hash = sha256(source)
        original_size = source.stat().st_size
        with source.open("rb") as src, target.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as dst:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(block)
        source.unlink()
        rows.append(
            {
                "source_path": str(source.relative_to(ROOT)),
                "release_path": str(target.relative_to(ROOT)),
                "original_sha256": original_hash,
                "release_sha256": sha256(target),
                "original_bytes": original_size,
                "release_bytes": target.stat().st_size,
                "transformation": "deterministic gzip; decompressed bytes are identical",
            }
        )
    # A prior interrupted packaging pass may already have created archives.
    for target in sorted(simulation_root.rglob("*.jsonl.gz")):
        if any(row["release_path"] == str(target.relative_to(ROOT)) for row in rows):
            continue
        digest = hashlib.sha256()
        original_size = 0
        with gzip.open(target, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                original_size += len(block)
        source = target.with_suffix("")
        rows.append(
            {
                "source_path": str(source.relative_to(ROOT)),
                "release_path": str(target.relative_to(ROOT)),
                "original_sha256": digest.hexdigest(),
                "release_sha256": sha256(target),
                "original_bytes": original_size,
                "release_bytes": target.stat().st_size,
                "transformation": "deterministic gzip; decompressed bytes are identical",
            }
        )
    return rows


def pseudonymize_metadata() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw_results = ROOT / "experiments/hardware_lih_final_evidence/raw/results"
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if raw_results in path.parents:
            continue
        try:
            before = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        after = PROJECT_PATH.sub("${ARTIFACT_ROOT}", before)
        after = HOME_PATH.sub("${ANONYMIZED_HOME}", after)
        after = CRN.sub("<REDACTED_IBM_INSTANCE>", after)
        if after == before:
            continue
        before_hash = hashlib.sha256(before.encode()).hexdigest()
        path.write_text(after, encoding="utf-8")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "source_sha256": before_hash,
                "release_sha256": sha256(path),
                "transformation": "identity-bearing path/account metadata pseudonymized; measurements unchanged",
            }
        )
    return rows


def recover_prior_transformations(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Recover provenance if packaging was interrupted after text rewriting."""
    known = {row["path"] for row in rows}
    source_repository = ROOT.parent
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = str(path.relative_to(ROOT))
        if relative in known:
            continue
        original = source_repository / relative
        if not original.is_file():
            continue
        try:
            released_text = path.read_text(encoding="utf-8")
            original_text = original.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if released_text == original_text:
            continue
        if not any(
            marker in released_text
            for marker in ("${ARTIFACT_ROOT}", "<REDACTED_IBM_INSTANCE>", "<REDACTED_SAVED_ACCOUNT_NAME>")
        ):
            continue
        rows.append(
            {
                "path": relative,
                "source_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                "release_sha256": sha256(path),
                "transformation": "identity-bearing path/account metadata pseudonymized; measurements unchanged",
            }
        )
    return rows


def update_hardware_hashes() -> None:
    hw_root = ROOT / "experiments/hardware_lih_final_evidence"
    manifest = hw_root / "manifests/artifact_hashes.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for relative in sorted(payload["artifacts"]):
        path = hw_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen hardware artifact missing: {path}")
        payload["artifacts"][relative] = "sha256:" + sha256(path)
    payload["release_note"] = (
        "Hashes regenerated for this anonymous release after path/account metadata pseudonymization; "
        "source hashes are retained in manifests/source_to_release.csv."
    )
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    compressed = compress_jsonl()
    transformed = recover_prior_transformations(pseudonymize_metadata())
    update_hardware_hashes()
    write_rows(ROOT / "manifests/compressed_sources.csv", compressed)
    write_rows(ROOT / "manifests/source_to_release.csv", transformed)
    print(json.dumps({"compressed_jsonl": len(compressed), "pseudonymized_files": len(transformed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
