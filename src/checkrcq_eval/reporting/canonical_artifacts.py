"""Paper-facing artifacts generated only from authoritative canonical data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from checkrcq_eval.common.campaigns import canonical_json, select_authoritative_campaign
from checkrcq_eval.common.stats import summarize_binary_success, summarize_numeric


def generate_artifact_bundle(
    records: Iterable[Mapping[str, Any]],
    *,
    registry: Mapping[str, Any],
    analysis_name: str,
    metric_specs: Mapping[str, Mapping[str, str]],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate neutral CSV/JSON/LaTeX/figure/text/numeric artifacts."""
    selection = select_authoritative_campaign(registry, analysis_name)
    campaign_id = str(selection["campaign_id"])
    config_digest = str(selection["config_hash"])
    selected = [record for record in records if _campaign_id(record) == campaign_id]
    if not selected:
        raise ValueError(f"No canonical records for authoritative campaign {campaign_id}.")
    if any(record.get("campaign_state") in {"legacy", "planned", "smoke", "superseded"} for record in selected):
        raise ValueError("Non-evaluation campaign data cannot feed final paper artifacts.")

    source_hash = _source_hash(selected)
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, spec in metric_specs.items():
        raw_values = [_raw_path(record, spec["field"]) for record in selected]
        if spec.get("measured_only") in {True, "true"} and any(
            isinstance(value, Mapping) and value.get("provenance") != "measured"
            for value in raw_values
            if value is not None
        ):
            raise ValueError(f"Measured-only artifact metric {name!r} contains non-measured provenance.")
        values = [_path(record, spec["field"]) for record in selected]
        values = [value for value in values if value is not None]
        if spec["kind"] == "binary":
            summary = summarize_binary_success((bool(value) for value in values), denominator=len(selected)).as_dict()
        elif spec["kind"] == "continuous":
            summary = summarize_numeric((float(value) for value in values), denominator=len(selected)).as_dict()
        else:
            raise ValueError(f"Unknown metric kind: {spec['kind']}")
        summaries[name] = summary
        rows.append({"metric": name, **summary})

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{analysis_name}.csv"
    json_path = output_dir / f"{analysis_name}.json"
    tex_path = output_dir / f"{analysis_name}.tex"
    figure_path = output_dir / f"{analysis_name}_figure_data.json"
    summary_path = output_dir / f"{analysis_name}_summary.txt"
    numbers_path = output_dir / "paper_numbers.json"
    manifest_path = output_dir / "artifact_manifest.json"

    _write_csv(csv_path, rows)
    metadata = {
        "analysis": analysis_name,
        "campaign_id": campaign_id,
        "config_hash": config_digest,
        "canonical_source_sha256": source_hash,
        "neutral_facts_only": True,
    }
    _write_json(json_path, {"metadata": metadata, "metrics": summaries})
    _write_json(figure_path, {"metadata": metadata, "series": rows})
    _write_json(numbers_path, {"metadata": metadata, analysis_name: summaries})
    tex_path.write_text(_latex_table(rows, metadata), encoding="utf-8")
    summary_path.write_text(_neutral_summary(rows, metadata), encoding="utf-8")
    files = (csv_path, json_path, tex_path, figure_path, summary_path, numbers_path)
    manifest = {
        **metadata,
        "files": {str(path.name): _file_hash(path) for path in files},
    }
    _write_json(manifest_path, manifest)
    return {
        "csv": csv_path,
        "json": json_path,
        "latex": tex_path,
        "figure_data": figure_path,
        "text_summary": summary_path,
        "paper_numbers": numbers_path,
        "manifest": manifest_path,
    }


def detect_stale_artifacts(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if manifest.get("canonical_source_sha256") != _source_hash(records):
        issues.append("canonical_source_changed")
    for name, expected in manifest.get("files", {}).items():
        path = manifest_path.parent / name
        if not path.exists() or _file_hash(path) != expected:
            issues.append(f"stale:{name}")
    return issues


def assert_caption_uses_figure_data(caption_metadata: Mapping[str, Any], figure_path: Path) -> None:
    if caption_metadata.get("figure_data_sha256") != _file_hash(figure_path):
        raise ValueError("Caption metadata does not match machine-readable figure source data.")


def _campaign_id(record: Mapping[str, Any]) -> str:
    identity = record.get("identity", {})
    return str(record.get("campaign_id") or (identity.get("campaign_id") if isinstance(identity, Mapping) else ""))


def _source_hash(records: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(record) for record in records), key=lambda item: canonical_json(item))
    return "sha256:" + hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _path(record: Mapping[str, Any], dotted: str) -> object | None:
    value = _raw_path(record, dotted)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _raw_path(record: Mapping[str, Any], dotted: str) -> object | None:
    value: object = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(rows: list[dict[str, Any]], metadata: Mapping[str, Any]) -> str:
    lines = [
        f"% campaign_id={metadata['campaign_id']} config_hash={metadata['config_hash']}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Metric & Estimate & Q1/low & Q3/high & n \\\\",
        "\\midrule",
    ]
    for row in rows:
        estimate = row.get("median", row.get("rate"))
        low = row.get("iqr_low", row.get("ci_low"))
        high = row.get("iqr_high", row.get("ci_high"))
        lines.append(f"{row['metric']} & {_fmt(estimate)} & {_fmt(low)} & {_fmt(high)} & {row.get('count', 0)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _neutral_summary(rows: list[dict[str, Any]], metadata: Mapping[str, Any]) -> str:
    lines = [
        f"Campaign: {metadata['campaign_id']}",
        f"Config hash: {metadata['config_hash']}",
        "Neutral observed statistics:",
    ]
    for row in rows:
        value = row.get("median", row.get("rate"))
        lines.append(f"- {row['metric']}: observed estimate={_fmt(value)}, sample_count={row.get('count', 0)}")
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6g}"
