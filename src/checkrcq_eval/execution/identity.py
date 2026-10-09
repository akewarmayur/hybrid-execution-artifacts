"""Fail-closed identity manifests for resumable campaign and plan output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from checkrcq_eval.common.campaigns import canonical_json, config_hash
from checkrcq_eval.io_utils import write_json
from checkrcq_eval.schemas.campaigns import ExpandedRun
from checkrcq_eval.schemas.sigmetrics import SIGMETRICS_RECORD_SCHEMA_VERSION_V4


CAMPAIGN_OUTPUT_IDENTITY_SCHEMA = "checkrcq-campaign-output-identity-v1"
PLAN_OUTPUT_IDENTITY_SCHEMA = "checkrcq-plan-output-identity-v1"


def campaign_identity_path(output: Path) -> Path:
    return output / "manifests" / "identity.json"


def expected_campaign_identity(
    config: Mapping[str, Any],
    runs: Iterable[ExpandedRun],
    *,
    root: Path,
    git_commit: str,
) -> dict[str, Any]:
    items = tuple(runs)
    run_ids = sorted(run.run_id for run in items)
    dependency_specs = sorted(
        (
            {
                "campaign_id": str(item["campaign_id"]),
                "config_hash": str(item["config_hash"]),
                "schema_version": str(item["schema_version"]),
            }
            for item in config.get("dependencies", ())
        ),
        key=lambda item: item["campaign_id"],
    )
    dependency_artifacts = []
    for item in dependency_specs:
        path = root / "outputs" / "sigmetrics" / "dependencies" / f"{item['campaign_id']}.json"
        dependency_artifacts.append(
            {
                "campaign_id": item["campaign_id"],
                "present": path.is_file(),
                "sha256": _hash_file(path) if path.is_file() else None,
            }
        )
    axes = config.get("axes", {})
    return {
        "schema_version": CAMPAIGN_OUTPUT_IDENTITY_SCHEMA,
        "scientific_record_schema": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "campaign_id": str(config["campaign_id"]),
        "config_hash": str(config.get("_config_hash") or config_hash(config)),
        "git_commit": git_commit,
        "execution_freeze_tag": str(config.get("_execution_freeze_tag", "sigmetrics-2027-execution-v1")),
        "plan_name": config.get("_plan_name"),
        "plan_item_id": config.get("_plan_item_id"),
        "execution_plan_hash": config.get("_execution_plan_hash"),
        "output_root": str(config["output_root"]),
        "expected_run_count": len(run_ids),
        "expected_run_ids_sha256": _hash_json(run_ids),
        "axes_sha256": _hash_json(axes),
        "workloads": sorted(str(value) for value in axes.get("workload", ())),
        "dependency_specs": dependency_specs,
        "dependency_artifacts": dependency_artifacts,
    }


def ensure_campaign_identity(
    output: Path,
    expected: Mapping[str, Any],
    *,
    resume: bool,
) -> Path:
    path = campaign_identity_path(output)
    if output.exists():
        if not resume:
            raise FileExistsError(f"Campaign output already exists; use --resume: {output}")
        if not path.is_file():
            raise RuntimeError(f"Resume collision: campaign output lacks identity manifest: {output}")
        actual = _load_mapping(path)
        _assert_identity_matches(actual, expected, context=str(output))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(expected))
    return path


def validate_existing_campaign_identity(output: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = campaign_identity_path(output)
    if not output.exists():
        return {"exists": False, "compatible": True, "identity_path": str(path), "reason": None}
    if not path.is_file():
        return {
            "exists": True,
            "compatible": False,
            "identity_path": str(path),
            "reason": "campaign output lacks identity manifest",
        }
    try:
        actual = _load_mapping(path)
        _assert_identity_matches(actual, expected, context=str(output))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {"exists": True, "compatible": False, "identity_path": str(path), "reason": str(exc)}
    return {"exists": True, "compatible": True, "identity_path": str(path), "reason": None}


def plan_identity_path(root: Path, plan_name: str) -> Path:
    return root / "outputs" / "sigmetrics" / "plan_manifests" / f"{plan_name}.identity.json"


def expected_plan_identity(
    *,
    plan_name: str,
    plan_hash: str,
    git_commit: str,
    freeze_tag: str,
    items: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "item_id": str(item["item_id"]),
            "campaign_id": str(item["campaign_id"]),
            "config_hash": str(item["config_hash"]),
            "output_root": str(item["output_root"]),
            "dependencies": list(item.get("dependencies", ())),
        }
        for item in items
    ]
    return {
        "schema_version": PLAN_OUTPUT_IDENTITY_SCHEMA,
        "scientific_record_schema": SIGMETRICS_RECORD_SCHEMA_VERSION_V4,
        "plan": plan_name,
        "execution_plan_hash": plan_hash,
        "git_commit": git_commit,
        "execution_freeze_tag": freeze_tag,
        "items": rows,
        "items_sha256": _hash_json(rows),
    }


def ensure_plan_identity(path: Path, expected: Mapping[str, Any], *, resume: bool) -> Path:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Plan output identity already exists; use --resume: {path}")
        actual = _load_mapping(path)
        _assert_identity_matches(actual, expected, context=str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(expected))
    return path


def _assert_identity_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    if canonical_json(actual) != canonical_json(expected):
        mismatched = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(f"Resume collision for {context}; identity differs in {mismatched}.")


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Identity manifest must contain an object: {path}")
    return value


def _hash_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
