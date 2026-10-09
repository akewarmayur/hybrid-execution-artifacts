"""Aggregation helpers with strict setting separation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from checkrcq_eval.common.stats import detect_repetition_unit, summarize_binary_success, summarize_numeric
from checkrcq_eval.constants import EVALUATIONS_BY_SETTING
from checkrcq_eval.io_utils import read_jsonl
from checkrcq_eval.schemas.summaries import AggregatedSummaryRecord


def load_processed_records(path: Path) -> pd.DataFrame:
    """Load processed records from CSV or JSONL."""
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix == ".jsonl":
        frame = pd.DataFrame(read_jsonl(path))
    else:
        raise ValueError(f"Unsupported processed data format: {path}")
    if frame.empty:
        return frame
    if "artifact_presence" in frame.columns:
        frame["artifact_presence"] = frame["artifact_presence"].apply(_normalize_artifact_presence)
    return frame


def _normalize_artifact_presence(value: object) -> dict[str, bool]:
    """Normalize serialized artifact maps after CSV load."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        import ast
        import json

        normalized = text.replace("'", '"')
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Artifact presence did not parse to a mapping: {value!r}")
            return {str(key): bool(val) for key, val in parsed.items()}
    raise ValueError(f"Could not parse artifact presence map: {value!r}")


def validate_setting(frame: pd.DataFrame, setting: str, evaluation_question: str) -> None:
    """Ensure a frame contains only the expected setting/evaluation combination."""
    if frame.empty:
        raise ValueError("Processed data frame is empty.")
    expected_evals = EVALUATIONS_BY_SETTING.get(setting)
    if expected_evals is None:
        raise ValueError(f"Unknown setting: {setting}")
    if evaluation_question not in expected_evals:
        raise ValueError(f"{evaluation_question} does not belong to {setting}")
    settings = set(frame["setting"].astype(str).unique())
    evals = set(frame["evaluation_question"].astype(str).unique())
    if settings != {setting}:
        raise ValueError(f"Expected only setting {setting!r}, found {sorted(settings)}")
    if evals != {evaluation_question}:
        raise ValueError(f"Expected only evaluation {evaluation_question!r}, found {sorted(evals)}")


def aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    include_ci: bool = False,
) -> pd.DataFrame:
    """Aggregate selected metrics over canonical groups."""
    repetition_unit = detect_repetition_unit(frame)
    rows: list[AggregatedSummaryRecord] = []
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        grouping = dict(zip(group_columns, group_key))
        for metric_name in metrics:
            if metric_name not in group_frame.columns:
                raise ValueError(f"Missing metric column {metric_name!r}")
            summary = summarize_numeric(group_frame[metric_name].astype(float), include_ci=include_ci)
            rows.append(
                AggregatedSummaryRecord(
                    setting=str(group_frame["setting"].iloc[0]),
                    evaluation_question=str(group_frame["evaluation_question"].iloc[0]),
                    grouping=grouping,
                    metric_name=metric_name,
                    median=summary.median,
                    iqr_low=summary.iqr_low,
                    iqr_high=summary.iqr_high,
                    count=summary.count,
                    denominator=summary.denominator,
                    ci_low=summary.ci_low,
                    ci_high=summary.ci_high,
                    repetition_unit=repetition_unit,
                )
            )
    return pd.DataFrame(
        [
            {
                "setting": row.setting,
                "evaluation_question": row.evaluation_question,
                **row.grouping,
                "metric_name": row.metric_name,
                "median": row.median,
                "iqr_low": row.iqr_low,
                "iqr_high": row.iqr_high,
                "count": row.count,
                "denominator": row.denominator,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "repetition_unit": row.repetition_unit,
            }
            for row in rows
        ]
    )


def aggregate_binary_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate binary metrics as proportions with confidence intervals."""
    repetition_unit = detect_repetition_unit(frame)
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        grouping = dict(zip(group_columns, group_key))
        denominator = int(group_frame.shape[0])
        for metric_name in metrics:
            if metric_name not in group_frame.columns:
                raise ValueError(f"Missing metric column {metric_name!r}")
            summary = summarize_binary_success(
                group_frame[metric_name].astype(float),
                denominator=denominator,
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "setting": str(group_frame["setting"].iloc[0]),
                    "evaluation_question": str(group_frame["evaluation_question"].iloc[0]),
                    **grouping,
                    "metric_name": metric_name,
                    "rate": summary.rate,
                    "count": summary.count,
                    "denominator": summary.denominator,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                    "repetition_unit": repetition_unit,
                }
            )
    return pd.DataFrame(rows)
