"""Command-line interface for the CheckRCQ evaluation repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from checkrcq_eval.analysis.hardware.e3_hw_analysis import analyze as analyze_e3_hw
from checkrcq_eval.analysis.ideal.e1_analysis import analyze as analyze_e1
from checkrcq_eval.analysis.ideal.e2_analysis import analyze as analyze_e2
from checkrcq_eval.analysis.ideal.e4_analysis import analyze as analyze_e4
from checkrcq_eval.analysis.noisy.e3_analysis import analyze as analyze_e3
from checkrcq_eval.common.aggregation import load_processed_records, validate_setting
from checkrcq_eval.common.config import load_common_config, load_experiment_config
from checkrcq_eval.common.stats import detect_repetition_unit
from checkrcq_eval.common.validation import require_columns
from checkrcq_eval.constants import EVALUATIONS_BY_SETTING, OUTPUTS_DIR, PROCESSED_DATA_DIR
from checkrcq_eval.experiments.hardware.e3_hw_validation import run as run_e3_hw
from checkrcq_eval.experiments.ideal.e1_contract_cost import run as run_e1
from checkrcq_eval.experiments.ideal.e2_resume import run as run_e2
from checkrcq_eval.experiments.ideal.e4_ablations import run as run_e4
from checkrcq_eval.experiments.noisy.e3_replay_migration import run as run_e3
from checkrcq_eval.logging_utils import build_logger
from checkrcq_eval.provenance import prepare_campaign_manifest
from checkrcq_eval.reporting.paper_artifacts import build as build_paper_artifacts_bundle


RUNNERS = {
    ("ideal", "e1"): run_e1,
    ("ideal", "e1_review"): run_e1,
    ("ideal", "e2"): run_e2,
    ("ideal", "e4"): run_e4,
    ("noisy", "e3"): run_e3,
    ("noisy", "e3_review"): run_e3,
    ("hardware", "e3_hw"): run_e3_hw,
}

ANALYZERS = {
    ("ideal", "e1"): analyze_e1,
    ("ideal", "e1_review"): lambda: analyze_e1("e1_review"),
    ("ideal", "e2"): analyze_e2,
    ("ideal", "e4"): analyze_e4,
    ("noisy", "e3"): analyze_e3,
    ("noisy", "e3_review"): lambda: analyze_e3("e3_review"),
    ("hardware", "e3_hw"): analyze_e3_hw,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(description="CheckRCQ evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--setting", required=True, choices=EVALUATIONS_BY_SETTING)
    run_parser.add_argument("--exp", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze processed experiment results")
    analyze_parser.add_argument("--setting", required=True, choices=EVALUATIONS_BY_SETTING)
    analyze_parser.add_argument("--exp", required=True)

    subparsers.add_parser("build-paper-artifacts", help="Regenerate all paper artifacts from processed data")
    campaign_parser = subparsers.add_parser(
        "prepare-campaign",
        help="Snapshot an experiment config and write a provenance-first campaign manifest",
    )
    campaign_parser.add_argument("--config", required=True)
    campaign_parser.add_argument("--campaign-id", required=True)
    campaign_parser.add_argument("--paper-stage", required=True, choices=("fast_legacy", "sigmetrics"))
    campaign_parser.add_argument("--rq", action="append", required=True, help="RQ1 through RQ6; repeat as needed")
    campaign_parser.add_argument(
        "--execution-provenance",
        required=True,
        choices=("ideal_sim", "noisy_sim", "live_hardware", "cached_hardware", "mock_hardware"),
    )
    campaign_parser.add_argument(
        "--measurement-provenance",
        action="append",
        required=True,
        choices=("measured", "modeled", "derived"),
        help="Repeat when a campaign contains more than one measurement provenance class.",
    )
    campaign_parser.add_argument("--calibration-evaluation-split", default="unknown")
    subparsers.add_parser("validate-results", help="Validate processed result files")
    subparsers.add_parser("list-outputs", help="List generated output files")
    return parser


def _print_result_paths(result: dict[str, object]) -> None:
    """Print path-bearing result values in a stable way."""
    for key in sorted(result):
        print(f"{key}: {result[key]}")


def command_run(args: argparse.Namespace) -> int:
    """Handle the run command."""
    key = (args.setting, args.exp)
    if key not in RUNNERS:
        raise ValueError(f"Unsupported run target: {args.setting}/{args.exp}")
    common = load_common_config()
    config = load_experiment_config(Path(args.config))
    if config.setting != args.setting or config.evaluation_question != args.exp:
        raise ValueError(
            f"Config {args.config} is for {config.setting}/{config.evaluation_question}, "
            f"not {args.setting}/{args.exp}"
        )
    log_path = OUTPUTS_DIR / args.setting / args.exp / "logs" / ("run_dry.log" if args.dry_run else "run.log")
    logger = build_logger(f"checkrcq_eval.run.{args.setting}.{args.exp}", log_path)
    logger.info("Running %s/%s using %s", args.setting, args.exp, args.config)
    print(f"running: {args.setting}/{args.exp}", flush=True)
    print(f"config: {Path(args.config)}", flush=True)
    print(f"log_path: {log_path}", flush=True)
    result = RUNNERS[key](config=config, common=common, dry_run=args.dry_run)
    _print_result_paths(result)
    logger.info("Wrote outputs to %s", json.dumps({k: str(v) for k, v in result.items()}, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    """Handle the analyze command."""
    key = (args.setting, args.exp)
    if key not in ANALYZERS:
        raise ValueError(f"Unsupported analysis target: {args.setting}/{args.exp}")
    log_path = OUTPUTS_DIR / args.setting / args.exp / "logs" / "analysis.log"
    logger = build_logger(f"checkrcq_eval.analyze.{args.setting}.{args.exp}", log_path)
    logger.info("Analyzing %s/%s from processed records", args.setting, args.exp)
    print(f"analyzing: {args.setting}/{args.exp}", flush=True)
    print(f"log_path: {log_path}", flush=True)
    result = ANALYZERS[key]()
    _print_result_paths(result)
    return 0


def command_build_paper_artifacts() -> int:
    """Handle the build-paper-artifacts command."""
    log_path = OUTPUTS_DIR / "paper_artifacts" / "logs" / "paper_artifacts.log"
    logger = build_logger("checkrcq_eval.paper_artifacts", log_path)
    logger.info("Building paper artifacts from processed data")
    result = build_paper_artifacts_bundle()
    _print_result_paths(result)
    return 0


def command_prepare_campaign(args: argparse.Namespace) -> int:
    """Prepare a campaign manifest without executing the experiment."""
    result = prepare_campaign_manifest(
        config_path=Path(args.config),
        campaign_id=args.campaign_id,
        paper_stage=args.paper_stage,
        rqs=args.rq,
        execution_provenance=args.execution_provenance,
        measurement_provenance=args.measurement_provenance,
        calibration_evaluation_split=args.calibration_evaluation_split,
    )
    _print_result_paths(result)
    return 0


def command_validate_results() -> int:
    """Handle the validate-results command."""
    validated = 0
    missing = 0
    for setting, evaluations in EVALUATIONS_BY_SETTING.items():
        for exp in evaluations:
            path = PROCESSED_DATA_DIR / setting / f"{exp}_records.csv"
            if not path.exists():
                print(f"missing: {path}")
                missing += 1
                continue
            frame = load_processed_records(path)
            require_columns(frame)
            validate_setting(frame, setting, exp)
            detect_repetition_unit(frame)
            print(f"validated: {path}")
            validated += 1
    print(f"validated_count: {validated}")
    print(f"missing_count: {missing}")
    return 0


def command_list_outputs() -> int:
    """Handle the list-outputs command."""
    if not OUTPUTS_DIR.exists():
        print(f"outputs directory does not exist: {OUTPUTS_DIR}")
        return 0
    for path in sorted(OUTPUTS_DIR.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "analyze":
        return command_analyze(args)
    if args.command == "build-paper-artifacts":
        return command_build_paper_artifacts()
    if args.command == "prepare-campaign":
        return command_prepare_campaign(args)
    if args.command == "validate-results":
        return command_validate_results()
    if args.command == "list-outputs":
        return command_list_outputs()
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
