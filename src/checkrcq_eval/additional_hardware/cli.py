"""CLI for supplemental hardware experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_MPLCONFIGDIR = Path(__file__).resolve().parents[3] / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from checkrcq_eval.additional_hardware import SUPPLEMENTAL_CONFIG_DIR
from checkrcq_eval.additional_hardware.analyze import analyze
from checkrcq_eval.additional_hardware.combined_report import build_combined_table
from checkrcq_eval.additional_hardware.config import load_supplemental_hardware_config, resolve_config_path
from checkrcq_eval.additional_hardware.runner import run, supplemental_paths
from checkrcq_eval.common.config import load_common_config
from checkrcq_eval.logging_utils import build_logger


def build_parser() -> argparse.ArgumentParser:
    """Build the supplemental-hardware CLI parser."""
    parser = argparse.ArgumentParser(description="CheckRCQ supplemental hardware CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a supplemental hardware campaign")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one supplemental hardware campaign")
    analyze_parser.add_argument("--config", required=True)

    subparsers.add_parser("list-configs", help="List supplemental hardware configs")
    subparsers.add_parser("build-combined-table", help="Build one compact combined supplemental hardware table")
    return parser


def command_run(args: argparse.Namespace) -> int:
    """Handle supplemental run."""
    config_path = resolve_config_path(args.config)
    config = load_supplemental_hardware_config(config_path)
    common = load_common_config()
    log_path = supplemental_paths(config.results_subdir)["logs_dir"] / (
        "run_dry.log" if args.dry_run else "run.log"
    )
    logger = build_logger(f"checkrcq_eval.additional_hardware.run.{config.results_subdir}", log_path)
    logger.info("Running supplemental hardware experiment using %s", config_path)
    print(f"running: supplemental_hardware/{config.experiment_name}", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"log_path: {log_path}", flush=True)
    result = run(config=config, common=common, config_path=config_path, dry_run=args.dry_run)
    for key in sorted(result):
        print(f"{key}: {result[key]}")
    logger.info("Wrote outputs to %s", json.dumps({k: str(v) for k, v in result.items()}, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    """Handle supplemental analysis."""
    config_path = resolve_config_path(args.config)
    config = load_supplemental_hardware_config(config_path)
    log_path = supplemental_paths(config.results_subdir)["logs_dir"] / "analysis.log"
    logger = build_logger(f"checkrcq_eval.additional_hardware.analyze.{config.results_subdir}", log_path)
    logger.info("Analyzing supplemental hardware experiment from %s", config_path)
    print(f"analyzing: supplemental_hardware/{config.experiment_name}", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"log_path: {log_path}", flush=True)
    result = analyze(config, config_path=config_path)
    for key in sorted(result):
        print(f"{key}: {result[key]}")
    return 0


def command_list_configs() -> int:
    """List available supplemental configs."""
    for path in sorted(SUPPLEMENTAL_CONFIG_DIR.glob("*.yaml")):
        print(path)
    return 0


def command_build_combined_table() -> int:
    """Build the combined supplemental-hardware CSV and LaTeX table."""
    result = build_combined_table()
    for key in sorted(result):
        print(f"{key}: {result[key]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Supplemental CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "analyze":
        return command_analyze(args)
    if args.command == "list-configs":
        return command_list_configs()
    if args.command == "build-combined-table":
        return command_build_combined_table()
    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
