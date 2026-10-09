"""Command-line interface for the controlled LiH IBM hardware campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from checkrcq_eval.hardware_vertical.campaign import (
    estimate_campaign,
    freeze_evaluation,
    run_analysis,
    run_calibration,
    run_core,
    run_dry,
    run_optional_scale,
    run_overnight,
    record_review_large_budget_stop,
    run_pilot,
    run_preflight,
)
from checkrcq_eval.hardware_vertical.config import (
    DEFAULT_CONFIG,
    campaign_paths,
    load_config,
)
from checkrcq_eval.hardware_vertical.util import read_json
from checkrcq_eval.hardware_vertical.runtime import QPUBudgetExceeded, SubmissionThrottleReached
from checkrcq_eval.hardware_vertical.scaled_followup import (
    FOLLOWUP_CONFIG,
    followup_paths,
    load_followup_config,
    run_scaled_followup,
)
from checkrcq_eval.hardware_vertical.final_evidence import (
    DEFAULT_CONFIG as FINAL_EVIDENCE_CONFIG,
    campaign_output_paths,
    load_final_config,
    run_final_evidence_campaign,
)


LIVE_MODES = {
    "pilot",
    "calibrate",
    "run_core",
    "run_optional_scale",
    "run_overnight",
    "run_scaled_followup",
    "run_final_evidence_campaign",
}
THROTTLED_MODES = {"pilot", "calibrate", "run_core", "run_optional_scale"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged, resumable RES-Q LiH validation on IBM Quantum hardware."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_const", const="preflight", dest="mode")
    mode.add_argument("--dry-run", action="store_const", const="dry_run", dest="mode")
    mode.add_argument("--pilot", action="store_const", const="pilot", dest="mode")
    mode.add_argument("--calibrate", action="store_const", const="calibrate", dest="mode")
    mode.add_argument("--freeze-evaluation", action="store_const", const="freeze", dest="mode")
    mode.add_argument("--run-core", action="store_const", const="run_core", dest="mode")
    mode.add_argument("--run-overnight", action="store_const", const="run_overnight", dest="mode")
    mode.add_argument(
        "--run-scaled-followup",
        action="store_const",
        const="run_scaled_followup",
        dest="mode",
    )
    mode.add_argument(
        "--run-final-evidence-campaign",
        action="store_const",
        const="run_final_evidence_campaign",
        dest="mode",
    )
    mode.add_argument("--analyze", action="store_const", const="analyze", dest="mode")
    mode.add_argument(
        "--run-optional-scale",
        action="store_const",
        const="run_optional_scale",
        dest="mode",
    )
    mode.add_argument("--estimate", action="store_const", const="estimate", dest="mode")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-backend")
    parser.add_argument("--target-a")
    parser.add_argument("--target-b")
    parser.add_argument("--qpu-budget-seconds", type=float)
    parser.add_argument(
        "--max-new-live-jobs",
        type=int,
        help=(
            "Operational per-invocation submission limit; it does not change the scientific plan. "
            "Use --resume in a later invocation to continue deterministic pending job keys."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-live-hardware",
        action="store_true",
        help="Required acknowledgement for modes that can submit IBM QPU jobs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "run_scaled_followup":
        config = load_followup_config(args.config or FOLLOWUP_CONFIG)
    elif args.mode == "run_final_evidence_campaign":
        config = load_final_config(args.config or FINAL_EVIDENCE_CONFIG)
    else:
        config = load_config(args.config or DEFAULT_CONFIG)
    live_paths = campaign_paths(namespace="live")
    dry_paths = campaign_paths(namespace="dry_run")
    optional_paths = campaign_paths(namespace="optional_scale")
    scaled_paths = followup_paths()
    final_evidence_paths = campaign_output_paths()
    budget = config.qpu_budget_seconds if args.qpu_budget_seconds is None else args.qpu_budget_seconds
    if args.qpu_budget_seconds is None and args.mode in {
        "run_core",
        "run_optional_scale",
        "run_overnight",
    }:
        budget = 1500.0

    if args.mode in LIVE_MODES and not args.allow_live_hardware:
        raise PermissionError(
            f"--{args.mode.replace('_', '-')} can submit IBM QPU jobs; "
            "rerun only after review with --allow-live-hardware."
        )
    if args.max_new_live_jobs is not None:
        if args.mode not in THROTTLED_MODES:
            raise ValueError("--max-new-live-jobs is valid only for direct live submission modes.")
        if args.max_new_live_jobs < 0:
            raise ValueError("--max-new-live-jobs must be nonnegative.")
    if args.mode not in {"preflight", "run_scaled_followup", "run_final_evidence_campaign"}:
        _verify_frozen_backend_overrides(args, live_paths.manifests / "preflight_manifest.json")
    if args.mode == "run_scaled_followup" and any(
        (args.source_backend, args.target_a, args.target_b)
    ):
        raise ValueError(
            "The scaled follow-up backend rule is frozen in its config; CLI backend overrides are forbidden."
        )
    if args.mode == "run_final_evidence_campaign" and any(
        (args.source_backend, args.target_a, args.target_b)
    ):
        raise ValueError(
            "The final-evidence backend-selection algorithm is frozen; CLI overrides are forbidden."
        )

    if args.mode == "preflight":
        result = run_preflight(
            config,
            live_paths,
            source_backend=args.source_backend,
            target_a=args.target_a,
            target_b=args.target_b,
        )
    elif args.mode == "dry_run":
        result = run_dry(config, dry_paths, resume=args.resume)
    elif args.mode == "pilot":
        try:
            result = run_pilot(
                config,
                live_paths,
                allow_live_hardware=True,
                resume=args.resume,
                qpu_budget_seconds=budget,
                max_new_live_jobs=args.max_new_live_jobs,
            )
        except SubmissionThrottleReached as exc:
            result = exc.as_dict()
    elif args.mode == "calibrate":
        try:
            result = run_calibration(
                config,
                live_paths,
                allow_live_hardware=True,
                resume=args.resume,
                qpu_budget_seconds=budget,
                max_new_live_jobs=args.max_new_live_jobs,
            )
        except SubmissionThrottleReached as exc:
            result = exc.as_dict()
    elif args.mode == "freeze":
        result = freeze_evaluation(config, live_paths)
    elif args.mode == "run_core":
        try:
            result = run_core(
                config,
                live_paths,
                allow_live_hardware=True,
                resume=args.resume,
                qpu_budget_seconds=budget,
                max_new_live_jobs=args.max_new_live_jobs,
            )
        except SubmissionThrottleReached as exc:
            result = exc.as_dict()
    elif args.mode == "run_overnight":
        result = run_overnight(
            config,
            live_paths,
            optional_paths,
            allow_live_hardware=True,
            resume=args.resume,
            qpu_budget_seconds=budget,
        )
    elif args.mode == "run_scaled_followup":
        result = run_scaled_followup(
            config,
            scaled_paths,
            allow_live_hardware=True,
            resume=args.resume,
        )
    elif args.mode == "run_final_evidence_campaign":
        result = run_final_evidence_campaign(
            config,
            final_evidence_paths,
            allow_live_hardware=True,
            resume=args.resume,
        )
    elif args.mode == "analyze":
        result = run_analysis(config, live_paths)
    elif args.mode == "run_optional_scale":
        try:
            result = run_optional_scale(
                config,
                optional_paths,
                allow_live_hardware=True,
                resume=args.resume,
                qpu_budget_seconds=budget,
                max_new_live_jobs=args.max_new_live_jobs,
            )
        except SubmissionThrottleReached as exc:
            result = exc.as_dict()
        except QPUBudgetExceeded as exc:
            result = record_review_large_budget_stop(
                config,
                live_paths,
                optional_paths,
                reason=str(exc),
                budget_limit_seconds=budget,
            )
    else:
        result = estimate_campaign(config)
    print(json.dumps(_jsonable(_display_result(args.mode, result)), indent=2, sort_keys=True))
    return 0


def _verify_frozen_backend_overrides(args: argparse.Namespace, manifest_path: Path) -> None:
    supplied = {
        "source": args.source_backend,
        "target_a": args.target_a,
        "target_b": args.target_b,
    }
    if not any(supplied.values()):
        return
    if not manifest_path.is_file():
        raise FileNotFoundError("Backend overrides require a completed --preflight manifest.")
    selected = read_json(manifest_path).get("selected_backends", {})
    mismatches = {
        role: {"requested": value, "frozen": selected.get(role)}
        for role, value in supplied.items()
        if value is not None and value != selected.get(role)
    }
    if mismatches:
        raise ValueError(
            "Backend overrides differ from the preflight-frozen selection: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _display_result(mode: str, result: Any) -> Any:
    if mode != "preflight" or not isinstance(result, dict):
        return result
    compilation = result.get("compilation", {})
    return {
        "schema_version": result.get("schema_version"),
        "account_detected": result.get("account_detected"),
        "instance_detected": result.get("instance_detected"),
        "authentication": result.get("authentication"),
        "accessible_candidate_backends": result.get("selection_rationale", {}).get(
            "accessible_operational_backends"
        ),
        "selected_backends": result.get("selected_backends"),
        "selection_rationale": result.get("selection_rationale", {}).get("selection_inputs"),
        "workload": result.get("workload"),
        "compilation": {
            name: {
                "circuit_count": values.get("circuit_count"),
                "aggregate": values.get("aggregate"),
                "lih_executable": values.get("lih_executable"),
                "required_primitive_available": values.get("required_primitive_available"),
                "compiled_qpy": values.get("compiled_qpy"),
            }
            for name, values in compilation.items()
        },
        "estimates": result.get("estimates"),
        "manifest": "experiments/hardware_lih_vertical/manifests/preflight_manifest.json",
        "live_jobs_submitted": result.get("live_jobs_submitted"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
