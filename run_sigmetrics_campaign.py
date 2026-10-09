"""One canonical entry point for final SIGMETRICS campaign execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkrcq_eval.common.campaigns import dry_run_campaign, load_campaign_config
from checkrcq_eval.common.execution_plans import load_execution_plan
from checkrcq_eval.constants import ROOT
from checkrcq_eval.execution.dispatcher import (
    execute_campaign_config,
    execute_plan,
    validate_campaign_output,
)
from checkrcq_eval.execution.registry import resolve_executor
from checkrcq_eval.reporting.plan_review import (
    analyze_plan_outputs,
    build_review_package,
    evaluate_plan_claim_guards,
    validate_plan_outputs,
)


DEFAULT_PLAN = ROOT / "configs" / "campaigns" / "execution_plan.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CheckRC-Q final SIGMETRICS campaign dispatcher")
    commands = parser.add_subparsers(dest="command", required=True)

    campaign = commands.add_parser("campaign", help="Run or inspect one campaign config")
    campaign.add_argument("--config", required=True)
    campaign.add_argument("--campaign-id")
    campaign.add_argument("--dry-run", action="store_true")
    campaign.add_argument("--validate-only", action="store_true")
    campaign.add_argument("--resume", action="store_true")
    campaign.add_argument("--max-runs", type=int)
    campaign.add_argument("--allow-live-hardware", action="store_true")

    plan = commands.add_parser("execute-plan", help="Execute a frozen plan in dependency order")
    plan.add_argument("--plan", required=True, choices=("core", "recommended", "exhaustive"))
    plan.add_argument("--execution-plan", default=str(DEFAULT_PLAN))
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument("--resume", action="store_true")
    for name, help_text in (
        ("validate-plan", "Validate and canonicalize every regular campaign in a frozen plan"),
        ("analyze-plan", "Build centralized statistics and RQ1-RQ6 factual summaries"),
        ("build-review", "Build a non-authoritative review package from validated plan output"),
        ("claim-guards", "Evaluate aggregate evidence and authority claim gates"),
    ):
        post = commands.add_parser(name, help=help_text)
        post.add_argument("--plan", required=True, choices=("core", "recommended", "exhaustive"))
        post.add_argument("--execution-plan", default=str(DEFAULT_PLAN))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "execute-plan":
        plan = load_execution_plan(Path(args.execution_plan), root=ROOT)
        result = execute_plan(plan, args.plan, root=ROOT, resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command in {"validate-plan", "analyze-plan", "build-review", "claim-guards"}:
        plan = load_execution_plan(Path(args.execution_plan), root=ROOT)
        operation = {
            "validate-plan": validate_plan_outputs,
            "analyze-plan": analyze_plan_outputs,
            "build-review": build_review_package,
            "claim-guards": evaluate_plan_claim_guards,
        }[args.command]
        result = operation(plan, args.plan, root=ROOT)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.command == "validate-plan" and result["valid"] is not True:
            return 2
        return 0

    config = load_campaign_config(Path(args.config))
    if args.campaign_id and args.campaign_id != config["campaign_id"]:
        raise ValueError("--campaign-id does not match the campaign config.")
    binding = resolve_executor(config)
    if args.dry_run:
        print(json.dumps({**dry_run_campaign(config), "executor": binding.campaign_type}, indent=2, sort_keys=True))
        return 0
    if args.validate_only:
        print(json.dumps(validate_campaign_output(config, root=ROOT), indent=2, sort_keys=True))
        return 0
    result = execute_campaign_config(
        config,
        root=ROOT,
        resume=args.resume,
        allow_live_hardware=args.allow_live_hardware,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
