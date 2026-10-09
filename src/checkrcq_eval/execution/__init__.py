"""Final SIGMETRICS campaign execution and lifecycle management."""

from checkrcq_eval.execution.dispatcher import (
    execute_campaign_config,
    execute_plan,
    validate_campaign_output,
)
from checkrcq_eval.execution.registry import EXECUTOR_REGISTRY, resolve_executor

__all__ = [
    "EXECUTOR_REGISTRY",
    "execute_campaign_config",
    "execute_plan",
    "resolve_executor",
    "validate_campaign_output",
]
