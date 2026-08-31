"""Execution-first runtime API.

The public runtime surface is organized around four steps:

- build an [`ExecutionRequest`][prejectory.runtime.ExecutionRequest]
- resolve it into an [`ExecutionPlan`][prejectory.runtime.ExecutionPlan]
- execute the request directly with [`execute_request`][prejectory.runtime.execute_request]
- or execute a resolved plan with [`execute_plan`][prejectory.runtime.execute_plan]
"""

from __future__ import annotations

from prejectory.runtime.api import execute_plan, execute_request, resolve_request
from prejectory.runtime.types import (
    CleanupRemovalSummary,
    CleanupSummary,
    ExecutionPlan,
    ExecutionRequest,
    ExecutionResult,
    OutputTransform,
)

__all__ = [
    "CleanupRemovalSummary",
    "CleanupSummary",
    "ExecutionPlan",
    "ExecutionRequest",
    "ExecutionResult",
    "OutputTransform",
    "execute_plan",
    "execute_request",
    "resolve_request",
]
