"""Execution-first runtime API.

The public runtime surface is organized around four steps:

- build an [`ExecutionRequest`][dronalize.runtime.ExecutionRequest]
- resolve it into an [`ExecutionPlan`][dronalize.runtime.ExecutionPlan]
- execute the request directly with [`execute_request`][dronalize.runtime.execute_request]
- or execute a resolved plan with [`execute_plan`][dronalize.runtime.execute_plan]
"""

from __future__ import annotations

from dronalize.runtime.api import execute_plan, execute_request, resolve_request
from dronalize.runtime.types import ExecutionPlan, ExecutionRequest, ExecutionResult, OutputSample

__all__ = [
    "ExecutionPlan",
    "ExecutionRequest",
    "ExecutionResult",
    "OutputSample",
    "execute_plan",
    "execute_request",
    "resolve_request",
]
