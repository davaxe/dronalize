"""Orchestration layer for dataset processing.

This package ties together dataset discovery (registry), configuration,
processing (loaders + pipelines), and output (writers) into a single
cohesive workflow.

Public API
----------
- `process_dataset` — process a single dataset end-to-end.
"""

from dronalize.processing.runner import process_dataset

__all__ = ["process_dataset"]
