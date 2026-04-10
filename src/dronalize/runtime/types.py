"""Public runtime data models shared by API and internal execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.io.formats import StorageBackend


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Final result of a processing run."""

    dataset: str
    output_dir: Path
    storage_backend: StorageBackend
    processed_sources: int
    processed_scenes: int
    split_counts: dict[str, int]
