"""Public request model used by both CLI and Python execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dronalize.config.runtime import RuntimeOverride
from dronalize.io.formats import StorageBackend

if TYPE_CHECKING:
    from dronalize.datasets.registry import DatasetSpec


class ProcessRequest(BaseModel):
    """Normalized user request for one dataset processing job."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    input_dir: Path
    output_dir: Path
    storage_backend: StorageBackend | str = StorageBackend.PICKLE
    config_path: Path | None = None
    overrides: RuntimeOverride = Field(default_factory=RuntimeOverride)
    include_map: bool | None = None
    limit: int | None = None
    seed: int | None = None
    input_dir_exists: bool = True


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Initial internal request used to compile a runtime plan."""

    descriptor: DatasetSpec
    config_path: Path | None = None
    overrides: RuntimeOverride = field(default_factory=RuntimeOverride)
    include_map: bool | None = None
    seed: int | None = None
