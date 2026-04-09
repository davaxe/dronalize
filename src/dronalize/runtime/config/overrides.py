"""Runtime-only overrides layered on top of authoring configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from dronalize.io.config import TrajectorySchemaLike  # noqa: TC001
from dronalize.processing.loading.splits import (  # noqa: TC001
    NativeSplitStrategySelection,
    SplitStrategyName,
)


class PlanOverrides(BaseModel):
    """CLI/runtime-only overrides layered on top of authoring config."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    trajectory_schema: TrajectorySchemaLike | None = None
    jobs: int | None = None
    split: SplitStrategyName | None = None
    read_split: NativeSplitStrategySelection = None
    ratio: tuple[float, float, float] | None = None
    gap: int | None = None
    segments: int | None = None
    include_map: bool | None = None
