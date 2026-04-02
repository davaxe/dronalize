"""Runtime-only overrides layered on top of authoring configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from dronalize.io.config import SceneSchemaLike  # noqa: TC001
from dronalize.processing.ingest.splits import (  # noqa: TC001
    NativeSplitSelection,
    SplitModeName,
)


class PlanOverrides(BaseModel):
    """CLI/runtime-only overrides layered on top of authoring config."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    scene_schema: SceneSchemaLike | None = None
    jobs: int | None = None
    split: SplitModeName | None = None
    read_split: NativeSplitSelection = None
    ratio: tuple[float, float, float] | None = None
    gap: int | None = None
    segments: int | None = None
    include_map: bool | None = None
