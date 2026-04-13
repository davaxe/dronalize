from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import AliasChoices, Field

from dronalize.config.base import FullConfig, PartialConfig
from dronalize.core.scene import CANONICAL, TrajectorySchema
from dronalize.core.scene.schema import TrajectorySchemaDefinition

FloatDType = type[np.float32] | type[np.float64]
OutputPrecision = Literal["float32", "float64"]
TrajectorySchemaLike = TrajectorySchema | str | TrajectorySchemaDefinition


class MDSOutputConfig(FullConfig):
    """Backend-specific tuning for Mosaic Streaming output."""

    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int = 67_108_864
    exist_ok: bool = False


class PartialMDSOutputConfig(PartialConfig[MDSOutputConfig]):
    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int | None = None
    exist_ok: bool | None = None
    full_config_type: type[MDSOutputConfig] = MDSOutputConfig


class OutputConfig(FullConfig):
    """Resolved output configuration shared by storage backends."""

    trajectory_schema: TrajectorySchemaLike = Field(
        default=CANONICAL,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    precision: OutputPrecision = "float32"
    recenter_positions: bool = True
    mds: MDSOutputConfig = Field(default_factory=MDSOutputConfig)


class PartialOutputConfig(PartialConfig[OutputConfig]):
    trajectory_schema: TrajectorySchemaLike | None = Field(
        default=None,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    precision: OutputPrecision | None = None
    recenter_positions: bool | None = None
    mds: PartialMDSOutputConfig | None = None
    full_config_type: type[OutputConfig] = OutputConfig
