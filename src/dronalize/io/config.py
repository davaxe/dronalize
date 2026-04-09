"""Export-side configuration models and shared type aliases."""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dronalize.core.scene import CANONICAL, TrajectorySchema, get_trajectory_schema
from dronalize.core.scene.schema import TrajectorySchemaDefinition

FloatDType = type[np.float32] | type[np.float64]
ExportPrecision = Literal["float32", "float64"]
TrajectorySchemaLike = TrajectorySchema | str | TrajectorySchemaDefinition


class MDSBackendConfig(BaseModel):
    """Backend-specific tuning for the Mosaic Streaming format."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int = 67_108_864
    exist_ok: bool = False


class ExportConfig(BaseModel):
    """Resolved runtime export configuration shared by scene writers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    trajectory_schema: TrajectorySchema = CANONICAL
    precision: ExportPrecision = "float32"
    recenter_positions: bool = True
    mds: MDSBackendConfig = Field(default_factory=MDSBackendConfig)

    @classmethod
    def create(
        cls,
        trajectory_schema: TrajectorySchemaLike,
        *,
        precision: ExportPrecision = "float32",
        recenter_positions: bool = True,
    ) -> ExportConfig:
        """Create an export configuration from a flexible schema reference.

        Parameters
        ----------
        trajectory_schema : TrajectorySchemaLike
            Persisted trajectory schema as a concrete `TrajectorySchema`, registered
            schema name, or inline schema-definition payload.
        precision : {"float32", "float64"}, optional
            Floating-point precision used for persisted feature tensors.
        recenter_positions : bool, optional
            Whether position features should be stored relative to the scene
            mean position.

        Returns
        -------
        ExportConfig
            Fully resolved export configuration.
        """
        resolved_schema = get_trajectory_schema(trajectory_schema)
        return cls(
            trajectory_schema=resolved_schema,
            precision=precision,
            recenter_positions=recenter_positions,
        )

    def with_trajectory_schema(self, trajectory_schema: TrajectorySchemaLike) -> ExportConfig:
        """Return a copy with a different persisted trajectory schema."""
        return self.model_copy(
            update={"trajectory_schema": get_trajectory_schema(trajectory_schema)}
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the persisted feature columns in tensor order."""
        return self.trajectory_schema.feature_columns()

    @property
    def feature_dim(self) -> int:
        """Return the number of persisted per-agent features."""
        return self.trajectory_schema.feature_dim

    @property
    def float_dtype(self) -> FloatDType:
        """Return the NumPy dtype implied by `precision`."""
        return np.float32 if self.precision == "float32" else np.float64
