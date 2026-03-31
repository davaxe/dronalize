from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dronalize.core.scene import CANONICAL_V1, SceneSchema, get_scene_schema
from dronalize.core.scene.schema import SceneSchemaDefinition

FloatDType = type[np.float32] | type[np.float64]
WriterPrecision = Literal["float32", "float64"]
SceneSchemaLike = SceneSchema | str | SceneSchemaDefinition


class MDSFormatConfig(BaseModel):
    """Backend-specific tuning for the Mosaic Streaming format."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int = 67_108_864
    exist_ok: bool = False


class WriterConfig(BaseModel):
    """Resolved runtime configuration shared by scene writers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    scene_schema: SceneSchema = CANONICAL_V1
    precision: WriterPrecision = "float32"
    offset_positions: bool = True
    mds: MDSFormatConfig = Field(default_factory=MDSFormatConfig)

    @classmethod
    def create(
        cls,
        scene_schema: SceneSchemaLike,
        *,
        precision: WriterPrecision = "float32",
        offset_positions: bool = True,
    ) -> WriterConfig:
        """Flexible constructor for WriterConfig."""
        resolved_schema = get_scene_schema(scene_schema)
        return cls(
            scene_schema=resolved_schema, precision=precision, offset_positions=offset_positions
        )

    def with_scene_schema(self, scene_schema: SceneSchemaLike) -> WriterConfig:
        """Return a copy with a different persisted scene schema."""
        return self.model_copy(update={"scene_schema": get_scene_schema(scene_schema)})

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the persisted feature columns in tensor order."""
        return self.scene_schema.feature_columns()

    @property
    def feature_dim(self) -> int:
        """Return the number of persisted per-agent features."""
        return self.scene_schema.feature_dim

    @property
    def float_dtype(self) -> FloatDType:
        """Return the NumPy dtype implied by `precision`."""
        return np.float32 if self.precision == "float32" else np.float64
