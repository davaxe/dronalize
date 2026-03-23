from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from dronalize.scene import CANONICAL_V1, SceneSchema, get_scene_schema

FloatDType = type[np.float32] | type[np.float64]
WriterPrecision = Literal["float32", "float64"]
SceneSchemaLike = SceneSchema | str | dict[str, object]
ResolvedSceneSchema = Annotated[SceneSchema, BeforeValidator(get_scene_schema)]


class ZarrFormatConfig(BaseModel):
    """Backend-specific tuning for the Zarr storage format."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    scene_chunk: int = Field(default=128, gt=0)
    agent_chunk: int = Field(default=4096, gt=0)
    map_node_chunk: int = Field(default=4096, gt=0)
    map_edge_chunk: int = Field(default=4096, gt=0)


class MDSFormatConfig(BaseModel):
    """Backend-specific tuning for the Mosaic Streaming format."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int = 67_108_864
    exist_ok: bool = False


class WriterConfig(BaseModel):
    """User-facing configuration shared by scene writers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    scene_schema: ResolvedSceneSchema = CANONICAL_V1
    precision: WriterPrecision = "float32"
    offset_positions: bool = True
    zarr: ZarrFormatConfig = Field(default_factory=ZarrFormatConfig)
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
            scene_schema=resolved_schema,
            precision=precision,
            offset_positions=offset_positions,
        )

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
