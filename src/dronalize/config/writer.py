from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict

from dronalize.scene import CANONICAL_V1, SceneSchema, get_scene_schema

FloatDType = type[np.float32] | type[np.float64]
WriterPrecision = Literal["float32", "float64"]
SceneSchemaLike = SceneSchema | str | dict[str, object]
ResolvedSceneSchema = Annotated[SceneSchema, BeforeValidator(get_scene_schema)]


class WriterConfig(BaseModel):
    """User-facing configuration shared by scene writers."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    scene_schema: ResolvedSceneSchema = CANONICAL_V1
    precision: WriterPrecision = "float32"
    offset_positions: bool = True

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
        """Return the NumPy dtype implied by ``precision``."""
        return np.float32 if self.precision == "float32" else np.float64
