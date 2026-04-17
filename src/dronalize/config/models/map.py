from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from dronalize.config.base import FullConfig, PartialConfig


class SceneExtentExtraction(FullConfig):
    """Configuration for extraction around the scene trajectory extent."""

    mode: Literal["scene_extent"] = Field("scene_extent", repr=False, init=False)
    padding: float = Field(gt=1.0, default=1.05)


class CircularExtraction(FullConfig):
    """Configuration for circular map extraction mode."""

    mode: Literal["circle"] = Field("circle", repr=False, init=False)
    radius: float = Field(gt=0)


class BoundingBoxExtraction(FullConfig):
    """Configuration for bounding-box map extraction mode."""

    mode: Literal["bounding_box"] = Field("bounding_box", repr=False, init=False)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class FullMapExtraction(FullConfig):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


MapExtraction = Annotated[
    CircularExtraction | BoundingBoxExtraction | FullMapExtraction | SceneExtentExtraction,
    Field(discriminator="mode"),
]


class MapConfig(FullConfig):
    """Configuration for map data processing."""

    min_distance: float | None = Field(gt=0, default=1.5)
    interp_distance: float | None = Field(gt=0, default=4.0)
    extraction: MapExtraction = Field(default_factory=FullMapExtraction)

    @model_validator(mode="after")
    def _validate_distances(self) -> MapConfig:
        if self.interp_distance is None or self.min_distance is None:
            return self
        if self.interp_distance < self.min_distance:
            msg = (
                f"interp_distance ({self.interp_distance}) must be greater "
                f"than or equal to min_distance ({self.min_distance})."
            )
            raise ValueError(msg)
        return self


class PartialMapConfig(PartialConfig[MapConfig]):
    min_distance: float | None = Field(gt=0, default=None)
    interp_distance: float | None = Field(gt=0, default=None)
    extraction: MapExtraction | None = None
    full_config_type: type[MapConfig] = MapConfig
