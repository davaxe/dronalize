"""Map processing configuration models for dataset generation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from dronalize.config.base import FullConfig, PartialConfig


class SceneExtentExtraction(FullConfig):
    """Configuration for extraction around the scene trajectory extent."""

    mode: Literal["scene_extent"] = Field("scene_extent", repr=False, init=False)
    padding: float = Field(gt=1.0, default=1.05)
    """Scale factor applied around the scene extent before cropping the map."""


class CircularExtraction(FullConfig):
    """Configuration for circular map extraction mode."""

    mode: Literal["circle"] = Field("circle", repr=False, init=False)
    radius: float = Field(gt=0)
    """Radius of the circular crop centered on the scene in map units."""


class BoundingBoxExtraction(FullConfig):
    """Configuration for bounding-box map extraction mode."""

    mode: Literal["bounding_box"] = Field("bounding_box", repr=False, init=False)
    width: float = Field(gt=0)
    """Width of the extracted map crop in map units."""
    height: float = Field(gt=0)
    """Height of the extracted map crop in map units."""


class FullMapExtraction(FullConfig):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


MapExtraction = Annotated[
    CircularExtraction | BoundingBoxExtraction | FullMapExtraction | SceneExtentExtraction,
    Field(discriminator="mode"),
]
"""Discriminated union of the supported map extraction strategies.

The `mode` field selects whether map geometry is cropped around the scene,
within a circle, within a bounding box, or retained in full.
"""


class MapConfig(FullConfig):
    """Configuration for map data processing."""

    min_distance: float | None = Field(gt=0, default=2)
    """Minimum spacing allowed between neighboring map samples after simplification."""
    interp_distance: float | None = Field(gt=0, default=5.0)
    """Target spacing used when interpolating map geometry."""
    extraction: MapExtraction = Field(default_factory=FullMapExtraction)
    """Map extraction strategy used to crop or retain source map geometry."""

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
    """Patch model for partially overriding :class:`MapConfig`."""

    min_distance: float | None = Field(gt=0, default=None)
    """Replacement minimum spacing for simplified map samples."""
    interp_distance: float | None = Field(gt=0, default=None)
    """Replacement interpolation spacing for map geometry."""
    extraction: MapExtraction | None = Field(default=None)
    """Replacement map extraction strategy."""
    full_config_type: type[MapConfig] = Field(MapConfig, repr=False, init=False)
