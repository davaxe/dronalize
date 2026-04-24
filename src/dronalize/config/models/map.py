"""Map processing configuration models for dataset generation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator
from typing_extensions import override

from dronalize.config.base import FullConfig, PartialConfig, apply_optional
from dronalize.core.categories import EdgeType, EdgeTypeLike, coerce_edge_types

EdgeTypes = Annotated[
    frozenset[EdgeType], BeforeValidator(lambda v: coerce_edge_types(v, frozenset))
]


class SceneExtentExtraction(FullConfig):
    """Configuration for extraction around the scene trajectory extent."""

    mode: Literal["scene_extent"] = Field("scene_extent", repr=False, init=False)
    padding: float = Field(ge=1.0, default=1.0)
    """Scale factor applied around the scene extent before cropping the map."""
    shape: Literal["circle", "bounding_box"] = Field(default="circle")
    """Adaptive crop shape used around the scene extent."""


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


class TrajectoryBufferExtraction(FullConfig):
    """Configuration for buffering the scene trajectories directly."""

    mode: Literal["trajectory_buffer"] = Field("trajectory_buffer", repr=False, init=False)
    radius: float = Field(gt=0)
    """Buffer radius around each relevant trajectory point in map units."""


class FullMapExtraction(FullConfig):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


class MapEdgeTypesConfig(FullConfig):
    """Semantic edge-type filtering and remapping rules."""

    include: EdgeTypes | None = Field(default=None)
    """Optional allow-list of edge types to keep after remapping."""
    exclude: EdgeTypes = Field(default_factory=frozenset)
    """Edge types to drop after remapping."""
    remap: dict[EdgeType, EdgeType] = Field(default_factory=dict)
    """Mapping applied to edge types before include/exclude filters."""

    @model_validator(mode="after")
    def _validate_no_conflicts(self) -> MapEdgeTypesConfig:
        if self.include is not None and self.exclude.intersection(self.include):
            overlap = self.exclude.intersection(self.include)
            msg = f"Conflict in edge type rules: {overlap}."
            raise ValueError(msg)
        if self.remap.keys() & self.remap.values():
            overlap = self.remap.keys() & self.remap.values()
            msg = f"Conflict in edge type remapping: {overlap}."
            raise ValueError(msg)
        return self

    @field_validator("remap", mode="before")
    @classmethod
    def _coerce_remap_keys_and_values(
        cls, v: dict[EdgeTypeLike, EdgeTypeLike]
    ) -> dict[EdgeType, EdgeType]:
        return {EdgeType.from_value(k): EdgeType.from_value(val) for k, val in v.items()}


MapExtraction = Annotated[
    CircularExtraction
    | BoundingBoxExtraction
    | FullMapExtraction
    | SceneExtentExtraction
    | TrajectoryBufferExtraction,
    Field(discriminator="mode"),
]
"""Discriminated union of the supported map extraction strategies.

The `mode` field selects whether map geometry is cropped around the scene,
within a circle, within a bounding box, around the trajectory points directly,
or retained in full.
"""


class MapConfig(FullConfig):
    """Configuration for map data processing."""

    min_distance: float | None = Field(gt=0, default=2)
    """Minimum spacing allowed between neighboring map samples after simplification."""
    interp_distance: float | None = Field(gt=0, default=5.0)
    """Target spacing used when interpolating map geometry."""
    extraction: MapExtraction = Field(default_factory=FullMapExtraction)
    """Map extraction strategy used to crop or retain source map geometry."""
    edge_types: MapEdgeTypesConfig | None = Field(default=None)
    """Optional edge-type remapping and filtering rules."""

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
    edge_types: PartialMapEdgeTypesConfig | Literal[False] | None = Field(default=None)
    """Replacement edge-type rules, or `false` to clear inherited rules."""
    full_config_type: type[MapConfig] = Field(MapConfig, repr=False, init=False)

    @override
    def apply_to(self, target: MapConfig | None, *, exclude_none: bool = True) -> MapConfig:
        if exclude_none is False:
            return super().apply_to(target, exclude_none=exclude_none)

        base = MapConfig() if target is None else target
        return MapConfig(
            min_distance=self.min_distance if self.min_distance is not None else base.min_distance,
            interp_distance=(
                self.interp_distance if self.interp_distance is not None else base.interp_distance
            ),
            extraction=self.extraction if self.extraction is not None else base.extraction,
            edge_types=apply_optional(self.edge_types, base.edge_types),
        )


class PartialMapEdgeTypesConfig(PartialConfig[MapEdgeTypesConfig]):
    """Patch model for partially overriding :class:`MapEdgeTypesConfig`."""

    include: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement allow-list of edge types to keep."""
    exclude: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement deny-list of edge types to drop."""
    remap: dict[EdgeTypeLike, EdgeTypeLike] | None = Field(default=None)
    """Replacement edge-type remapping rules."""
    full_config_type: type[MapEdgeTypesConfig] = Field(MapEdgeTypesConfig, repr=False, init=False)
