from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelevantAreaExtraction(BaseModel):
    """Configuration for extraction around relevant scene positions."""

    mode: Literal["relevant"] = Field("relevant", repr=False, init=False)
    padding_factor: float = Field(gt=1.0, default=1.2)


class CircularExtraction(BaseModel):
    """Configuration for circular map extraction mode."""

    mode: Literal["circle"] = Field("circle", repr=False, init=False)
    radius: float = Field(gt=0)


class BoundingBoxExtraction(BaseModel):
    """Configuration for bounding-box map extraction mode."""

    mode: Literal["bounding_box"] = Field("bounding_box", repr=False, init=False)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class FullMapExtraction(BaseModel):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


MapExtraction = Annotated[
    CircularExtraction | BoundingBoxExtraction | FullMapExtraction | RelevantAreaExtraction,
    Field(discriminator="mode"),
]


class MapConfig(BaseModel):
    """Configuration for map data processing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    min_distance: float | None = Field(gt=0, default=1.75)
    interp_distance: float | None = Field(gt=0, default=3.0)
    include_map: bool = True
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

    @classmethod
    def default(cls) -> MapConfig:
        """Return the default map configuration."""
        return cls()

    @classmethod
    def full_map(
        cls, min_distance: float | None = 1.75, interp_distance: float | None = 3.0
    ) -> MapConfig:
        """Return a map configuration that keeps the full map."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=FullMapExtraction(),
        )

    @classmethod
    def relevant_area_extraction(
        cls,
        padding_factor: float = 1.15,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration extracted around relevant scene positions."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=RelevantAreaExtraction(padding_factor=padding_factor),
        )

    @classmethod
    def circular_extraction(
        cls, radius: float, min_distance: float | None = 1.75, interp_distance: float | None = 3.0
    ) -> MapConfig:
        """Return a map configuration with circular extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=CircularExtraction(radius=radius),
        )

    @classmethod
    def bounding_box_extraction(
        cls,
        width: float,
        height: float,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with bounding-box extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=BoundingBoxExtraction(width=width, height=height),
        )

    @classmethod
    def no_map(cls) -> MapConfig:
        """Return a map configuration that indicates no map should be used."""
        return cls(
            min_distance=None,
            interp_distance=None,
            include_map=False,
            extraction=FullMapExtraction(),
        )
