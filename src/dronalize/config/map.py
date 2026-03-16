from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoExtraction(BaseModel):
    """Configuration for automatic map extraction mode."""

    mode: Literal["auto", "automatic", "dynamic", "heurstic"] = "auto"
    padding_factor: float = Field(gt=1.0, default=1.2)


class RadialExtraction(BaseModel):
    """Configuration for radial map extraction mode."""

    mode: Literal["radial", "circular", "radius", "circle"] = "radial"
    radius: float


class RectangularExtraction(BaseModel):
    """Configuration for rectangular map extraction mode."""

    mode: Literal["rectangular", "box", "rectangle"] = "rectangular"
    width: float
    height: float


class SquareExtraction(BaseModel):
    """Configuration for square map extraction mode."""

    mode: Literal["square"] = "square"
    size: float

    @property
    def width(self) -> float:
        """Alias width to size for square extraction."""
        return self.size

    @property
    def height(self) -> float:
        """Alias height to size for square extraction."""
        return self.size


class NoneExtraction(BaseModel):
    """Configuration for no map extraction."""

    mode: Literal["none", "null", "false"] = "none"


MapExtraction = Annotated[
    RadialExtraction | RectangularExtraction | SquareExtraction | NoneExtraction | AutoExtraction,
    Field(discriminator="mode"),
]


class MapConfig(BaseModel):
    """Configuration for map data processing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    min_distance: float | None = Field(
        gt=0,
        description="Target minimum distance between two adjacent map nodes.",
        default=1.75,
    )
    interp_distance: float | None = Field(
        gt=0,
        description="Distance between interpolated nodes when densifying the map graph.",
        default=3.0,
    )
    include_map: bool = Field(
        default=True,
        description="Whether to include map data at all.",
    )

    extraction: MapExtraction = Field(default_factory=NoneExtraction)

    @model_validator(mode="after")
    def _validate_distances(self) -> MapConfig:
        if self.interp_distance is None or self.min_distance is None:
            return self

        if self.interp_distance < self.min_distance:
            # Fixed the string concatenation and variable name bugs here
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
    def no_extraction(
        cls,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with no extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=NoneExtraction(),
        )

    @classmethod
    def auto_extraction(
        cls,
        padding_factor: float = 1.15,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with automatic extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=AutoExtraction(padding_factor=padding_factor),
        )

    @classmethod
    def radial_extraction(
        cls,
        radius: float,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with radial extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=RadialExtraction(radius=radius),
        )

    @classmethod
    def rectangular_extraction(
        cls,
        width: float,
        height: float,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with rectangular extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=RectangularExtraction(width=width, height=height),
        )

    @classmethod
    def square_extraction(
        cls,
        size: float,
        min_distance: float | None = 1.75,
        interp_distance: float | None = 3.0,
    ) -> MapConfig:
        """Return a map configuration with square extraction."""
        return cls(
            min_distance=min_distance,
            interp_distance=interp_distance,
            extraction=SquareExtraction(size=size),
        )

    @classmethod
    def no_map(cls) -> MapConfig:
        """Return a map configuration that indicates no map should be used."""
        return cls(
            min_distance=None,
            interp_distance=None,
            include_map=False,
            extraction=NoneExtraction(),
        )
