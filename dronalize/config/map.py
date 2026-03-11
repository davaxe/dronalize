from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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

    @computed_field
    @property
    def width(self) -> float:
        """Alias width to size for square extraction."""
        return self.size

    @computed_field
    @property
    def height(self) -> float:
        """Alias height to size for square extraction."""
        return self.size


class NoneExtraction(BaseModel):
    """Configuration for no map extraction."""

    mode: Literal["none", "null", "false"] = "none"


MapExtraction = Annotated[
    RadialExtraction | RectangularExtraction | SquareExtraction | NoneExtraction,
    Field(discriminator="mode"),
]


class MapConfig(BaseModel):
    """Configuration for map data processing."""

    model_config = ConfigDict(frozen=True)

    min_distance: float = Field(
        gt=0,
        description="Target minimum distance between two adjacent map nodes.",
        default=1.75,
    )
    interp_distance: float = Field(
        gt=0,
        description="Distance between interpolated nodes when densifying the map graph.",
        default=3.0,
    )
    include_map: bool = Field(
        default=True,
        description="Whether to include map data at all.",
    )

    # Add the extracted configuration here
    extraction: MapExtraction = Field(default_factory=NoneExtraction)

    @model_validator(mode="after")
    def _validate_distances(self) -> MapConfig:
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
