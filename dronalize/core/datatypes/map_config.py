from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MapConfig(BaseModel):
    """Configuration for map data processing."""

    model_config = ConfigDict(frozen=True)

    min_distance: float = Field(
        gt=0,
        description="Target minimum distance between to adjacent map nodes.",
        default=1.75,
    )
    interp_distance: float = Field(
        gt=0,
        description="Distance between interpolated nodes when densifying the map graph.",
        default=3,
    )

    include_map: bool = Field(
        default=True,
        description="Whether to include map data at all.",
    )

    @model_validator(mode="after")
    def _validate(self) -> MapConfig:
        if self.interp_distance < self.min_distance:
            msg = f"interp_distance ({self.interp_distance}) must be greater"
            "than or equal to min_dist ({self.min_dist})."
            raise ValueError(msg)
        return self

    @classmethod
    def default(cls) -> MapConfig:
        """Return the default map configuration."""
        return cls()
