"""Dataset-owned loader option models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


class DatasetOptionsModel(BaseModel):
    """Base model for dataset-specific declarative config."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def parse(cls, payload: dict[str, object] | None = None) -> Self:
        """Validate one plain dataset-owned config mapping."""
        return cls(**(payload or {}))


class NoDatasetOptions(DatasetOptionsModel):
    """Empty dataset config for datasets without dataset-owned settings."""
