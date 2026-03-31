from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronalize.io.config import WriterConfig
from dronalize.processing.ingest.config import LoaderConfig  # noqa: TC001
from dronalize.processing.ingest.splits import SplitConfig
from dronalize.processing.maps.config import MapConfig


class ResolvedExecutionConfig(BaseModel):
    """Resolved execution settings used by runtime executors."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    jobs: int | None = 1
    chunksize: int | None = None

    @model_validator(mode="after")
    def _validate_jobs(self) -> ResolvedExecutionConfig:
        if self.jobs is not None and self.jobs < 1:
            msg = "Resolved execution jobs must be at least 1 or None for auto."
            raise ValueError(msg)
        return self

    @property
    def parallel(self) -> bool:
        """Return whether execution should use the parallel executor."""
        return self.jobs is None or self.jobs > 1

    @property
    def workers(self) -> int | None:
        """Return the concrete worker count, or None for auto selection."""
        return None if self.jobs is None else max(1, self.jobs)


class ResolvedConfig(BaseModel):
    """Fully resolved, execution-ready runtime configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    loader: LoaderConfig
    map: MapConfig | None = Field(default_factory=MapConfig.default)
    split: SplitConfig = Field(default_factory=SplitConfig)
    execution: ResolvedExecutionConfig = Field(default_factory=ResolvedExecutionConfig)
    writer: WriterConfig = Field(default_factory=WriterConfig)
