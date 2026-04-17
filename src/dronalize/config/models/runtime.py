"""Runtime configuration models for dronalize."""

from __future__ import annotations

import multiprocessing as mp
from typing import Literal

from pydantic import Field
from typing_extensions import override

from dronalize.config.base import FullConfig, PartialConfig


class RuntimeConfig(FullConfig):
    """Execution controls for processing jobs.

    Attributes
    ----------
    jobs : int | Literal["auto"]
        Number of worker processes to use. Set to `"auto"` to let runtime
        choose an appropriate value.
    chunksize : int | None
        Optional per-worker batch size for scene dispatch.
    """

    jobs: int = Field(default=1, gt=0)
    """Number of worker processes to use."""
    chunksize: int | None = Field(default=None, gt=0)
    """Optional per-worker batch size for scene dispatch."""


class PartialRuntimeConfig(PartialConfig[RuntimeConfig]):
    """Patch model for partially overriding :class:`RuntimeConfig`."""

    jobs: int | Literal["auto"] | None = None
    """Replacement worker count or `"auto"` to use the current CPU count."""
    chunksize: int | None = Field(default=None, gt=0)
    """Replacement per-worker batch size for scene dispatch."""
    full_config_type: type[RuntimeConfig] = Field(default=RuntimeConfig, init=False, repr=False)

    @override
    def apply_to(self, target: RuntimeConfig | None, *, exclude_none: bool = True) -> RuntimeConfig:
        partial = self.model_copy(update={"jobs": mp.cpu_count()}) if self.jobs == "auto" else self
        return PartialConfig[RuntimeConfig].apply_to(partial, target, exclude_none=exclude_none)
