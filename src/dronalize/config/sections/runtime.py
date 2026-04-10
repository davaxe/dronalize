from __future__ import annotations

from typing import Literal

from pydantic import Field

from dronalize.config.base import FullConfig, PartialConfig


class RuntimeConfig(FullConfig):
    jobs: int | Literal["auto"] = 1
    chunksize: int | None = Field(default=None, gt=0)


class PartialRuntimeConfig(PartialConfig[RuntimeConfig]):
    jobs: int | Literal["auto"] | None = None
    chunksize: int | None = Field(default=None, gt=0)
    full_config_type: type[RuntimeConfig] = RuntimeConfig
