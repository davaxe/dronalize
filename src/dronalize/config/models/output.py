"""Configuration models for output data formatting and storage."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import AliasChoices, Field

from dronalize.config.base import FullConfig, PartialConfig
from dronalize.core.scene import CANONICAL, TrajectorySchema
from dronalize.core.scene.schema import TrajectorySchemaDefinition

FloatDType = type[np.float32] | type[np.float64]
OutputPrecision = Literal["float32", "float64"]
"""Accepted floating-point precision labels for serialized numeric arrays.

The chosen label is later resolved to the corresponding NumPy dtype when
writers materialize trajectory tensors.
"""

TrajectorySchemaLike = TrajectorySchema | str | TrajectorySchemaDefinition
"""User-facing trajectory schema inputs accepted by output-related config.

Callers may provide a registered schema object directly, reference a schema by
name, or inline a full schema definition that resolves to a
[`TrajectorySchema`][dronalize.core.scene.TrajectorySchema].
"""


class MDSOutputConfig(FullConfig):
    """Backend-specific tuning for Mosaic Streaming output."""

    compression: str | None = None
    """Compression algorithm to use for each Mosaic shard, if any."""
    hashes: tuple[str, ...] | None = None
    """Hash algorithms recorded for each shard, if hashing is enabled."""
    size_limit: str | int = 67_108_864
    """Maximum shard size accepted by the writer before starting a new shard."""
    exist_ok: bool = False
    """Whether an existing output location may be reused instead of raising an error."""


class PartialMDSOutputConfig(PartialConfig[MDSOutputConfig]):
    """Patch model for partially overriding Mosaic Streaming writer settings."""

    compression: str | None = None
    """Replacement compression algorithm for Mosaic shards."""
    hashes: tuple[str, ...] | None = None
    """Replacement hash algorithms recorded for Mosaic shards."""
    size_limit: str | int | None = None
    """Replacement maximum shard size for the Mosaic writer."""
    exist_ok: bool | None = None
    """Replacement policy for whether existing output locations are allowed."""
    full_config_type: type[MDSOutputConfig] = Field(default=MDSOutputConfig, init=False, repr=False)


class OutputConfig(FullConfig):
    """Resolved output configuration shared by storage backends."""

    trajectory_schema: TrajectorySchemaLike = Field(
        default=CANONICAL,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    """Trajectory schema used when encoding scene records."""
    precision: OutputPrecision = "float32"
    """Floating-point precision used for serialized numeric arrays."""
    recenter_positions: bool = True
    """Whether scene positions are translated into a local origin before writing."""
    mds: MDSOutputConfig = Field(default_factory=MDSOutputConfig)
    """Backend-specific tuning for Mosaic Streaming outputs."""


class PartialOutputConfig(PartialConfig[OutputConfig]):
    """Patch model for partially overriding shared output settings."""

    trajectory_schema: TrajectorySchemaLike | None = Field(
        default=None,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    """Replacement trajectory schema used when encoding scene records."""
    precision: OutputPrecision | None = None
    """Replacement floating-point precision for serialized numeric arrays."""
    recenter_positions: bool | None = None
    """Replacement policy for recentering scene positions before writing."""
    mds: PartialMDSOutputConfig | None = None
    """Partial backend-specific overrides for Mosaic Streaming outputs."""
    full_config_type: type[OutputConfig] = Field(default=OutputConfig, init=False, repr=False)
