"""Internal processing request and plan models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dronalize.config.models import (
    NativeSplitConfig,
    NoSplitConfig,
    SceneSplitConfig,
    ShuffledTimeSplitConfig,
    SourceSplitConfig,
    TimeSplitConfig,
)
from dronalize.core.categories import DatasetSplit

if TYPE_CHECKING:
    from dronalize.config.models import (
        MapConfig,
        ScenesConfig,
        ScreeningConfig,
        SplitConfig,
        SplitConfigUnion,
    )
    from dronalize.processing.loading.options import DatasetOptionsModel


@dataclass(frozen=True, slots=True)
class SplitRequest:
    """Request for dataset splitting, derived from a resolved dataset config."""

    config: SplitConfigUnion
    read: tuple[DatasetSplit, ...] | None = None
    seed: int | None = None

    @classmethod
    def from_config(
        cls,
        split: SplitConfig,
        *,
        read: tuple[DatasetSplit, ...] | None = None,
        seed: int | None = None,
    ) -> SplitRequest:
        """Build a runtime split request from the public config wrapper."""
        split_root = split.root
        native_read = (
            tuple(split_root.splits) if isinstance(split_root, NativeSplitConfig) else None
        )
        return cls(config=split_root, read=read or native_read, seed=seed)

    def active(self) -> tuple[tuple[DatasetSplit, float], ...]:
        """Return the active splits based on the configuration."""
        if isinstance(self.config, NoSplitConfig):
            return ()
        if isinstance(self.config, NativeSplitConfig):
            return ((split, 1.0) for split in self.read) if self.read is not None else ()
        active: list[tuple[DatasetSplit, float]] = []
        if self.config.ratio.train > 0:
            active.append((DatasetSplit.TRAIN, self.config.ratio.train))
        if self.config.ratio.val > 0:
            active.append((DatasetSplit.VAL, self.config.ratio.val))
        if self.config.ratio.test > 0:
            active.append((DatasetSplit.TEST, self.config.ratio.test))
        return tuple(active)

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the active splits based on the configuration."""
        return tuple(split for split, _ in self.active())

    def active_weights(self) -> tuple[float, ...]:
        """Return the active split weights based on the configuration."""
        return tuple(weight for _, weight in self.active())

    def uses_time_partition(self) -> bool:
        """Return whether the split strategy uses time-based partitioning."""
        return isinstance(self.config, (TimeSplitConfig, ShuffledTimeSplitConfig))

    def uses_weighted_assignment(self) -> bool:
        """Return whether the split strategy uses weighted random assignment."""
        return isinstance(self.config, (SceneSplitConfig, SourceSplitConfig))

    def output_splits(
        self, *, available_native_splits: tuple[DatasetSplit, ...] | None = None
    ) -> tuple[DatasetSplit, ...] | None:
        """Return the split directories expected for this request."""
        if isinstance(self.config, NoSplitConfig):
            return None
        if isinstance(self.config, NativeSplitConfig):
            return self.read or available_native_splits
        return self.active_splits()

    @property
    def gap(self) -> int | None:
        """Return the configured temporal split gap, if any."""
        return getattr(self.config, "gap", None)

    @property
    def segments(self) -> int | None:
        """Return the configured shuffled-time segment count, if any."""
        return getattr(self.config, "segments", None)

    @property
    def strategy(self) -> str:
        """Return the split strategy name."""
        return self.config.strategy


@dataclass(frozen=True, slots=True)
class LoaderRequest:
    """Narrow loader-facing request derived from a resolved dataset config."""

    scenes: ScenesConfig
    dataset: DatasetOptionsModel
    screening: ScreeningConfig | None = None
    split: SplitRequest = SplitRequest(config=NoSplitConfig())
    map: MapConfig | None = None
    native_splits: tuple[DatasetSplit, ...] | None = None

    @property
    def history_frames(self) -> int:
        """Return the number of history frames per scene."""
        return self.scenes.history_frames

    @property
    def future_frames(self) -> int:
        """Return the number of future frames per scene."""
        return self.scenes.future_frames


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Trajectory-processing plan for one run."""

    scenes: ScenesConfig
    screening: ScreeningConfig | None = None
    split: SplitRequest | None = None
