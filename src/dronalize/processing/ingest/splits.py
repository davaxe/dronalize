"""Dataset split configuration and runtime planning.

This module defines the loader-side split strategies used across the project:
time-block splits, shuffled time-block splits, scene-level assignment, and
source-level assignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import (
    ConfigurationError,
    SplitConflictError,
    SplitNotSupportedError,
    SplitStrategyNotSupportedError,
)

SplitStrategyName = Literal[
    "time_blocks", "shuffled_time_blocks", "by_scene", "by_source", "auto", "native", "unsplit"
]
NativeSplitSelection = Sequence[DatasetSplit | str] | DatasetSplit | str | None


class TimeBlockSplit(BaseModel):
    """Time-block split specification.

    Splits a full recording into separate chronological temporal partitions
    based on split ratios. A non-zero gap leaves a buffer of excluded frames
    between neighboring partitions to mitigate leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    gap: int = Field(ge=0, default=0)
    type: Literal["time_blocks"] = Field("time_blocks", repr=False, init=False)


class ShuffledTimeBlockSplit(BaseModel):
    """Shuffled time-block split specification.

    Partitions recordings into multiple contiguous temporal segments (with
    optional gaps) and assigns those segments to splits according to the
    specified ratios.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    type: Literal["shuffled_time_blocks"] = Field("shuffled_time_blocks", repr=False, init=False)


class BySceneSplit(BaseModel):
    """Scene-based split specification.

    Randomly assigns each individual scene to a split based on ratios,
    ensuring entire scenes are kept together to avoid data leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["by_scene"] = Field("by_scene", repr=False, init=False)


class BySourceSplit(BaseModel):
    """Source-based split specification.

    Assigns data by source (e.g., recording or agent track) rather than scene,
    keeping all data from a given source together to avoid leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["by_source"] = Field("by_source", repr=False, init=False)


class NativeSplit(BaseModel):
    """Native split specification.

    Uses dataset-defined partitions (e.g., 'train', 'val', 'test') as-is without
    custom assignment.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    train: bool = Field(default=False)
    val: bool = Field(default=False)
    test: bool = Field(default=False)
    type: Literal["native"] = Field("native", repr=False, init=False)

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the dataset splits that should receive data for this request."""
        return tuple(split for split in DatasetSplit if getattr(self, split.value, False))


class Unsplit(BaseModel):
    """Unsplit specification.

    Processes all data without custom assignment or dataset-native split routing.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["unsplit"] = Field("unsplit", repr=False, init=False)


SplitStrategy = Annotated[
    TimeBlockSplit | BySceneSplit | ShuffledTimeBlockSplit | BySourceSplit | NativeSplit | Unsplit,
    Field(discriminator="type"),
]


class SplitWeights(BaseModel):
    """Weights used when routing data into train/val/test splits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    train: float = Field(ge=0)
    val: float = Field(ge=0)
    test: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _set_none_to_zero(cls, values: dict[str, float]) -> dict[str, float]:
        for split in ("train", "val", "test"):
            if values.get(split) is None:
                values[split] = 0.0
        return values

    @model_validator(mode="after")
    def _validate_total_weight(self) -> SplitWeights:
        if self.train + self.val + self.test <= 0:
            msg = "At least one split weight must be greater than zero."
            raise ValueError(msg)
        return self

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return dataset splits with non-zero weights in train/val/test order."""
        splits = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
        return tuple(s for s, w in zip(splits, self.values(), strict=True) if w > 0)

    def active_weights(self, *, normalize: bool = True) -> tuple[float, ...]:
        """Return non-zero weights in train/val/test order."""
        return tuple(w for w in self.values(normalize=normalize) if w > 0)

    def values(self, *, normalize: bool = False) -> tuple[float, float, float]:
        """Return the raw train/val/test weights."""
        if normalize:
            total = self.train + self.val + self.test
            return (self.train / total, self.val / total, self.test / total)
        return (self.train, self.val, self.test)

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float]) -> SplitWeights:
        """Construct from a tuple of train/val/test weights."""
        return cls(train=values[0], val=values[1], test=values[2])


class SplitRequest(BaseModel):
    """Resolved loader-side custom split request."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    strategy: SplitStrategy
    weights: SplitWeights | None = None
    seed: int | None = None

    @property
    def strategy_name(self) -> SplitStrategyName:
        """Return the resolved split strategy name."""
        return self.strategy.type

    @property
    def uses_block_split(self) -> bool:
        """Return whether the split strategy uses block-based splitting."""
        return type(self.strategy) in {TimeBlockSplit, ShuffledTimeBlockSplit}

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the dataset splits that should receive data for this request."""
        return self.weights.active_splits() if self.weights else ()

    def active_weights(self) -> tuple[float, ...]:
        """Return the non-zero weights corresponding to `active_splits()`."""
        return self.weights.active_weights() if self.weights else ()

    def active(self) -> tuple[tuple[DatasetSplit, float], ...]:
        """Return tuples of (split, weight) for splits with non-zero weights."""
        return tuple(zip(self.active_splits(), self.active_weights(), strict=True))

    def loader_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Get splits to pass to the loader."""
        if isinstance(self.strategy, NativeSplit):
            return self.strategy.active_splits() or None
        return None

    def writer_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Get splits to pass to the writer."""
        if isinstance(self.strategy, NativeSplit):
            return self.strategy.active_splits() or None
        if isinstance(self.strategy, Unsplit):
            return None
        return self.active_splits() or None


class SplitConfig(BaseModel):
    """Runtime split configuration resolved from config files or CLI overrides."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    strategy: SplitStrategy | None = None
    weights: SplitWeights | None = None

    def request(self, seed: int | None = None) -> SplitRequest:
        """Get a loader-side split request based on this configuration."""
        return SplitRequest(strategy=self.strategy or Unsplit(), weights=self.weights, seed=seed)

    def resolve_runtime_overrides(
        self,
        split: Sequence[DatasetSplit | str] | DatasetSplit | str | None,
        split_strategy_name: SplitStrategyName | None,
        split_weights: tuple[float, float, float] | None,
        split_gap: int | None = None,
        split_n_segments: int | None = None,
        *,
        dataset_name: str,
        predefined_splits: Sequence[DatasetSplit] = (),
        supported_split_strategies: Sequence[SplitStrategyName] = (),
        recommended_split_strategy: SplitStrategyName | None = None,
    ) -> SplitConfig:
        """Return a copy with runtime split overrides applied."""
        if split is not None and (
            split_weights is not None
            or split_gap is not None
            or split_n_segments is not None
            or split_strategy_name not in {None, "native"}
        ):
            msg = "Native splits and custom split assignment are mutually exclusive."
            raise SplitConflictError(msg)

        norm_splits = ()
        if split is not None:
            requested = [split] if isinstance(split, (DatasetSplit, str)) else list(split)
            norm_splits = tuple(dict.fromkeys(DatasetSplit(v) for v in requested))

        strat_name = split_strategy_name
        if strat_name == "auto" or (strat_name is None and split_weights is not None):
            strat_name = _resolve_auto_strategy(
                dataset_name, supported_split_strategies, recommended_split_strategy
            )

        strategy = _resolve_strategy(
            strat_name, split_gap, split_n_segments, norm_splits, self.strategy
        )

        if isinstance(strategy, Unsplit):
            return SplitConfig(strategy=strategy)

        if isinstance(strategy, NativeSplit):
            req_splits = strategy.active_splits() or tuple(predefined_splits)
            if not req_splits:
                msg = f"{dataset_name} does not expose native dataset splits."
                raise ConfigurationError(msg)

            unsupported = [s for s in req_splits if s not in predefined_splits]
            if unsupported or not predefined_splits:
                raise SplitNotSupportedError(dataset_name, unsupported or list(req_splits))

            return SplitConfig(strategy=strategy)

        if strategy.type not in supported_split_strategies:
            raise SplitStrategyNotSupportedError(
                dataset_name, strategy.type, tuple(supported_split_strategies)
            )

        weights = (
            SplitWeights.from_tuple(split_weights) if split_weights is not None else self.weights
        )
        if weights is None:
            msg = "Custom split assignment requires train/val/test split weights."
            raise ConfigurationError(msg)

        return SplitConfig(strategy=strategy, weights=weights)


def _resolve_strategy(
    strategy_name: SplitStrategyName | None,
    split_gap: int | None = None,
    split_n_segments: int | None = None,
    splits: Sequence[DatasetSplit] = (),
    previous_strategy: SplitStrategy | None = None,
) -> SplitStrategy:
    split_vals = {s.value for s in splits}
    strategy_map: dict[str | None, SplitStrategy] = {
        "time_blocks": TimeBlockSplit(gap=split_gap or 0),
        "shuffled_time_blocks": ShuffledTimeBlockSplit(
            segments=split_n_segments or 1, gap=split_gap or 0
        ),
        "by_scene": BySceneSplit(),
        "by_source": BySourceSplit(),
        "unsplit": Unsplit(),
        "native": NativeSplit(
            train="train" in split_vals, val="val" in split_vals, test="test" in split_vals
        ),
    }
    strategy = strategy_map.get(strategy_name) or previous_strategy
    if strategy is None and splits:
        return strategy_map["native"]

    return strategy or Unsplit()


def _resolve_auto_strategy(
    dataset_name: str,
    supported_strategies: Sequence[SplitStrategyName],
    recommended_strategy: SplitStrategyName | None,
) -> SplitStrategyName:
    if not supported_strategies:
        msg = f"{dataset_name} does not support custom split assignment."
        raise ConfigurationError(msg)
    if recommended_strategy:
        return recommended_strategy
    if len(supported_strategies) == 1:
        return supported_strategies[0]
    msg = "Specify a split strategy explicitly."
    raise ConfigurationError(msg)
