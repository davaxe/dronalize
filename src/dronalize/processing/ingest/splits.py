"""Dataset split configuration and runtime planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitError

SplitModeName = Literal["time", "shuffled-time", "scene", "source", "auto", "native", "none"]
NativeSplitSelection = Sequence[DatasetSplit | str] | DatasetSplit | str | None


class TimeBlockSplit(BaseModel):
    """Time-block split specification.

    Splits a full recording into separate chronological temporal partitions
    based on split ratios. A non-zero gap leaves a buffer of excluded frames
    between neighboring partitions to mitigate leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    gap: int = Field(ge=0, default=0)
    type: Literal["time"] = Field("time", repr=False, init=False)


class ShuffledTimeBlockSplit(BaseModel):
    """Shuffled time-block split specification.

    Partitions recordings into multiple contiguous temporal segments (with
    optional gaps) and assigns those segments to splits according to the
    specified ratios.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    type: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)


class BySceneSplit(BaseModel):
    """Scene-based split specification.

    Randomly assigns each individual scene to a split based on ratios,
    ensuring entire scenes are kept together to avoid data leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["scene"] = Field("scene", repr=False, init=False)


class BySourceSplit(BaseModel):
    """Source-based split specification.

    Assigns data by source (e.g., recording or agent track) rather than scene,
    keeping all data from a given source together to avoid leakage.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["source"] = Field("source", repr=False, init=False)


class NativeSplit(BaseModel):
    """Native split specification.

    Uses dataset-defined partitions (e.g., 'train', 'val', 'test') as-is without
    custom assignment.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    splits: tuple[DatasetSplit, ...] = Field(default_factory=tuple)
    read: tuple[DatasetSplit, ...] = Field(default_factory=tuple)
    type: Literal["native"] = Field("native", repr=False, init=False)

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the dataset splits that should receive data for this request."""
        return self.read


class Unsplit(BaseModel):
    """Unsplit specification.

    Processes all data without custom assignment or dataset-native split routing.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    type: Literal["none"] = Field("none", repr=False, init=False)


SplitMode = Annotated[
    TimeBlockSplit | BySceneSplit | ShuffledTimeBlockSplit | BySourceSplit | NativeSplit | Unsplit,
    Field(discriminator="type"),
]


class SplitWeights(BaseModel):
    """Weights used when routing data into train/val/test splits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    train: float = Field(ge=0, default=0.0)
    val: float = Field(ge=0, default=0.0)
    test: float = Field(ge=0, default=0.0)

    @model_validator(mode="after")
    def _validate_total_weight(self) -> SplitWeights:
        if self.train + self.val + self.test <= 0:
            msg = "At least one split weight must be greater than zero."
            raise SplitError(msg)
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


class SplitConfig(BaseModel):
    """Runtime split configuration resolved from config files or CLI overrides."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    mode: SplitMode = Field(default_factory=Unsplit)
    ratio: SplitWeights | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> SplitConfig:
        if isinstance(self.mode, (NativeSplit, Unsplit)):
            if self.ratio is not None:
                msg = "Resolved native or unsplit configs must not carry split ratios."
                raise SplitError(msg)
            return self

        if self.ratio is None:
            msg = "Resolved custom split configs require train/val/test split ratios."
            raise SplitError(msg)
        return self

    @property
    def mode_name(self) -> SplitModeName:
        """Return the resolved split mode name."""
        return self.mode.type

    @property
    def uses_block_split(self) -> bool:
        """Return whether the split mode uses block-based splitting."""
        return type(self.mode) in {TimeBlockSplit, ShuffledTimeBlockSplit}

    def active_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the dataset splits that should receive data for this request."""
        return self.ratio.active_splits() if self.ratio else ()

    def active_weights(self) -> tuple[float, ...]:
        """Return the non-zero weights corresponding to `active_splits()`."""
        return self.ratio.active_weights() if self.ratio else ()

    def active(self) -> tuple[tuple[DatasetSplit, float], ...]:
        """Return tuples of (split, weight) for splits with non-zero weights."""
        return tuple(zip(self.active_splits(), self.active_weights(), strict=True))

    def loader_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Get splits to pass to the loader."""
        if isinstance(self.mode, NativeSplit):
            return self.mode.active_splits() or None
        return None

    def writer_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Get splits to pass to the writer."""
        if isinstance(self.mode, NativeSplit):
            return self.mode.active_splits() or None
        if isinstance(self.mode, Unsplit):
            return None
        return self.active_splits() or None

    def with_seed(self, seed: int | None) -> SplitConfig:
        """Return a copy with the plan-specific random seed attached."""
        return self.model_copy(update={"seed": seed})
