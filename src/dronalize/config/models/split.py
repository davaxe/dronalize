from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, RootModel, model_validator

from dronalize.config.base import FullConfig
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import ConfigurationError


class SplitWeights(FullConfig):
    """Weights used when routing data into train/val/test splits."""

    train: float = Field(ge=0, default=0.0)
    val: float = Field(ge=0, default=0.0)
    test: float = Field(ge=0, default=0.0)

    @model_validator(mode="after")
    def _validate_sum(self) -> SplitWeights:
        """Validate that the weights sum to 1."""
        total = self.train + self.val + self.test
        if total == 0:
            v = f"train={self.train}, val={self.val}, test={self.test}"
            msg = f"At least one split weight must be greater than 0, got {v}"
            raise ConfigurationError(msg)
        return self


class TimeSplitConfig(FullConfig):
    """Time-block split specification."""

    gap: int = Field(ge=0, default=0)
    strategy: Literal["time"] = Field("time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class ShuffledTimeSplitConfig(FullConfig):
    """Shuffled time-block split specification."""

    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    strategy: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class SceneSplitConfig(FullConfig):
    """Scene-based split specification."""

    strategy: Literal["scene"] = Field("scene", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class SourceSplitConfig(FullConfig):
    """Source-based split specification."""

    strategy: Literal["source"] = Field("source", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class NativeSplitConfig(FullConfig):
    """Dataset-defined split specification."""

    strategy: Literal["native"] = Field("native", repr=False, init=False)
    splits: frozenset[DatasetSplit] = Field(
        default=frozenset((DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)),
        max_length=3,
        min_length=1,
    )


class NoSplitConfig(FullConfig):
    """No-split strategy specification."""

    strategy: Literal["none"] = Field("none", repr=False, init=False)


SplitConfigUnion = Annotated[
    TimeSplitConfig
    | ShuffledTimeSplitConfig
    | SceneSplitConfig
    | SourceSplitConfig
    | NativeSplitConfig
    | NoSplitConfig,
    Field(discriminator="strategy"),
]


class SplitConfig(RootModel[SplitConfigUnion]):
    """Split configuration wrapper model for discriminated union behavior."""
