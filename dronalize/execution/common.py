from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from dronalize.categories import DatasetSplit


@dataclass(frozen=True, slots=True)
class Progress:
    """Snapshot of the executor's current progress."""

    running: bool
    processed_sources: int
    processed_scenes: int
    total_sources: int | None
    total_scenes: int | None
    active_workers: int


class SingleSplit(BaseModel):
    """Use a single split."""

    mode: Literal["single"] = "single"
    split: DatasetSplit

    def splits(
        self, predefined_splits: list[DatasetSplit] | None = None
    ) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        return [self.split]


class PredefinedSplit(BaseModel):
    """Use predefined splits defined by the dataset descriptor."""

    mode: Literal["predefined"] = "predefined"

    @staticmethod
    def splits(predefined_splits: list[DatasetSplit] | None = None) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        if predefined_splits is None:
            return None
        return predefined_splits


class CustomSplit(BaseModel):
    """Custom split with defined weights."""

    mode: Literal["custom"] = "custom"
    split_weights: tuple[float, float, float]

    @field_validator("split_weights")
    @classmethod
    def _validate_weights(cls, values: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(v < 0 for v in values):
            msg = "All split weights must be positive or zero."
            raise ValueError(msg)

        return values

    def weights(self) -> list[float]:
        """Get split weights (proportions) for train, val, and test splits."""
        return [w for w in self.split_weights if abs(w) < 1e-12]

    def splits(
        self, predefined_splits: list[DatasetSplit] | None = None
    ) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        order = [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]
        groups: list[DatasetSplit] = []
        for i, _ in enumerate(self.split_weights):
            groups.append(order[i])
        return groups


class NoSplit(BaseModel):
    """No split, process all data together."""

    mode: Literal["no_split"] = "no_split"

    @staticmethod
    def splits(predefined_splits: list[DatasetSplit] | None = None) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        return None


SplitMode = Annotated[
    SingleSplit | PredefinedSplit | CustomSplit | NoSplit, Field(discriminator="mode")
]


def resolve_split_mode(
    single_split: DatasetSplit | Literal["train", "test", "val", "all"] | None,
    custom_split_weights: tuple[float, float, float] | None,
) -> SplitMode:
    """Resolve the split mode based on the provided arguments.

    Parameters
    ----------
    single_split : DatasetSplit | Literal["train", "test", "val", "all"] | None
        If specified, use a single split for processing. Can be one of the
        predefined splits or 'all' to process all data together.
    custom_split_weights : tuple[float, float, float] | None
        If specified, use custom split weights for train, val, and test splits.

    Returns
    -------
    SplitMode
        The resolved split mode based on the provided arguments.
    """
    if not isinstance(single_split, DatasetSplit) and single_split is not None:
        single_split = DatasetSplit(single_split)

    if custom_split_weights is not None and single_split not in {DatasetSplit.ALL, None}:
        msg = (
            "When using custom split weights, the split specified should either be unset or 'all'."
        )
        raise ValueError(msg)

    if custom_split_weights is not None:
        return CustomSplit(split_weights=custom_split_weights)
    if single_split is not None:
        return SingleSplit(split=single_split)
    return NoSplit()
