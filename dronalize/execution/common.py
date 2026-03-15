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


class MultiSplit(BaseModel):
    """Use multiple splits."""

    mode: Literal["multi"] = "multi"
    split: list[DatasetSplit]

    def splits(
        self, predefined_splits: list[DatasetSplit] | None = None
    ) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        return list(self.split)


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
        if sum(values) <= 0:
            msg = "At least one custom split weight must be greater than zero."
            raise ValueError(msg)

        return values

    def _active_groups(self) -> list[tuple[DatasetSplit, float]]:
        order = [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]
        return [
            (group, weight)
            for group, weight in zip(order, self.split_weights, strict=True)
            if weight > 0
        ]

    def weights(self) -> list[float]:
        """Get split weights (proportions) for train, val, and test splits."""
        return [weight for _, weight in self._active_groups()]

    def splits(
        self, predefined_splits: list[DatasetSplit] | None = None
    ) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        return [group for group, _ in self._active_groups()]


class NoSplit(BaseModel):
    """No split, process all data together."""

    mode: Literal["no_split"] = "no_split"

    @staticmethod
    def splits(predefined_splits: list[DatasetSplit] | None = None) -> list[DatasetSplit] | None:
        """Get splits for this mode."""
        _ = predefined_splits
        return None


SplitMode = Annotated[
    SingleSplit | MultiSplit | PredefinedSplit | CustomSplit | NoSplit, Field(discriminator="mode")
]


def resolve_split_mode(
    single_split: DatasetSplit | str | list[DatasetSplit | str] | None,
    custom_split_weights: tuple[float, float, float] | None,
) -> SplitMode:
    """Resolve the split mode based on the provided arguments.

    Parameters
    ----------
    single_split : DatasetSplit | str | Sequence[DatasetSplit | str] | None
        If specified, use a single split or multiple splits for processing.
        Can be one of the predefined splits or a sequence of splits. `None`
        processes all available data together.
    custom_split_weights : tuple[float, float, float] | None
        If specified, use custom split weights for train, val, and test splits.

    Returns
    -------
    SplitMode
        The resolved split mode based on the provided arguments.
    """
    parsed_split: DatasetSplit | list[DatasetSplit] | None = None

    if isinstance(single_split, list) and len(single_split) == 1:
        single_split = single_split[0]
    if isinstance(single_split, str):
        parsed_split = DatasetSplit(single_split)
    elif isinstance(single_split, list):
        # Strings are also sequences, so checking for str first is required
        parsed_split = [DatasetSplit(s) for s in single_split]

    # 2. Validate against custom_split_weights conflicts
    if custom_split_weights is not None:
        # If a sequence or explicit predefined split is provided, raise an error.
        is_invalid_for_custom = isinstance(parsed_split, list) or parsed_split is not None
        if is_invalid_for_custom:
            msg = "Custom split weights cannot be used with predefined splits."
            raise ValueError(msg)
        return CustomSplit(split_weights=custom_split_weights)

    # 3. Resolve and return the correct SplitMode
    if isinstance(parsed_split, list):
        return MultiSplit(split=parsed_split)
    if parsed_split is not None:
        return SingleSplit(split=parsed_split)

    return NoSplit()
