from collections.abc import Sequence
from dataclasses import dataclass

from dronalize.categories import DatasetSplit
from dronalize.exceptions import SplitConflictError, SplitNotSupportedError

_STANDARD_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


@dataclass(frozen=True, slots=True)
class Progress:
    """Snapshot of the executor's current progress."""

    running: bool
    processed_sources: int
    processed_scenes: int
    total_sources: int | None
    total_scenes: int | None
    active_workers: int


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Resolved split routing for one processing request."""

    requested_splits: tuple[DatasetSplit, ...] | None = None
    custom_weights: tuple[float, float, float] | None = None

    def loader_splits(self, predefined_splits: Sequence[DatasetSplit]) -> list[DatasetSplit] | None:
        """Return dataset-defined splits that the loader should read."""
        if self.custom_weights is not None or self.requested_splits is None:
            return None
        if len(predefined_splits) == 0:
            return None
        return list(self.requested_splits)

    def writer_splits(self) -> list[DatasetSplit] | None:
        """Return split directories that the writer should create."""
        if self.custom_weights is not None:
            return [group for group, _ in self.active_custom_groups()]
        if self.requested_splits is None:
            return None
        return list(self.requested_splits)

    def active_custom_groups(self) -> list[tuple[DatasetSplit, float]]:
        """Return non-zero custom split groups and weights."""
        if self.custom_weights is None:
            return []
        return [
            (group, weight)
            for group, weight in zip(_STANDARD_SPLITS, self.custom_weights, strict=True)
            if weight > 0
        ]

    def weights(self) -> list[float]:
        """Return active custom weights in split order."""
        return [weight for _, weight in self.active_custom_groups()]


SplitType = str | DatasetSplit


def resolve_split_plan(
    single_split: SplitType | Sequence[SplitType] | None,
    custom_split_weights: tuple[float, float, float] | None,
) -> SplitPlan:
    """Resolve a split plan from the provided arguments."""
    parsed_split = _parse_requested_splits(single_split)

    if custom_split_weights is not None:
        if parsed_split is not None:
            msg = "Custom split weights cannot be used with predefined splits."
            raise SplitConflictError(msg)
        _validate_split_weights(custom_split_weights)
        return SplitPlan(custom_weights=custom_split_weights)

    if parsed_split is None:
        return SplitPlan()

    requested_splits = parsed_split if isinstance(parsed_split, list) else [parsed_split]
    return SplitPlan(requested_splits=tuple(requested_splits))


def validate_split_plan(
    plan: SplitPlan,
    *,
    dataset_name: str,
    predefined_splits: Sequence[DatasetSplit],
) -> None:
    """Validate a split plan against the dataset's built-in split support."""
    if plan.requested_splits is None:
        return

    if len(predefined_splits) == 0:
        if len(plan.requested_splits) > 1:
            raise SplitNotSupportedError(dataset_name, list(plan.requested_splits))
        return

    unsupported = [split for split in plan.requested_splits if split not in predefined_splits]
    if unsupported:
        raise SplitNotSupportedError(dataset_name, unsupported)


def _parse_requested_splits(
    single_split: SplitType | Sequence[SplitType] | None,
) -> DatasetSplit | list[DatasetSplit] | None:
    """Normalize requested splits into a single value or a concrete list."""
    if single_split is None:
        return None

    if isinstance(single_split, (str, DatasetSplit)):
        return DatasetSplit(single_split)

    parsed_splits = [DatasetSplit(split) for split in single_split]
    if len(parsed_splits) == 1:
        return parsed_splits[0]
    return parsed_splits


def _validate_split_weights(values: tuple[float, float, float]) -> None:
    if any(v < 0 for v in values):
        msg = "All split weights must be positive or zero."
        raise ValueError(msg)
    if sum(values) <= 0:
        msg = "At least one custom split weight must be greater than zero."
        raise ValueError(msg)
