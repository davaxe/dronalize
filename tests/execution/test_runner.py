from pathlib import Path
from typing import Literal, TypedDict

import pytest

from dronalize.categories import DatasetSplit
from dronalize.execution import runner


class _CommonArgs(TypedDict):
    dataset: str
    input_dir: Path
    output_dir: Path
    output_format: Literal["mds", "dummy"]


def _a43_args() -> _CommonArgs:
    return {
        "dataset": "a43",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "output_format": "dummy",
    }


def test_splits_a43() -> None:
    """Test that the runner correctly handles the a43 dataset, which has no predefined splits."""
    common_args = _a43_args()
    args = runner.prepare_dataset(**common_args, split=None)

    # a43 have no predefined splits, so the runner should not assign any.
    assert args.descriptor.predefined_splits == []
    assert args.loader_splits() is None
    assert args.writer_splits() is None

    args = runner.prepare_dataset(**common_args, split=["train"])

    assert args.descriptor.predefined_splits == []
    assert args.loader_splits() == [DatasetSplit.TRAIN]
    assert args.writer_splits() == [DatasetSplit.TRAIN]

    args = runner.prepare_dataset(**common_args, split=["train", "val"])

    assert args.descriptor.predefined_splits == []
    assert args.loader_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]
    assert args.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]

    args = runner.prepare_dataset(**common_args, custom_split=(0.7, 0.2, 0.1))

    assert args.descriptor.predefined_splits == []
    assert args.loader_splits() is None
    assert args.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]

    # If a custom split is provided, predefined splits should not be allowed.
    with pytest.raises(
        ValueError,
        match=r"Custom split weights cannot be used with predefined splits\.",
    ):
        args = runner.prepare_dataset(**common_args, custom_split=(0.5, 0.5, 0.0), split=["train"])


def test_prepare_dataset_accepts_sequence_splits() -> None:
    """Tuple-based split requests should behave the same as list-based ones."""
    args = runner.prepare_dataset(**_a43_args(), split=("train", "val"))

    assert args.loader_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]
    assert args.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]


def test_prepare_dataset_overrides_worker_count() -> None:
    """An explicit jobs override should update both execution mode and worker count."""
    parallel_args = runner.prepare_dataset(**_a43_args(), jobs=3)
    assert parallel_args.parallel is True
    assert parallel_args.config.execution.parallel is True
    assert parallel_args.config.execution.workers == 3

    sequential_args = runner.prepare_dataset(**_a43_args(), jobs=1)
    assert sequential_args.parallel is False
    assert sequential_args.config.execution.parallel is False
    assert sequential_args.config.execution.workers == 1

    with pytest.raises(ValueError, match=r"jobs must be at least 1\."):
        _ = runner.prepare_dataset(**_a43_args(), jobs=0)


def test_splits_waymo() -> None:
    """Test that the runner correctly handles the waymo dataset, which has predefined splits."""
    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "output_format": "dummy",
    }
    args = runner.prepare_dataset(**common_args, split=None)

    # Waymo has predefined splits, but if `split=None` is used it means just
    # process all available data without assigning splits.
    assert sorted(args.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert args.loader_splits() is None
    assert args.writer_splits() is None

    args = runner.prepare_dataset(**common_args, split=["train", "val"])

    assert sorted(args.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert args.loader_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]
    assert args.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]

    # Custom splits should not be allowed when also trying to use predefined splits.
    with pytest.raises(
        ValueError,
        match=r"Custom split weights cannot be used with predefined splits\.",
    ):
        args = runner.prepare_dataset(**common_args, custom_split=(0.5, 0.5, 0.0), split=["train"])

    args = runner.prepare_dataset(**common_args, custom_split=(0.5, 0.4, 0.1), split=None)

    assert args.loader_splits() is None
    writer_splits = args.writer_splits()
    assert writer_splits is not None
    assert sorted(writer_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
