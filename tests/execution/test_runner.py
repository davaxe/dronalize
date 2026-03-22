from pathlib import Path
from typing import Literal, TypedDict

import pytest

from dronalize.categories import DatasetSplit
from dronalize.datasets import DatasetDescriptor
from dronalize.datasets.a43.loader import A43Loader
from dronalize.execution.runner import prepare_dataset


class _CommonArgs(TypedDict):
    dataset: str
    input_dir: Path
    output_dir: Path
    output_format: Literal["dummy"]


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
    job = prepare_dataset(**common_args, split=None)

    # a43 has no predefined splits, so the runner should not assign any.
    assert job.descriptor.predefined_splits == []
    assert job.loader_splits() is None
    assert job.writer_splits() is None

    job = prepare_dataset(**common_args, split=["train"])

    assert job.descriptor.predefined_splits == []
    assert job.loader_splits() is None
    assert job.writer_splits() == [DatasetSplit.TRAIN]

    with pytest.raises(
        ValueError,
        match=r"a43 does not support split 'train, val'\.",
    ):
        _ = prepare_dataset(**common_args, split=["train", "val"])

    job = prepare_dataset(**common_args, custom_split=(0.7, 0.2, 0.1))

    assert job.descriptor.predefined_splits == []
    assert job.loader_splits() is None
    assert job.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]

    # If a custom split is provided, predefined splits should not be allowed.
    with pytest.raises(
        ValueError,
        match=r"Custom split weights cannot be used with predefined splits\.",
    ):
        _ = prepare_dataset(**common_args, custom_split=(0.5, 0.5, 0.0), split=["train"])


def test_prepare_dataset_accepts_sequence_splits() -> None:
    """Tuple-based split requests should behave the same as list-based ones."""
    job = prepare_dataset(**_a43_args(), split=("train",))

    assert job.loader_splits() is None
    assert job.writer_splits() == [DatasetSplit.TRAIN]


def test_prepare_dataset_overrides_worker_count() -> None:
    """An explicit jobs override should update both execution mode and worker count."""
    parallel_args = prepare_dataset(**_a43_args(), jobs=3)
    assert parallel_args.parallel is True
    assert parallel_args.config.execution.parallel is True
    assert parallel_args.config.execution.workers == 3

    sequential_args = prepare_dataset(**_a43_args(), jobs=1)
    assert sequential_args.parallel is False
    assert sequential_args.config.execution.parallel is False
    assert sequential_args.config.execution.workers == 1

    with pytest.raises(ValueError, match=r"jobs must be at least 1\."):
        _ = prepare_dataset(**_a43_args(), jobs=0)


def test_prepare_dataset_overrides_scene_schema() -> None:
    """An explicit scene schema override should update the resolved writer config."""
    job = prepare_dataset(**_a43_args(), scene_schema="positions_only")

    assert job.config.writer.scene_schema.name == "positions_only"
    assert job.config.writer.feature_dim == 2
    assert job.config.writer.feature_columns == ("x", "y")


def test_descriptor_from_loader_populates_common_metadata() -> None:
    """Descriptor helpers should derive repeated metadata from the loader class."""
    descriptor = DatasetDescriptor.from_loader("a43-test", A43Loader, A43Loader, has_map=True)

    assert descriptor.name == "a43-test"
    assert descriptor.loader_factory is A43Loader
    assert descriptor.default_config == A43Loader.default_config()
    assert descriptor.default_map_config == A43Loader.default_map_config()
    assert descriptor.native_schema == A43Loader.native_scene_schema()
    assert descriptor.predefined_splits == list(A43Loader.predefined_splits())
    assert descriptor.has_map is True


def test_dataset_job_build_loader_uses_resolved_runtime_config() -> None:
    """Job-local loader construction should carry resolved runtime settings."""
    job = prepare_dataset(**_a43_args(), scene_schema="positions_only")
    loader = job.build_loader()

    assert loader.loader_config == job.config.loader
    assert loader.map_config == job.config.map
    assert loader.requested_scene_schema == job.config.writer.scene_schema
    assert loader.splits is None


def test_dataset_job_open_exposes_live_run() -> None:
    """Prepared jobs should open into live runs with observable executors."""
    job = prepare_dataset(**_a43_args())

    with job.open() as run:
        assert run.job is job
        assert callable(run.executor.progress)
        assert callable(run.executor.progress_event)


def test_dataset_job_run_executes_directly() -> None:
    """Prepared jobs should be executable directly without helper wrappers."""
    job = prepare_dataset(**_a43_args())

    job.run()


def test_splits_waymo() -> None:
    """Test that the runner correctly handles the waymo dataset, which has predefined splits."""
    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "output_format": "dummy",
    }
    job = prepare_dataset(**common_args, split=None)

    # Waymo has predefined splits, but if `split=None` is used it means just
    # process all available data without assigning splits.
    assert sorted(job.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert job.loader_splits() is None
    assert job.writer_splits() is None

    job = prepare_dataset(**common_args, split=["train", "val"])

    assert sorted(job.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert job.loader_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]
    assert job.writer_splits() == [DatasetSplit.TRAIN, DatasetSplit.VAL]

    # Custom splits should not be allowed when also trying to use predefined splits.
    with pytest.raises(
        ValueError,
        match=r"Custom split weights cannot be used with predefined splits\.",
    ):
        _ = prepare_dataset(**common_args, custom_split=(0.5, 0.5, 0.0), split=["train"])

    job = prepare_dataset(**common_args, custom_split=(0.5, 0.4, 0.1), split=None)

    assert job.loader_splits() is None
    writer_splits = job.writer_splits()
    assert writer_splits is not None
    assert sorted(writer_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
