from pathlib import Path
from typing import Literal, TypedDict

import pytest

from dronalize.categories import DatasetSplit
from dronalize.config import SplitConfig
from dronalize.config.split import (
    BySceneSplit,
    NativeSplit,
    ShuffledTimeBlockSplit,
    TimeBlockSplit,
    Unsplit,
)
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
    """Test that the runner correctly handles the a43 dataset, which has no native splits."""
    common_args = _a43_args()
    job = prepare_dataset(**common_args, split=None)

    assert job.descriptor.predefined_splits == []
    assert isinstance(job.split_request.strategy, Unsplit)
    assert job.loader_splits() is None
    assert job.writer_splits() is None

    with pytest.raises(ValueError, match=r"a43 does not support split 'train'\."):
        _ = prepare_dataset(**common_args, split=["train"])

    job = prepare_dataset(
        **common_args,
        split_method="time_blocks",
        split_weights=(0.7, 0.2, 0.1),
    )

    assert job.descriptor.predefined_splits == []
    assert job.loader_splits() is None
    assert job.writer_splits() == (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
    assert isinstance(job.split_request.strategy, TimeBlockSplit)
    assert job.split_request.method == "time_blocks"

    with pytest.raises(
        ValueError,
        match=r"Specify a split method explicitly\.",
    ):
        _ = prepare_dataset(**common_args, split_weights=(0.7, 0.2, 0.1))

    with pytest.raises(
        ValueError,
        match=r"Native splits and custom split assignment are mutually exclusive\.",
    ):
        _ = prepare_dataset(
            **common_args,
            split=["train"],
            split_method="time_blocks",
            split_weights=(0.5, 0.5, 0.0),
        )


def test_prepare_dataset_accepts_sequence_splits_for_native_datasets() -> None:
    """Tuple-based split requests should behave the same as list-based ones."""
    job = prepare_dataset(
        dataset="waymo",
        input_dir=Path(__file__).parent,
        output_dir=Path(__file__).parent / "output",
        output_format="dummy",
        split=("train",),
    )

    assert job.loader_splits() == (DatasetSplit.TRAIN,)
    assert job.writer_splits() == (DatasetSplit.TRAIN,)
    assert isinstance(job.split_request.strategy, NativeSplit)


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


def test_prepare_dataset_accepts_auto_worker_mode() -> None:
    """The `-1` worker sentinel should keep parallel mode while deferring worker count."""
    job = prepare_dataset(**_a43_args(), jobs=-1)

    assert job.parallel is True
    assert job.config.execution.parallel is True
    assert job.config.execution.workers is None


def test_prepare_dataset_overrides_scene_schema() -> None:
    """An explicit scene schema override should update the resolved writer config."""
    job = prepare_dataset(**_a43_args(), scene_schema="positions_only")

    assert job.config.writer.scene_schema.name == "positions_only"
    assert job.config.writer.feature_dim == 2
    assert job.config.writer.feature_columns == ("x", "y")


def test_descriptor_from_loader_populates_common_metadata() -> None:
    """Descriptor helpers should derive repeated metadata from the loader class."""
    descriptor = DatasetDescriptor.from_loader("a43-test", A43Loader, has_map=True)

    assert descriptor.name == "a43-test"
    assert descriptor.loader_factory is A43Loader
    assert descriptor.default_config == A43Loader.default_config()
    assert descriptor.default_map_config == A43Loader.default_map_config()
    assert descriptor.native_schema == A43Loader.native_scene_schema()
    assert descriptor.predefined_splits == list(A43Loader.predefined_splits())
    assert descriptor.supported_split_methods == list(A43Loader.supported_split_methods())
    assert descriptor.recommended_split_method == A43Loader.recommended_split_method()
    assert descriptor.has_map is True


def test_dataset_job_build_loader_uses_resolved_runtime_config() -> None:
    """Job-local loader construction should carry resolved runtime settings."""
    job = prepare_dataset(
        **_a43_args(),
        scene_schema="positions_only",
        split_method="time_blocks",
        split_weights=(0.7, 0.2, 0.1),
    )
    loader = job.build_loader()

    assert loader.loader_config == job.config.loader
    assert loader.map_config == job.config.map
    assert loader.requested_scene_schema == job.config.writer.scene_schema
    assert loader.splits is None
    assert loader.split_request is not None
    assert loader.split_request.method == "time_blocks"


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
    """Test that the runner correctly handles the waymo dataset, which has native splits."""
    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "output_format": "dummy",
    }
    job = prepare_dataset(**common_args, split=None)

    assert sorted(job.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert isinstance(job.split_request.strategy, Unsplit)
    assert job.loader_splits() is None
    assert job.writer_splits() is None

    job = prepare_dataset(**common_args, split=["train", "val"])

    assert sorted(job.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert job.loader_splits() == (DatasetSplit.TRAIN, DatasetSplit.VAL)
    assert job.writer_splits() == (DatasetSplit.TRAIN, DatasetSplit.VAL)
    assert isinstance(job.split_request.strategy, NativeSplit)

    with pytest.raises(
        ValueError,
        match=r"Native splits and custom split assignment are mutually exclusive\.",
    ):
        _ = prepare_dataset(**common_args, split_weights=(0.5, 0.5, 0.0), split=["train"])

    job = prepare_dataset(**common_args, split_weights=(0.5, 0.4, 0.1), split=None)

    assert job.loader_splits() is None
    assert isinstance(job.split_request.strategy, BySceneSplit)
    assert job.split_request.method == "by_scene"
    writer_splits = job.writer_splits()
    assert writer_splits is not None
    assert sorted(writer_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])


def test_prepare_dataset_carries_split_config_from_runtime_options() -> None:
    """Runtime split options should be normalized into the resolved config model."""
    job = prepare_dataset(
        **_a43_args(),
        split_method="shuffled_time_blocks",
        split_weights=(0.6, 0.2, 0.2),
        split_gap=5,
        split_n_segments=8,
    )

    assert isinstance(job.config.split, SplitConfig)
    assert isinstance(job.config.split.strategy, ShuffledTimeBlockSplit)
    assert job.config.split.strategy.gap == 5
    assert job.config.split.strategy.segments == 8
    assert job.config.split.weights is not None
    assert job.config.split.weights.values() == (0.6, 0.2, 0.2)


def test_prepare_dataset_uses_split_config_file_until_cli_overrides(tmp_path: Path) -> None:
    """Split config should load from TOML and allow CLI overrides on top."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[a43.split.weights]
train = 0.7
val = 0.2
test = 0.1

[a43.split.strategy]
type = "time_blocks"
gap = 2
""",
        encoding="utf-8",
    )

    job = prepare_dataset(**_a43_args(), config_path=config_path)

    assert isinstance(job.config.split.strategy, TimeBlockSplit)
    assert job.config.split.strategy.gap == 2
    assert job.config.split.weights is not None
    assert job.config.split.weights.values() == (0.7, 0.2, 0.1)

    overridden = prepare_dataset(
        **_a43_args(),
        config_path=config_path,
        split_method="shuffled_time_blocks",
        split_weights=(0.6, 0.2, 0.2),
        split_gap=5,
        split_n_segments=8,
    )

    assert isinstance(overridden.config.split.strategy, ShuffledTimeBlockSplit)
    assert overridden.config.split.strategy.gap == 5
    assert overridden.config.split.strategy.segments == 8
    assert overridden.config.split.weights is not None
    assert overridden.config.split.weights.values() == (0.6, 0.2, 0.2)


def test_prepare_dataset_cli_can_clear_configured_splits(tmp_path: Path) -> None:
    """An explicit CLI no-split request should override split config from file."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[waymo.split.weights]
train = 0.7
val = 0.2
test = 0.1

[waymo.split.strategy]
type = "by_scene"
""",
        encoding="utf-8",
    )

    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "output_format": "dummy",
    }
    job = prepare_dataset(**common_args, config_path=config_path, split_method="unsplit")

    assert isinstance(job.split_request.strategy, Unsplit)
    assert job.loader_splits() is None
    assert job.writer_splits() is None
