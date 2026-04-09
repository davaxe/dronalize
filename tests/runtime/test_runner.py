# pyright: standard
# ruff: noqa: PLC2701
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import BaseModel, ValidationError

import dronalize
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import ConfigurationError, UnsupportedStorageBackendError
from dronalize.datasets import DatasetCapabilities, DatasetSpec
from dronalize.datasets.a43 import DATASET_SPEC as A43_DATASET_SPEC
from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.argoverse1 import DATASET_SPEC as ARGOVERSE1_DATASET_SPEC
from dronalize.datasets.argoverse1.loader import Argoverse1Loader, Argoverse1LoaderOptions
from dronalize.datasets.argoverse2.loader import Argoverse2LoaderOptions
from dronalize.datasets.eth_ucy import DATASET_SPECS as ETH_UCY_DATASET_SPECS
from dronalize.datasets.interact.loader import InteractionLoaderOptions
from dronalize.datasets.lyft.loader import LyftLoaderOptions
from dronalize.io.formats import StorageBackend
from dronalize.processing.loading import (
    NativeSplitStrategy,
    NoLoaderOptions,
    NoSplitStrategy,
    SceneSplitStrategy,
    ShuffledTimeBlockStrategy,
    SplitConfig,
    TimeBlockStrategy,
)
from dronalize.runtime import DatasetPlan, DatasetRun, plan_dataset, summarize_plan
from dronalize.runtime.models import _build_executor, _build_loader, _build_writer_factory
from dronalize.runtime.parallel.executor import ParallelExecutor
from dronalize.runtime.sequential import SequentialExecutor


class _CommonArgs(TypedDict):
    dataset: str
    input_dir: Path
    output_dir: Path
    storage_backend: StorageBackend


def _a43_args() -> _CommonArgs:
    return {
        "dataset": "a43",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "storage_backend": StorageBackend.NULL,
    }


def test_public_import_surface() -> None:
    """The root package should stay a namespace instead of re-exporting helpers."""
    assert not hasattr(dronalize, "DatasetPlan")
    assert not hasattr(dronalize, "plan_dataset")
    with pytest.raises(AttributeError):
        _ = dronalize.DatasetPlan


def test_dataset_package_roots_expose_descriptors_only() -> None:
    """Dataset package roots should publish descriptors, not loader internals."""
    assert A43_DATASET_SPEC.name == "a43"
    assert ARGOVERSE1_DATASET_SPEC.name == "argoverse1"
    assert set(ETH_UCY_DATASET_SPECS) == {"eth", "hotel", "univ", "zara1", "zara2"}
    assert not hasattr(__import__("dronalize.datasets.a43", fromlist=["A43Loader"]), "A43Loader")
    assert not hasattr(
        __import__("dronalize.datasets.argoverse1", fromlist=["Argoverse1MapBuilder"]),
        "Argoverse1MapBuilder",
    )


def test_a43_splits() -> None:
    """A43 should support custom split strategies but not native dataset splits."""
    common_args = _a43_args()
    plan_obj = plan_dataset(**common_args, split=None)

    assert isinstance(plan_obj, DatasetPlan)
    assert plan_obj.descriptor.predefined_splits == []
    assert isinstance(plan_obj.split_request.strategy, NoSplitStrategy)
    assert plan_obj.loader_splits() is None
    assert plan_obj.output_splits() is None

    with pytest.raises(ValueError, match=r"does not expose native dataset splits"):
        _ = plan_dataset(**common_args, split="native", read_split=["train"])

    plan_obj = plan_dataset(**common_args, split="time", ratio=(0.7, 0.2, 0.1))

    assert plan_obj.descriptor.predefined_splits == []
    assert plan_obj.loader_splits() is None
    assert plan_obj.output_splits() == (
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    )
    assert isinstance(plan_obj.split_request.strategy, TimeBlockStrategy)
    assert plan_obj.split_request.strategy_name == "time"

    with pytest.raises(ValueError, match=r"--ratio is only valid with custom split strategies"):
        _ = plan_dataset(**common_args, ratio=(0.7, 0.2, 0.1))

    with pytest.raises(ValueError, match=r"Specify a split strategy explicitly"):
        _ = plan_dataset(**common_args, split="auto", ratio=(0.7, 0.2, 0.1))


def test_plan_dataset_native_splits() -> None:
    """Native split requests should read dataset-defined partitions explicitly."""
    plan_obj = plan_dataset(
        dataset="waymo",
        input_dir=Path(__file__).parent,
        output_dir=Path(__file__).parent / "output",
        storage_backend=StorageBackend.NULL,
        split="native",
        read_split="train",
    )

    assert plan_obj.loader_splits() == (DatasetSplit.TRAIN,)
    assert plan_obj.output_splits() == (DatasetSplit.TRAIN,)
    assert isinstance(plan_obj.split_request.strategy, NativeSplitStrategy)


def test_plan_dataset_workers() -> None:
    """An explicit jobs override should update both execution mode and worker count."""
    parallel_plan = plan_dataset(**_a43_args(), jobs=3)
    assert parallel_plan.parallel is True
    assert parallel_plan.config.execution.parallel is True
    assert parallel_plan.config.execution.jobs == 3

    sequential_plan = plan_dataset(**_a43_args(), jobs=1)
    assert sequential_plan.parallel is False
    assert sequential_plan.config.execution.parallel is False
    assert sequential_plan.config.execution.jobs == 1

    with pytest.raises(ValueError, match=r"jobs must be at least 1\."):
        _ = plan_dataset(**_a43_args(), jobs=0)


def test_plan_dataset_auto_workers() -> None:
    """The `-1` worker sentinel should keep parallel mode while deferring worker count."""
    plan_obj = plan_dataset(**_a43_args(), jobs=-1)

    assert plan_obj.parallel is True
    assert plan_obj.config.execution.parallel is True
    assert plan_obj.config.execution.jobs is None


def test_plan_dataset_schema() -> None:
    """An explicit trajectory-schema override should update the resolved export config."""
    plan_obj = plan_dataset(**_a43_args(), trajectory_schema="positions_only")

    assert plan_obj.config.export.trajectory_schema.name == "positions_only"
    assert plan_obj.config.export.feature_dim == 2
    assert plan_obj.config.export.feature_columns == ("x", "y")


def test_plan_dataset_include_map_overrides() -> None:
    """Map inclusion overrides should preserve or disable the resolved map config correctly."""
    default_plan = plan_dataset(**_a43_args())
    explicit_true = plan_dataset(**_a43_args(), include_map=True)
    explicit_false = plan_dataset(**_a43_args(), include_map=False)

    assert default_plan.config.map == A43Loader.default_map_config()
    assert explicit_true.config.map == A43Loader.default_map_config()
    assert explicit_false.config.map is None


def test_descriptor_from_loader() -> None:
    """Dataset-spec helpers should derive repeated metadata from the loader class."""
    descriptor = DatasetSpec.from_loader(
        "a43-test",
        A43Loader,
        capabilities=DatasetCapabilities.MAP_AVAILABLE,
        infer_capabilities=True,
    )

    assert descriptor.name == "a43-test"
    assert descriptor.loader_factory is A43Loader
    assert descriptor.default_loader_config == A43Loader.default_config()
    assert descriptor.loader_options_type is NoLoaderOptions
    assert descriptor.default_loader_options == A43Loader.default_loader_options()
    assert descriptor.default_map_config == A43Loader.default_map_config()
    assert descriptor.native_schema == A43Loader.native_trajectory_schema()
    assert descriptor.predefined_splits == list(A43Loader.predefined_splits())
    assert descriptor.supported_split_strategies == list(A43Loader.supported_split_strategies())
    assert descriptor.recommended_split_strategy == A43Loader.recommended_split_strategy()
    assert descriptor.has_map is True
    assert descriptor.capabilities & DatasetCapabilities.MAP_AVAILABLE
    assert descriptor.capabilities & DatasetCapabilities.CUSTOM_SPLITS


def test_descriptor_infers_map_capability_from_default_map_config() -> None:
    """Dataset specs built from loaders with map defaults should advertise map support."""
    descriptor = DatasetSpec.from_loader("a43-test", A43Loader, infer_capabilities=True)

    assert descriptor.has_map is True
    assert descriptor.capabilities & DatasetCapabilities.MAP_AVAILABLE


def test_descriptor_infers_loader_option_capability() -> None:
    """Dataset specs should advertise typed dataset-specific loader options explicitly."""
    descriptor = DatasetSpec.from_loader(
        "argoverse1-test",
        Argoverse1Loader,
        infer_capabilities=True,
    )

    assert descriptor.loader_options_type is Argoverse1LoaderOptions
    assert descriptor.default_loader_options == Argoverse1Loader.default_loader_options()
    assert descriptor.capabilities & DatasetCapabilities.LOADER_OPTIONS


def test_build_loader_uses_config() -> None:
    """Plan-local loader construction should carry resolved runtime settings."""
    plan_obj = plan_dataset(
        **_a43_args(),
        trajectory_schema="positions_only",
        split="time",
        ratio=(0.7, 0.2, 0.1),
    )
    loader = _build_loader(plan_obj)

    assert loader.loader_config == plan_obj.config.loader
    assert loader.loader_options == plan_obj.config.loader_options
    assert loader.map_config == plan_obj.config.map
    assert loader.requested_trajectory_schema == plan_obj.config.export.trajectory_schema
    assert loader.splits is None
    assert loader.split_request is not None
    assert loader.split_request.strategy_name == "time"


def test_build_executor_uses_plan_execution_mode() -> None:
    """Executor construction should follow the resolved execution config."""
    sequential_plan = plan_dataset(**_a43_args(), jobs=1)
    parallel_plan = plan_dataset(**_a43_args(), jobs=2)

    assert isinstance(
        _build_executor(sequential_plan, _build_loader(sequential_plan)),
        SequentialExecutor,
    )
    assert isinstance(
        _build_executor(parallel_plan, _build_loader(parallel_plan)), ParallelExecutor
    )


def test_build_writer_factory_uses_plan_storage_backend() -> None:
    """Writer-factory construction should follow the resolved storage backend."""
    plan_obj = plan_dataset(**_a43_args())
    writer_factory = _build_writer_factory(plan_obj)
    writer = writer_factory(0)

    assert callable(writer_factory)
    assert type(writer).__name__ == "NullWriter"


def test_plan_dataset_rejects_unknown_storage_backend() -> None:
    """Planning should surface unsupported storage backends as a typed domain error."""
    args = {**_a43_args(), "storage_backend": "bogus"}
    with pytest.raises(
        UnsupportedStorageBackendError, match=r"Unsupported storage backend 'bogus'"
    ):
        _ = plan_dataset(**args)


def test_plan_open_returns_live_run() -> None:
    """Prepared plans should open into live runs with observable executors."""
    plan_obj = plan_dataset(**_a43_args())

    with plan_obj.open() as run:
        assert isinstance(run, DatasetRun)
        assert run.plan is plan_obj
        assert callable(run.executor.progress)
        assert callable(run.executor.progress_event)


def test_plan_run() -> None:
    """Prepared plans should be executable directly without helper wrappers."""
    plan_obj = plan_dataset(**_a43_args())
    plan_obj.run()


def test_waymo_splits() -> None:
    """Waymo should support both native and custom split assignment."""
    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "storage_backend": StorageBackend.NULL,
    }
    plan_obj = plan_dataset(**common_args, split=None)

    assert sorted(plan_obj.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert isinstance(plan_obj.split_request.strategy, NoSplitStrategy)
    assert plan_obj.loader_splits() is None
    assert plan_obj.output_splits() is None

    plan_obj = plan_dataset(**common_args, split="native", read_split=["train", "val"])

    assert sorted(plan_obj.descriptor.predefined_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])
    assert plan_obj.loader_splits() == (DatasetSplit.TRAIN, DatasetSplit.VAL)
    assert plan_obj.output_splits() == (DatasetSplit.TRAIN, DatasetSplit.VAL)
    assert isinstance(plan_obj.split_request.strategy, NativeSplitStrategy)

    with pytest.raises(ValueError, match=r"--read-split is only valid with --split native"):
        _ = plan_dataset(**common_args, read_split=["train"])

    plan_obj = plan_dataset(**common_args, split="scene", ratio=(0.5, 0.4, 0.1))

    assert plan_obj.loader_splits() is None
    assert isinstance(plan_obj.split_request.strategy, SceneSplitStrategy)
    assert plan_obj.split_request.strategy_name == "scene"
    writer_splits = plan_obj.output_splits()
    assert writer_splits is not None
    assert sorted(writer_splits) == sorted([
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ])


def test_plan_dataset_split_config() -> None:
    """Runtime split options should be normalized into the resolved config model."""
    plan_obj = plan_dataset(
        **_a43_args(),
        split="shuffled-time",
        ratio=(0.6, 0.2, 0.2),
        gap=5,
        segments=8,
    )

    assert isinstance(plan_obj.config.split, SplitConfig)
    assert isinstance(plan_obj.config.split.strategy, ShuffledTimeBlockStrategy)
    assert plan_obj.config.split.strategy.gap == 5
    assert plan_obj.config.split.strategy.segments == 8
    assert plan_obj.config.split.ratio is not None
    assert plan_obj.config.split.ratio.values() == (0.6, 0.2, 0.2)


def test_plan_dataset_config_file_splits(tmp_path: Path) -> None:
    """Split config should load from TOML and allow CLI overrides on top."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[datasets.a43.split]
strategy = "time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
gap = 2
""",
        encoding="utf-8",
    )

    plan_obj = plan_dataset(**_a43_args(), config_path=config_path)

    assert isinstance(plan_obj.config.split.strategy, TimeBlockStrategy)
    assert plan_obj.config.split.strategy.gap == 2
    assert plan_obj.config.split.ratio is not None
    assert plan_obj.config.split.ratio.values() == (0.7, 0.2, 0.1)

    overridden = plan_dataset(
        **_a43_args(),
        config_path=config_path,
        split="shuffled-time",
        ratio=(0.6, 0.2, 0.2),
        gap=5,
        segments=8,
    )

    assert isinstance(overridden.config.split.strategy, ShuffledTimeBlockStrategy)
    assert overridden.config.split.strategy.gap == 5
    assert overridden.config.split.strategy.segments == 8
    assert overridden.config.split.ratio is not None
    assert overridden.config.split.ratio.values() == (0.6, 0.2, 0.2)


def test_plan_dataset_accepts_real_sample_config() -> None:
    """The checked-in sample config should be a valid end-to-end planning example."""
    config_path = Path(__file__).resolve().parents[2] / "config.toml"

    plan_obj = plan_dataset(**_a43_args(), config_path=config_path)

    assert plan_obj.config.loader.input_len == 20
    assert plan_obj.config.loader.output_len == 60
    assert plan_obj.config.loader.window is not None
    assert plan_obj.config.loader.window.size == 80
    assert isinstance(plan_obj.config.split.strategy, ShuffledTimeBlockStrategy)


def test_plan_dataset_rejects_loader_options_for_datasets_without_extra_args(
    tmp_path: Path,
) -> None:
    """Datasets without extra loader args should reject free-form loader options."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[datasets.a43.loader.options]
custom_flag = "debug"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigurationError, match=r"a43 does not expose dataset-specific loader options"
    ):
        _ = plan_dataset(**_a43_args(), config_path=config_path)


def test_plan_dataset_resolves_typed_loader_options(tmp_path: Path) -> None:
    """Datasets with loader options should resolve them into the typed runtime config."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[datasets.argoverse1.loader.options]
file_batch_size = 7
""",
        encoding="utf-8",
    )

    plan_obj = plan_dataset(
        dataset="argoverse1",
        input_dir=tmp_path,
        output_dir=tmp_path / "output",
        storage_backend=StorageBackend.NULL,
        input_dir_exists=False,
        config_path=config_path,
    )

    assert plan_obj.config.loader_options == Argoverse1LoaderOptions(file_batch_size=7)


@pytest.mark.parametrize(
    ("model_type", "field_name"),
    [
        (Argoverse1LoaderOptions, "file_batch_size"),
        (Argoverse2LoaderOptions, "file_batch_size"),
        (InteractionLoaderOptions, "file_batch_size"),
        (LyftLoaderOptions, "scene_batch_size"),
    ],
)
def test_batch_size_loader_options_reject_none(
    model_type: type[BaseModel], field_name: str
) -> None:
    """Batch-size loader options should require positive integers."""
    with pytest.raises(ValidationError):
        _ = model_type.model_validate({field_name: None})


def test_plan_dataset_rejects_highway_config_for_unsupported_datasets(tmp_path: Path) -> None:
    """Datasets without highway support should fail during planning, not execution."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[datasets.waymo.loader.lane_change_sampling]
persist = 2
negative_keep_every = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=r"waymo does not support lane-change sampling"):
        _ = plan_dataset(
            dataset="waymo",
            input_dir=Path(__file__).parent,
            output_dir=Path(__file__).parent / "output",
            storage_backend=StorageBackend.NULL,
            config_path=config_path,
        )


def test_plan_dataset_clears_splits(tmp_path: Path) -> None:
    """An explicit CLI no-split request should override split config from file."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """[datasets.waymo.split]
strategy = "scene"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
""",
        encoding="utf-8",
    )

    common_args: _CommonArgs = {
        "dataset": "waymo",
        "input_dir": Path(__file__).parent,
        "output_dir": Path(__file__).parent / "output",
        "storage_backend": StorageBackend.NULL,
    }
    plan_obj = plan_dataset(**common_args, config_path=config_path, split="none")

    assert isinstance(plan_obj.split_request.strategy, NoSplitStrategy)
    assert plan_obj.loader_splits() is None
    assert plan_obj.output_splits() is None


def test_plan_summary_exposes_sections_and_rows() -> None:
    """Processing summaries should be structured but still flatten to rows."""
    plan_obj = plan_dataset(
        **_a43_args(),
        split="shuffled-time",
        ratio=(0.6, 0.2, 0.2),
        gap=5,
        segments=8,
        include_map=False,
    )

    summary = summarize_plan(plan_obj)

    assert summary.title == "Processing Plan"
    assert tuple(section.title for section in summary.sections) == (
        "Overview",
        "Transformations",
        "Execution",
        "Splits",
    )
    assert ("Map", "disabled") in summary.rows
    assert ("Split strategy", "shuffled-time") in summary.rows
    assert ("Time split settings", "segments=8, gap=5 frames") in summary.rows
