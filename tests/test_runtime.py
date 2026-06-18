from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import pytest

from dronalize.config import RuntimeOverride
from dronalize.core.errors import (
    CliError,
    ConfigurationError,
    DatasetNotFoundError,
    UnsupportedStorageBackendError,
)
from dronalize.datasets import (
    DatasetFeatureSupport,
    DatasetTemporalSupport,
    DatasetWindowingSupport,
    FrameBounds,
    list_datasets,
)
from dronalize.datasets.registry import (
    _REGISTRY,  # pyright: ignore[reportPrivateUsage]
    dataset_names_by_id,
)
from dronalize.io import StorageBackend, read_manifest
from dronalize.io.backends.null import NullWriter
from dronalize.io.base import WorkerWriterProvider
from dronalize.io.readers import PickleReader
from dronalize.runtime import ExecutionRequest, OutputTransform, execute_request, resolve_request
from dronalize.runtime.executor import open_executor
from dronalize.runtime.processor import RuntimeProcessor
from tests.support import (
    DemoOptions,
    cleanup_demo_descriptor,
    demo_descriptor,
    stale_kinematics_demo_descriptor,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.scene import Scene
    from dronalize.datasets import DatasetDescriptor
    from dronalize.io.records import SceneRecord


def _request(tmp_path: Path, **kwargs: object) -> ExecutionRequest:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base: dict[str, object] = {
        "dataset": "demo",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "storage_backend": StorageBackend.NULL,
    }
    base.update(kwargs)
    return ExecutionRequest.model_validate(base)


def _cli_app_and_runner() -> tuple[Any, Any]:
    pytest.importorskip("typer")
    pytest.importorskip("rich")

    from typer.testing import CliRunner  # noqa: PLC0415

    import dronalize.runtime.cli.app as cli_app  # noqa: PLC0415

    return cli_app.app, CliRunner()


def _patch_descriptor(monkeypatch: pytest.MonkeyPatch, descriptor: DatasetDescriptor) -> None:
    def provider(_name: str) -> DatasetDescriptor:
        return descriptor

    monkeypatch.setattr("dronalize.runtime.api.get_dataset", provider)


def _patch_get_demo_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_descriptor(monkeypatch, demo_descriptor())


def _create_null_writer(_worker_id: int) -> NullWriter:
    return NullWriter()


class FailingWriter:
    def write(self, scene: Scene) -> None:  # noqa: PLR6301
        _ = scene
        msg = "intentional writer failure"
        raise RuntimeError(msg)

    def finish_local(self) -> None:  # noqa: PLR6301
        return


def _create_failing_writer(_worker_id: int) -> FailingWriter:
    return FailingWriter()


def test_resolve_request_builds_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    plan = resolve_request(_request(tmp_path, include_map=False))

    assert plan.dataset == "demo"
    assert plan.storage_backend == StorageBackend.NULL
    dataset_options = cast("DemoOptions", plan.loader.loader_options)
    assert dataset_options.batch_size == 2
    assert plan.map is None
    assert plan.effective_horizon_frames == 3
    assert plan.effective_default_observation_length == 2


def test_resolve_request_rejects_unknown_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    with pytest.raises(UnsupportedStorageBackendError, match="Unsupported storage backend"):
        _ = resolve_request(_request(tmp_path, storage_backend="bad-backend"))


def test_resolve_request_rejects_lane_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    config_path = tmp_path / "dronalize.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support lane-change sampling"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_resolve_request_requires_window_for_lane_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "dronalize.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes]
window = { op = "clear" }

[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires window sampling"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_resolve_request_rejects_long_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=2, confidence="documented"),
            windowing=DatasetWindowingSupport(enabled_by_default=True),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)

    with pytest.raises(ConfigurationError, match="supports windows up to 2 source frames"):
        _ = resolve_request(_request(tmp_path))


def test_resampling_recomputes_native_kinematics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = stale_kinematics_demo_descriptor()
    _patch_descriptor(monkeypatch, descriptor)
    plan = resolve_request(_request(tmp_path, dataset=descriptor.name))
    loader = descriptor.build_loader(root=plan.data_root, request=plan.loader)
    processor = RuntimeProcessor.from_plan(plan, loader)
    source = next(iter(processor.iter_sources()))
    candidate = next(iter(processor.iter_candidates(source)))

    scene = processor.materialize(candidate, scene_number=0)

    assert scene.horizon_frames == 2
    assert scene.frame["frame"].to_list() == [0, 1]
    assert scene.frame["vx"].to_list() == pytest.approx([1.0, 1.0])
    assert scene.frame["vy"].to_list() == pytest.approx([0.0, 0.0])
    assert scene.frame["ax"].to_list() == pytest.approx([0.0, 0.0])
    assert set(plan.manifest().derived_features) == {"vx", "vy", "ax", "ay", "yaw"}


def test_resolve_request_rejects_window_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=10, confidence="documented"),
            windowing=DatasetWindowingSupport(
                enabled_by_default=True, supported_policies=("strict",)
            ),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.window]
step = 1
policy = "partial"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support window policy 'partial'"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_execute_request_surfaces_unknown_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path, dataset="this-dataset-does-not-exist")

    with pytest.raises(DatasetNotFoundError):
        _ = execute_request(request)


def test_execute_request_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    request = _request(tmp_path)
    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.storage_backend == StorageBackend.NULL
    assert result.stats.processed_sources == 1

    manifest = read_manifest(result.output_dir)
    assert manifest.storage_backend == "null"
    assert manifest.dataset_names == ("demo",)
    assert manifest.source_trajectory_schema_fields == (
        "frame",
        "id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "yaw",
        "agent_category",
    )
    assert manifest.horizon_frames == 3
    assert manifest.default_observation_length == 2


def test_execute_request_rejects_non_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    request = _request(tmp_path)
    request.output_dir.mkdir()
    marker = request.output_dir / "old-data"
    _ = marker.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _ = execute_request(request, show_progress=False)

    assert marker.exists()


def test_execute_request_overwrites_output_when_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    request = _request(tmp_path, overwrite=True)
    request.output_dir.mkdir()
    marker = request.output_dir / "old-data"
    _ = marker.write_text("stale", encoding="utf-8")

    result = execute_request(request, show_progress=False)

    assert not marker.exists()
    assert read_manifest(result.output_dir).dataset == "demo"


def test_builtin_manifest_uses_global_dataset_name_table(tmp_path: Path) -> None:
    request = ExecutionRequest(
        dataset="a43",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        storage_backend=StorageBackend.NULL,
        input_dir_exists=False,
    )
    plan = resolve_request(request)

    assert plan.manifest().dataset_names == dataset_names_by_id()


def test_execute_request_applies_record_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset_id": record.dataset_id,
            "feature_shape": record.features.shape,
        }

    output_transform = OutputTransform(record_transform=transform)
    request = _request(
        tmp_path, storage_backend=StorageBackend.PICKLE, output_transform=output_transform
    )

    result = execute_request(request)
    record = cast("dict[str, object]", PickleReader(result.output_dir, record_type=dict)[0])

    assert record == {"scene_number": 0, "dataset_id": None, "feature_shape": (1, 3, 7)}


def test_execute_request_writes_custom_mds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "streaming", reason="Requires streaming package for custom MDS output record format"
    )
    from dronalize.io.readers import MDSReader  # noqa: PLC0415

    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset_id": -1 if record.dataset_id is None else record.dataset_id,
            "feature_shape": record.features.shape,
        }

    output_transform = OutputTransform(
        record_transform=transform,
        mds_columns={"scene_number": "int", "dataset_id": "int", "feature_shape": "json"},
    )
    request = _request(
        tmp_path, storage_backend=StorageBackend.MDS, output_transform=output_transform
    )
    result = execute_request(request)
    reader = MDSReader(path=result.output_dir, convert_raw=dict)
    record = reader[0]
    assert record == {"scene_number": 0, "dataset_id": -1, "feature_shape": [1, 3, 7]}


def test_parallel_execution_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    _patch_get_demo_descriptor(monkeypatch)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=2),
    )

    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.stats.processed_sources == 1
    assert result.stats.candidate_scenes == 1
    assert result.stats.written_scenes == 1
    assert result.stats.split_counts["unsplit"] == 1

    manifest = read_manifest(output_dir)
    assert manifest.horizon_frames == 3
    assert manifest.default_observation_length == 2
    assert manifest.dataset_names == ("demo",)


def test_execute_request_reports_cleanup_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    request = _request(tmp_path, dataset="cleanup-demo")
    result = execute_request(request)

    assert result.cleanup_summary is not None
    assert result.cleanup_summary.overall.scene_count == 1
    assert result.cleanup_summary.overall.total_rows_removed == 3
    assert result.cleanup_summary.overall.total_agents_removed == 1
    assert result.cleanup_summary.overall.average_rows_removed_per_scene == pytest.approx(3.0)
    assert result.cleanup_summary.overall.min_rows_removed_per_scene == 3
    assert result.cleanup_summary.overall.max_rows_removed_per_scene == 3
    assert "trim_unimportant" in result.cleanup_summary.by_rule
    rule_summary = result.cleanup_summary.by_rule["trim_unimportant"]
    assert rule_summary.total_rows_removed == 3
    assert rule_summary.total_agents_removed == 1


def test_parallel_execution_reports_cleanup_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="cleanup-demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=2),
    )

    result = execute_request(request)

    assert result.cleanup_summary is not None
    assert result.cleanup_summary.overall.total_rows_removed == 3
    assert result.cleanup_summary.overall.total_agents_removed == 1
    assert result.cleanup_summary.by_rule["trim_unimportant"].scene_count == 1


@pytest.mark.parametrize("jobs", [None, 2], ids=["sequential", "parallel"])
def test_execution_progress_reports_cleanup_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int | None
) -> None:
    _patch_descriptor(monkeypatch, cleanup_demo_descriptor())

    request_kwargs: dict[str, object] = {"dataset": "cleanup-demo"}
    if jobs is not None:
        request_kwargs["overrides"] = RuntimeOverride.from_inputs(jobs=jobs)

    plan = resolve_request(_request(tmp_path, **request_kwargs))
    writer_provider = WorkerWriterProvider(_create_null_writer)

    with open_executor(plan) as executor:
        progress = executor.execute(writer_provider)

    assert progress.cleanup.rows_total == 6
    assert progress.cleanup.rows_removed == 3
    assert progress.cleanup.agents_total == 2
    assert progress.cleanup.agents_removed == 1


def test_failed_writer_is_not_counted_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    plan = resolve_request(_request(tmp_path))
    writer_provider = WorkerWriterProvider(_create_failing_writer)

    with open_executor(plan) as executor:
        with pytest.raises(RuntimeError, match="intentional writer failure"):
            _ = executor.execute(writer_provider)
        progress = executor.snapshot()

    assert progress.stats.written_scenes == 0
    assert progress.stats.split_counts["unsplit"] == 0


@pytest.mark.parametrize(
    "name",
    ["process", "available", "inspect", "show-config", "split-support"],
    ids=["process", "available", "inspect", "show-config", "split-support"],
)
def test_cli_commands_smoke(tmp_path: Path, name: str) -> None:
    app, runner = _cli_app_and_runner()
    for dataset_name in list_datasets():
        output_dir = tmp_path / "cli-output"
        args_by_command: dict[str, list[str]] = {
            "process": [
                "process",
                dataset_name,
                "--input",
                ".",
                "--output",
                str(output_dir),
                "--plan",
            ],
            "available": ["available", "--no-details"],
            "inspect": ["inspect", dataset_name],
            "show-config": ["show-config", dataset_name],
            "split-support": ["split-support", dataset_name],
        }
        args = args_by_command[name]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{name} failed: {result.output}"


def test_cli_help_smoke() -> None:
    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0


def test_inspect_reports_temporal_support() -> None:
    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["inspect", "argoverse1"])

    assert result.exit_code == 0, result.output
    assert "Configured horizon" in result.output
    assert "Sliding windows" in result.output
    assert "Source bounds" in result.output
    assert "Configured horizon fits" in result.output
    assert "Supported policies" in result.output


def test_cli_imports_dataset_module_before_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = tmp_path / "custom_datasets.py"
    _ = module_path.write_text(
        (
            "from dataclasses import replace\n"
            "from tests.support import demo_descriptor\n"
            "\n"
            "\n"
            "def register_dronalize_datasets():\n"
            '    return replace(demo_descriptor(), name="cli_demo")\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _ = _REGISTRY.pop("cli_demo", None)

    try:
        app, runner = _cli_app_and_runner()
        available_result = runner.invoke(
            app, ["--dataset-module", "custom_datasets", "available", "--no-details"]
        )
        inspect_result = runner.invoke(
            app, ["--dataset-module", "custom_datasets", "inspect", "cli_demo"]
        )
    finally:
        _ = _REGISTRY.pop("cli_demo", None)

    assert available_result.exit_code == 0, available_result.output
    assert "cli_demo" in available_result.output
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "cli_demo" in inspect_result.output


@pytest.mark.parametrize(
    ("module_body", "expected"),
    [
        ("register_dronalize_datasets = 1\n", "non-callable"),
        ("def register_dronalize_datasets():\n    return 1\n", "unsupported value"),
        ("def register_dronalize_datasets():\n    return [1]\n", "returned int"),
    ],
)
def test_cli_rejects_invalid_dataset_module_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_body: str, expected: str
) -> None:
    module_path = tmp_path / "bad_datasets.py"
    _ = module_path.write_text(module_body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    _ = sys.modules.pop("bad_datasets", None)

    app, runner = _cli_app_and_runner()
    result = runner.invoke(app, ["--dataset-module", "bad_datasets", "available", "--no-details"])

    assert result.exit_code != 0
    message = result.output or str(result.exception)
    assert isinstance(result.exception, CliError)
    assert expected in message
