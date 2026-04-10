from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import Field
from typer.testing import CliRunner
from typing_extensions import override

from dronalize.config.runtime import RuntimeOverride
from dronalize.config.sections import (
    DatasetConfig,
    MapConfig,
    OutputConfig,
    RuntimeConfig,
    ScenesConfig,
    SceneSplitConfig,
    SplitConfig,
    SplitWeights,
    WindowConfig,
)
from dronalize.core.categories import DatasetSplit
from dronalize.core.scene import CANONICAL
from dronalize.datasets.registry import DatasetSpec
from dronalize.io.manifest import read_manifest
from dronalize.processing.loading.base import BaseSceneLoader, DatasetOptionsModel
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.runtime import ProcessRequest
from dronalize.runtime._internal.runner import open_job
from dronalize.runtime.api import resolve_job, run_job
from dronalize.runtime.cli.app import app

if TYPE_CHECKING:
    import pytest

    from dronalize.processing.loading import DatasetResources
    from dronalize.processing.models import LoaderRequest


class DemoOptions(DatasetOptionsModel):
    batch_size: int = Field(default=2, gt=0)
    use_cache: bool = False


class DemoLoader(BaseSceneLoader[Path, DemoOptions]):
    @classmethod
    @override
    def loader_options_model(cls) -> type[DemoOptions]:
        return DemoOptions

    @classmethod
    @override
    def native_trajectory_schema(cls):
        return CANONICAL

    @override
    def discover_sources(self):
        yield Source(identifier="source-1", data=self.root / "source.parquet")

    @override
    def load_source(self, source: Source[Path]):
        _ = source
        yield LoadedSourceData(
            pl.DataFrame({
                "frame": [0, 1, 2],
                "id": [1, 1, 1],
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
                "agent_category": [1, 1, 1],
            }).lazy()
        )

    @override
    def num_sources(self) -> int | None:
        return 1


def _descriptor() -> DatasetSpec:
    default_config = DatasetConfig(
        scenes=ScenesConfig(
            history_frames=4, future_frames=6, sample_time=0.5, window=WindowConfig(step=1)
        ),
        runtime=RuntimeConfig(jobs=1),
        dataset={"batch_size": 2, "use_cache": False},
        output=OutputConfig(),
        map=MapConfig(),
        split=SplitConfig(SceneSplitConfig(ratio=SplitWeights(train=0.7, val=0.2, test=0.1))),
    )
    return DatasetSpec(
        name="demo",
        loader_factory=DemoLoader.unified_factory,
        default_config=default_config,
        native_schema=CANONICAL,
        native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
        dataset_options_model=DemoOptions,
        has_map=True,
    )


def _write_config(path: Path) -> Path:
    config_path = path / "config.toml"
    _ = config_path.write_text(
        """
        [profiles.fast.runtime]
        jobs = 2

        [datasets.demo]
        uses = ["fast"]

        [datasets.demo.scenes]
        future_frames = 7

        [datasets.demo.dataset]
        batch_size = 8
        use_cache = true
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_build_job_resolves_config_and_runtime_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _descriptor()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    monkeypatch.setattr("dronalize.runtime.api.get", lambda _=descriptor: descriptor)

    job = resolve_job(
        ProcessRequest(
            dataset="demo",
            input_dir=input_dir,
            output_dir=output_dir,
            storage_backend="null",
            config_path=_write_config(tmp_path),
            overrides=RuntimeOverride.model_validate({
                "runtime": {"jobs": "auto"},
                "split": {"strategy": "native", "splits": (DatasetSplit.TRAIN,)},
            }),
            include_map=False,
        )
    )

    assert job.loader.scenes.future_frames == 7
    assert job.runtime.jobs == "auto"
    assert isinstance(job.loader.dataset, DemoOptions)
    assert job.loader.dataset.batch_size == 8
    assert job.loader.dataset.use_cache is True
    assert job.loader.split is not None
    assert job.loader.split.strategy == "native"
    assert job.loader.split.read == (DatasetSplit.TRAIN,)
    assert job.map is None


def test_open_job_builds_loader_from_loader_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    seen: dict[str, Any] = {}

    def _build_loader(
        data_root: Path | str, request: LoaderRequest, resources: DatasetResources | None = None
    ) -> BaseSceneLoader[Any, Any]:
        seen["root"] = data_root
        seen["request"] = request
        seen["resources"] = resources
        return DemoLoader(data_root=data_root, request=request, resources=resources)

    descriptor = DatasetSpec(
        name="demo",
        loader_factory=_build_loader,
        default_config=_descriptor().default_config,
        native_schema=CANONICAL,
        native_splits=(),
        dataset_options_model=DemoOptions,
        has_map=True,
    )
    monkeypatch.setattr("dronalize.runtime.api.get", lambda _=descriptor: descriptor)

    job = resolve_job(
        ProcessRequest(
            dataset="demo", input_dir=input_dir, output_dir=output_dir, storage_backend="null"
        )
    )

    with open_job(job) as run:
        assert seen["root"] == input_dir
        assert seen["request"] is job.loader
        assert run.job is job
        assert run.executor.progress().total_sources == 1


def test_cli_show_config_and_process_plan_use_run_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _descriptor()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    runner = CliRunner()

    monkeypatch.setattr("dronalize.runtime.api.get", lambda _=descriptor: descriptor)
    monkeypatch.setattr("dronalize.datasets.registry.get", lambda _=descriptor: descriptor)
    monkeypatch.setattr("dronalize.datasets.available", lambda _=descriptor: ["demo"])

    show_result = runner.invoke(
        app,
        [
            "show-config",
            "demo",
            "--config",
            str(_write_config(tmp_path)),
            "--scene-schema",
            "canonical",
        ],
    )
    assert show_result.exit_code == 0
    assert "Resolved Config" in show_result.stdout
    assert "future_frames=7" in show_result.stdout

    plan_result = runner.invoke(
        app,
        [
            "process",
            "demo",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--storage-backend",
            "null",
            "--plan",
            "--split",
            "native",
            "--read-split",
            "train",
        ],
    )
    assert plan_result.exit_code == 0
    assert "Processing plan: demo" in plan_result.stdout
    assert "Split strategy" in plan_result.stdout


def test_cli_registry_commands_render_from_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor()
    runner = CliRunner()

    monkeypatch.setattr("dronalize.datasets.registry.get", lambda _=descriptor: descriptor)
    monkeypatch.setattr("dronalize.datasets.available", lambda _=descriptor: ["demo"])

    available_result = runner.invoke(app, ["available"])
    assert available_result.exit_code == 0
    assert "Available datasets" in available_result.stdout
    assert "demo" in available_result.stdout

    inspect_result = runner.invoke(app, ["inspect", "demo"])
    assert inspect_result.exit_code == 0
    assert "Dataset inspect: demo" in inspect_result.stdout

    split_result = runner.invoke(app, ["split-support", "demo"])
    assert split_result.exit_code == 0
    assert "Split support: demo" in split_result.stdout


def test_run_job_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base = _descriptor().default_config
    descriptor = DatasetSpec(
        name="demo",
        loader_factory=DemoLoader.unified_factory,
        default_config=base,
        native_schema=CANONICAL,
        native_splits=(),
        dataset_options_model=DemoOptions,
        has_map=True,
    )
    monkeypatch.setattr("dronalize.runtime.api.get", lambda _=descriptor: descriptor)

    job = resolve_job(
        ProcessRequest(
            dataset="demo", input_dir=input_dir, output_dir=output_dir, storage_backend="null"
        )
    )

    _ = run_job(job)
    manifest = read_manifest(output_dir)
    assert manifest.history_frames == 4
    assert manifest.future_frames == 6
    assert manifest.has_map is True
