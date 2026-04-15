from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dronalize.config import ProcessingConfig, RuntimeOverride, load_project_config
from dronalize.config.models import DatasetConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, body: str) -> Path:
    config_path = path / "config.toml"
    _ = config_path.write_text(body.strip() + "\n", encoding="utf-8")
    return config_path


def test_load_project_config_parses_profiles_and_dataset_entries(tmp_path: Path) -> None:
    cfg = load_project_config(
        _write(
            tmp_path,
            """
            [profiles.fast.runtime]
            jobs = 2

            [datasets.demo]
            uses = ["fast"]
            """,
        )
    )

    assert isinstance(cfg, ProcessingConfig)
    assert "fast" in cfg.profiles
    assert "demo" in cfg.datasets


def test_resolve_raises_for_missing_profile(tmp_path: Path) -> None:
    cfg = load_project_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            uses = ["missing"]
            """,
        )
    )

    with pytest.raises(ValueError, match="Profile 'missing' not found"):
        _ = cfg.resolve(
            "demo",
            DatasetConfig.model_validate({
                "scenes": {
                    "history_frames": 1,
                    "future_frames": 1,
                    "sample_time": 0.1,
                }
            }),
        )


def test_load_project_config_surfaces_invalid_toml(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        [datasets.demo
        uses = ["fast"]
        """,
    )

    with pytest.raises(ValueError, match=r".+"):
        _ = load_project_config(path)


def test_runtime_override_from_inputs_only_sets_provided_sections() -> None:
    override = RuntimeOverride.from_inputs(
        split_strategy="scene",
        read_split=None,
        jobs=3,
        trajectory_schema="canonical",
        ratio=(0.7, 0.2, 0.1),
        gap=None,
        segments=None,
    )

    assert override.runtime is not None
    assert override.runtime.jobs == 3
    assert override.output is not None
    assert override.output.trajectory_schema == "canonical"
    assert override.split is not None

    empty = RuntimeOverride.from_inputs(
        split_strategy=None,
        read_split=None,
        jobs=None,
        trajectory_schema=None,
        ratio=None,
        gap=None,
        segments=None,
    )
    assert empty.runtime is None
    assert empty.output is None
    assert empty.split is None
