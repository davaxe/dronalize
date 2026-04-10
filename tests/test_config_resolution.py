from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dronalize.config.project import ProjectConfig
from dronalize.config.reader import load_project_config
from dronalize.config.runtime import RuntimeOverride

if TYPE_CHECKING:
    from pathlib import Path


def _load(tmp_path: Path, body: str) -> ProjectConfig:
    path = tmp_path / "config.toml"
    _ = path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_project_config(path)


def test_load() -> None:
    """Load valid project config."""
    cfg = ProjectConfig.model_validate({
        "profiles": {
            "base": {
                "runtime": {"jobs": 2},
                "split": {"strategy": "scene"},
            }
        },
        "datasets": {
            "demo": {
                "uses": ["base"],
                "runtime": {"jobs": 1},
            }
        },
    })

    assert "base" in cfg.profiles
    assert "demo" in cfg.datasets


def test_merge_profiles(tmp_path: Path) -> None:
    """Merge used profiles in order."""
    cfg = _load(
        tmp_path,
        """
        [profiles.base.runtime]
        jobs = 2

        [profiles.extra.runtime]
        jobs = 4

        [datasets.demo]
        uses = ["base", "extra"]
        """,
    )

    assert cfg.datasets["demo"].uses == ("base", "extra")


def test_missing_profile(tmp_path: Path) -> None:
    """Reject unknown used profile."""
    cfg = _load(
        tmp_path,
        """
        [datasets.demo]
        uses = ["missing"]
        """,
    )

    from dronalize.config.sections.dataset import DatasetConfig

    with pytest.raises(ValueError, match="Profile 'missing' not found"):
        _ = cfg.resolve(
            "demo",
            DatasetConfig.model_validate({
                "scenes": {"history_frames": 1, "future_frames": 1, "sample_time": 0.1}
            }),
        )


def test_override() -> None:
    """Build runtime override from inputs."""
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
    assert type(override.split.root).__name__ == "SceneSplitConfig"


def test_override_empty() -> None:
    """Keep override empty when inputs missing."""
    override = RuntimeOverride.from_inputs(
        split_strategy=None,
        read_split=None,
        jobs=None,
        trajectory_schema=None,
        ratio=None,
        gap=None,
        segments=None,
    )

    assert override.runtime is None
    assert override.output is None
    assert override.split is None
