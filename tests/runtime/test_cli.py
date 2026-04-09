from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dronalize.core.categories import DatasetSplit
from dronalize.runtime.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


runner = CliRunner()


@dataclass
class _DummyPlan:
    config: str = "dummy-config"


def test_show_config_passes_custom_split_strategy_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI parsing should forward custom split-strategy options to planning intact."""
    captured: dict[str, object] = {}

    def _fake_plan_dataset(**kwargs: object) -> _DummyPlan:
        captured.update(kwargs)
        return _DummyPlan()

    monkeypatch.setattr("dronalize.runtime.plan_dataset", _fake_plan_dataset)

    result = runner.invoke(
        app,
        [
            "show-config",
            "a43",
            "--split",
            "shuffled-time",
            "--ratio",
            "0.8",
            "0.1",
            "0.1",
            "--gap",
            "5",
            "--segments",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert captured["split"] == "shuffled-time"
    assert captured["read_split"] is None
    assert captured["ratio"] == (0.8, 0.1, 0.1)
    assert captured["gap"] == 5
    assert captured["segments"] == 8


def test_show_config_passes_native_split_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI parsing should forward native read-split options to planning intact."""
    captured: dict[str, object] = {}

    def _fake_plan_dataset(**kwargs: object) -> _DummyPlan:
        captured.update(kwargs)
        return _DummyPlan()

    monkeypatch.setattr("dronalize.runtime.plan_dataset", _fake_plan_dataset)

    result = runner.invoke(
        app,
        [
            "show-config",
            "waymo",
            "--split",
            "native",
            "--read-split",
            "train",
            "--read-split",
            "val",
        ],
    )

    assert result.exit_code == 0
    assert captured["split"] == "native"
    assert captured["read_split"] == [DatasetSplit.TRAIN, DatasetSplit.VAL]
    assert captured["ratio"] is None


def test_show_config_rejects_read_split_without_native_mode() -> None:
    """CLI validation should reject --read-split unless native mode is active."""
    result = runner.invoke(app, ["show-config", "waymo", "--read-split", "train"])

    assert result.exit_code == 2
    assert "--read-split is only valid with --split native" in result.output
    assert "Traceback" not in result.output


def test_show_config_rejects_ratio_without_split_strategy() -> None:
    """CLI validation should reject ratios when no custom split strategy was selected."""
    result = runner.invoke(app, ["show-config", "a43", "--ratio", "0.8", "0.1", "0.1"])

    assert result.exit_code == 2
    assert "--ratio is only valid with custom split strategies" in result.output
    assert "Traceback" not in result.output


def test_process_reports_missing_input_without_traceback() -> None:
    """Missing input paths should render as clean CLI errors."""
    result = runner.invoke(
        app,
        ["process", "a43", "--input", "DOES_NOT_EXIST", "--output", "out", "--force"],
    )

    assert result.exit_code == 2
    assert "Input directory DOES_NOT_EXIST does not exist." in result.output
    assert "Traceback" not in result.output


def test_show_config_reports_invalid_config_without_traceback(tmp_path: Path) -> None:
    """Invalid config files should render a structured usage error instead of a traceback."""
    config_path = tmp_path / "bad.toml"
    _ = config_path.write_text(
        """[datasets.a43.split]
gap = 2
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-config", "a43", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "Invalid configuration:" in result.output
    assert "datasets.a43.split" in result.output
    assert "Traceback" not in result.output


def test_show_config_rejects_highway_config_for_unsupported_dataset_without_traceback(
    tmp_path: Path,
) -> None:
    """Unsupported highway config should fail during planning with a clean CLI error."""
    config_path = tmp_path / "bad.toml"
    _ = config_path.write_text(
        """[datasets.waymo.loader.lane_change_sampling]
persist = 2
negative_keep_every = 3
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["show-config", "waymo", "--config", str(config_path)])

    assert result.exit_code == 2
    assert "waymo does not support lane-change sampling" in result.output
    assert "Traceback" not in result.output
