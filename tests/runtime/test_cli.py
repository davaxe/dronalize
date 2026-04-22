from __future__ import annotations

import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

from typing import TYPE_CHECKING

from typer.testing import CliRunner

import dronalize.runtime.cli.app as cli_app
from dronalize.datasets import available

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "name",
    ["process", "available", "inspect", "show-config", "split-support"],
    ids=["process", "available", "inspect", "show-config", "split-support"],
)
def test_cli_commands_smoke(tmp_path: Path, name: str) -> None:
    runner = CliRunner()
    for dataset_name in available():
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
        result = runner.invoke(cli_app.app, args)

        assert result.exit_code == 0, f"{name} failed: {result.output}"


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_app.app, ["--help"])

    assert result.exit_code == 0
