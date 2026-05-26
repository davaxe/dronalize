from __future__ import annotations

import pytest

pytest.importorskip("typer")
pytest.importorskip("rich")

import sys
from typing import TYPE_CHECKING

from typer.testing import CliRunner

import dronalize.runtime.cli.app as cli_app
from dronalize.datasets import list_datasets
from dronalize.datasets.registry import _REGISTRY  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "name",
    ["process", "available", "inspect", "show-config", "split-support"],
    ids=["process", "available", "inspect", "show-config", "split-support"],
)
def test_cli_commands_smoke(tmp_path: Path, name: str) -> None:
    runner = CliRunner()
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
        result = runner.invoke(cli_app.app, args)

        assert result.exit_code == 0, f"{name} failed: {result.output}"


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_app.app, ["--help"])

    assert result.exit_code == 0


def test_inspect_reports_temporal_support() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_app.app, ["inspect", "a43"])

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
        runner = CliRunner()
        available_result = runner.invoke(
            cli_app.app, ["--dataset-module", "custom_datasets", "available", "--no-details"]
        )
        inspect_result = runner.invoke(
            cli_app.app, ["--dataset-module", "custom_datasets", "inspect", "cli_demo"]
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

    runner = CliRunner()
    result = runner.invoke(
        cli_app.app, ["--dataset-module", "bad_datasets", "available", "--no-details"]
    )

    assert result.exit_code != 0
    assert expected in result.output
