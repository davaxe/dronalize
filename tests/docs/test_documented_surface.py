from __future__ import annotations

import re
from pathlib import Path

import pytest

from dronalize import runtime
from dronalize.config import load_project_config
from dronalize.io import StorageBackend
from dronalize.runtime import (
    ExecutionRequest,
    execute_plan,
    execute_request,
    resolve_request,
    stream_plan,
)

DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
TOML_BLOCK = re.compile(r"```toml\n(.*?)\n```", re.DOTALL)
CONFIG_DOCS = (
    "reference/configuration/index.md",
    "reference/configuration/output.md",
    "reference/configuration/scenes.md",
    "reference/configuration/split.md",
    "reference/configuration/map.md",
    "formats/backends/mds.md",
)


def test_documented_runtime_imports_match_public_api() -> None:
    assert ExecutionRequest is runtime.ExecutionRequest
    assert resolve_request is runtime.resolve_request
    assert execute_request is runtime.execute_request
    assert execute_plan is runtime.execute_plan
    assert stream_plan is runtime.stream_plan
    assert not hasattr(runtime, "ProcessRequest")
    assert not hasattr(runtime, "process_dataset")
    assert not hasattr(runtime, "resolve_job")
    assert ExecutionRequest.model_fields["storage_backend"].default == StorageBackend.PICKLE


def _concrete_toml_blocks(doc_path: Path) -> list[str]:
    blocks: list[str] = []
    for match in TOML_BLOCK.findall(doc_path.read_text(encoding="utf-8")):
        snippet = match.strip()
        if "<" in snippet or "..." in snippet:
            continue
        blocks.append(snippet)
    return blocks


def _documented_config_params() -> list[str]:
    params: list[str] = []
    for relative_path in CONFIG_DOCS:
        doc_path = DOCS_ROOT / relative_path
        blocks = _concrete_toml_blocks(doc_path)
        assert blocks, f"No concrete TOML blocks found in {relative_path}"
        params.extend(blocks)
    return params


@pytest.mark.parametrize("snippet", _documented_config_params())
def test_documented_configuration_examples_validate(tmp_path: Path, snippet: str) -> None:
    config_path = tmp_path / "example.toml"
    _ = config_path.write_text(snippet + "\n", encoding="utf-8")
    _ = load_project_config(config_path)
