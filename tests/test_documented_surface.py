from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

from dronalize import runtime
from dronalize.config import parse_config
from dronalize.io import StorageBackend
from dronalize.runtime import (
    ExecutionRequest,
    OutputSample,
    execute_plan,
    execute_request,
    resolve_request,
)

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCED_BLOCK = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)\n```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
SKIP_VALIDATE_MARKER = "<!-- no-validate -->"
PYTHON_INFO_STRINGS = {"py", "python"}


def test_documented_runtime_imports_match_api() -> None:
    assert ExecutionRequest is runtime.ExecutionRequest
    assert OutputSample is runtime.OutputSample
    assert resolve_request is runtime.resolve_request
    assert execute_request is runtime.execute_request
    assert execute_plan is runtime.execute_plan
    assert not hasattr(runtime, "ProcessRequest")
    assert not hasattr(runtime, "process_dataset")
    assert not hasattr(runtime, "resolve_job")
    assert ExecutionRequest.model_fields["storage_backend"].default == StorageBackend.PICKLE


def _documented_code_params(language: str | set[str]) -> list[tuple[str, str, str]]:
    languages = {language} if isinstance(language, str) else language
    params: list[tuple[str, str, str]] = []
    for doc_path in sorted(DOCS_ROOT.rglob("*.md")):
        relative_path = doc_path.relative_to(DOCS_ROOT).as_posix()
        text = doc_path.read_text(encoding="utf-8")
        for block_index, match in enumerate(FENCED_BLOCK.finditer(text), start=1):
            info = match.group("info").strip().split()
            prefix = text[: match.start()].rstrip()
            marker_line = prefix.rsplit("\n", maxsplit=1)[-1] if prefix else ""
            if not languages.intersection(info) or marker_line == SKIP_VALIDATE_MARKER:
                continue
            block_id = f"{relative_path}:block-{block_index}"
            params.append((block_id, relative_path, match.group("body").strip()))
    return params


def _documented_config_params() -> list[tuple[str, str, str]]:
    params = _documented_code_params("toml")
    assert params, "No TOML documentation examples found"
    return params


def _documented_python_params() -> list[tuple[str, str, str]]:
    params = _documented_code_params(PYTHON_INFO_STRINGS)
    assert params, "No Python documentation examples found"
    return params


@pytest.mark.parametrize(
    ("block_id", "relative_path", "snippet"), _documented_config_params(), ids=lambda value: value
)
def test_documented_config_examples_validate(
    tmp_path: Path, block_id: str, relative_path: str, snippet: str
) -> None:
    """Test that TOML code blocks in the docs can be parsed as valid config."""
    _ = block_id
    _ = relative_path
    config_path = tmp_path / "example.toml"
    _ = config_path.write_text(snippet + "\n", encoding="utf-8")
    _ = parse_config(config_path)


@pytest.mark.parametrize(
    ("block_id", "relative_path", "snippet"), _documented_python_params(), ids=lambda value: value
)
def test_documented_python_examples_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_id: str, relative_path: str, snippet: str
) -> None:
    """Test that Python code blocks in the docs run without error."""
    _ = relative_path
    example_path = tmp_path / f"{block_id.replace('/', '__').replace(':', '_')}.py"
    _ = example_path.write_text(snippet + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _ = runpy.run_path(str(example_path), run_name="__main__")


def test_config_file_parses() -> None:
    path = Path("dronalize.toml")
    _ = parse_config(path)


def test_markdown_links_resolve() -> None:
    failures: list[str] = []
    for doc_path in sorted(DOCS_ROOT.rglob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group("target").strip()
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", maxsplit=1)[0].strip()
            if not target:
                continue
            if not target.endswith(".md"):
                continue
            linked_path = (doc_path.parent / target).resolve()
            if not linked_path.is_file():
                source = doc_path.relative_to(DOCS_ROOT).as_posix()
                failures.append(f"{source}: missing link target {target}")

    assert not failures, "\n".join(failures)
