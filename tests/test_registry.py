# pyright: standard
# ruff: noqa: PLC2701 SLF001
from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets import _registry as registry

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_extract_builtin_metadata_supports_annotated_assignment(tmp_path: Path) -> None:
    """Read builtin dataset metadata from an annotated module-level assignment."""
    package_init = tmp_path / "__init__.py"
    package_init.write_text(
        (
            "__dronalize_builtin__: dict[str, object] = {\n"
            '    "datasets": ["demo"],\n'
            '    "optional_dependencies": ["demo.dep"],\n'
            '    "extra": "demo",\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    metadata = registry._extract_builtin_metadata(package_init)

    assert metadata.datasets == ["demo"]
    assert metadata.optional_dependencies == ["demo.dep"]
    assert metadata.extra == "demo"


def test_available_caches_optional_dependency_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated availability checks should not re-scan the same optional dependencies."""
    spec = registry._BuiltinDatasetSpec(
        module="dronalize.datasets.demo",
        optional_dependencies=("demo.dep",),
    )
    calls = 0

    registry._missing_optional_dependencies.cache_clear()
    registry._has_module.cache_clear()

    def fake_has_module(module_name: str) -> bool:
        nonlocal calls
        calls += 1
        assert module_name == "demo.dep"
        return False

    monkeypatch.setattr(registry, "_builtin_datasets", lambda: {"demo": spec})
    monkeypatch.setattr(registry, "_has_module", fake_has_module)
    monkeypatch.setattr(registry, "_REGISTRY", {})

    assert registry.available() == []
    assert registry.available() == []
    assert calls == 1
