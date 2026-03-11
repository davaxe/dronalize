from __future__ import annotations

import sys

from dronalize.datasets import available
from dronalize.datasets.registry import _REGISTRY, get  # noqa: PLC2701


def test_available_lists_builtins_without_importing_modules() -> None:
    """Built-in datasets should be discoverable without eager module import."""
    _REGISTRY.pop("a43", None)
    sys.modules.pop("dronalize.datasets.a43", None)

    dataset_names = available()

    assert "a43" in dataset_names
    assert "dronalize.datasets.a43" not in sys.modules


def test_get_lazily_imports_dataset_module() -> None:
    """A built-in dataset module should only be imported when explicitly resolved."""
    _REGISTRY.pop("a43", None)
    sys.modules.pop("dronalize.datasets.a43", None)

    descriptor = get("a43")

    assert descriptor.name == "a43"
    assert "dronalize.datasets.a43" in sys.modules


def test_get_supports_multiple_dataset_names_from_one_manifest() -> None:
    """One manifest should be able to expose multiple registered dataset names."""
    for dataset_name in ("eth", "hotel", "zara1"):
        _REGISTRY.pop(dataset_name, None)
    sys.modules.pop("dronalize.datasets.eth_ucy", None)

    descriptor = get("hotel")

    assert descriptor.name == "hotel"
    assert "dronalize.datasets.eth_ucy" in sys.modules
    assert get("eth").name == "eth"
    assert get("zara1").name == "zara1"


def test_available_filters_missing_optional_dependencies(monkeypatch) -> None:
    """Datasets with missing optional imports should not appear as available."""
    _REGISTRY.pop("lyft", None)

    def _mock_has_module(module_name: str) -> bool:
        return module_name not in {"zarr.creation", "numcodecs"}

    monkeypatch.setattr("dronalize.datasets.registry._has_module", _mock_has_module)

    dataset_names = available()

    assert "lyft" not in dataset_names
