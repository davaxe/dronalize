"""Loader implementation for the View-of-Delft dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest

VodLoaderOptions = NuScenesLoaderOptions


class VodLoader(NuScenesLoader):
    """Loader for the View-of-Delft dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> VodLoader:
        return cls(data_root=data_root, request=request, resources=resources)
