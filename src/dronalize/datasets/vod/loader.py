"""Loader implementation for the View-of-Delft dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


class VodLoader(NuScenesLoader):
    """Loader for the View-of-Delft dataset."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the VOD loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)

    @override
    @classmethod
    def default_loader_options(cls) -> NuScenesLoaderOptions:
        # vod contains dulplicate ego vehicle data
        return NuScenesLoaderOptions(drop_full_category_regex=["vehicle.ego"])
