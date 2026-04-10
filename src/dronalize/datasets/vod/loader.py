"""Loader implementation for the View-of-Delft dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.nuscenes.loader import NuScenesLoader

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
        self._full_category_contains = ["vehicle.ego"]
