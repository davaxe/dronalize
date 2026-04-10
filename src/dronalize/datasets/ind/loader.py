"""Loader implementation for the inD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.shared.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


class InDLoader(LevelXDataLoader):
    """Loader for the inD dataset."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the inD loader."""
        super().__init__(data_root=Path(data_root) / "data", request=request, resources=resources)
