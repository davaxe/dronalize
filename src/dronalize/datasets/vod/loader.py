"""Loader implementation for the View-of-Delft dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dronalize.core.categories import DatasetSplit
from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions
from dronalize.datasets.vod.splits import get_split
from dronalize.processing.loading.loader import Source

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest

VodLoaderOptions = NuScenesLoaderOptions

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class VodLoader(NuScenesLoader):
    """Loader for the View-of-Delft dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, resources=resources)
        self._data_dirs: list[Path] = [self.root / "v1.0-trainval", self.root / "v1.0-test"]

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> VodLoader:
        return cls(data_root=data_root, request=request, resources=resources)

    @override
    def _build_sources_manifest(self) -> dict[DatasetSplit, list[Source[str]]]:
        sources: dict[DatasetSplit, list[Source[str]]] = {split: [] for split in _NATIVE_SPLITS}

        log_records: list[dict[str, Any]] = []
        scene_records: list[dict[str, Any]] = []
        for data_dir in self._data_dirs:
            with (data_dir / "log.json").open("r", encoding="utf-8") as f:
                log_records.extend(json.load(f))
            with (data_dir / "scene.json").open("r", encoding="utf-8") as f:
                scene_records.extend(json.load(f))

        log_to_map = {
            str(row["token"]): str(row["location"]) if row.get("location") else None
            for row in log_records
            if "token" in row
        }

        for row in scene_records:
            if "token" not in row or "log_token" not in row or "name" not in row:
                continue
            token = str(row["token"])
            sources[get_split(token)].append(
                Source(identifier=token, data=token, map_key=log_to_map.get(str(row["log_token"])))
            )
        return sources
