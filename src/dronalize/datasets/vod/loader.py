"""Loader implementation for the View-of-Delft dataset."""

from collections.abc import Iterable
from pathlib import Path

from typing_extensions import override

from dronalize.core.categories import DatasetSplit
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.splits import SplitConfig
from dronalize.processing.maps.config import MapConfig


class VodLoader(NuScenesLoader):
    """Loader for the View-of-Delft dataset.

    This shares the same base loading logic as `NuScenesLoader`, but with
    dataset-specific category filtering. For further details, see
    `NuScenesLoader`.

    """

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the VOD loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the extracted VOD dataset.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        # Overrides the internal nuScenes category filtering
        self._full_category_contains: list[str] = ["vehicle.ego"]

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=5, output_len=30, sample_time=0.1)
            .with_window(5)
            .with_filter(Filter.define(agent_rules=[RequireFrames.define(frames=[4])]))
        )


if __name__ == "__main__":
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env
    from dronalize.datasets.vod import DESCRIPTOR

    root = resolve_dataset_root_from_env("vod")
    _ = debug_descriptor(DESCRIPTOR, root)
