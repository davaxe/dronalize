from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY_V1
from dronalize.datasets.opendd.maps.builder import OpenDDMapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.filters import Filter, RequireAgentFrames
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.pipeline.functional.resample import ResampleSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps.graph import MapGraph
    from dronalize.core.scene import Scene, SceneSchema
    from dronalize.processing.ingest.splits import SplitRequest
    from dronalize.processing.maps.resolver import MapResolver


def _table_query(table_name: str) -> str:
    return f"""
    SELECT
        OBJID as id,
        TIMESTAMP,
        UTM_X as x,
        UTM_Y as y,
        CLASS
    FROM {table_name}
    """


class OpenDDLoader(BaseSceneLoader[tuple[Path, str]]):
    """Loader for OpenDD data split across multiple SQLite databases."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True,
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the multi-database OpenDD loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory containing the extracted OpenDD SQLite files.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)

    def _db_paths(self) -> Iterable[Path]:
        for db_file in self._data_root.rglob("trajectories_*_v3.sqlite"):
            level = len(db_file.relative_to(self._data_root).parts)
            if level > 3:
                continue
            yield db_file

    @override
    def discover_sources(self) -> Iterable[Source[tuple[Path, str]]]:
        for db_path in self._db_paths():
            for table_name in _list_table_names(db_path):
                identifier = f"{db_path.stem}:{table_name}"
                yield Source(
                    identifier=identifier,
                    data=(db_path, table_name),
                    map_key=str(db_path.parent),
                )

    @override
    def num_sources(self) -> int | None:
        return sum(_count_tables(db_path) for db_path in self._db_paths())

    @override
    def ingest(self, source: Source[tuple[Path, str]]) -> Iterable[IngestedData]:
        db_path, table_name = source.data
        with sqlite3.connect(db_path) as connection:
            yield IngestedData(
                pl
                .read_database(_table_query(table_name), connection)
                .lazy()
                .with_columns(
                    ((pl.col("TIMESTAMP") * 1000).round(4).rank(method="dense") - 1)
                    .cast(pl.Int64)
                    .alias("frame"),
                    pl
                    .col("CLASS")
                    .replace_strict(
                        {
                            "Car": AgentCategory.CAR,
                            "Medium Vehicle": AgentCategory.CAR,
                            "Heavy Vehicle": AgentCategory.TRUCK,
                            "Trailer": AgentCategory.TRUCK,
                            "Bus": AgentCategory.BUS,
                            "Motorcycle": AgentCategory.MOTORCYCLE,
                            "Pedestrian": AgentCategory.PEDESTRIAN,
                            "Bicycle": AgentCategory.BICYCLE,
                        },
                        default=AgentCategory.UNKNOWN,
                    )
                    .alias("agent_category"),
                )
                .drop("CLASS", "TIMESTAMP"),
            )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=60, output_len=150, sample_time=1 / 30)
            .with_resampling(ResampleSpec(up=1, down=3))
            .with_window(75)
            .with_filters(Filter.define(filter_rules=[RequireAgentFrames.define(frames=[59])]))
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.default()

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None

            return utils.extract_based_on_scene(
                self._get_map(
                    scene.map_key,
                    self.map_config.min_distance,
                    self.map_config.interp_distance,
                ),
                scene,
                self.map_config.extraction,
            )

        return _resolver

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _get_map(
        key: str,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> MapGraph:
        database: str = Path(key).name
        map_path = Path(key) / f"map_{database}" / f"map_{database}.sqlite"
        return OpenDDMapBuilder.from_sqlite_file(map_path).build(min_distance, interp_distance)


def _list_table_names(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        return [name for (name,) in cursor]


def _count_tables(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table';",
        ).fetchone()
    return row[0] if row is not None else 0
