from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import override

import dronalize.core.transforms as tr
from dronalize.common.plotting import plot_trajectories
from dronalize.core import BaseSceneLoader, LoaderConfig
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.pipeline import Pipeline
from dronalize.core.pipelines import trajectory_pipeline
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class EthUcyLoader(BaseSceneLoader[str, Path]):
    """Processor for ETH/UCY pedestrian trajectory datasets."""

    def __init__(
        self,
        data_dir: Path,
        dataset: str | Sequence[str],
        loader_config: LoaderConfig | None = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        """Initialize with the given configuration.

        Parameters
        ----------
        data_dir : Path
            Path to the root directory containing the ETH/UCY data.
        dataset : str or Sequence[str]
            Name(s) of the dataset(s) to load (e.g., "hotel", "eth").
        loader_config : LoaderConfig, optional
            Configuration override. If None, default configuration will be used.
        split : {"train", "val", "test"}, optional
            Data split to load. Defaults to "train".

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_root = data_dir
        self._dataset = {dataset} if isinstance(dataset, str) else set(dataset)
        self._split = split

    @override
    def sources(self) -> Iterable[Source[str, Path]]:
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            # Sort to ensure consistent order across runs and systems.
            for data_file in sorted(data_dir.iterdir()):
                yield Source(identifier=data_file.name, inner=data_file)

    @override
    def ingest(self, source: Source[str, Path]) -> Iterable[pl.LazyFrame]:
        yield EthUcyLoader._read_data_file(source.inner)

    @override
    def pipeline(self) -> Pipeline:
        config = self.loader_config
        return (
            Pipeline()
            .compose(trajectory_pipeline(config, derivative_rename=self.derivative_names()))
            .then(tr.yaw_from_vel())
            .then(tr.with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN)))
        )

    @override
    def num_sources(self) -> int | None:
        num_sources: int = 0
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            num_sources += sum(1 for _ in data_dir.iterdir())

        return num_sources

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(
                input_len=8,
                output_len=12,
                sample_time=0.4,
            )
            .with_window(step_size=1)
            .with_filtering(require_all_valid=True)
            .with_resampling(4, 1, method="fast")
        )

    @staticmethod
    def _read_data_file(path: Path) -> pl.LazyFrame:
        return pl.scan_csv(
            path,
            has_header=False,
            separator="\t",
            new_columns=["frame", "id", "x", "y"],
            schema={
                "frame": pl.Int32,
                "id": pl.Int32,
                "x": pl.Float64,
                "y": pl.Float64,
            },
        ).with_columns(
            ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
            pl.col("id").cast(pl.Int32),
        )


if __name__ == "__main__":
    import altair as alt

    path = Path("data")
    alt.renderers.enable("browser")
    loader = EthUcyLoader(path, dataset=["hotel"], split="train")
    count = 0
    for scene in loader.scenes():
        count += 1

    print(f"Total scenes: {count}")
