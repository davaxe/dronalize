from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import override

import dronalize.core.transforms as tr
from dronalize.core import BaseSceneLoader, LoaderConfig
from dronalize.core.datatypes import map_context as mc
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.pipeline import Pipeline
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.core.datatypes.loader_config import WindowParams


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
    def pipeline(self) -> Pipeline:
        config = self.loader_config
        window = config.window_params
        has_window = window is not None
        step_size, window_size = self._window_params(window)
        return (
            Pipeline()
            # 1. Drop agents with only a single observation in the raw file.
            #    This runs on the smallest possible data (pre-window explosion)
            #    so agents that would be invalid in every window are never
            #    duplicated across the ~N windows produced by step 2.
            .then(tr.require_min("id", minimum=2))
            # 2. Add window_index column (does NOT fan-out yet, keeps lazy).
            #    With step_size=1 this multiplies the row count by ~window_size,
            #    so it is important to reduce data before reaching this step.
            .then(
                tr.window(window_size, step_size, offset_sliding_col=True),
                when=has_window,
            )
            # 3. Scene-level filter, scoped per window when windowing is active.
            #    Must run after windowing because its semantics (require_all_valid,
            #    min_agents, require_frames) are all per-window concepts.
            .then(
                tr.filter_scene(
                    config.scene_filtering,
                    group_by="window_index" if has_window else None,
                ),
                when=config.scene_filtering is not None,
            )
            # 4. Drop agents that appear only once *within* a window.  Step 1
            #    already removed globally single-sample agents; this catches
            #    agents whose track starts or ends mid-window, leaving them
            #    with just one sample inside that particular window.
            .then(
                tr.require_min(["window_index", "id"] if has_window else "id", minimum=2),
                when=has_window,
            )
            # 5. Resample + derivatives, grouped so each agent * window is
            #    treated as an independent track.
            .then(
                tr.resample(
                    config.resampling,
                    config.sample_time,
                    group_by=["window_index", "id"] if has_window else ["id"],
                    add_derivative=True,
                    add_second_derivative=True,
                    derivative_rename=self.derivative_names(),
                )
            )
            # 6. Fan-out: one LazyFrame per window, drop the window_index col.
            #    Then zero-offset the frame column within each yielded group.
            .then_flat_map(tr.group_by_yield("window_index"), when=has_window)
            # 7. Add missing columns
            .then(tr.yaw_from_vel())
            .then(tr.with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN)))
        )

    @staticmethod
    def _window_params(window_params: WindowParams | None) -> tuple[int, int]:
        step_size, window_size = 0, 0
        if window_params is not None:
            step_size = window_params.step_size
            window_size = window_params.window_size
        return step_size, window_size

    @override
    def num_sources(self) -> int | None:
        num_sources: int = 0
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            num_sources += sum(1 for _ in data_dir.iterdir())

        return num_sources

    @override
    def load_raw(self, source: Source[str, Path]) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        yield EthUcyLoader._read_data_file(source.inner), mc.NoMap()

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
