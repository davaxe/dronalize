"""Lane-change-aware pipeline extensions for highway-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from dronalize.processing.pipeline import transforms as tr
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from dronalize.processing.pipeline.extensions.base import PipelineBuildContext

_LANE_CHANGE_EVENT_COLUMN = "valid_lane_change"
_SCENE_LANE_CHANGE_COUNT_COLUMN = "_scene_lane_change_count"


@dataclass(frozen=True, slots=True)
class LaneChangeSamplingExtension:
    """Keep lane-change scenes and deterministically thin no-change scenes."""

    negative_keep_every: int = 3
    min_lane_change_events: int = 1
    persist: int = 1
    margin_before: int = 0
    margin_after: int = 0

    def _validate(self, builder: PipelineBuildContext) -> str:
        if self.negative_keep_every < 1:
            msg = "negative_keep_every must be at least 1."
            raise ValueError(msg)
        if self.min_lane_change_events < 1:
            msg = "min_lane_change_events must be at least 1."
            raise ValueError(msg)
        if builder.columns.lane_id is None:
            msg = "Lane-change sampling requires a lane_id column."
            raise ValueError(msg)
        return builder.columns.lane_id

    def extend(self, builder: PipelineBuildContext) -> None:
        """Attach lane-change detection and deterministic scene sampling."""
        if not builder.has_window:
            return
        lane_id = self._validate(builder)

        builder.add_pre_window(
            Pipeline().then(
                tr.valid_lane_change(
                    persist=self.persist,
                    margin_before=self.margin_before,
                    margin_after=self.margin_after,
                    frame_column=builder.columns.frame,
                    agent_id_column=builder.columns.agent_id,
                    lane_id_column=lane_id,
                    group_by=builder.window_group_columns() or None,
                    valid_column=_LANE_CHANGE_EVENT_COLUMN,
                )
            )
        )

        def _label_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .col(_LANE_CHANGE_EVENT_COLUMN)
                .sum()
                .over(builder.scene_id_column)
                .alias(_SCENE_LANE_CHANGE_COUNT_COLUMN)
            )

        def _sample_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
            is_positive = (
                pl.col(_SCENE_LANE_CHANGE_COUNT_COLUMN).fill_null(value=0)
                >= self.min_lane_change_events
            )
            keep_negative = (
                pl.lit(value=True)
                if self.negative_keep_every == 1
                else (pl.col(builder.scene_id_column) % self.negative_keep_every) == 0
            )
            return df.filter(is_positive | keep_negative).select(
                pl.all().exclude(_LANE_CHANGE_EVENT_COLUMN, _SCENE_LANE_CHANGE_COUNT_COLUMN)
            )

        builder.add_post_screening(
            Pipeline()
            .then(_label_scenes, name="label_lane_change_scenes")
            .then(_sample_scenes, name="sample_lane_change_scenes")
        )
