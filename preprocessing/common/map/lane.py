from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl
from scipy.interpolate import UnivariateSpline
from typing_extensions import Self, override

from preprocessing.core.datatypes.categories import EdgeType
from preprocessing.core.graph.builder import GraphBuilder
from preprocessing.core.graph.nodes import IntIDNode


@dataclass(slots=True, frozen=True)
class LaneDescription:
    """Describes the lanes on a highway, including their IDs and directions."""

    ids: list
    direction: list[bool]


class HighWayLaneGraphBuilder(GraphBuilder[int, IntIDNode]):
    """A graph builder that constructs a lane graph for a highway based on vehicle trajectory data.

    This makes three major assumptions about the structure of the highway and the data:
        1. The highway is mostly straight and oriented along either the Y-axis or X-axis.
        2. Drivers drive in the center of their lanes, so lane centers can be estimated by aggregating
           vehicle positions within each longitudinal bin.
    """

    def __init__(
        self,
        data: pl.LazyFrame,
        id_col: str = "id",
        x_col: str = "x",
        y_col: str = "y",
        lane_id_col: str = "lane_id",
        *,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        bin_size: float = 8.0,
        include_outer_borders: bool = True,
        smoothing: float | None = None,
    ) -> None:
        """Initialize the graph builder.

        Args:
            data: A Polars LazyFrame containing vehicle trajectory data.
            id_col: Name of the column containing unique identifiers for each vehicle.
            x_col: Name of the column containing X coordinates.
            y_col: Name of the column containing Y coordinates.
            lane_id_col: Name of the column containing lane identifiers.
            orientation: The dominant axis of the highway ("vertical" for Y-axis, "horizontal" for X-axis).
            bin_size: The size of the longitudinal bins used to group vehicle positions.
            include_outer_borders: Whether to include outer lane borders in the graph.
            smoothing: Optional smoothing factor for lane center estimation (spline parameter).

        """
        super().__init__()
        self._data = data
        self._id_col = id_col
        self._x_col = x_col
        self._y_col = y_col
        self._lane_id_col = lane_id_col
        self._orientation = orientation
        self._bin_size = bin_size
        self._include_outer_borders = include_outer_borders
        self._smoothing_factor = smoothing
        self._lane_description = None

        if orientation == "vertical":
            self._long_col = self._y_col
            self._lat_col = self._x_col
        elif orientation == "horizontal":
            self._long_col = self._x_col
            self._lat_col = self._y_col

    def lane_description(self, lane_description: LaneDescription) -> Self:
        """Provide lane description to enable more accurate edge type classification."""
        self._lane_description = lane_description
        return self

    @override
    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        return IntIDNode(x, y, z)

    @override
    def build_impl(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> None:
        if self._lane_description is not None:
            mapping = {lane_id: idx for idx, lane_id in enumerate(self._lane_description.ids)}
            self._data = self._data.with_columns(
                pl.col(self._lane_id_col).replace_strict(mapping).alias(self._lane_id_col)
            )

        lane_centers = self._get_lane_centers(self._data, bin_size=self._bin_size)
        if self._smoothing_factor is not None:
            lane_centers = _smooth_lines(
                lane_centers,
                group_col=self._lane_id_col,
                x_col="lat_center",
                y_col="long_bin",
                smoothing_factor=self._smoothing_factor,
            ).lazy()

        lane_borders = self._get_inner_borders(lane_centers)

        # Add inner lane boundaries
        for (left, right), group_data in lane_borders.collect().group_by([
            "left_lane",
            "right_lane",
        ]):
            if self._orientation == "vertical":
                x_vals, y_vals = group_data["border_lat"], group_data["long_bin"]
            else:
                x_vals, y_vals = group_data["long_bin"], group_data["border_lat"]

            nodes = list(map(self.new_node, x_vals, y_vals))
            self.add_path_lazy(nodes, self._get_border_type(left, right))

        if self._include_outer_borders:
            outer_borders = self._get_outer_borders(lane_centers, lane_borders)
            for _, group_data in outer_borders.collect().group_by("border_type"):
                if self._orientation == "vertical":
                    x_vals, y_vals = group_data["border_lat"], group_data["long_bin"]
                else:
                    x_vals, y_vals = group_data["long_bin"], group_data["border_lat"]

                nodes = list(map(self.new_node, x_vals, y_vals))
                self.add_path_lazy(nodes, EdgeType.CURB)

    def _get_lane_centers(self, data: pl.LazyFrame, bin_size: float = 8.0) -> pl.LazyFrame:
        return (
            data
            .with_columns((pl.col(self._long_col) // bin_size * bin_size).alias("long_bin"))
            .group_by([self._lane_id_col, "long_bin"])
            # Use median to ignore lane-changing vehicles
            .agg(pl.col(self._lat_col).median().alias("lat_center"))
            # Ensure the points are ordered sequentially along the highway
            .sort([self._lane_id_col, "long_bin"])
        )

    def _get_inner_borders(self, lane_centers: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lane_centers
            .with_columns((pl.col(self._lane_id_col) + 1).alias("next_lane_id"))
            .join(
                lane_centers,
                left_on=["next_lane_id", "long_bin"],
                right_on=[self._lane_id_col, "long_bin"],
                suffix="_right",
            )
            .select([
                "long_bin",
                pl.col(self._lane_id_col).alias("left_lane"),
                pl.col("next_lane_id").alias("right_lane"),
                ((pl.col("lat_center") + pl.col("lat_center_right")) / 2.0).alias("border_lat"),
            ])
        )

    def _get_outer_borders(self, centers: pl.LazyFrame, borders: pl.LazyFrame) -> pl.LazyFrame:
        # Calculate a single average lane half-width for projection
        avg_half_width = (
            borders
            .join(
                centers, left_on=["left_lane", "long_bin"], right_on=[self._lane_id_col, "long_bin"]
            )
            .select((pl.col("border_lat") - pl.col("lat_center")).abs())
            .collect()
            .to_series()
            .mean()
        )

        extreme_lanes = centers.with_columns([
            pl.col(self._lane_id_col).min().alias("min_id"),
            pl.col(self._lane_id_col).max().alias("max_id"),
        ]).filter(
            (pl.col(self._lane_id_col) == pl.col("min_id"))
            | (pl.col(self._lane_id_col) == pl.col("max_id"))
        )

        return extreme_lanes.select([
            "long_bin",
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.lit("left_outer"))
            .otherwise(pl.lit("right_outer"))
            .alias("border_type"),
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.col("lat_center") - avg_half_width)
            .otherwise(pl.col("lat_center") + avg_half_width)
            .alias("border_lat"),
        ])

    def _get_border_type(self, left: Any, right: Any) -> EdgeType:  # noqa: ANN401
        if self._lane_description is not None:
            left_idx = self._lane_description.ids.index(left)
            right_idx = self._lane_description.ids.index(right)
            if (
                self._lane_description.direction[left_idx]
                != self._lane_description.direction[right_idx]
            ):
                return EdgeType.ROAD_BORDER

            return EdgeType.LINE_THICK_DASHED

        return EdgeType.VIRTUAL


def _smooth_lines(
    df: pl.LazyFrame | pl.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    smoothing_factor: float = 1.0,
) -> pl.DataFrame:
    eager_df = df.collect() if isinstance(df, pl.LazyFrame) else df

    def _apply_spline(group: pl.DataFrame) -> pl.DataFrame:
        # Spline requires strictly increasing independent variable
        unique_group = group.unique(subset=[y_col]).sort(y_col)
        if len(unique_group) < 4:  # Spline degree k=3 needs at least 4 points
            return group

        y = unique_group[y_col].to_numpy()
        x = unique_group[x_col].to_numpy()

        spline = UnivariateSpline(y, x, s=smoothing_factor)
        # Apply back to original coordinate mapping to maintain row count
        return group.with_columns(pl.Series(x_col, spline(group[y_col].to_numpy())))

    return eager_df.group_by(group_col, maintain_order=True).map_groups(_apply_spline)
