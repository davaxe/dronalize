from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """IDs of the lanes in the order they appear on the highway, from left to right.

    The IDs must match the ones in the dataframe input (often integers) but could be any type as
    long as they are consistent and can be compared for equality.
    """
    direction: list[bool]
    """Directions of the lanes in the same order as `ids`. True for forward, False for backward.

    It is only important to know if two adjacent lanes have the same direction or not, so the actual
    meaning of True/False is not important as long as it is consistent.
    """


class HighWayLaneGraphBuilder(GraphBuilder[int, IntIDNode]):
    """A graph builder that constructs a lane graph for a highway based on vehicle trajectory data.

    This makes three major asumptions about the structure of the highway and the data:
        1. The highway is mostly straight and oriented along the Y-axis, so lanes can be identified by
           binning the Y coordinates (also works for lanes oriented fully in the X-axis, by swapping the columns).
        2. Drivers drive in the center of their lanes, so lane centers can be estimated by aggregating
           vehicle positions within each Y-bin.
    """

    def __init__(
        self,
        data: pl.LazyFrame,
        id_col: str = "id",
        x_col: str = "x",
        y_col: str = "y",
        lane_id_col: str = "lane_id",
        *,
        bin_size: float = 8.0,
        include_outer_borders: bool = True,
        smoothing: float | None = None,
    ) -> None:
        """Initialize the graph builder.

        The input dataframe is expected to contain the following:
            - A unique identifier for each vehicle (e.g., "id" column).
            - X and Y coordinates of the vehicle positions (e.g., "x" and "y" columns).
            - A lane identifier for each position (e.g., "lane_id" column), which indicates which lane
                the vehicle is in. If a `LandeDescription` is not provided it is assumed
                tha the lane identifier is integers where adjacent lanes have consecutive integers
                (e.g., lane 0 is adjacent to lane 1, lane 1 is adjacent to lanes 0 and 2, etc.). If
                this is not the case use `lane_description` method to provide a mapping from lane IDs
                to their order on the highway and their direction.

        Args:
            data: A Polars LazyFrame containing vehicle trajectory data with columns for ID, X and
                Y coordinates, and lane ID.
            id_col: Name of the column containing unique identifiers for each vehicle. Defaults to "id".
            x_col: Name of the column containing X coordinates. Defaults to "x".
            y_col: Name of the column containing Y coordinates. Defaults to "y".
            lane_id_col: Name of the column containing lane identifiers. Defaults to "lane_id".
            bin_size: The size of the Y-axis bins used to group vehicle positions for lane center
            estimation. Defaults to 8.0 units (meters).
            include_outer_borders: Whether to include outer lane borders in the graph. Defaults to True.
            smoothing: Optional smoothing factor for lane center estimation. If provided, applies a
                spline smoothing to the lane center points to create smoother lane borders. Higher
                values gives more smoothing.

        """
        super().__init__()
        self._data = data
        self._id_col = id_col
        self._x_col = x_col
        self._y_col = y_col
        self._lane_id_col = lane_id_col
        self._bin_size = bin_size
        self._include_outer_borders = include_outer_borders
        self._smoothing_factor = smoothing is not None
        self._lane_description = None

    def lane_description(self, lane_description: LaneDescription) -> Self:
        """Provide lane description to enable more accurate edge type classification.

        If not provided, all lane border edges will be classified as `EdgeType.VIRTUAL` since the
        builder cannot determine whether adjacent lanes have the same direction or not.
        """
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
                x_col="x_center",
                y_col="y_bin",
                smoothing_factor=self._smoothing_factor,
            ).lazy()

        lane_borders = self._get_inner_borders(lane_centers)

        # Add inner lane boundaries
        for (left, right), group_data in lane_borders.collect().group_by([
            "left_lane",
            "right_lane",
        ]):
            nodes = list(map(self.new_node, group_data["border_x"], group_data["y_bin"]))
            self.add_path_lazy(nodes, self._get_border_type(left, right))

        if self._include_outer_borders:
            outer_borders = self._get_outer_borders(lane_centers, lane_borders)
            for _, group_data in outer_borders.collect().group_by("border_type"):
                nodes = list(map(self.new_node, group_data["border_x"], group_data["y_bin"]))
                self.add_path_lazy(nodes, EdgeType.CURB)

    def _get_lane_centers(self, data: pl.LazyFrame, bin_size: float = 8.0) -> pl.LazyFrame:
        return (
            data
            .with_columns((pl.col(self._y_col) // bin_size * bin_size).alias("y_bin"))
            .group_by([self._lane_id_col, "y_bin"])
            # Use median to ignore lane-changing vehicles
            .agg(pl.col(self._x_col).median().alias("x_center"))
            # Ensure the points are ordered sequentially along the highway
            .sort([self._lane_id_col, "y_bin"])
        )

    def _get_inner_borders(self, lane_centers: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lane_centers
            .with_columns((pl.col(self._lane_id_col) + 1).alias("next_lane_id"))
            .join(
                lane_centers,
                left_on=["next_lane_id", "y_bin"],
                right_on=[self._lane_id_col, "y_bin"],
                suffix="_right",
            )
            .select([
                "y_bin",
                pl.col(self._lane_id_col).alias("left_lane"),
                pl.col("next_lane_id").alias("right_lane"),
                ((pl.col("x_center") + pl.col("x_center_right")) / 2.0).alias("border_x"),
            ])
        )

    def _get_outer_borders(self, centers: pl.LazyFrame, borders: pl.LazyFrame) -> pl.LazyFrame:
        # Calculate a single average lane half-width for projection
        avg_half_width = (
            borders
            .join(centers, left_on=["left_lane", "y_bin"], right_on=[self._lane_id_col, "y_bin"])
            .select((pl.col("border_x") - pl.col("x_center")).abs())
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
            "y_bin",
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.lit("left_outer"))
            .otherwise(pl.lit("right_outer"))
            .alias("border_type"),
            pl
            .when(pl.col(self._lane_id_col) == pl.col("min_id"))
            .then(pl.col("x_center") - avg_half_width)
            .otherwise(pl.col("x_center") + avg_half_width)
            .alias("border_x"),
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
        # Spline requires strictly increasing Y values
        unique_group = group.unique(subset=[y_col]).sort(y_col)
        if len(unique_group) < 4:  # Spline degree k=3 needs at least 4 points
            return group

        y = unique_group[y_col].to_numpy()
        x = unique_group[x_col].to_numpy()

        spline = UnivariateSpline(y, x, s=smoothing_factor)
        # Apply back to original Y coordinates to maintain row count
        return group.with_columns(pl.Series(x_col, spline(group[y_col].to_numpy())))

    return eager_df.group_by(group_col, maintain_order=True).map_groups(_apply_spline)
