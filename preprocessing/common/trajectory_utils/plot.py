from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from preprocessing.common.trajectory_utils.resample import resample_tracks

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


def plot_trajectories(
    data: pl.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    group_by: str | None = None,
    n_groups: int | None = None,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: Axes | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    group_sample_seed: int | None = None,
) -> Axes:
    """Plot a set of trajectories from a Polars dataframe.

    Args:
        data: underlying dataframe containing trajectory data.
        x_col: column name for x-axis values. Defaults to "x".
        y_col: column name for y-axis values. Defaults to "y".
        group_by: column name to group trajectories by (e.g., "track_id").
        n_groups: number of unique groups to plot. If None, plots all groups. Defaults to None.
        x_label: label for x-axis. If None, uses x_col. Defaults to None
        y_label: label for y-axis. If None, uses y_col. Defaults to None.
        ax: Optional Matplotlib Axes to plot on. If None, a new figure and axes are created.
        save_path: Optional path to save the plot image. If None, the plot is not saved.
        title: Optional title for the plot. Defaults to None.
        group_sample_seed: Optional random seed for sampling groups when n_groups is specified.

    Returns:
        The Matplotlib Axes object containing the figure.

    """
    # 1. Filter logic
    if n_groups is not None:
        selected_ids = data.select(group_by).unique().sample(n_groups, seed=group_sample_seed)
        data = data.join(selected_ids, on=group_by, how="semi")

    # 2. Create a consistent color mapping
    unique_ids = data.select(group_by).unique().to_series().to_list()
    palette_name = "tab10" if len(unique_ids) <= 10 else "husl"
    colors = sns.color_palette(palette_name, n_colors=len(unique_ids))
    color_map = dict(zip(unique_ids, colors, strict=True))

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 8))

    df_pd = data.to_pandas()

    # 3. Plot the main trajectory lines
    sns.lineplot(
        data=df_pd,
        x=x_col,
        y=y_col,
        hue=group_by,
        ax=ax,
        palette=color_map,
        alpha=0.8,
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.5,
        sort=False,
        legend=False,
        estimator=None,
    )

    # 4. Add markers and annotations
    unique_groups = df_pd[group_by].unique()
    for tid in unique_groups:
        subset = df_pd[df_pd[group_by] == tid]
        if len(subset) == 0:
            continue

        x, y = list(subset[x_col]), list(subset[y_col])

        # Start marker (Green Circle)
        ax.plot(
            x[0],
            y[0],
            marker="o",
            color="#2ecc71",
            markersize=8,
            markeredgecolor="black",
            zorder=10,
        )

        ax.plot(
            x[-1],
            y[-1],
            marker="X",
            color="#e74c3c",
            markersize=8,
            markeredgecolor="black",
            zorder=10,
        )

    if title is not None:
        ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    return ax


def _generate_test() -> pl.DataFrame:
    t = np.linspace(0, 2 * np.pi, 50)
    frames = np.arange(len(t))
    # Track 1
    return pl.concat([
        pl.DataFrame({
            "frame": frames,
            "x": np.sin(2 * t),
            "y": np.cos(3 * t),
            "track_id": 1,
        }),
        pl.DataFrame({
            "frame": frames,
            "x": np.sin(3 * t),
            "y": np.cos(4 * t),
            "track_id": 2,
        }),
        pl.DataFrame({
            "frame": frames,
            "x": np.cos(1 * t),
            "y": np.sin(1 * t),
            "track_id": 3,
        }),
    ]).cast({
        "x": pl.Float64,
        "y": pl.Float64,
        "frame": pl.Int64,
        "track_id": pl.Int64,
    })


if __name__ == "__main__":
    # 1. Generate the complex data
    df_complex = _generate_test()
    print("Original DataFrame:")
    print(df_complex)

    df_resampled = resample_tracks(df_complex, up=3, down=1, group_by="track_id", method="spline")

    print(df_resampled.filter(pl.col("track_id") == 1))

    plot_trajectories(df_complex, group_by="track_id", n_groups=1)
    plot_trajectories(df_resampled, group_by="track_id", n_groups=1)
    plt.show()
