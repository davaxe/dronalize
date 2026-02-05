from collections.abc import Sequence
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from scipy.interpolate import CubicSpline


def resample_tracks(
    data: pl.DataFrame,
    dt: float,
    org_dt: float,
    frame_column: str = "frame",
    interpolate_columns: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    fraction = Fraction(dt / org_dt).limit_denominator()
    up, down = fraction.denominator, fraction.numerator
    if down == 1 and up == 1:
        return data

    # Ensure group_by is a list for consistency
    group_cols = [group_by] if isinstance(group_by, str) else list(group_by or [])
    return (
        data
        .sort([*group_cols, frame_column])
        .group_by(group_cols)
        .agg([
            # Resample all float columns
            cs.float().map_batches(lambda s: _resample_track(s, up, down)),
            pl
            .col(frame_column)
            .map_batches(lambda s: _generate_new_frames(s, up, down))
            .alias(frame_column),
        ])
        .explode(cs.all().exclude(group_cols))
    )


def _resample_track(series: pl.Series, up: int, down: int) -> pl.Series:
    if series.len() < 2:
        return series

    x_old = np.arange(len(series))
    # Calculate new length to match your dt logic
    new_len = int(np.ceil(len(series) * up / down))
    x_new = np.linspace(0, len(series) - 1, new_len)

    # Create the spline and evaluate at new points
    cs = CubicSpline(x_old, series.to_numpy(), bc_type="clamped")
    return pl.Series(cs(x_new))


def _generate_new_frames(series: pl.Series, up: int, down: int) -> pl.Series:
    if series.len() < 2:
        return series
    new_len = int(np.ceil(len(series) * up / down))
    start_frame = series.item(0)
    return pl.Series(np.arange(new_len) + start_frame)


def plot_comparison(
    original_df: pl.DataFrame,
    resampled_df: pl.DataFrame,
    group_by: str = "track_id",
    n_groups: int | None = None,
):
    # 1. Filter logic
    if n_groups is not None:
        selected_ids = original_df.select(group_by).unique().head(n_groups)
        original_df = original_df.join(selected_ids, on=group_by, how="semi")
        resampled_df = resampled_df.join(selected_ids, on=group_by, how="semi")

    # 2. Create a consistent color mapping
    # This ensures Track 5 is the same color in both plots
    unique_ids = original_df.select(group_by).unique().to_series().to_list()

    # "tab10" is high contrast for few items; "husl" is distinct for many items
    palette_name = "tab10" if len(unique_ids) <= 10 else "husl"
    colors = sns.color_palette(palette_name, n_colors=len(unique_ids))
    color_map = dict(zip(unique_ids, colors))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    dataframes = [original_df.to_pandas(), resampled_df.to_pandas()]
    titles = ["Original (Low Frequency)", "Resampled (Spline Interpolation)"]

    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        sns.lineplot(
            data=df,
            x="x",
            y="y",
            hue=group_by,
            ax=ax,
            palette=color_map,  # Use the fixed dictionary here
            alpha=0.8,  # Increased opacity for better visibility
            linewidth=2,  # Slightly thicker lines
            marker="o",
            markersize=4,
            markeredgecolor="white",  # Adds a tiny border to points to pop against background
            markeredgewidth=0.5,
            sort=False,
            legend=False,
        )

        # Plot Start/End markers
        unique_groups = df[group_by].unique()
        for tid in unique_groups:
            subset = df[df[group_by] == tid]
            if len(subset) == 0:
                continue

            # Start: Green Circle with black edge for contrast
            ax.plot(
                subset["x"].iloc[0],
                subset["y"].iloc[0],
                marker="o",
                color="#2ecc71",
                markersize=8,
                markeredgecolor="black",
                label="Start" if tid == unique_groups[0] and i == 0 else "",
                zorder=10,
            )
            # End: Red X with black edge
            ax.plot(
                subset["x"].iloc[-1],
                subset["y"].iloc[-1],
                marker="X",
                color="#e74c3c",
                markersize=8,
                markeredgecolor="black",
                label="End" if tid == unique_groups[0] and i == 0 else "",
                zorder=10,
            )

        ax.set_title(titles[i], fontsize=12, pad=15, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("X Position")

    axes[0].set_ylabel("Y Position")

    # Add single legend
    if n_groups is not None and n_groups > 0:
        fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)

    plt.tight_layout()
    plt.show()


def generate_test_suite():
    # 1. Lissajous Curve (Complex curvature)
    t1 = np.linspace(0, 2 * np.pi, 50)
    df1 = pl.DataFrame({
        "frame": np.arange(len(t1)),
        "x": 10 * np.sin(2 * t1),
        "y": 10 * np.cos(3 * t1),
        "track_id": 1,
    })

    # 2. Damped Spiral (Increasing frequency/tightening turns)
    t2 = np.linspace(0, 4 * np.pi, 50)
    r = 15 / (1 + 0.1 * t2)
    df2 = pl.DataFrame({
        "frame": np.arange(len(t2)),
        "x": r * np.cos(t2) + 25,
        "y": r * np.sin(t2),
        "track_id": 2,
    })

    # 3. Straight Line (Testing for artifacts/overshoot)
    # We use the same number of points to keep it consistent
    t3 = np.linspace(0, 1, 50)
    df3 = pl.DataFrame({
        "frame": np.arange(len(t3)),
        "x": np.linspace(0, 30, 50),
        "y": np.linspace(-15, -15, 50),  # Constant Y
        "track_id": 3,
    })

    return pl.concat([df1, df2, df3]).with_columns([
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        pl.col("frame").cast(pl.Int64),
        pl.col("track_id").cast(pl.Int64),
    ])


if __name__ == "__main__":
    # 1. Generate the complex data
    df_complex = generate_test_suite()

    # 3. Resample (dt=0.2, org_dt=1.0 means 5x upsampling)
    dt_new = 0.4
    dt_org = 1.0

    df_resampled = resample_tracks(
        df_complex, dt=dt_new, org_dt=dt_org, group_by="track_id"
    )
    print(df_resampled)

    plot_comparison(df_complex, df_resampled)

    print(f"Original row count: {len(df_complex)}")
    print(f"Resampled row count: {len(df_resampled)}")
