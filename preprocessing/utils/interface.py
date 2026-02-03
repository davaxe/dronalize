import datetime
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import IntEnum, auto
from typing import Generic, TypedDict, TypeVar, cast

import numpy as np
import polars as pl
from attr import asdict, dataclass


class Category(IntEnum):
    """Enumeration of categories of agents / objects."""

    CAR = auto()
    VAN = auto()
    TRAILER = auto()
    TRUCK = auto()
    TRAM = auto()
    BUS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    PEDESTRIAN = auto()
    TRICYCLE = auto()
    ANIMAL = auto()
    STATIC_OBJECT = auto()
    MOVEABLE_OBJECT = auto()
    UNKNOWN = auto()


class Status(IntEnum):
    """Enumeration of agent status."""

    UNKNOWN = auto()
    MOVING = auto()
    STOPPED = auto()


class FrameDict(TypedDict):
    """TypedDict representing a single frame of an object's trajectory."""

    frame: int
    """Frame index."""
    track_id: int
    """Unique track identifier."""
    x: float
    """x position in meters."""
    y: float
    """y position in meters."""
    vx: float
    """x velocity in m/s."""
    vy: float
    """y velocity in m/s."""
    yaw: float
    """Orientation in radians."""
    agent_class: Category
    """Class of the agent/object."""
    status: Status
    """Status of the agent/object."""


T_Dict = TypeVar("T_Dict", bound="FrameDict")


@dataclass(slots=True, frozen=True)
class BaseFrame(Generic[T_Dict]):
    """Data class representing a single frame of an object's trajectory.

    Can be extended for different data representations.
    """

    frame: int
    """Frame index."""
    track_id: int
    """Unique track identifier."""
    x: float
    """x position in meters."""
    y: float
    """y position in meters."""
    vx: float | None = None
    """x velocity in m/s."""
    vy: float | None = None
    """y velocity in m/s."""
    yaw: float | None = None
    """Orientation in radians."""
    category: Category = Category.UNKNOWN
    """Category of the agent/object."""
    status: Status = Status.UNKNOWN
    """Status of the agent/object."""

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        if self.vx is not None and self.vy is not None and self.yaw is None:
            object.__setattr__(
                self,
                "yaw",
                math.atan2(self.vy, self.vx),
            )

    def to_dict(self) -> T_Dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the Frame.

        """
        return cast("T_Dict", asdict(self))


class Frame(BaseFrame[FrameDict]):
    """Data class representing a single frame of an object's trajectory.

    This is equivalent to a single row in the common dataframe format.
    """


T_Frame = TypeVar("T_Frame", bound=BaseFrame)


class DataProcessor(ABC, Generic[T_Frame]):
    """Common interface for data processing classes."""

    @abstractmethod
    def parse_raw_data(self) -> Iterable[T_Frame]:
        """Parse the raw data into an iterable of Frame objects.

        Returns:
            An iterable of Frame objects.

        """
        ...

    @abstractmethod
    def process(self) -> pl.DataFrame:
        """Process the raw data into a common `polars.Dataframe` format.

        Returns:
            Processed data as dataframe

        """
        return pl.DataFrame([frame.to_dict() for frame in self.parse_raw_data()])


def estimate_velocity(data: pl.DataFrame, dt: float) -> pl.DataFrame:
    """Estimate velocities and yaw if missing in the input dataframe.

    Args:
        data: Input dataframe in common format.
        dt: Time difference between frames in seconds.

    Returns:
        Dataframe with estimated velocities and yaw.

    """
    # 1. Ensure columns exist to prevent schema errors
    df = data
    for col in ["vx", "vy", "yaw"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    # Sort for temporal consistency
    df = df.sort(["track_id", "frame"])

    df = df.with_columns(
        pl.col("x").diff().over("track_id").fill_null(0.0).alias("_dx"),
        pl.col("y").diff().over("track_id").fill_null(0.0).alias("_dy"),
    )

    df = df.with_columns(
        pl.coalesce([pl.col("vx"), pl.col("_dx") / dt]).alias("vx"),
        pl.coalesce([pl.col("vy"), pl.col("_dy") / dt]).alias("vy"),
    )

    df = df.with_columns(
        pl.coalesce([
            pl.col("yaw"),
            pl.arctan2(pl.col("vy"), pl.col("vx")),
        ]).alias("yaw")
    )

    return df.drop(["_dx", "_dy"])


def upsample(
    data: pl.DataFrame,
    org_dt: float,
    target_dt: float,
    group_by: str = "track_id",
) -> pl.DataFrame:
    """Upsample data input from `org_dt` to `target_dt`.

    Uses linear interpolation.

    Args:
        data: data to upsample
        org_dt: original time difference between frames in seconds
        target_dt: target time difference between frames in seconds
        group_by: column name to group by for upsampling. Defaults to "track_id".

    Returns:
        upsampled dataframe

    """
    us_per_frame = int(org_dt * 1e6)
    base_time = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    data = data.with_columns([
        ((pl.col("vx") ** 2 + pl.col("vy") ** 2) ** 0.5).alias("_speed"),
        pl
        .struct(["vx", "vy"])
        .map_batches(
            lambda s: np.unwrap(
                np.arctan2(s.struct.field("vy"), s.struct.field("vx"))
            )
        )
        .alias("_v_angle"),
        pl.col("yaw").map_batches(lambda s: np.unwrap(s.to_numpy())).alias("yaw"),
    ])

    interp_cols = ["x", "y", "frame", "_speed", "_v_angle", "yaw"]
    fill_cols = [c for c in data.columns if c not in interp_cols and c != "time"]
    data = (
        data
        .with_columns(
            (
                (pl.col("frame") * us_per_frame).cast(pl.Duration("us")) + base_time
            ).alias("time")
        )
        .upsample(
            time_column="time",
            every=datetime.timedelta(seconds=target_dt),
            group_by=group_by,
        )
        .with_columns([
            pl.col(interp_cols).interpolate(),
            pl.col(fill_cols).forward_fill(),
        ])
    )

    return data.with_columns([
        # Reconstruct Velocities
        (pl.col("_speed") * pl.col("_v_angle").cos()).alias("vx"),
        (pl.col("_speed") * pl.col("_v_angle").sin()).alias("vy"),
        # Wrap Yaw and Velocity Angle
        *(
            (pl.col(c) + math.pi) % (2 * math.pi) - math.pi
            for c in ["yaw", "_v_angle"]
        ),
        # Fix Frame Index
        (pl.col("frame") * (org_dt / target_dt)).cast(pl.Int64).alias("frame"),
    ]).drop(["time", "_speed", "_v_angle"])


def downsample(
    data: pl.DataFrame,
    org_dt: float,
    target_dt: float,
    group_by: str = "track_id",
) -> pl.DataFrame:
    """Downsample data input from `org_dt` to `target_dt`.

    Will be downsampled to the nearest integer factor of the original framerate.

    Args:
        data: data to downsample
        org_dt: original time difference between frames in seconds
        target_dt: target time difference between frames in seconds
        group_by: column name to group by for downsampling. Defaults to "track_id".

    Returns:
        downsampled dataframe

    """
    factor = int(target_dt / org_dt)

    if factor <= 1:
        return data

    data = data.filter((pl.col("frame").over(group_by) % factor) == 0)
    return data.with_columns(
        (pl.col("frame") / factor).cast(pl.Int64).alias("frame")
    )
