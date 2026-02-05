import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum, auto
from typing import Generic, TypedDict, TypeVar, cast, override

import polars as pl
from attr import asdict, dataclass


class Category(StrEnum):
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


class Status(StrEnum):
    """Enumeration of agent status."""

    UNKNOWN = auto()
    MOVING = auto()
    STOPPED = auto()


CategoryPL = pl.Enum(Category)
StatusPL = pl.Enum(Status)


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


T_Dict = TypeVar("T_Dict", bound=FrameDict)


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


T_Frame = TypeVar("T_Frame", bound=BaseFrame[FrameDict])
T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID")


@dataclass(slots=True, frozen=True)
class Scene(Generic[T_ID, T_Frame]):
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to atleast contain all columns defined in FrameDict.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    identifier: T_ID
    """Identifier for the scene (e.g., file name, index)."""

    def __post_init__(self) -> None:
        """Validate the inner DataFrame schema."""
        required_cols = FrameDict.__annotations__.keys()
        missing = set(required_cols) - set(self.inner.columns)

        if missing:
            msg = f"Scene DataFrame missing required columns: {missing}"
            raise ValueError(msg)

    def frames(self) -> Iterable[T_Frame]:
        """Yield frames as data class instances."""
        for row in self.inner.iter_rows(named=True):
            yield cast("T_Frame", row)


@dataclass(slots=True)
class DataProcessor(ABC, Generic[T_ID, T_Source, T_Frame]):
    """Interface for processing raw data sources into a standardized format.

    Generic over the data scene source type and identifier type. Source type can
    be anything (e.g., file path, URL, database connection, raw data).
    Identifier type is used to unique identify each scene source (e.g., str,
    int).
    """

    # --- Abstract Steps (The "Blanks" to fill) ---

    @abstractmethod
    def sources(self) -> Iterable[tuple[T_ID, T_Source]]:
        """Discover and yield identifiers for each scene to process."""

    @abstractmethod
    def load_raw(self, source: T_Source) -> Iterable[pl.DataFrame]:
        """Read the raw data source into one or more DataFrame(s) (any schema)."""

    @abstractmethod
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert the raw DataFrame into the common schema."""

    def post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Post-process step after normalization and before validation.

        This is an optional step that can be overridden by subclasses.
        """
        return df

    def process_next(
        self,
        input_data: T_Source,
        *,
        validate_output: bool = True,
        validate_intermediate: bool = False,
    ) -> Iterable[pl.DataFrame]:
        """Process a single data item through the pipeline.

        Steps:
        1. Load raw data.
        2. Normalize to common schema.
        3. Validate intermediate schema (Optional, default False).
        4. Post-process the data.
        5. Validate output schema (Optional, default True).
        """

        def _step(df: pl.DataFrame) -> pl.DataFrame:
            df = self.normalize(df)
            if validate_intermediate:
                self._validate_schema(df)

            df = self.post_process(df)
            if validate_output:
                self._validate_schema(df)
            return df

        for df in self.load_raw(input_data):
            yield _step(df)

    def process_scenes(self) -> Iterable[Scene[T_ID, T_Frame]]:
        """Iterate over all sources and process them.

        This will lazy-load and process each source one at a time.
        """
        for source_id, source in self.sources():
            for df in self.process_next(source):
                yield Scene[T_ID, T_Frame](inner=df, identifier=source_id)

    def _validate_schema(self, df: pl.DataFrame) -> None:
        # FrameDict keys are required
        required_cols = FrameDict.__annotations__.keys()
        missing = set(required_cols) - set(df.columns)

        if missing:
            msg = f"processor output missing required columns: {missing}"
            raise ValueError(msg)


@dataclass(slots=True)
class FrameStreamProcessor(DataProcessor[T_ID, T_Source, T_Frame], ABC):
    """DataProcessor specialized for frame-by-frame data sources."""

    @abstractmethod
    def iter_frames(self, source: T_Source) -> Iterable[T_Frame]:
        """Yield frames one by one from the raw source."""

    @override
    def load_raw(self, source: T_Source) -> pl.DataFrame:
        frames = [f.to_dict() for f in self.iter_frames(source)]

        if not frames:
            return pl.DataFrame(schema=T_Frame.__annotations__)

        return pl.DataFrame(frames)

    @override
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select([
            pl.col("frame").cast(pl.Int64),
            pl.col("track_id").cast(pl.Int64),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("vx").cast(pl.Float64),
            pl.col("vy").cast(pl.Float64),
            pl.col("yaw").cast(pl.Float64),
            pl.col("agent_class").cast(pl.Int64),
            pl.col("status").cast(pl.Int64),
        ])
