import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Generic, Self, TypedDict, TypeVar, cast, override

import polars as pl
from attr import asdict, dataclass

from preprocessing.trajectory.utils import (
    AgentData,
    Category,
    convert_to_agent_data_dict,
)


@dataclass(slots=True)
class SceneFiltering:
    """Configuration for filtering scenes based on pedestrian presence and validity."""

    min_agents: int = 2
    """Minimum number of agents required in a scene to be valid."""

    require_all_valid: bool = False
    """If True, requires all agents in a scene to have valid positions for all
    time-steps."""

    require_prediction_frame: bool = True
    """If True, requires all agents to have valid positions at the first prediction
    frame."""


@dataclass(slots=True)
class WindowParams:
    """Configuration for sliding window sampling of scenes."""

    window_size: int
    """Number of frames in each window."""

    step_size: int
    """Number of frames to skip between windows."""


@dataclass(slots=True)
class ProcessorConfig:
    """Base configuration dataclass for trajectory data processing.

    This can be extended by specific dataset processors to include additional
    parameters.
    """

    input_len: int
    """Observation length in frames."""

    output_len: int
    """Prediction length in frames."""

    sample_time: float
    """Time interval between frames in seconds."""

    target_sample_time: float | None = None
    """The target time interval for resampling the trajectories, in seconds. If
    None, no resampling will be performed."""

    window_params: WindowParams | None = None
    """Used for datasets where multiple samples can be generated from a single
    scene by using a sliding window approach. If None, it is assumed that each
    scene corresponds to exactly one sample."""

    scene_filtering: SceneFiltering | None = None
    """Configuration for filtering scenes based on pedestrian presence and validity."""

    def window_parameters(
        self, step_size: int, window_size: int | None = None
    ) -> Self:
        """Set the window parameters for sliding window sampling."""
        self.window_params = WindowParams(
            window_size=window_size
            if window_size is not None
            else self.input_len + self.output_len,
            step_size=step_size,
        )
        return self

    def scene_filtering_parameters(
        self,
        min_agents: int = 2,
        *,
        require_all_valid: bool = False,
        require_prediction_frame: bool = True,
    ) -> Self:
        """Set the scene filtering parameters."""
        self.scene_filtering = SceneFiltering(
            min_agents=min_agents,
            require_all_valid=require_all_valid,
            require_prediction_frame=require_prediction_frame,
        )
        return self


class FrameDict(TypedDict):
    """TypedDict representing a single frame of an object's trajectory."""

    frame: int
    """Frame index."""
    id: int
    """Unique track identifier."""
    x: float
    """x position in meters."""
    y: float
    """y position in meters."""
    vx: float
    """x velocity in m/s."""
    vy: float
    """y velocity in m/s."""
    ax: float
    """x acceleration in m/s^2."""
    ay: float
    """y acceleration in m/s^2."""
    yaw: float
    """Orientation in radians."""
    agent_class: Category
    """Class of the agent/object."""


T_Dict = TypeVar("T_Dict", bound=FrameDict)


@dataclass(slots=True, frozen=True)
class BaseFrame(Generic[T_Dict]):
    """Data class representing a single frame of an object's trajectory.

    Can be extended for different data representations.
    """

    frame: int
    """Frame index."""
    id: int
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
    source_identifier: T_ID
    """Identifier for the scene (e.g., file name, index)."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""

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

    def to_agent_data(
        self,
        target_id: int | None = None,
    ) -> AgentData:
        """Convert `Scene` into a agent dictionary.

        The dictionary is in format that is later compatible with pytorch
        geometric HeteroData.

        Args:
            target_id: Optional track ID to use as the target node. If None, the
                first valid track will be used as the target.

        Returns:
            Dictionary containing the agent data according to the
            AgentData TypedDict.

        """
        return convert_to_agent_data_dict(
            self.inner,
            input_len=self.input_len,
            output_len=self.output_len,
            target_agent=target_id,
        )


IDMapping = Callable[[int, T_ID], T_ID]


class DataProcessor(ABC, Generic[T_ID, T_Source, T_Frame]):
    """Interface for processing raw data sources into a standardized format.

    Generic over the data scene source type and identifier type. Source type can
    be anything (e.g., file path, URL, database connection, raw data).
    Identifier type is used to unique identify each scene source (e.g., str,
    int).
    """

    def __init__(
        self,
        processor_config: ProcessorConfig,
        id_mapping: IDMapping[T_ID] | None = None,
        *,
        validate_output: bool = True,
        validate_intermediate: bool = False,
    ) -> None:
        """Initialize internal state."""
        self._count: int = 0
        self._source_counter: int = 0
        self._id_mapping = id_mapping
        self._validate_output = validate_output
        self._validate_intermediate = validate_intermediate
        self._processor_config = processor_config

    # --- Abstract Steps (The "Blanks" to fill) ---

    @property
    def processor_config(self) -> ProcessorConfig:
        """Return the processor configuration."""
        return self._processor_config

    @property
    def original_input_len(self) -> int:
        """Original observation length in frames (before resampling)."""
        return self.processor_config.input_len

    @property
    def original_output_len(self) -> int:
        """Original prediction length in frames (before resampling)."""
        return self.processor_config.output_len

    @property
    def sequence_length(self) -> int:
        """Total sequence length (observation + prediction) in frames."""
        return self.processor_config.input_len + self.processor_config.output_len

    @property
    def resampling_ratio(self) -> float:
        """Ratio of target sample time to original sample time."""
        if self.processor_config.target_sample_time is None:
            return 1.0
        return (
            self.processor_config.sample_time
            / self.processor_config.target_sample_time
        )

    def input_len(self) -> int:
        """Observation length in frames (resulting value in Scene)."""
        return int((self.original_input_len - 1) * self.resampling_ratio + 1)

    def output_len(self) -> int:
        """Prediction length in frames (resulting value in Scene)."""
        total_len = int((self.sequence_length - 1) * self.resampling_ratio + 1)
        return total_len - self.input_len()

    @abstractmethod
    def sources(self) -> Iterable[tuple[T_ID, T_Source]]:
        """Discover and yield identifiers for each scene to process."""

    @abstractmethod
    def load_raw(self, source: T_Source) -> Iterable[pl.DataFrame]:
        """Read the raw data source into one or more DataFrame(s) (any schema)."""

    @abstractmethod
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Convert the raw DataFrame into the common schema."""

    def post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Post-process step after normalization and before validation.

        This is an optional step that can be overridden by subclasses.
        """
        return df

    def process_next(
        self,
        source: T_Source,
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

        def _step(raw_df: pl.DataFrame) -> pl.DataFrame | None:
            normalized_df = self.normalize(raw_df)
            if normalized_df is None:
                return None
            if validate_intermediate:
                self._validate_schema(normalized_df)
            df = self.post_process(normalized_df)
            if validate_output:
                self._validate_schema(df)
            return df

        for raw_df in self.load_raw(source):
            df = _step(raw_df)
            if df is not None:
                yield df

    def process_scenes(self) -> Iterable[Scene[T_ID, T_Frame]]:
        """Iterate over all sources and process them.

        This will lazy-load and process each source one at a time.
        """
        for source_id, source in self.sources():
            self._source_counter += 1
            for scene_df in self.process_next(
                source,
                validate_intermediate=self._validate_intermediate,
                validate_output=self._validate_output,
            ):
                yield Scene[T_ID, T_Frame](
                    inner=scene_df,
                    source_identifier=source_id
                    if self._id_mapping is None
                    else self._id_mapping(self._source_counter, source_id),
                    scene_number=self._count,
                    input_len=self.input_len(),
                    output_len=self.output_len(),
                )
                self._count += 1

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
    def load_raw(self, source: T_Source) -> Iterable[pl.DataFrame]:
        frames = [f.to_dict() for f in self.iter_frames(source)]

        if not frames:
            return []
        return [pl.DataFrame(frames)]

    @override
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame | None:
        return df.select([
            pl.col("frame").cast(pl.Int32),
            pl.col("track_id").cast(pl.Int32),
            pl.col("x").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("vx").cast(pl.Float64),
            pl.col("vy").cast(pl.Float64),
            pl.col("yaw").cast(pl.Float64),
            pl.col("agent_class").cast(pl.Int32),
        ])
