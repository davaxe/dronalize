from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
)

import polars as pl

from preprocessing.common.trajectory_utils.convert import (
    convert_to_agent_data_dict,
)

if TYPE_CHECKING:
    from preprocessing.common.agent_data import AgentData
    from preprocessing.core.categories import AgentCategory
    from preprocessing.core.map_graph import MapGraph


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

    require_frames: Collection[int] | None = None
    """Specific frames offset required for the scene to be considered valid."""

    filter_agent_category: Collection[AgentCategory] | None = None
    """Set of agent categories to filter out from scenes."""

    def __post_init__(self) -> None:
        """Convert to set(s) to avoid duplicates."""
        if self.filter_agent_category is not None and not isinstance(
            self.filter_agent_category, set
        ):
            self.filter_agent_category = set(self.filter_agent_category)
        if self.require_frames is not None and not isinstance(self.require_frames, set):
            self.require_frames = set(self.require_frames)


@dataclass(slots=True)
class WindowParams:
    """Configuration for sliding window sampling of scenes."""

    window_size: int
    """Number of frames in each window."""

    step_size: int
    """Number of frames to skip between windows."""


@dataclass(slots=True)
class Resampling:
    """Configuration for resampling trajectories."""

    up: int
    """Upsampling factor."""
    down: int
    """Downsampling factor."""
    method: Literal["fast", "spline"] = "fast"
    """Method used for resampling."""

    def __post_init__(self) -> None:
        """Simplify the resampling ratio to its smallest integer ratio form."""
        self.up, self.down = Fraction(self.up, self.down).as_integer_ratio()

    @property
    def factors(self) -> tuple[int, int]:
        """(up, down) resampling factors."""
        return (self.up, self.down)


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

    resampling: Resampling | None = None
    """Resampling config if applicable."""

    window_params: WindowParams | None = None
    """Used for datasets where multiple samples can be generated from a single
    scene by using a sliding window approach. If None, it is assumed that each
    scene corresponds to exactly one sample."""

    scene_filtering: SceneFiltering | None = None
    """Configuration for filtering scenes based on pedestrian presence and validity."""

    def window_parameters(self, step_size: int, window_size: int | None = None) -> Self:
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
        require_frames: Collection[int] | None = None,
        filter_agent_category: Collection[AgentCategory] | None = None,
    ) -> Self:
        """Set the scene filtering parameters.

        The inputs are passed int `SceneFiltering` dataclass. Note that using
        this function support negative indices in `require_frames` where
        negative -1 indicate last frame etc.
        """
        if require_frames is not None:
            require_frames = {
                frame if frame > 0 else (self.input_len + self.output_len + frame)
                for frame in require_frames
            }

        self.scene_filtering = SceneFiltering(
            min_agents=min_agents,
            require_all_valid=require_all_valid,
            require_prediction_frame=require_prediction_frame,
            require_frames=require_frames,
            filter_agent_category=filter_agent_category,
        )
        return self

    def resampling_parameters(
        self,
        up: int,
        down: int,
        method: Literal["fast", "spline"] = "fast",
    ) -> Self:
        """Set the resampling parameters."""
        self.resampling = Resampling(up=up, down=down, method=method)
        return self


T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID", bound=(Hashable))


@dataclass(slots=True, frozen=True)
class Scene(Generic[T_ID]):
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to atleast contain all columns defined in FrameDict.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    identifier: T_ID
    """Identifier for the scene (e.g., file name, index, scene name/token)."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    map: MapGraph | None = None
    """Map graph associated with the scene. In some cases (e.g., Waymo), the map
    and trajectory are stored in the same file, and it can be useful to include
    the map graph in the scene data to avoid recomputing it for each scene."""
    map_information: str | None = None
    """Additional map information that can be used to determine what map corresponds to the scene."""

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

    def enforce_schema(self, schema: pl.Schema | None = None) -> Self:
        """Enforce the scene dataframe to follow a specified schema.

        This will select relevant columns and try to cast if needed/possible.
        If it is not possible to enforce schema, an error will be raised.

        Args:
            schema: schema to follow.

        Returns:
            Scene with enforced schema.

        """
        if schema is None:
            schema = Scene._base_schema()
        return replace(
            self,
            inner=self.inner.select([pl.col(name).cast(dtype) for name, dtype in schema.items()]),
        )

    @staticmethod
    def _base_schema() -> pl.Schema:
        return pl.Schema({
            "frame": pl.UInt32(),
            "id": pl.Int32(),
            "x": pl.Float32(),
            "y": pl.Float32(),
            "vx": pl.Float32(),
            "vy": pl.Float32(),
            "ax": pl.Float32(),
            "ay": pl.Float32(),
            "yaw": pl.Float32(),
            "agent_category": pl.Int32(),
        })


# TODO: Possibly rework how map and other metadata is attached to Scene.


class DataProcessor(ABC, Generic[T_ID, T_Source]):
    """Interface for processing raw data sources into a standardized format.

    Generic over the data scene source type and identifier type. Source type can
    be anything (e.g., file path, URL, database connection, raw data).
    Identifier type is used to unique identify each scene source (e.g., str,
    int).
    """

    def __init__(
        self,
        processor_config: ProcessorConfig | None = None,
        *,
        enforce_schema: bool = True,
    ) -> None:
        """Initialize internal state."""
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema = enforce_schema
        self._processor_config = processor_config or self.default_config()
        self._attach_scene_properties: dict[str, Any] = {}
        self._attach_scene_expr: dict[str, pl.Expr] = {}

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
    def input_len(self) -> int:
        """Observation length in frames (resulting value in Scene)."""
        up, down = (
            self.processor_config.resampling.factors if self.processor_config.resampling else (1, 1)
        )
        ratio = up / down
        return int((self.original_input_len - 1) * ratio + 1)

    @property
    def output_len(self) -> int:
        """Prediction length in frames (resulting value in Scene)."""
        up, down = (
            self.processor_config.resampling.factors if self.processor_config.resampling else (1, 1)
        )
        ratio = up / down
        total_len = int((self.sequence_length - 1) * ratio + 1)
        return total_len - self.input_len

    @property
    def post_sample_time(self) -> float:
        """Time interval between frames after resampling."""
        if self.processor_config.resampling is None:
            return self.processor_config.sample_time
        up, down = self.processor_config.resampling.factors
        ratio = up / down
        return self.processor_config.sample_time / ratio

    @abstractmethod
    def sources(self) -> Iterable[tuple[T_ID, T_Source]]:
        """Discover and yield identifiers for each scene to process."""

    @abstractmethod
    def load_raw(self, source: T_Source) -> Iterable[pl.LazyFrame]:
        """Read the raw data source into one or more DataFrame(s) (any schema)."""

    @abstractmethod
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert the raw DataFrame into the common schema."""

    @abstractmethod
    def default_config(self) -> ProcessorConfig:
        """Return the default processor configuration for this dataset."""

    @staticmethod
    def derivative_names() -> dict[int, list[str]]:
        """Return the names of the derivatives for velocity and acceleration."""
        return {
            1: ["vx", "vy"],
            2: ["ax", "ay"],
        }

    def attach_to_scene(
        self,
        select_expr: dict[str, pl.Expr] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Attach additional properties to the scene that can be used in `modify_scene`."""
        if properties is not None:
            self._attach_scene_properties.update(properties)
        if select_expr is not None:
            self._attach_scene_expr.update(select_expr)

    def process_next(self, source: T_Source) -> Iterable[pl.DataFrame]:
        """Process a single data item through the pipeline.

        Steps:
        1. Load raw data for all sources.
        2. Normalize to common schema.
        """

        def _step(raw_df: pl.LazyFrame) -> pl.DataFrame | None:
            normalized_df = self.normalize(raw_df)
            try:
                df = normalized_df.collect()
            except ValueError:
                # Error cases can be caused by:
                # 1. Empty scenes (due to filtering or missing data)
                return None
            return df if not df.is_empty() else None

        for raw_df in self.load_raw(source):
            df = _step(raw_df)
            if df is not None:
                yield df

    def scenes_iter(self) -> Iterable[Scene[T_ID]]:
        """Iterate over all sources and process them into scenes.

        This will lazy-load and process each source one at a time.
        """
        for source_id, source in self.sources():
            self._source_counter += 1
            for scene_df in self.process_next(source):
                yield self._create_scene(scene_df, source_id)

    def _create_scene(self, df: pl.DataFrame, source_id: T_ID) -> Scene[T_ID]:
        scene = Scene[T_ID](
            inner=df,
            identifier=source_id,
            scene_number=self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            **self._attach_scene_properties,
            **{key: df.select(expr).item() for key, expr in self._attach_scene_expr.items()},
        )
        self._attach_scene_properties.clear()
        self._count += 1
        return scene if not self._enforce_schema else scene.enforce_schema()
