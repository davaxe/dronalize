from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Hashable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeVar,
)

from typing_extensions import override

from preprocessing.core.datatypes.scene import Scene

if TYPE_CHECKING:
    import polars as pl

    from preprocessing.core.datatypes.categories import AgentCategory
    from preprocessing.core.datatypes.map_context import MapContext


@dataclass(slots=True)
class FilteringConfig:
    """Configuration for filtering scenes based on pedestrian presence and validity."""

    min_agents: int = 2
    """Minimum number of agents required in a scene to be valid."""

    require_all_valid: bool = False
    """If True, requires all agents in a scene to have valid positions for all
    time-steps (input and output)."""

    require_prediction_frame: bool = True
    """If True, requires all agents to have valid positions at the first prediction
    frame."""

    require_frames: Collection[int] | None = None
    """Specific frames offset required for the scene to be considered valid."""

    filter_agent_category: Collection[AgentCategory] | None = None
    """Set of agent categories to filter out from scenes."""

    filter_slow_agents: float | None = None
    """Filter out agents with an average speed below this threshold."""

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

    @property
    def no_resampling(self) -> bool:
        """Whether no resampling is applied."""
        return self.up == 1 and self.down == 1


@dataclass(slots=True)
class LoaderConfig:
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

    scene_filtering: FilteringConfig | None = None
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
        filter_slow_agents: float | None = None,
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

        self.scene_filtering = FilteringConfig(
            min_agents=min_agents,
            require_all_valid=require_all_valid,
            require_prediction_frame=require_prediction_frame,
            require_frames=require_frames,
            filter_agent_category=filter_agent_category,
            filter_slow_agents=filter_slow_agents,
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
T_Source_co = TypeVar("T_Source_co", covariant=True)
T_ID = TypeVar("T_ID", bound=Hashable)
P = ParamSpec("P")


@dataclass(slots=True, frozen=True)
class Source(Generic[T_ID, T_Source]):
    """Represents a raw data source for a scene, identified by a unique identifier."""

    identifier: T_ID
    """Generic identifier for the source, e.g., file name, URL, database key."""
    inner: T_Source
    """The actual source data, which can be of any type (e.g., file path, raw data)."""
    map_context: MapContext | None = None
    """Optional map context associated with the source.

    This can be useful if all scenes generated from this source share the same map context.
    """
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the source."""


class SceneLoader(Protocol, Generic[T_ID]):
    """Minimal protocol for scene loading, used for type hinting."""

    def scenes(self) -> Iterable[Scene[T_ID]]:
        """Yield processed scenes one at a time."""
        ...

    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[T_ID], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Process scenes and call the provided callback on each scene.

        This is an alternative to `scenes()` that allows for more flexible
        processing of scenes without needing to yield them. The callback will be
        called with each processed scene, allowing for custom handling (e.g.,
        saving to disk, feeding into a model) without needing to store all scenes
        in memory at once.

        Args:
            callback: A function that takes a Scene and additional arguments, and
                processes it (e.g., saves to disk, feeds into a model).
            *args: Additional positional arguments to pass to the callback.
            **kwargs: Additional keyword arguments to pass to the callback.

        """
        ...


class BaseSceneLoader(ABC, SceneLoader[T_ID], Generic[T_ID, T_Source]):
    """ABC interface for processing raw data sources into a standardized format.

    Generic over the data scene source type and identifier type. Source type can
    be anything (e.g., file path, URL, database connection, raw data).
    Identifier type is used to unique identify each scene source (e.g., str,
    int).
    """

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        *,
        enforce_schema: bool = True,
    ) -> None:
        """Initialize internal state."""
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema = enforce_schema
        self._loader_config = loader_config or self.default_config()
        self._attach_scene_properties: dict[str, Any] = {}
        self._attach_scene_expr: dict[str, pl.Expr] = {}

    # --- Abstract Steps (The "Blanks" to fill) ---

    @abstractmethod
    def sources(self) -> Iterable[Source[T_ID, T_Source]]:
        """Discover and yield identifiers for each scene to process."""

    @abstractmethod
    def load_raw(self, source: Source[T_ID, T_Source]) -> Iterable[tuple[pl.LazyFrame, MapContext]]:
        """Read the raw data source into one or more DataFrame(s) (any schema)."""

    @abstractmethod
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert the raw DataFrame into the common schema."""

    @abstractmethod
    def default_config(self) -> LoaderConfig:
        """Return the default processor configuration for this dataset."""

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[T_ID], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

    def num_scenes(self) -> int | None:
        """Get the total number of scenes that will be processed.

        In some cases this can be expensive to compute or not known in advanced, in that case `None`
        is returned.
        """
        _self = self  # To satisfy Ruff, since `self` will most likely be used in subclasses.
        return None

    def num_sources(self) -> int | None:
        """Get the total number of sources that will be processed.

        This is different from `number_of_scenes()` since each source can potentially generate
        multiple scenes (e.g., by using sliding window sampling). In some cases this can be
        expensive to compute or not known in advanced, in that case `None` is returned.

        This could trivially be implemented as `len(list(self.sources()))`, but that would require
        loading all sources into memory which can be expensive for large datasets.
        """
        _self = self  # To satisfy Ruff, since `self` will most likely be used in subclasses.
        return None

    def process_next(
        self, source: Source[T_ID, T_Source]
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
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

        for raw_df, map_context in self.load_raw(source):
            df = _step(raw_df)
            if df is not None:
                yield df, map_context

    def scenes(self) -> Iterable[Scene[T_ID]]:
        """Iterate over all sources and process them into scenes.

        This will lazy-load and process each source one at a time.
        """
        for source in self.sources():
            self._source_counter += 1
            for scene_df, map_context in self.process_next(source):
                yield self.create_scene(scene_df, source.identifier, map_context)

    def create_scene(
        self, df: pl.DataFrame, source_id: T_ID, map_context: MapContext
    ) -> Scene[T_ID]:
        """Create a Scene object from the processed DataFrame and source identifier.

        This method also calls `Scene.enforce_schema()` if `self._enforce_schema` is True to ensure
        the scene follows the expected schema. If overriding this method, make sure to follow
        the expected behavior regarding schema enforcement.

        Args:
            df: Processed DataFrame for the scene, expected to follow the common schema.
            source_id: Identifier for the scene source (e.g., file name, index).
            map_context: Map context associated with the scene.

        """
        scene = Scene(
            inner=df,
            identifier=source_id,
            scene_number=self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            map_context=map_context,
        )
        self._attach_scene_properties.clear()
        self._count += 1
        return scene if not self._enforce_schema else scene.enforce_schema()

    @staticmethod
    def derivative_names() -> dict[int, list[str]]:
        """Return the names of the derivatives for velocity and acceleration."""
        return {
            1: ["vx", "vy"],
            2: ["ax", "ay"],
        }

    @property
    def loader_config(self) -> LoaderConfig:
        """Return the loader configuration."""
        return self._loader_config

    @property
    def original_input_len(self) -> int:
        """Original observation length in frames (before resampling)."""
        return self.loader_config.input_len

    @property
    def original_output_len(self) -> int:
        """Original prediction length in frames (before resampling)."""
        return self.loader_config.output_len

    @property
    def sequence_length(self) -> int:
        """Total sequence length (observation + prediction) in frames."""
        return self.loader_config.input_len + self.loader_config.output_len

    @property
    def input_len(self) -> int:
        """Observation length in frames (resulting value in Scene)."""
        up, down = (
            self.loader_config.resampling.factors if self.loader_config.resampling else (1, 1)
        )
        ratio = up / down
        return int((self.original_input_len - 1) * ratio + 1)

    @property
    def output_len(self) -> int:
        """Prediction length in frames (resulting value in Scene)."""
        up, down = (
            self.loader_config.resampling.factors if self.loader_config.resampling else (1, 1)
        )
        ratio = up / down
        total_len = int((self.sequence_length - 1) * ratio + 1)
        return total_len - self.input_len

    @property
    def post_sample_time(self) -> float:
        """Time interval between frames after resampling."""
        if self.loader_config.resampling is None:
            return self.loader_config.sample_time
        up, down = self.loader_config.resampling.factors
        ratio = up / down
        return self.loader_config.sample_time / ratio
