"""Runtime source processing and scene materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitAssignmentError
from dronalize.core.scene import Scene
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.loading.assigner import StatelessWeightedAssigner
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
from dronalize.processing.models import PipelinePlan, SplitRequest
from dronalize.processing.pipeline import spec as pipeline_spec
from dronalize.processing.screening.apply import AGENT_PASS_COLUMN, SCENE_PASS_COLUMN

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.base import BaseSceneLoader
    from dronalize.processing.loading.options import DatasetOptionsModel
    from dronalize.processing.maps.resolver import MapKey, MapResolver
    from dronalize.processing.pipeline.pipeline import Pipeline
    from dronalize.runtime.types import ExecutionPlan


@dataclass(frozen=True, slots=True)
class SceneIdentifier:
    """Stable identifier for one candidate scene within a source."""

    source_identifier: object
    source_local_scene_index: int


@dataclass(frozen=True, slots=True)
class SceneCandidate:
    """Runtime scene candidate produced from one processed source frame."""

    source: Source[Any]
    stable_identifier: SceneIdentifier
    frame: pl.DataFrame
    passes_screening: bool
    map_binding: MapBinding = field(default_factory=MapBinding)
    passed_agent_ids: frozenset[int] | None = None
    split_assignment: DatasetSplit | None = None


@dataclass(slots=True)
class SourcePlanner:
    """Resolve the effective source stream for one runtime plan."""

    loader: BaseSceneLoader[Any, DatasetOptionsModel]
    descriptor: DatasetSpec
    split_request: SplitRequest | None

    def iter_sources(self) -> Iterable[Source[Any]]:
        split = self.split_request
        if split is not None and split.strategy == "native":
            read = split.read or self.descriptor.native_splits
            for native_split in read or ():
                for source in self.loader.sources_for_split(native_split):
                    yield source.with_predefined_split(native_split)
            return

        if self.descriptor.native_splits:
            for native_split in self.descriptor.native_splits:
                for source in self.loader.sources_for_split(native_split):
                    yield source.with_predefined_split(native_split)
            return

        yield from self.loader.discover_sources()

    def total_sources(self) -> int | None:
        return self.loader.num_sources()


@dataclass(slots=True)
class SplitAssigner:
    """Assign output splits to selected scene candidates."""

    request: SplitRequest | None
    _weighted_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

    def __post_init__(self) -> None:
        split = self.request
        if split is None or not split.uses_weighted_assignment():
            return
        self._weighted_assigner = StatelessWeightedAssigner(
            split.active_splits(), split.active_weights(), seed=split.seed
        )

    def assign(
        self, *, source: Source[Any], stable_identifier: SceneIdentifier, frame: pl.DataFrame
    ) -> DatasetSplit | None:
        split = self.request
        if split is None:
            return None

        strategy = split.strategy
        if strategy in {"time", "shuffled-time"}:
            return self._resolve_time_split(frame)
        if strategy == "scene":
            return self._resolve_scene_split(source, stable_identifier)
        if strategy == "source":
            return self._resolve_source_split(source)
        if strategy == "native":
            return source.predefined_split
        return None

    @staticmethod
    def _resolve_time_split(frame: pl.DataFrame) -> DatasetSplit:
        if "split" not in frame:
            msg = "Did not get split column in dataframe."
            raise SplitAssignmentError(msg)

        split_str = str(frame["split"].first()).lower()
        try:
            return DatasetSplit(split_str)
        except ValueError as exc:
            msg = f"Invalid split assignment '{split_str}' in dataframe."
            raise SplitAssignmentError(msg) from exc

    def _resolve_scene_split(
        self, source: Source[Any], stable_identifier: SceneIdentifier
    ) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None
        return self._weighted_assigner.assign(
            stable_identifier.source_local_scene_index, str(source.identifier)
        )

    def _resolve_source_split(self, source: Source[Any]) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None
        return self._weighted_assigner.assign(str(source.identifier))


@dataclass(slots=True)
class SceneExtractor:
    """Extract runtime scene candidates from dataset sources."""

    loader: BaseSceneLoader[Any, DatasetOptionsModel]
    split_assigner: SplitAssigner
    _pipeline: Pipeline | None = None

    def screening_enabled(self) -> bool:
        config = self.loader.screening_config
        return config is not None and bool(config.agent or config.scene)

    def iter_candidates(self, source: Source[Any]) -> Iterable[SceneCandidate]:
        source_local_scene_index = 0
        for data in self.loader.load_source(source):
            effective_source = self._effective_source(source, data)
            for processed_frame in self._pipeline_for_loader().execute(
                data.frame, collect=True, filter_empty=True
            ):
                passes_screening, frame = self._extract_scene_pass(processed_frame)
                passed_agent_ids, frame = self._extract_agent_pass(frame)
                stable_identifier = SceneIdentifier(
                    source_identifier=effective_source.identifier,
                    source_local_scene_index=source_local_scene_index,
                )
                split_assignment = (
                    self.split_assigner.assign(
                        source=effective_source, stable_identifier=stable_identifier, frame=frame
                    )
                    if passes_screening
                    else None
                )
                yield SceneCandidate(
                    source=effective_source,
                    stable_identifier=stable_identifier,
                    frame=frame,
                    passes_screening=passes_screening,
                    map_binding=data.map_binding,
                    passed_agent_ids=passed_agent_ids,
                    split_assignment=split_assignment,
                )
                source_local_scene_index += 1

    def _pipeline_for_loader(self) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        plan = PipelinePlan(
            scenes=self.loader.scenes_config,
            screening=self.loader.screening_config,
            split=self.loader.split_config,
        )
        columns = TrajectoryColumns.from_schema(self.loader.native_trajectory_schema())
        self._pipeline = pipeline_spec.trajectory_pipeline(
            pipeline_spec.lane_change_sampling(plan, columns=columns)
            if self.loader.scenes_config.lane_change is not None
            else pipeline_spec.standard(plan, columns=columns)
        )
        return self._pipeline

    @staticmethod
    def _effective_source(source: Source[Any], data: LoadedSourceData) -> Source[Any]:
        if data.predefined_split is None:
            return source
        return source.with_predefined_split(data.predefined_split)

    @staticmethod
    def _extract_scene_pass(frame: pl.DataFrame) -> tuple[bool, pl.DataFrame]:
        if SCENE_PASS_COLUMN not in frame.columns:
            return True, frame
        passes = bool(frame.get_column(SCENE_PASS_COLUMN).first())
        return passes, frame.drop(SCENE_PASS_COLUMN)

    @staticmethod
    def _extract_agent_pass(frame: pl.DataFrame) -> tuple[frozenset[int] | None, pl.DataFrame]:
        if AGENT_PASS_COLUMN not in frame.columns:
            return None, frame
        passed_ids = frame.filter(frame[AGENT_PASS_COLUMN]).get_column("id").unique().to_list()
        passed_agent_ids = frozenset(int(agent_id) for agent_id in passed_ids)
        return passed_agent_ids, frame.drop(AGENT_PASS_COLUMN)


@dataclass(slots=True)
class SceneMaterializer:
    """Materialize selected scene candidates into final scene objects."""

    loader: BaseSceneLoader[Any, DatasetOptionsModel]
    source_schema: TrajectorySchema
    target_schema: TrajectorySchema
    history_frames: int
    future_frames: int
    sample_time: float

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        map_key, map_resolver = self._resolve_scene_map(candidate)
        scene = Scene.create(
            frame=candidate.frame,
            scene_number=scene_number,
            history_frames=self.history_frames,
            future_frames=self.future_frames,
            schema=self.source_schema,
            sample_time=self.sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            passed_agent_ids=candidate.passed_agent_ids,
            split_assignment=candidate.split_assignment,
        )
        if self.target_schema == self.source_schema:
            return scene
        return scene.as_schema(self.target_schema)

    def _resolve_scene_map(self, candidate: SceneCandidate) -> tuple[MapKey, MapResolver | None]:
        if self.loader.map_config is None:
            return None, None

        map_key = candidate.map_binding.map_key or candidate.source.map_key

        def _resolver(scene: Scene) -> MapGraph | None:
            return self.loader.resolve_map(scene, candidate.map_binding)

        return map_key, _resolver


@dataclass(slots=True)
class RuntimeProcessor:
    """Own the full runtime source-to-scene processing flow for one plan."""

    planner: SourcePlanner
    extractor: SceneExtractor
    materializer: SceneMaterializer

    @classmethod
    def from_plan(cls, plan: ExecutionPlan, loader: object) -> RuntimeProcessor:
        typed_loader = cast("BaseSceneLoader[Any, DatasetOptionsModel]", loader)
        split_assigner = SplitAssigner(plan.loader.split)
        return cls(
            planner=SourcePlanner(
                loader=typed_loader, descriptor=plan.descriptor, split_request=plan.loader.split
            ),
            extractor=SceneExtractor(loader=typed_loader, split_assigner=split_assigner),
            materializer=SceneMaterializer(
                loader=typed_loader,
                source_schema=plan.descriptor.native_schema,
                target_schema=plan.output.trajectory_schema,
                history_frames=plan.effective_history_frames,
                future_frames=plan.effective_future_frames,
                sample_time=plan.effective_sample_time,
            ),
        )

    def iter_sources(self) -> Iterable[Source[Any]]:
        yield from self.planner.iter_sources()

    def total_sources(self) -> int | None:
        return self.planner.total_sources()

    def screening_enabled(self) -> bool:
        return self.extractor.screening_enabled()

    def iter_candidates(self, source: Source[Any]) -> Iterable[SceneCandidate]:
        yield from self.extractor.iter_candidates(source)

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        return self.materializer.materialize(candidate, scene_number)
