# ruff: noqa: D102,D105
"""Runtime DatasetSource processing and scene materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitAssignmentError
from dronalize.core.scene import Scene
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.loading.assigner import StatelessWeightedAssigner
from dronalize.processing.loading.models import DatasetSource, LoadedSourceFrame, MapReference
from dronalize.processing.models import SplitAssignmentPlan, TrajectoryPipelinePlan
from dronalize.processing.pipeline.trajectory import build_trajectory_pipeline
from dronalize.processing.screening.screen import (
    AGENT_SCREENING_PASS_COLUMN,
    SCENE_SCREENING_PASS_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.base import SceneLoader
    from dronalize.processing.loading.models import DatasetOptionsModel
    from dronalize.processing.maps import MapKey, MapResolver
    from dronalize.processing.pipeline.pipeline import Pipeline
    from dronalize.runtime.types import ExecutionPlan


@dataclass(frozen=True, slots=True)
class SceneIdentifier:
    """Stable identifier for one candidate scene within a DatasetSource."""

    source_identifier: object
    source_local_scene_index: int


@dataclass(frozen=True, slots=True)
class SceneCandidate:
    """Runtime scene candidate produced from one processed DatasetSource frame."""

    source: DatasetSource[Any]
    stable_identifier: SceneIdentifier
    frame: pl.DataFrame
    passes_screening: bool
    map_binding: MapReference = field(default_factory=MapReference)
    passed_agent_ids: frozenset[int] | None = None
    split_assignment: DatasetSplit | None = None


@dataclass(frozen=True, slots=True)
class DeferredMapResolver:
    """Picklable scene map resolver for deferred map materialization."""

    loader: SceneLoader[Any, DatasetOptionsModel]
    map_binding: MapReference | None = None

    def __call__(self, scene: Scene) -> MapGraph | None:
        return self.loader.resolve_map(scene, self.map_binding)


@dataclass(slots=True)
class SourcePlanner:
    """Resolve the effective DatasetSource stream for one runtime plan."""

    loader: SceneLoader[Any, DatasetOptionsModel]

    def iter_sources(self) -> Iterable[DatasetSource[Any]]:
        yield from self.loader.iter_sources()

    def total_sources(self) -> int | None:
        return self.loader.count_sources()


@dataclass(slots=True)
class SplitAssigner:
    """Assign output splits to selected scene candidates."""

    request: SplitAssignmentPlan | None
    _weighted_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

    def __post_init__(self) -> None:
        split = self.request
        if split is None or not split.uses_weighted_assignment():
            return
        self._weighted_assigner = StatelessWeightedAssigner(
            split.active_splits(), split.active_weights(), seed=split.seed
        )

    def assign(
        self, *, source: DatasetSource[Any], stable_identifier: SceneIdentifier, frame: pl.DataFrame
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
        if strategy == "preserve-native":
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
        self, source: DatasetSource[Any], stable_identifier: SceneIdentifier
    ) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None
        return self._weighted_assigner.assign(
            stable_identifier.source_local_scene_index, str(source.identifier)
        )

    def _resolve_source_split(self, source: DatasetSource[Any]) -> DatasetSplit | None:
        if self._weighted_assigner is None:
            return None
        return self._weighted_assigner.assign(str(source.identifier))


@dataclass(slots=True)
class SceneExtractor:
    """Extract runtime scene candidates from dataset sources."""

    loader: SceneLoader[Any, DatasetOptionsModel]
    split_assigner: SplitAssigner
    _pipeline: Pipeline | None = None

    def screening_enabled(self) -> bool:
        config = self.loader.screening_config
        return config is not None and bool(config.agent or config.scene)

    def iter_candidates(self, source: DatasetSource[Any]) -> Iterable[SceneCandidate]:
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

        plan = TrajectoryPipelinePlan(
            scenes=self.loader.scenes_config,
            screening=self.loader.screening_config,
            assignment=self.split_assigner.request,
        )
        columns = TrajectoryColumns.from_schema(self.loader.native_trajectory_schema())
        self._pipeline = build_trajectory_pipeline(plan, columns=columns)
        return self._pipeline

    @staticmethod
    def _effective_source(
        source: DatasetSource[Any], data: LoadedSourceFrame
    ) -> DatasetSource[Any]:
        if data.predefined_split is None:
            return source
        return source.with_predefined_split(data.predefined_split)

    @staticmethod
    def _extract_scene_pass(frame: pl.DataFrame) -> tuple[bool, pl.DataFrame]:
        if SCENE_SCREENING_PASS_COLUMN not in frame.columns:
            return True, frame
        passes = bool(frame.get_column(SCENE_SCREENING_PASS_COLUMN).first())
        return passes, frame.drop(SCENE_SCREENING_PASS_COLUMN)

    @staticmethod
    def _extract_agent_pass(frame: pl.DataFrame) -> tuple[frozenset[int] | None, pl.DataFrame]:
        if AGENT_SCREENING_PASS_COLUMN not in frame.columns:
            return None, frame
        passed_ids = (
            frame.filter(frame[AGENT_SCREENING_PASS_COLUMN]).get_column("id").unique().to_list()
        )
        passed_agent_ids = frozenset(int(agent_id) for agent_id in passed_ids)
        return passed_agent_ids, frame.drop(AGENT_SCREENING_PASS_COLUMN)


@dataclass(slots=True)
class SceneMaterializer:
    """Materialize selected scene candidates into final scene objects."""

    dataset: str
    loader: SceneLoader[Any, DatasetOptionsModel]
    source_schema: TrajectorySchema
    target_schema: TrajectorySchema
    horizon_frames: int
    sample_time: float

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        map_key, map_resolver = self._resolve_scene_map(candidate)
        scene: Scene = Scene.create(
            frame=candidate.frame,
            scene_number=scene_number,
            horizon_frames=self.horizon_frames,
            schema=self.source_schema,
            sample_time=self.sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            passed_agent_ids=candidate.passed_agent_ids,
            split_assignment=candidate.split_assignment,
            dataset=self.dataset,
        )
        if self.target_schema == self.source_schema:
            return scene
        return scene.as_schema(self.target_schema)

    def _resolve_scene_map(self, candidate: SceneCandidate) -> tuple[MapKey, MapResolver | None]:
        if self.loader.map_config is None:
            return None, None

        map_key = candidate.map_binding.map_key or candidate.source.map_key
        return map_key, DeferredMapResolver(self.loader, candidate.map_binding)


@dataclass(slots=True)
class RuntimeProcessor:
    """Own the full runtime DatasetSource-to-scene processing flow for one plan."""

    planner: SourcePlanner
    extractor: SceneExtractor
    materializer: SceneMaterializer

    @classmethod
    def from_plan(cls, plan: ExecutionPlan, loader: object) -> RuntimeProcessor:
        typed_loader = cast("SceneLoader[Any, DatasetOptionsModel]", loader)
        split_assigner = SplitAssigner(plan.assignment)
        return cls(
            planner=SourcePlanner(loader=typed_loader),
            extractor=SceneExtractor(loader=typed_loader, split_assigner=split_assigner),
            materializer=SceneMaterializer(
                dataset=plan.dataset,
                loader=typed_loader,
                source_schema=plan.descriptor.native_schema,
                target_schema=plan.output.trajectory_schema,
                horizon_frames=plan.effective_horizon_frames,
                sample_time=plan.effective_sample_time,
            ),
        )

    def iter_sources(self) -> Iterable[DatasetSource[Any]]:
        yield from self.planner.iter_sources()

    def total_sources(self) -> int | None:
        return self.planner.total_sources()

    def screening_enabled(self) -> bool:
        return self.extractor.screening_enabled()

    def iter_candidates(self, source: DatasetSource[Any]) -> Iterable[SceneCandidate]:
        yield from self.extractor.iter_candidates(source)

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        return self.materializer.materialize(candidate, scene_number)
