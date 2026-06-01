"""Runtime DatasetSource processing and scene materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
        """Resolve the map for the given scene."""
        return self.loader.resolve_map(scene, self.map_binding)


class SplitAssigner:
    """Assign output splits to selected scene candidates."""

    def __init__(self, request: SplitAssignmentPlan | None) -> None:
        self.request: SplitAssignmentPlan | None = request
        self._weighted_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

        if request is not None and request.uses_weighted_assignment():
            self._weighted_assigner = StatelessWeightedAssigner(
                request.active_splits(), request.active_weights(), seed=request.seed
            )

    def assign(
        self, *, source: DatasetSource[Any], stable_identifier: SceneIdentifier, frame: pl.DataFrame
    ) -> DatasetSplit | None:
        """Assign split to scene candidate."""
        if self.request is None:
            return None

        match self.request.strategy:
            case "time" | "shuffled-time":
                return self._resolve_time_split(frame)
            case "scene":
                return self._resolve_scene_split(source, stable_identifier)
            case "source":
                return self._resolve_source_split(source)
            case "preserve-native":
                return source.predefined_split
            case _:
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
class RuntimeProcessor:
    """Own the full runtime DatasetSource-to-scene processing flow for one plan."""

    dataset: str
    loader: SceneLoader[Any, DatasetOptionsModel]
    source_schema: TrajectorySchema
    target_schema: TrajectorySchema
    horizon_frames: int
    sample_time: float
    split_assigner: SplitAssigner
    _pipeline: Pipeline | None = None

    @classmethod
    def from_plan(
        cls, plan: ExecutionPlan, loader: SceneLoader[Any, DatasetOptionsModel]
    ) -> RuntimeProcessor:
        """Create processor from execution plan and scene loader."""
        return cls(
            dataset=plan.dataset,
            loader=loader,
            source_schema=plan.descriptor.native_schema,
            target_schema=plan.output.trajectory_schema,
            horizon_frames=plan.effective_horizon_frames,
            sample_time=plan.effective_sample_time,
            split_assigner=SplitAssigner(plan.assignment),
        )

    def iter_sources(self) -> Iterable[DatasetSource[Any]]:
        """Iterate all DatasetSources to process."""
        yield from self.loader.iter_sources()

    def total_sources(self) -> int | None:
        """Return total number of sources if known."""
        return self.loader.count_sources()

    def screening_enabled(self) -> bool:
        """Return whether screening is enabled for this processor."""
        config = self.loader.screening_config
        return config is not None and bool(config.agent or config.scene)

    def iter_candidates(self, source: DatasetSource[Any]) -> Iterable[SceneCandidate]:
        """Iterate scene candidates for one DatasetSource."""
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

                split_assignment = None
                if passes_screening:
                    split_assignment = self.split_assigner.assign(
                        source=effective_source, stable_identifier=stable_identifier, frame=frame
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

    def materialize(self, candidate: SceneCandidate, scene_number: int) -> Scene:
        """Materialize one scene from candidate."""
        map_key, map_resolver = self._resolve_scene_map(candidate)

        scene = Scene.create(
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

    def _resolve_scene_map(self, candidate: SceneCandidate) -> tuple[MapKey, MapResolver | None]:
        if self.loader.map_config is None:
            return None, None

        map_key = candidate.map_binding.map_key or candidate.source.map_key
        return map_key, DeferredMapResolver(self.loader, candidate.map_binding)

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
