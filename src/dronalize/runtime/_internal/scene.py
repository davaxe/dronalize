"""Internal scene building for runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitAssignmentError
from dronalize.core.scene import Scene
from dronalize.processing.loading.assigner import StatelessWeightedAssigner
from dronalize.processing.loading.loader import PreparedSceneData, SceneIdentifier, Source
from dronalize.processing.screening.apply import AGENT_PASS_COLUMN

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.base import BaseSceneLoader, DatasetOptionsModel
    from dronalize.processing.maps.resolver import MapKey, MapResolver
    from dronalize.processing.models import SplitRequest
    from dronalize.processing.pipeline.pipeline import Pipeline
    from dronalize.runtime.types import ExecutionPlan


@dataclass(slots=True)
class SceneBuilder:
    """Runtime helper that turns loaded source frames into final scenes."""

    spec: DatasetSpec
    split_request: SplitRequest | None
    source_schema: TrajectorySchema
    target_schema: TrajectorySchema
    history_frames: int
    future_frames: int
    sample_time: float
    _pipeline: Pipeline | None = None
    _scene_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None
    _source_assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> SceneBuilder:
        """Build a runtime scene builder from one resolved run plan."""
        return cls(
            spec=plan.descriptor,
            split_request=plan.loader.split,
            source_schema=plan.descriptor.native_schema,
            target_schema=plan.output.trajectory_schema,
            history_frames=plan.effective_history_frames,
            future_frames=plan.effective_future_frames,
            sample_time=plan.effective_sample_time,
        )

    def __post_init__(self) -> None:
        split = self.split_request
        if split is None or not split.uses_weighted_assignment():
            return
        if split.strategy == "scene":
            self._scene_assigner = StatelessWeightedAssigner(
                split.active_splits(), split.active_weights(), seed=split.seed
            )
            return
        if split.strategy == "source":
            self._source_assigner = StatelessWeightedAssigner(
                split.active_splits(), split.active_weights(), seed=split.seed
            )

    def prepare_source(
        self, loader: BaseSceneLoader[Any, DatasetOptionsModel], source: Source[Any]
    ) -> Iterable[PreparedSceneData]:
        if self._pipeline is None:
            self._pipeline = loader.pipeline()

        source_local_scene_index = 0
        for data in loader.load_source(source):
            effective_split = data.predefined_split or source.predefined_split
            for processed_frame in self._pipeline.execute(
                data.frame, collect=True, filter_empty=True
            ):
                passed_agent_ids: frozenset[int] | None = None
                frame = processed_frame
                if AGENT_PASS_COLUMN in frame.columns:
                    passed_ids = (
                        frame.filter(frame[AGENT_PASS_COLUMN]).get_column("id").unique().to_list()
                    )
                    passed_agent_ids = frozenset(int(agent_id) for agent_id in passed_ids)
                    frame = frame.drop(AGENT_PASS_COLUMN)

                yield PreparedSceneData(
                    frame=frame,
                    map_binding=data.map_binding,
                    predefined_split=effective_split,
                    passed_agent_ids=passed_agent_ids,
                    stable_identifier=SceneIdentifier(
                        source_identifier=source.identifier,
                        source_local_scene_index=source_local_scene_index,
                    ),
                )
                source_local_scene_index += 1

    def create_scene(
        self,
        loader: BaseSceneLoader[Any, Any],
        data: PreparedSceneData,
        source: Source[Any],
        scene_number: int,
    ) -> Scene:
        map_key, map_resolver = self._resolve_scene_map(loader, source, data)
        scene = Scene.create(
            frame=data.frame,
            scene_number=scene_number,
            history_frames=self.history_frames,
            future_frames=self.future_frames,
            schema=self.source_schema,
            sample_time=self.sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            passed_agent_ids=data.passed_agent_ids,
            split_assignment=self._resolve_split_assignment(data, source),
        )
        if self.target_schema == self.source_schema:
            return scene
        return scene.as_schema(self.target_schema)

    def _resolve_split_assignment(
        self, data: PreparedSceneData, source: Source[Any]
    ) -> DatasetSplit | None:
        split = self.split_request
        if split is None:
            return None

        strategy = split.strategy
        if strategy in {"time", "shuffled-time"}:
            return self._resolve_time_split(data)
        if strategy == "scene":
            return self._resolve_scene_split(data)
        if strategy == "source":
            return self._resolve_source_split(source)
        if strategy == "native":
            return source.predefined_split

        return None

    @staticmethod
    def _resolve_time_split(data: PreparedSceneData) -> DatasetSplit:
        if "split" not in data.frame:
            msg = "Did not get split column in dataframe."
            raise SplitAssignmentError(msg)

        split_str = str(data.frame["split"].first()).lower()
        try:
            return DatasetSplit(split_str)
        except ValueError as exc:
            msg = f"Invalid split assignment '{split_str}' in dataframe."
            raise SplitAssignmentError(msg) from exc

    def _resolve_scene_split(self, data: PreparedSceneData) -> DatasetSplit | None:
        if self._scene_assigner is None:
            return None

        identifier = data.stable_identifier
        return self._scene_assigner.assign(
            identifier.source_local_scene_index, str(identifier.source_identifier)
        )

    def _resolve_source_split(self, source: Source[Any]) -> DatasetSplit | None:
        if self._source_assigner is None:
            return None
        return self._source_assigner.assign(str(source.identifier))

    @staticmethod
    def _resolve_scene_map(
        loader: BaseSceneLoader[Any, DatasetOptionsModel],
        source: Source[Any],
        data: PreparedSceneData,
    ) -> tuple[MapKey, MapResolver | None]:
        if loader.map_config is None:
            return None, None
        map_key = (
            data.map_binding.map_key if data.map_binding.map_key is not None else source.map_key
        )

        def _resolver(scene: Scene) -> MapGraph | None:
            return loader.resolve_map(scene, data.map_binding)

        return map_key, _resolver
