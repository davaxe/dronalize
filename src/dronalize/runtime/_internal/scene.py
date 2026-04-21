"""Internal scene building for runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitAssignmentError
from dronalize.core.scene import Scene
from dronalize.processing.loading.assigner import StatelessWeightedAssigner
from dronalize.processing.loading.loader import PreparedSceneData, SceneIdentifier, Source
from dronalize.processing.screening.apply import AGENT_PASS_COLUMN, SCENE_PASS_COLUMN

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import polars as pl

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.base import RuntimeSceneLoader
    from dronalize.processing.maps.resolver import MapKey, MapResolver
    from dronalize.processing.models import SplitRequest
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
    _assigner: StatelessWeightedAssigner[DatasetSplit] | None = None

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
        if split.strategy in {"scene", "source"}:
            self._assigner = StatelessWeightedAssigner(
                split.active_splits(), split.active_weights(), seed=split.seed
            )
            return

    def prepare_source(
        self,
        loader: RuntimeSceneLoader,
        source: Source[Any],
        *,
        record_candidate_scene: Callable[[], object] | None = None,
        claim_selected_scene: Callable[[], int | None] | None = None,
    ) -> Iterable[PreparedSceneData]:
        pipeline = loader.pipeline()

        # Normalize optional callbacks to avoid inline None checks
        record_candidate = record_candidate_scene or (lambda: None)

        def _fallback_claim() -> int:
            nonlocal fallback_scene_number
            val = fallback_scene_number
            fallback_scene_number += 1
            return val

        fallback_scene_number: int = 0
        claim_scene = claim_selected_scene or _fallback_claim
        source_local_scene_index: int = 0

        for data in loader.load_source(source):
            effective_split = data.predefined_split or source.predefined_split

            for frame_ in pipeline.execute(data.frame, collect=True, filter_empty=True):
                _ = record_candidate()
                passes, frame = self._scene_pass(frame_)
                if not passes:
                    source_local_scene_index += 1
                    continue

                # Request a scene number; if None is returned, the allocation limit is reached
                scene_number = claim_scene()
                if scene_number is None:
                    return

                passed_agent_ids, frame = self._agent_pass(frame)

                yield PreparedSceneData(
                    scene_number=scene_number,
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

    @staticmethod
    def _scene_pass(data: pl.DataFrame) -> tuple[bool, pl.DataFrame]:
        if SCENE_PASS_COLUMN not in data.columns:
            return True, data
        return bool(data.get_column(SCENE_PASS_COLUMN).first()), data.drop(SCENE_PASS_COLUMN)

    @staticmethod
    def _agent_pass(data: pl.DataFrame) -> tuple[bool, frozenset[int] | None, pl.DataFrame]:
        if AGENT_PASS_COLUMN not in data.columns:
            return None, data
        passed_ids = data.filter(data[AGENT_PASS_COLUMN]).get_column("id").unique().to_list()
        passed_agent_ids = frozenset(int(agent_id) for agent_id in passed_ids)
        return passed_agent_ids, data.drop(AGENT_PASS_COLUMN)

    def create_scene(
        self, loader: RuntimeSceneLoader, data: PreparedSceneData, source: Source[Any]
    ) -> Scene:
        map_key, map_resolver = self._resolve_scene_map(loader, source, data)
        scene = Scene.create(
            frame=data.frame,
            scene_number=data.scene_number,
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
        if self._assigner is None:
            return None

        identifier = data.stable_identifier
        return self._assigner.assign(
            identifier.source_local_scene_index, str(identifier.source_identifier)
        )

    def _resolve_source_split(self, source: Source[Any]) -> DatasetSplit | None:
        if self._assigner is None:
            return None
        return self._assigner.assign(str(source.identifier))

    @staticmethod
    def _resolve_scene_map(
        loader: RuntimeSceneLoader, source: Source[Any], data: PreparedSceneData
    ) -> tuple[MapKey, MapResolver | None]:
        if loader.map_config is None:
            return None, None
        map_key = (
            data.map_binding.map_key if data.map_binding.map_key is not None else source.map_key
        )

        def _resolver(scene: Scene) -> MapGraph | None:
            return loader.resolve_map(scene, data.map_binding)

        return map_key, _resolver
