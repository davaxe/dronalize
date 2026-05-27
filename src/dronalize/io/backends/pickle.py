"""Pickle writer backend.

This backend writes one pickled full-horizon `SceneRecord` per scene inside split
subdirectories such as `train` or `unsplit`. It is the simplest persisted
backend, requires no optional storage dependency, and is a practical default
when inspectability matters more than shard-based streaming.
"""

from __future__ import annotations

import functools
import pickle  # noqa: S403
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.io.base import (
    DatasetWriter,
    RecordTransform,
    SceneTransform,
    split_directory_name,
    validate_transform_choice,
)
from dronalize.io.encoding import encode_scene_record

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.runtime.types import OutputPlan


def _create_writer(
    identifier: int | None,
    *,
    output_dir: Path,
    config: OutputPlan,
    splits: Iterable[DatasetSplit] | None,
    record_transform: RecordTransform[object] | None,
    scene_transform: SceneTransform[object] | None,
) -> PickleWriter:
    return PickleWriter(
        output_dir=output_dir,
        config=config,
        splits=splits,
        identifier=identifier,
        record_transform=record_transform,
        scene_transform=scene_transform,
    )


class PickleWriter(DatasetWriter):
    """Write one pickled sample per scene.

    By default each file contains a full-horizon `SceneRecord`. Advanced callers
    may provide `record_transform` to persist a custom pickleable sample derived
    from the encoded record, or `scene_transform` to bypass record encoding and
    persist a sample derived directly from the runtime `Scene`.
    """

    def __init__(
        self,
        output_dir: Path,
        identifier: str | int | None = None,
        *,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None = None,
        record_transform: RecordTransform[object] | None = None,
        scene_transform: SceneTransform[object] | None = None,
    ) -> None:
        validate_transform_choice(
            record_transform=record_transform, scene_transform=scene_transform
        )
        self._base_output_dir: Path = output_dir
        self._config: OutputPlan = config
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)
        self._record_transform: RecordTransform[object] | None = record_transform
        self._scene_transform: SceneTransform[object] | None = scene_transform

        self._dir_map: dict[DatasetSplit | None, Path] = {
            split: output_dir / split_directory_name(split) for split in splits or [None]
        }
        for sub_dir in self._dir_map.values():
            sub_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    @override
    def as_factory(
        cls,
        output_dir: Path,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None = None,
        record_transform: RecordTransform[object] | None = None,
        scene_transform: SceneTransform[object] | None = None,
    ) -> Callable[[int | None], PickleWriter]:
        """Create a worker-local pickle writer factory."""
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            splits=splits,
            record_transform=record_transform,
            scene_transform=scene_transform,
        )

    @override
    def write(self, scene: Scene) -> None:
        """Encode one scene and persist the configured pickle sample."""
        output_dir = self._dir_map[scene.split_assignment]
        file_path = output_dir / f"{scene.scene_number:06d}.pkl"
        sample = self._make_sample(scene)
        with file_path.open("wb", buffering=1024 * 1024) as file:
            pickle.dump(sample, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _make_sample(self, scene: Scene) -> object:
        if self._scene_transform is not None:
            return self._scene_transform(scene)

        record = encode_scene_record(
            scene,
            dtype=self._config.precision(),
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._config.trajectory_schema,
            default_observation_length=self._config.default_observation_length,
        )
        if self._record_transform is None:
            return record
        return self._record_transform(record)

    @override
    def finish_local(self) -> None: ...

    @override
    def finish_final(self) -> None: ...
