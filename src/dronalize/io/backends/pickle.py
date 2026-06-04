"""Pickle writer backend.

This backend writes one pickled full-horizon `SceneRecord` per scene inside split
subdirectories such as `train` or `unsplit`. It is the simplest persisted
backend, requires no optional storage dependency, and is a practical default
when inspectability matters more than shard-based streaming.
"""

from __future__ import annotations

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
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.runtime.types import OutputPlan


class PickleWriter(DatasetWriter):
    """Write one pickled record per scene.

    By default each file contains a full-horizon `SceneRecord`. Advanced callers
    may provide `record_transform` to persist a custom pickleable payload derived
    from the encoded record, or `scene_transform` to bypass record encoding and
    persist a payload derived directly from the runtime `Scene`.
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

    @override
    def write(self, scene: Scene) -> None:
        """Encode one scene and persist the configured pickle payload."""
        output_dir = self._dir_map[scene.split_assignment]
        file_path = output_dir / f"{scene.scene_number:06d}.pkl"
        payload = self._make_payload(scene)
        with file_path.open("wb", buffering=1024 * 1024) as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _make_payload(self, scene: Scene) -> object:
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
