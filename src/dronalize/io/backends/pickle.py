"""Pickle writer backend.

This backend writes one pickled `SceneRecord` per scene inside split
subdirectories such as `train` or `unsplit`. It is the simplest persisted
backend, requires no optional storage dependency, and is a practical default
when inspectability matters more than shard-based streaming.
"""

from __future__ import annotations

import functools
import pickle  # noqa: S403
from typing import TYPE_CHECKING, final

from typing_extensions import override

from dronalize.io.base import DatasetWriter, split_directory_name
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
) -> PickleWriter:
    return PickleWriter(output_dir=output_dir, config=config, splits=splits, identifier=identifier)


@final
class PickleWriter(DatasetWriter):
    """Write one pickled `SceneRecord` per scene."""

    def __init__(
        self,
        output_dir: Path,
        identifier: str | int | None = None,
        *,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None = None,
    ) -> None:
        self._base_output_dir: Path = output_dir
        self._config: OutputPlan = config
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)

        self._dir_map: dict[DatasetSplit | None, Path] = {
            split: output_dir / split_directory_name(split) for split in splits or [None]
        }
        for sub_dir in self._dir_map.values():
            sub_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    @override
    def as_factory(
        cls, output_dir: Path, config: OutputPlan, splits: Iterable[DatasetSplit] | None = None
    ) -> Callable[[int | None], PickleWriter]:
        """Create a worker-local pickle writer factory."""
        return functools.partial(
            _create_writer, output_dir=output_dir, config=config, splits=splits
        )

    @override
    def write(self, scene: Scene) -> None:
        """Encode one scene and persist it directly as a `SceneRecord` pickle."""
        output_dir = self._dir_map[scene.split_assignment]
        file_path = output_dir / f"{scene.scene_number:06d}.pkl"
        scene_data = encode_scene_record(
            scene,
            dtype=self._config.precision(),
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._config.trajectory_schema,
        )
        with file_path.open("wb", buffering=1024 * 1024) as file:
            pickle.dump(scene_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @override
    def finish_local(self) -> None: ...

    @override
    def finish_final(self) -> None: ...
