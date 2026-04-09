"""Persisted metadata models shared across storage backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, TypedDict

import numpy as np
import numpy.typing as npt

from dronalize._internal.typing import FloatScalarT
from dronalize.core.scene.model import derived_trajectory_fields

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.io.config import ExportConfig
    from dronalize.processing.loading.config import LoaderConfig


FORMAT_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"


class MapRecord(TypedDict, Generic[FloatScalarT]):
    """Simple persisted representation of one scene's map payload."""

    map_node_positions: npt.NDArray[FloatScalarT]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


class SceneRecord(TypedDict, Generic[FloatScalarT]):
    """Simple persisted representation of one scene."""

    scene_number: int
    position_offset: npt.NDArray[np.float64]
    agent_types: npt.NDArray[np.int32]
    features: npt.NDArray[FloatScalarT]
    mask: npt.NDArray[np.bool_]


SceneRecordF32 = SceneRecord[np.float32]
SceneRecordF64 = SceneRecord[np.float64]
AnySceneRecord = SceneRecord[np.float32] | SceneRecord[np.float64]
MapRecordF32 = MapRecord[np.float32]
MapRecordF64 = MapRecord[np.float64]
AnyMapRecord = MapRecord[np.float32] | MapRecord[np.float64]


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Format-agnostic metadata stored alongside exported datasets."""

    format_version: int
    source_trajectory_schema: str
    trajectory_schema: str
    derived_features: tuple[str, ...]
    feature_columns: tuple[str, ...]
    input_len: int
    output_len: int
    precision: str
    recenter_positions: bool
    has_map: bool
    sample_time: float
    original_sample_time: float

    @classmethod
    def from_configs(
        cls,
        *,
        loader_config: LoaderConfig,
        source_trajectory_schema: TrajectorySchema,
        export_config: ExportConfig,
        has_map: bool,
    ) -> DatasetManifest:
        """Create a manifest from resolved loader and export settings.

        Parameters
        ----------
        loader_config : LoaderConfig
            Effective loader configuration after all overrides have been
            applied.
        source_trajectory_schema : TrajectorySchema
            Schema produced natively by the dataset loader before optional
            conversion for storage.
        export_config : ExportConfig
            Effective export configuration used for persistence.
        has_map : bool
            Whether exported scenes are expected to carry map payloads.

        Returns
        -------
        DatasetManifest
            Manifest ready to serialize next to the exported dataset.
        """
        return cls(
            format_version=FORMAT_VERSION,
            source_trajectory_schema=source_trajectory_schema.name,
            trajectory_schema=export_config.trajectory_schema.name,
            derived_features=tuple(
                field.to_str()
                for field in derived_trajectory_fields(
                    source_trajectory_schema,
                    export_config.trajectory_schema,
                    sample_time=loader_config.post_sample_time,
                )
            ),
            feature_columns=export_config.feature_columns,
            input_len=loader_config.resampled_input_len,
            output_len=loader_config.resampled_output_len,
            precision=export_config.precision,
            recenter_positions=export_config.recenter_positions,
            has_map=has_map,
            sample_time=loader_config.post_sample_time,
            original_sample_time=loader_config.sample_time,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        """Create a manifest from previously serialized JSON data."""
        return cls(
            format_version=int(payload["format_version"]),
            source_trajectory_schema=str(
                payload.get("source_trajectory_schema", payload["trajectory_schema"])
            ),
            trajectory_schema=str(payload["trajectory_schema"]),
            derived_features=tuple(
                payload.get("derived_features", payload.get("derived_trajectory_fields", ()))
            ),
            feature_columns=tuple(payload["feature_columns"]),
            input_len=int(payload["input_len"]),
            output_len=int(payload["output_len"]),
            precision=str(payload["precision"]),
            recenter_positions=bool(payload["recenter_positions"]),
            has_map=bool(payload["has_map"]),
            sample_time=float(payload["sample_time"]),
            original_sample_time=float(payload["original_sample_time"]),
        )


def manifest_path(root: Path) -> Path:
    """Return the manifest path for one storage root."""
    return root / MANIFEST_FILENAME


def write_manifest(root: Path, manifest: DatasetManifest) -> None:
    """Write the storage manifest for one output root."""
    root.mkdir(parents=True, exist_ok=True)
    _ = manifest_path(root).write_text(
        json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8"
    )


def read_manifest(root: Path) -> DatasetManifest:
    """Read a previously written storage manifest."""
    payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return DatasetManifest.from_json_dict(payload)
