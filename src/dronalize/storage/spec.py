from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, TypedDict

import numpy as np
import numpy.typing as npt

from dronalize._internal._typing import FloatScalarT
from dronalize.scene._scene import derived_scene_fields

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig
    from dronalize.scene import SceneSchema


FORMAT_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"


class MapSample(TypedDict, Generic[FloatScalarT]):
    """Simple persisted representation of one scene's map payload."""

    map_node_positions: npt.NDArray[FloatScalarT]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


class SceneSample(TypedDict, Generic[FloatScalarT]):
    """Simple persisted representation of one scene sample."""

    scene_number: int
    global_origin: npt.NDArray[np.float64]
    agent_types: npt.NDArray[np.int32]
    features: npt.NDArray[FloatScalarT]
    mask: npt.NDArray[np.bool_]


SceneSampleF32 = SceneSample[np.float32]
SceneSampleF64 = SceneSample[np.float64]
AnySceneSample = SceneSample[np.float32] | SceneSample[np.float64]
MapSampleF32 = MapSample[np.float32]
MapSampleF64 = MapSample[np.float64]
AnyMapSample = MapSample[np.float32] | MapSample[np.float64]


@dataclass(slots=True, frozen=True)
class StorageManifest:
    """Format-agnostic metadata stored alongside exported datasets."""

    format_version: int
    source_scene_schema: str
    scene_schema: str
    derived_features: tuple[str, ...]
    feature_columns: tuple[str, ...]
    input_len: int
    output_len: int
    precision: str
    offset_positions: bool
    has_map: bool
    sample_time: float
    original_sample_time: float

    @classmethod
    def from_configs(
        cls,
        *,
        loader_config: LoaderConfig,
        source_scene_schema: SceneSchema,
        writer_config: WriterConfig,
        has_map: bool,
    ) -> StorageManifest:
        """Create a manifest from resolved loader and writer settings."""
        return cls(
            format_version=FORMAT_VERSION,
            source_scene_schema=source_scene_schema.name,
            scene_schema=writer_config.scene_schema.name,
            derived_features=tuple(
                field.to_str()
                for field in derived_scene_fields(
                    source_scene_schema,
                    writer_config.scene_schema,
                    sample_time=loader_config.post_sample_time,
                )
            ),
            feature_columns=writer_config.feature_columns,
            input_len=loader_config.resampled_input_len,
            output_len=loader_config.resampled_output_len,
            precision=writer_config.precision,
            offset_positions=writer_config.offset_positions,
            has_map=has_map,
            sample_time=loader_config.post_sample_time,
            original_sample_time=loader_config.sample_time,
        )

    def json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the manifest."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> StorageManifest:
        """Create a manifest from JSON data."""
        return cls(
            format_version=int(payload["format_version"]),
            source_scene_schema=str(payload.get("source_scene_schema", payload["scene_schema"])),
            scene_schema=str(payload["scene_schema"]),
            derived_features=tuple(payload.get("derived_scene_fields", ())),
            feature_columns=tuple(payload["feature_columns"]),
            input_len=int(payload["input_len"]),
            output_len=int(payload["output_len"]),
            precision=str(payload["precision"]),
            offset_positions=bool(payload["offset_positions"]),
            has_map=bool(payload["has_map"]),
            sample_time=float(payload["sample_time"]),
            original_sample_time=float(payload.get("original_sample_time", payload["sample_time"])),
        )


def manifest_path(root: Path) -> Path:
    """Return the manifest file path for one storage root."""
    return root / MANIFEST_FILENAME


def write_manifest(root: Path, manifest: StorageManifest) -> None:
    """Write the storage manifest for one output root."""
    root.mkdir(parents=True, exist_ok=True)
    _ = manifest_path(root).write_text(json.dumps(manifest.json_dict(), indent=2), encoding="utf-8")


def read_manifest(root: Path) -> StorageManifest:
    """Read a previously written storage manifest."""
    payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return StorageManifest.from_json_dict(payload)
