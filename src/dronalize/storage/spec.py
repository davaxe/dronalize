from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, TypedDict

import numpy as np
import numpy.typing as npt

from dronalize._internal._typing import FloatScalarT

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig


FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"


class MapSample(TypedDict, Generic[FloatScalarT]):
    """Logical persisted representation of one scene's map payload."""

    map_num_nodes: int
    map_num_edges: int
    map_node_positions: npt.NDArray[FloatScalarT]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


class SceneSample(TypedDict, Generic[FloatScalarT]):
    """Logical persisted representation of one scene sample."""

    scene_number: int
    global_origin: npt.NDArray[np.float64]
    num_agents: int
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
    scene_schema: str
    feature_columns: tuple[str, ...]
    input_len: int
    output_len: int
    precision: str
    offset_positions: bool
    has_map: bool

    @classmethod
    def from_configs(
        cls,
        *,
        loader_config: LoaderConfig,
        writer_config: WriterConfig,
        has_map: bool,
    ) -> StorageManifest:
        """Create a manifest from resolved loader and writer settings."""
        return cls(
            format_version=FORMAT_VERSION,
            scene_schema=writer_config.scene_schema.name,
            feature_columns=writer_config.feature_columns,
            input_len=loader_config.resampled_input_len,
            output_len=loader_config.resampled_output_len,
            precision=writer_config.precision,
            offset_positions=writer_config.offset_positions,
            has_map=has_map,
        )

    def json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the manifest."""
        return asdict(self)


def manifest_path(root: Path) -> Path:
    """Return the manifest file path for one storage root."""
    return root / MANIFEST_FILENAME


def write_manifest(root: Path, manifest: StorageManifest) -> None:
    """Write the storage manifest for one output root."""
    root.mkdir(parents=True, exist_ok=True)
    _ = manifest_path(root).write_text(json.dumps(manifest.json_dict(), indent=2), encoding="utf-8")


def read_manifest(root: Path) -> dict[str, Any]:
    """Read a previously written storage manifest."""
    return json.loads(manifest_path(root).read_text(encoding="utf-8"))
