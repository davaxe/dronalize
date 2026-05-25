"""MDS-specific scene-record serialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict

from dronalize.io.records import SceneRecord, make_scene_record

if TYPE_CHECKING:
    from collections.abc import Mapping


class MDSSample(TypedDict):
    """Serialized MDS payload for one scene sample."""

    scene_number: int
    dataset: str
    position_offset: npt.NDArray[np.float64]
    agent_types: npt.NDArray[np.int32]
    screened_agent_mask: npt.NDArray[np.uint8]
    features: npt.NDArray[np.float32 | np.float64]
    mask: npt.NDArray[np.uint8]
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


def encode_mds_sample(record: SceneRecord) -> MDSSample:
    """Convert one scene record to the MDS payload layout."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _encode_mds_map_arrays(
        record.map_node_positions,
        record.map_edge_indices,
        record.map_node_types,
        record.map_edge_types,
    )
    return {
        "scene_number": int(record.scene_number),
        "dataset": record.dataset or "",
        "position_offset": record.position_offset,
        "agent_types": record.agent_types,
        "screened_agent_mask": record.screened_agent_mask.astype(np.uint8, copy=False),
        "features": record.features,
        "mask": record.mask.astype(np.uint8, copy=False),
        "map_node_positions": map_node_positions,
        "map_edge_indices": map_edge_indices,
        "map_node_types": map_node_types,
        "map_edge_types": map_edge_types,
    }


def decode_mds_sample(sample: Mapping[str, Any]) -> SceneRecord:
    """Convert one MDS sample payload into the canonical scene record."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _decode_mds_map_arrays(
        np.asarray(sample["map_node_positions"]),
        np.asarray(sample["map_edge_indices"]),
        np.asarray(sample["map_node_types"]),
        np.asarray(sample["map_edge_types"]),
    )
    return make_scene_record(
        scene_number=int(sample["scene_number"]),
        dataset=str(sample["dataset"]) if sample.get("dataset") else None,
        position_offset=np.asarray(sample["position_offset"], dtype=np.float64),
        agent_types=np.asarray(sample["agent_types"], dtype=np.int32),
        screened_agent_mask=np.asarray(sample["screened_agent_mask"], dtype=bool),
        features=np.asarray(sample["features"]),
        mask=np.asarray(sample["mask"], dtype=bool),
        map_node_positions=map_node_positions,
        map_edge_indices=map_edge_indices,
        map_node_types=map_node_types,
        map_edge_types=map_edge_types,
    )


def mds_columns(dtype: str) -> dict[str, str]:
    """Return the MDS column schema for one serialized scene sample."""
    return {
        "scene_number": "int",
        "dataset": "str",
        "position_offset": "ndarray:float64:2",
        "agent_types": "ndarray:int32",
        "screened_agent_mask": "ndarray:uint8",
        "features": f"ndarray:{dtype}",
        "mask": "ndarray:uint8",
        "map_node_positions": f"ndarray:{dtype}",
        "map_edge_indices": "ndarray:int32",
        "map_node_types": "ndarray:int32",
        "map_edge_types": "ndarray:int32",
    }


def _encode_mds_map_arrays(
    map_node_positions: npt.NDArray[np.float32 | np.float64],
    map_edge_indices: npt.NDArray[np.int32],
    map_node_types: npt.NDArray[np.int32],
    map_edge_types: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    if map_node_positions.size == 0:
        map_node_positions = np.full((1, 2), dtype=map_node_positions.dtype, fill_value=np.nan)
    if map_edge_indices.size == 0:
        map_edge_indices = np.full((2, 1), dtype=np.int32, fill_value=-1)
    if map_node_types.size == 0:
        map_node_types = np.full((1,), dtype=np.int32, fill_value=-1)
    if map_edge_types.size == 0:
        map_edge_types = np.full((1,), dtype=np.int32, fill_value=-1)
    return map_node_positions, map_edge_indices, map_node_types, map_edge_types


def _decode_mds_map_arrays(
    map_node_positions: npt.NDArray[Any],
    map_edge_indices: npt.NDArray[Any],
    map_node_types: npt.NDArray[Any],
    map_edge_types: npt.NDArray[Any],
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    if map_node_positions.shape == (1, 2) and np.isnan(map_node_positions).all():
        map_node_positions = np.empty((0, 2), dtype=map_node_positions.dtype)
    if map_edge_indices.shape == (2, 1) and (map_edge_indices == -1).all():
        map_edge_indices = np.empty((2, 0), dtype=map_edge_indices.dtype)
    if map_node_types.shape == (1,) and (map_node_types == -1).all():
        map_node_types = np.empty((0,), dtype=map_node_types.dtype)
    if map_edge_types.shape == (1,) and (map_edge_types == -1).all():
        map_edge_types = np.empty((0,), dtype=map_edge_types.dtype)
    return map_node_positions, map_edge_indices, map_node_types, map_edge_types
