"""MDS-specific scene-record serialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict

from dronalize.io.records import (
    SceneRecord,
    UnsplitSceneRecord,
    make_unsplit_raw_scene_record,
    split_unsplit_raw_scene_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class MDSSample(TypedDict):
    """Serialized MDS payload for one scene sample."""

    scene_number: int
    observation_length: int
    position_offset: npt.NDArray[np.float64]
    agent_types: npt.NDArray[np.int32]
    passed_agent_mask: npt.NDArray[np.uint8]
    features: npt.NDArray[np.float32 | np.float64]
    mask: npt.NDArray[np.uint8]
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


def encode_mds_sample(record: UnsplitSceneRecord, *, observation_length: int) -> MDSSample:
    """Convert one unsplit raw scene record to the MDS payload layout."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _encode_mds_map_arrays(
        record.map_node_positions,
        record.map_edge_indices,
        record.map_node_types,
        record.map_edge_types,
    )
    return {
        "scene_number": int(record.scene_number),
        "observation_length": int(observation_length),
        "position_offset": record.position_offset,
        "agent_types": record.agent_types,
        "passed_agent_mask": record.passed_agent_mask.astype(np.uint8, copy=False),
        "features": record.features,
        "mask": record.mask.astype(np.uint8, copy=False),
        "map_node_positions": map_node_positions,
        "map_edge_indices": map_edge_indices,
        "map_node_types": map_node_types,
        "map_edge_types": map_edge_types,
    }


def decode_mds_sample(sample: Mapping[str, Any]) -> SceneRecord:
    """Convert one MDS sample payload into the canonical raw scene record."""
    map_node_positions, map_edge_indices, map_node_types, map_edge_types = _decode_mds_map_arrays(
        np.asarray(sample["map_node_positions"]),
        np.asarray(sample["map_edge_indices"]),
        np.asarray(sample["map_node_types"]),
        np.asarray(sample["map_edge_types"]),
    )
    record = make_unsplit_raw_scene_record(
        scene_number=int(sample["scene_number"]),
        position_offset=np.asarray(sample["position_offset"], dtype=np.float64),
        agent_types=np.asarray(sample["agent_types"], dtype=np.int32),
        passed_agent_mask=np.asarray(sample["passed_agent_mask"], dtype=bool),
        features=np.asarray(sample["features"]),
        mask=np.asarray(sample["mask"], dtype=bool),
        map_node_positions=map_node_positions,
        map_edge_indices=map_edge_indices,
        map_node_types=map_node_types,
        map_edge_types=map_edge_types,
    )
    return split_unsplit_raw_scene_record(
        record, observation_length=int(sample["observation_length"])
    )


def mds_columns(dtype: str) -> dict[str, str]:
    """Return the MDS column schema for one serialized scene sample."""
    return {
        "scene_number": "int",
        "observation_length": "int",
        "position_offset": "ndarray:float64:2",
        "agent_types": "ndarray:int32",
        "passed_agent_mask": "ndarray:uint8",
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
    if map_node_positions.size != 0:
        return map_node_positions, map_edge_indices, map_node_types, map_edge_types

    return (
        np.full((1, 2), dtype=map_node_positions.dtype, fill_value=np.nan),
        np.full((2, 1), dtype=np.int32, fill_value=-1),
        np.full((1,), dtype=np.int32, fill_value=-1),
        np.full((1,), dtype=np.int32, fill_value=-1),
    )


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
    is_empty_map = (
        map_node_positions.shape == (1, 2)
        and np.isnan(map_node_positions).all()
        and map_edge_indices.shape == (2, 1)
        and (map_edge_indices == -1).all()
        and map_node_types.shape == (1,)
        and (map_node_types == -1).all()
        and map_edge_types.shape == (1,)
        and (map_edge_types == -1).all()
    )
    if not is_empty_map:
        return map_node_positions, map_edge_indices, map_node_types, map_edge_types

    return (
        np.empty((0, 2), dtype=map_node_positions.dtype),
        np.empty((2, 0), dtype=map_edge_indices.dtype),
        np.empty((0,), dtype=map_node_types.dtype),
        np.empty((0,), dtype=map_edge_types.dtype),
    )
