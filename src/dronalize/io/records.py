"""Framework-neutral in-memory scene-record containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(slots=True)
class RawSceneRecord:
    """Raw scene record as loaded from persisted storage."""

    scene_number: int
    position_offset: npt.NDArray[np.float64]

    # Agent data
    agent_types: npt.NDArray[np.int32]
    input_features: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    input_mask: npt.NDArray[np.bool_]
    output_features: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    output_mask: npt.NDArray[np.bool_]

    # Map data
    map_node_positions: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    map_edge_indices: npt.NDArray[np.int32] | npt.NDArray[np.int64]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]
