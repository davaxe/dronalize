from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class AgentData(TypedDict):
    """TypedDict for a single pedestrian data sample."""

    num_nodes: int
    """Number of agents/nodes in the scene."""
    ta_index: int
    """Primary target agent index (integer in range [0, num_nodes-1])."""
    type: npt.NDArray[np.int32]
    """Integer tensor of shape (num_nodes,) indicating the category/type of each agent."""
    inp_pos: npt.NDArray[np.float32]
    """Position in meters, shape (num_nodes, input_len, 2)."""
    inp_vel: npt.NDArray[np.float32]
    """Velocity in m/s, shape (num_nodes, input_len, 2)."""
    inp_acc: npt.NDArray[np.float32]
    """Acceleration in m/s^2, shape (num_nodes, input_len, 2)."""
    inp_yaw: npt.NDArray[np.float32]
    """Orientation in radians, shape (num_nodes, input_len)."""
    trg_pos: npt.NDArray[np.float32]
    """Position in meters, shape (num_nodes, output_len, 2)."""
    trg_vel: npt.NDArray[np.float32]
    """Velocity in m/s, shape (num_nodes, output_len, 2)."""
    trg_acc: npt.NDArray[np.float32]
    """Acceleration in m/s^2, shape (num_nodes, output_len, 2)."""
    trg_yaw: npt.NDArray[np.float32]
    """Orientation in radians, shape (num_nodes, output_len)."""
    input_mask: npt.NDArray[np.bool]
    """Boolean mask indicating valid input data, shape (num_nodes, input_len)."""
    valid_mask: npt.NDArray[np.bool]
    """Boolean mask indicating valid data across output, shape (num_nodes, output_len)."""
    ma_mask: npt.NDArray[np.bool]
    sa_mask: npt.NDArray[np.bool]
