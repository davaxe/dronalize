"""Framework-neutral scene-record containers and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(slots=True)
class SceneRecord:
    """Canonical persisted full-horizon scene record.

    A `SceneRecord` contains one contiguous trajectory horizon for all agents in
    a scene. It deliberately does not encode an observation/prediction split;
    consumers that need split tensors can derive a [`SplitSceneRecord`][] with
    [`SceneRecord.split`][].

    Conventions:

    - `N`: number of agents in the scene
    - `M`: number of map nodes
    - `E`: number of map edges
    - `T`: number of time steps in the stored scene horizon
    - `F`: number of per-timestep agent features
    """

    # Agent data
    agent_types: npt.NDArray[np.int32]
    """Integer-encoded agent type for each agent, shape `(N,)`."""
    screened_agent_mask: npt.NDArray[np.bool_]
    """Mask indicating which agents passed screening, shape `(N,)`."""
    features: npt.NDArray[np.float32 | np.float64]
    """Contiguous per-agent trajectory features, shape `(N, T, F)`."""
    mask: npt.NDArray[np.bool_]
    """Validity mask for `features`, shape `(N, T)`."""

    # Map data
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    """2D map node coordinates, shape `(M, 2)`."""
    map_edge_indices: npt.NDArray[np.int32]
    """Directed map connectivity, shape `(2, E)`."""
    map_node_types: npt.NDArray[np.int32]
    """Integer-encoded map node types, shape `(M,)`."""
    map_edge_types: npt.NDArray[np.int32]
    """Integer-encoded map edge types, shape `(E,)`."""

    # Metadata
    scene_number: int
    """Scene identifier within the exported dataset."""
    position_offset: npt.NDArray[np.float64]
    """Global 2D translation offset with shape `(2,)`."""
    dataset_id: int | None = None
    """Integer dataset identifier associated with this record, if known."""
    default_observation_length: int | None = None
    """Default split point for reader/adaptor convenience, if known."""
    ego_agent_id: int | None = None
    """Optional agent ID of the ego vehicle, if known or applicable."""

    @property
    def horizon_frames(self) -> int:
        """Return the number of stored time steps."""
        return int(self.features.shape[1])

    def split(self, observation_length: int) -> SplitSceneRecord:
        """Split the full horizon into observation and prediction tensors."""
        total_length = self.horizon_frames
        if observation_length < 0 or observation_length > total_length:
            msg = (
                f"`observation_length` must be between 0 and {total_length}, "
                f"but got {observation_length}."
            )
            raise ValueError(msg)

        return SplitSceneRecord(
            scene_number=self.scene_number,
            position_offset=self.position_offset,
            agent_types=self.agent_types,
            screened_agent_mask=self.screened_agent_mask,
            history_features=self.features[:, :observation_length],
            history_mask=self.mask[:, :observation_length],
            future_features=self.features[:, observation_length:],
            future_mask=self.mask[:, observation_length:],
            map_node_positions=self.map_node_positions,
            map_edge_indices=self.map_edge_indices,
            map_node_types=self.map_node_types,
            map_edge_types=self.map_edge_types,
            dataset_id=self.dataset_id,
        )


@dataclass(slots=True)
class SplitSceneRecord:
    """Convenience scene record with explicit observation/prediction tensors.

    This type is intended for online reader/adaptor use. It is not the canonical
    persisted representation.
    """

    # Agent data
    agent_types: npt.NDArray[np.int32]
    """Integer-encoded agent type for each agent, shape `(N,)`."""
    screened_agent_mask: npt.NDArray[np.bool_]
    """Mask indicating which agents passed screening, shape `(N,)`."""
    history_features: npt.NDArray[np.float32 | np.float64]
    """Observed per-agent input features, shape `(N, T_in, F)`."""
    history_mask: npt.NDArray[np.bool_]
    """Validity mask for `history_features`, shape `(N, T_in)`."""
    future_features: npt.NDArray[np.float32 | np.float64]
    """Prediction-target features, shape `(N, T_out, F)`."""
    future_mask: npt.NDArray[np.bool_]
    """Validity mask for `future_features`, shape `(N, T_out)`."""

    # Map data
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    """2D map node coordinates, shape `(M, 2)`."""
    map_edge_indices: npt.NDArray[np.int32]
    """Directed map connectivity, shape `(2, E)`."""
    map_node_types: npt.NDArray[np.int32]
    """Integer-encoded map node types, shape `(M,)`."""
    map_edge_types: npt.NDArray[np.int32]
    """Integer-encoded map edge types, shape `(E,)`."""

    # Metadata
    scene_number: int
    """Scene identifier within the exported dataset."""
    position_offset: npt.NDArray[np.float64]
    """Global 2D translation offset applied to scene coordinates, shape `(2,)`."""
    dataset_id: int | None = None
    """Integer dataset identifier associated with this record, if known."""
    ego_agent_id: int | None = None
    """Optional agent ID of the ego vehicle, if known or applicable."""

    @property
    def observation_length(self) -> int:
        """Return the number of time steps in the observation tensors."""
        return int(self.history_features.shape[1])

    @property
    def future_length(self) -> int:
        """Return the number of time steps in the future tensors."""
        return int(self.future_features.shape[1])
