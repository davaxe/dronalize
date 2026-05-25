"""Framework-neutral scene-record containers and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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

    scene_number: int
    """Scene identifier within the exported dataset."""
    position_offset: npt.NDArray[np.float64]
    """Global 2D translation offset with shape `(2,)`."""

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

    dataset: str | None = None
    """Dataset label associated with this record, if known."""

    @property
    def horizon_frames(self) -> int:
        """Return the number of stored time steps."""
        return int(self.features.shape[1])

    def split(self, observation_length: int) -> SplitSceneRecord:
        """Split the full horizon into observation and prediction tensors."""
        return split_scene_record(self, observation_length=observation_length)


@dataclass(slots=True)
class SplitSceneRecord:
    """Convenience scene record with explicit observation/prediction tensors.

    This type is intended for online reader/adaptor use. It is not the canonical
    persisted representation.
    """

    scene_number: int
    """Scene identifier within the exported dataset."""
    position_offset: npt.NDArray[np.float64]
    """Global 2D translation offset applied to scene coordinates, shape `(2,)`."""

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

    dataset: str | None = None
    """Dataset label associated with this record, if known."""

    @property
    def observation_length(self) -> int:
        """Return the number of time steps in the observation tensors."""
        return int(self.history_features.shape[1])

    @property
    def future_length(self) -> int:
        """Return the number of time steps in the future tensors."""
        return int(self.future_features.shape[1])

    def join(self) -> SceneRecord:
        """Collapse the split tensors back into the full-horizon representation."""
        return join_split_scene_record(self)


def make_scene_record(
    *,
    scene_number: int,
    position_offset: npt.NDArray[np.float64],
    agent_types: npt.NDArray[np.int32],
    screened_agent_mask: npt.NDArray[np.bool_],
    features: npt.NDArray[np.float32 | np.float64],
    mask: npt.NDArray[np.bool_],
    map_node_positions: npt.NDArray[np.float32 | np.float64],
    map_edge_indices: npt.NDArray[np.int32],
    map_node_types: npt.NDArray[np.int32],
    map_edge_types: npt.NDArray[np.int32],
    dataset: str | None = None,
) -> SceneRecord:
    """Construct one canonical full-horizon `SceneRecord`."""
    return SceneRecord(
        scene_number=scene_number,
        position_offset=position_offset,
        agent_types=agent_types,
        screened_agent_mask=screened_agent_mask,
        features=features,
        mask=mask,
        map_node_positions=map_node_positions,
        map_edge_indices=map_edge_indices,
        map_node_types=map_node_types,
        map_edge_types=map_edge_types,
        dataset=dataset,
    )


def make_split_scene_record(
    *,
    scene_number: int,
    position_offset: npt.NDArray[np.float64],
    agent_types: npt.NDArray[np.int32],
    screened_agent_mask: npt.NDArray[np.bool_],
    history_features: npt.NDArray[np.float32 | np.float64],
    history_mask: npt.NDArray[np.bool_],
    future_features: npt.NDArray[np.float32 | np.float64],
    future_mask: npt.NDArray[np.bool_],
    map_node_positions: npt.NDArray[np.float32 | np.float64],
    map_edge_indices: npt.NDArray[np.int32],
    map_node_types: npt.NDArray[np.int32],
    map_edge_types: npt.NDArray[np.int32],
    dataset: str | None = None,
) -> SplitSceneRecord:
    """Construct one split convenience record."""
    return SplitSceneRecord(
        scene_number=scene_number,
        position_offset=position_offset,
        agent_types=agent_types,
        screened_agent_mask=screened_agent_mask,
        history_features=history_features,
        history_mask=history_mask,
        future_features=future_features,
        future_mask=future_mask,
        map_node_positions=map_node_positions,
        map_edge_indices=map_edge_indices,
        map_node_types=map_node_types,
        map_edge_types=map_edge_types,
        dataset=dataset,
    )


def split_scene_record(record: SceneRecord, *, observation_length: int) -> SplitSceneRecord:
    """Split one full-horizon scene record into observation and prediction tensors."""
    total_length = record.horizon_frames
    if observation_length < 0 or observation_length > total_length:
        msg = (
            f"`observation_length` must be between 0 and {total_length}, "
            f"but got {observation_length}."
        )
        raise ValueError(msg)

    return make_split_scene_record(
        scene_number=record.scene_number,
        position_offset=record.position_offset,
        agent_types=record.agent_types,
        screened_agent_mask=record.screened_agent_mask,
        history_features=record.features[:, :observation_length],
        history_mask=record.mask[:, :observation_length],
        future_features=record.features[:, observation_length:],
        future_mask=record.mask[:, observation_length:],
        map_node_positions=record.map_node_positions,
        map_edge_indices=record.map_edge_indices,
        map_node_types=record.map_node_types,
        map_edge_types=record.map_edge_types,
        dataset=record.dataset,
    )


def join_split_scene_record(record: SplitSceneRecord) -> SceneRecord:
    """Collapse one split convenience record into the full-horizon representation."""
    return make_scene_record(
        scene_number=record.scene_number,
        position_offset=record.position_offset,
        agent_types=record.agent_types,
        screened_agent_mask=record.screened_agent_mask,
        features=np.concatenate((record.history_features, record.future_features), axis=1),
        mask=np.concatenate((record.history_mask, record.future_mask), axis=1),
        map_node_positions=record.map_node_positions,
        map_edge_indices=record.map_edge_indices,
        map_node_types=record.map_node_types,
        map_edge_types=record.map_edge_types,
        dataset=record.dataset,
    )
