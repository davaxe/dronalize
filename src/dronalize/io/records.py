"""Framework-neutral scene-record containers and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(slots=True)
class SceneRecord:
    """Canonical scene-level record shared across all dataset reader backends.

    A `SceneRecord` contains all agent trajectories and map topology required
    to represent one training / validation / test scene after dataset-specific
    parsing and conversion into the common internal format.

    Conventions:

    - `N`: number of agents in the scene
    - `M`: number of map nodes
    - `E`: number of map edges
    - `T_in`: number of observed input time steps
    - `T_out`: number of predicted output time steps
    - `F`: number of per-timestep agent features

    Notes
    -----
    - Arrays are expected to be internally consistent in shape.
    - Masks use `True` for valid entries and `False` for invalid entries.
    - Invalid entries can arise from either missing DatasetSource data or padding to
        fixed-size tensors.
    """

    scene_number: int
    """Scene identifier within the DatasetSource dataset.

    This value is intended as a per-dataset scene key for indexing and tracing
    records back to their origin.

    Warning:
        The identifier is guaranteed to be unique only within a single dataset
        instance or export. It must not be treated as a globally unique or
        stable identifier across independent dataset exports or versions.
    """

    position_offset: npt.NDArray[np.float64]
    """Global 2D translation offset applied to scene coordinates.

    Shape:
        `(2,)`

    The offset is applied uniformly to all spatial quantities in the scene to
    place them in the canonical coordinate frame used by the reader pipeline.
    """

    # Agent data
    agent_types: npt.NDArray[np.int32]
    """Integer-encoded agent type for each agent.

    Shape:
        `(N,)`

    Each entry is an index into the dataset's agent-type vocabulary or the
    project's canonical agent-type mapping.
    """

    screened_agent_mask: npt.NDArray[np.bool_]
    """Mask indicating which agents passed dataset screening / validation.

    Shape:
        `(N,)`

    `True` indicates that the corresponding agent satisfied all validation
    rules applied during preprocessing. `False` indicates that the agent was
    retained in the record but flagged as invalid for downstream use that
    depends on quality-controlled trajectories.
    """

    history_features: npt.NDArray[np.float32 | np.float64]
    """Observed per-agent input features over the history horizon.

    Shape:
        `(N, T_in, F)`

    Contains the input trajectory/features for each agent over the observation
    interval. The feature dimension `F` is backend-dependent but must be
    consistent within the record.
    """

    history_mask: npt.NDArray[np.bool_]
    """Validity mask for `history_features`.

    Shape:
        `(N, T_in)`

    `True` marks a valid timestep for the corresponding agent. `False`
    marks an invalid timestep, typically due to missing measurements or padding
    introduced to obtain a fixed temporal length.
    """

    future_features: npt.NDArray[np.float32 | np.float64]
    """Target per-agent output features over the prediction horizon.

    Shape:
        `(N, T_out, F)`

    Contains the future trajectory/features associated with each agent over the
    prediction interval. The feature layout is expected to match that of
    `history_features` unless explicitly documented otherwise by the backend.
    """

    future_mask: npt.NDArray[np.bool_]
    """Validity mask for `future_features`.

    Shape:
        `(N, T_out)`

    `True` marks a valid future timestep for the corresponding agent.
    `False` marks an invalid timestep due to missing data or temporal padding.
    """

    # Map data
    map_node_positions: npt.NDArray[np.float32 | np.float64]
    """2D coordinates of map nodes in the scene.

    Shape:
        `(M, 2)`

    Each row stores the `(x, y)` position of one map node in the canonical
    scene coordinate frame.
    """

    map_edge_indices: npt.NDArray[np.int32]
    """Directed connectivity between map nodes.

    Shape:
        `(2, E)`

    Each column defines one edge as `[source_node, target_node]` using integer
    indices into `map_node_positions` / `map_node_types`.
    """

    map_node_types: npt.NDArray[np.int32]
    """Integer-encoded semantic type for each map node.

    Shape:
        `(M,)`

    Each entry is an index into the map-node type vocabulary used by the reader
    backend or the project's canonical map schema.
    """

    map_edge_types: npt.NDArray[np.int32]
    """Integer-encoded semantic type for each map edge.

    Shape:
        `(E,)`

    Each entry describes the type of the corresponding edge in
    `map_edge_indices` according to the edge-type encoding.
    """

    dataset: str | None = None
    """Dataset label associated with this record, if known."""

    def unsplit(self) -> FullHorizonSceneRecord:
        """Collapse the record into the unsplit representation."""
        return join_raw_scene_record(self)


@dataclass(slots=True)
class FullHorizonSceneRecord:
    """Canonical scene record before observation/prediction splitting.

    This is the storage-facing intermediary used by formats that store one
    contiguous trajectory tensor and split it into input/output horizons while
    reading. It follows the same conventions as `SceneRecord`, except that
    `features` has shape `(N, T, F)` and `mask` has shape `(N, T)`.
    """

    scene_number: int
    """Scene identifier within the DatasetSource dataset."""
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

    def split(self, observation_length: int) -> SceneRecord:
        """Split the record into observation and prediction tensors."""
        return split_unsplit_raw_scene_record(self, observation_length=observation_length)


def make_scene_record(
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
) -> SceneRecord:
    """Construct one canonical split `SceneRecord`."""
    return SceneRecord(
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


def make_unsplit_raw_scene_record(
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
) -> FullHorizonSceneRecord:
    """Construct one unsplit `FullHorizonSceneRecord`."""
    return FullHorizonSceneRecord(
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


def split_unsplit_raw_scene_record(
    record: FullHorizonSceneRecord, *, observation_length: int
) -> SceneRecord:
    """Split one unsplit scene record into observation and prediction tensors."""
    total_length = int(record.features.shape[1])
    if observation_length < 0 or observation_length > total_length:
        msg = (
            f"`observation_length` must be between 0 and {total_length}, "
            f"but got {observation_length}."
        )
        raise ValueError(msg)

    return make_scene_record(
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


def join_raw_scene_record(record: SceneRecord) -> FullHorizonSceneRecord:
    """Collapse one canonical record into the unsplit representation."""
    return make_unsplit_raw_scene_record(
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
