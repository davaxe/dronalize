"""Framework-neutral scene-record containers and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from prejectory.core.errors import MissingPredictionBoundsError

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class PredictionBounds:
    """Half-open prediction bounds for one stored scene horizon."""

    prediction_origin: int
    prediction_end: int

    def validate(self, *, horizon_frames: int) -> None:
        """Validate the bounds against a stored horizon."""
        if not 0 < self.prediction_origin < self.prediction_end <= horizon_frames:
            msg = (
                "Prediction bounds must satisfy "
                f"0 < origin < end <= {horizon_frames}, got "
                f"[{self.prediction_origin}, {self.prediction_end})."
            )
            raise ValueError(msg)


@dataclass(slots=True)
class SceneRecord:
    """Canonical persisted full-horizon scene record.

    A `SceneRecord` contains one contiguous trajectory horizon for all agents in
    a scene. It may carry prediction bounds but stores no duplicated split
    tensors; consumers derive those with [`SceneRecord.split`][].

    Conventions:

    - `N`: number of agents in the scene
    - `M`: number of map nodes
    - `E`: number of map edges
    - `T`: number of time steps in the stored scene horizon
    - `F`: number of per-timestep agent features
    """

    # Agent data
    agent_ids: npt.NDArray[np.int64]
    """Source agent identifier for each tensor row, shape `(N,)`."""
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
    prediction_origin: int | None = None
    """First prediction frame for this record, if a task was selected."""
    prediction_end: int | None = None
    """Exclusive prediction endpoint for this record, if a task was selected."""
    ego_agent_id: int | None = None
    """Optional agent ID of the ego vehicle, if known or applicable."""

    def __post_init__(self) -> None:
        """Validate record array shapes and core invariants."""
        _validate_full_record(self)

    @property
    def horizon_frames(self) -> int:
        """Number of stored time steps."""
        return int(self.features.shape[1])

    @property
    def prediction_bounds(self) -> PredictionBounds | None:
        """Persisted prediction bounds, if both are present."""
        if self.prediction_origin is None or self.prediction_end is None:
            return None
        return PredictionBounds(self.prediction_origin, self.prediction_end)

    def split(
        self,
        prediction_origin: int | None = None,
        prediction_end: int | None = None,
    ) -> SplitSceneRecord:
        """Create a forecast view using explicit or persisted prediction bounds."""
        if prediction_origin is None:
            bounds = self.prediction_bounds
            if bounds is None:
                raise MissingPredictionBoundsError(self.scene_number)
            if prediction_end is not None:
                msg = "`prediction_end` cannot be provided without `prediction_origin`."
                raise ValueError(msg)
        else:
            bounds = PredictionBounds(
                prediction_origin=prediction_origin,
                prediction_end=self.horizon_frames if prediction_end is None else prediction_end,
            )
        bounds.validate(horizon_frames=self.horizon_frames)

        return SplitSceneRecord(
            scene_number=self.scene_number,
            position_offset=self.position_offset,
            agent_ids=self.agent_ids,
            agent_types=self.agent_types,
            screened_agent_mask=self.screened_agent_mask,
            history_features=self.features[:, : bounds.prediction_origin],
            history_mask=self.mask[:, : bounds.prediction_origin],
            future_features=self.features[:, bounds.prediction_origin : bounds.prediction_end],
            future_mask=self.mask[:, bounds.prediction_origin : bounds.prediction_end],
            map_node_positions=self.map_node_positions,
            map_edge_indices=self.map_edge_indices,
            map_node_types=self.map_node_types,
            map_edge_types=self.map_edge_types,
            dataset_id=self.dataset_id,
            prediction_origin=bounds.prediction_origin,
            prediction_end=bounds.prediction_end,
            ego_agent_id=self.ego_agent_id,
        )


@dataclass(slots=True)
class SplitSceneRecord:
    """Convenience scene record with explicit observation/prediction tensors.

    This type is intended for online reader/adaptor use. It is not the canonical
    persisted representation.
    """

    # Agent data
    agent_ids: npt.NDArray[np.int64]
    """Source agent identifier for each tensor row, shape `(N,)`."""
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
    prediction_origin: int | None = None
    """First prediction frame used to create this view."""
    prediction_end: int | None = None
    """Exclusive prediction endpoint used to create this view."""
    ego_agent_id: int | None = None
    """Optional agent ID of the ego vehicle, if known or applicable."""

    def __post_init__(self) -> None:
        """Validate split-record array shapes and core invariants."""
        _validate_split_record(self)

    @property
    def observation_length(self) -> int:
        """Number of time steps in the observation tensors."""
        return int(self.history_features.shape[1])

    @property
    def future_length(self) -> int:
        """Number of time steps in the future tensors."""
        return int(self.future_features.shape[1])


def _validate_full_record(record: SceneRecord) -> None:
    if record.features.ndim != 3:
        msg = f"`features` must have shape (N, T, F), got {record.features.shape!r}."
        raise ValueError(msg)
    if record.mask.shape != record.features.shape[:2]:
        msg = f"`mask` must have shape {record.features.shape[:2]!r}, got {record.mask.shape!r}."
        raise ValueError(msg)
    _validate_optional_prediction_bounds(
        record.prediction_origin,
        record.prediction_end,
        horizon_frames=record.horizon_frames,
    )
    _validate_agent_arrays(
        num_agents=record.features.shape[0],
        agent_ids=record.agent_ids,
        agent_types=record.agent_types,
        screened_agent_mask=record.screened_agent_mask,
    )
    _validate_common_arrays(
        position_offset=record.position_offset,
        map_node_positions=record.map_node_positions,
        map_edge_indices=record.map_edge_indices,
        map_node_types=record.map_node_types,
        map_edge_types=record.map_edge_types,
    )


def _validate_optional_prediction_bounds(
    prediction_origin: int | None,
    prediction_end: int | None,
    *,
    horizon_frames: int,
) -> None:
    if (prediction_origin is None) != (prediction_end is None):
        msg = "`prediction_origin` and `prediction_end` must either both be set or both be absent."
        raise ValueError(msg)
    if prediction_origin is not None and prediction_end is not None:
        PredictionBounds(prediction_origin, prediction_end).validate(horizon_frames=horizon_frames)


def _validate_split_record(record: SplitSceneRecord) -> None:
    for name, features, mask in (
        ("history", record.history_features, record.history_mask),
        ("future", record.future_features, record.future_mask),
    ):
        if features.ndim != 3:
            msg = f"`{name}_features` must have shape (N, T, F), got {features.shape!r}."
            raise ValueError(msg)
        if mask.shape != features.shape[:2]:
            msg = f"`{name}_mask` must have shape {features.shape[:2]!r}, got {mask.shape!r}."
            raise ValueError(msg)
    if record.history_features.shape[0] != record.future_features.shape[0]:
        msg = "History and future tensors must contain the same number of agents."
        raise ValueError(msg)
    if record.history_features.shape[2] != record.future_features.shape[2]:
        msg = "History and future tensors must contain the same feature dimension."
        raise ValueError(msg)
    _validate_optional_prediction_bounds(
        record.prediction_origin,
        record.prediction_end,
        horizon_frames=record.observation_length + record.future_length,
    )
    _validate_agent_arrays(
        num_agents=record.history_features.shape[0],
        agent_ids=record.agent_ids,
        agent_types=record.agent_types,
        screened_agent_mask=record.screened_agent_mask,
    )
    _validate_common_arrays(
        position_offset=record.position_offset,
        map_node_positions=record.map_node_positions,
        map_edge_indices=record.map_edge_indices,
        map_node_types=record.map_node_types,
        map_edge_types=record.map_edge_types,
    )


def _validate_agent_arrays(
    *,
    num_agents: int,
    agent_ids: npt.NDArray[np.int64],
    agent_types: npt.NDArray[np.int32],
    screened_agent_mask: npt.NDArray[np.bool_],
) -> None:
    expected = (num_agents,)
    for name, array in (
        ("agent_ids", agent_ids),
        ("agent_types", agent_types),
        ("screened_agent_mask", screened_agent_mask),
    ):
        if array.shape != expected:
            msg = f"`{name}` must have shape {expected!r}, got {array.shape!r}."
            raise ValueError(msg)
    if np.unique(agent_ids).size != num_agents:
        msg = "`agent_ids` must contain one unique identifier per tensor row."
        raise ValueError(msg)


def _validate_common_arrays(
    *,
    position_offset: npt.NDArray[np.float64],
    map_node_positions: npt.NDArray[np.float32 | np.float64],
    map_edge_indices: npt.NDArray[np.int32],
    map_node_types: npt.NDArray[np.int32],
    map_edge_types: npt.NDArray[np.int32],
) -> None:
    if position_offset.shape != (2,) or not np.isfinite(position_offset).all():
        msg = f"`position_offset` must be a finite array with shape (2,), got {position_offset!r}."
        raise ValueError(msg)
    if map_node_positions.ndim != 2 or map_node_positions.shape[1:] != (2,):
        msg = f"`map_node_positions` must have shape (M, 2), got {map_node_positions.shape!r}."
        raise ValueError(msg)
    if map_edge_indices.ndim != 2 or map_edge_indices.shape[0] != 2:
        msg = f"`map_edge_indices` must have shape (2, E), got {map_edge_indices.shape!r}."
        raise ValueError(msg)
    if map_node_types.shape != (map_node_positions.shape[0],):
        msg = "`map_node_types` length must match the number of map nodes."
        raise ValueError(msg)
    if map_edge_types.shape != (map_edge_indices.shape[1],):
        msg = "`map_edge_types` length must match the number of map edges."
        raise ValueError(msg)
    if map_edge_indices.size:
        if map_node_positions.shape[0] == 0:
            msg = "Map edges cannot exist without map nodes."
            raise ValueError(msg)
        if (
            int(map_edge_indices.min()) < 0
            or int(map_edge_indices.max()) >= map_node_positions.shape[0]
        ):
            msg = "Map edge indices are outside the available map-node range."
            raise ValueError(msg)
