"""Torch dataset adapters built on top of framework-neutral scene readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from dronalize.core.optional import raise_missing_optional_dependency

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
    from typing_extensions import override
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(
        error, feature="The Torch scene dataset adapter", extra="torch"
    )


from dronalize.io.base import DatasetReader
from dronalize.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


ReaderT = TypeVar("ReaderT", bound=DatasetReader[SceneRecord])


@dataclass(slots=True)
class TorchSceneRecord:
    """Torch-backed equivalent of `SceneRecord`.

    Shapes match the framework-neutral record: agent features use
    `(N, T_in, F)` and `(N, T_out, F)`, masks use `(N, T_in)` and
    `(N, T_out)`, map node positions use `(M, 2)`, and map edges use `(2, E)`.
    """

    scene_number: int
    """Scene identifier within the DatasetSource dataset."""
    position_offset: torch.Tensor
    """Global 2D translation offset with shape `(2,)`."""
    agent_types: torch.Tensor
    """Integer-encoded agent types with shape `(N,)`."""
    screened_agent_mask: torch.Tensor
    """Mask of agents that passed screening, shape `(N,)`."""
    history_features: torch.Tensor
    """Observed agent features with shape `(N, T_in, F)`."""
    history_mask: torch.Tensor
    """Validity mask for `history_features`, shape `(N, T_in)`."""
    future_features: torch.Tensor
    """Prediction target features with shape `(N, T_out, F)`."""
    future_mask: torch.Tensor
    """Validity mask for `future_features`, shape `(N, T_out)`."""
    map_node_positions: torch.Tensor
    """2D map node coordinates with shape `(M, 2)`."""
    map_edge_indices: torch.Tensor
    """Directed map connectivity with shape `(2, E)`."""
    map_node_types: torch.Tensor
    """Integer-encoded map node types with shape `(M,)`."""
    map_edge_types: torch.Tensor
    """Integer-encoded map edge types with shape `(E,)`."""


class TorchSceneDataset(Dataset[TorchSceneRecord], Generic[ReaderT]):
    """Map-style Torch dataset wrapper over any `DatasetReader[SceneRecord]`.

    The dataset yields `TorchSceneRecord` instances converted from the wrapped
    reader's canonical `SceneRecord` outputs.

    !!! note "Batching"
        The dataset format is not natively compatible with
        `torch.utils.data.DataLoader` batching, since the records have variable
        numbers of agents and map nodes.

        If PyG-based batching is desired, consider using
        `HeteroSceneDataset` instead.
    """

    def __init__(self, reader: ReaderT, *, copy: bool = True) -> None:
        self.reader: ReaderT = reader
        self._copy: bool = copy

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)

    def __iter__(self) -> Iterator[TorchSceneRecord]:
        """Iterate over scene records converted to Torch tensors."""
        for record in self.reader:
            yield to_torch_scene_record(record, copy=self._copy)

    @override
    def __getitem__(self, index: int) -> TorchSceneRecord:
        """Return one scene record converted to Torch tensors."""
        return to_torch_scene_record(self.reader[index], copy=self._copy)


def to_torch_scene_record(record: SceneRecord, *, copy: bool = True) -> TorchSceneRecord:
    """Convert a framework-neutral scene record into Torch tensors."""
    # Some readers can expose read-only NumPy views; copy once so Torch receives
    # writable, stable buffers without emitting warnings.
    return TorchSceneRecord(
        scene_number=record.scene_number,
        position_offset=torch.asarray(record.position_offset, copy=copy),
        agent_types=torch.asarray(record.agent_types, copy=copy),
        screened_agent_mask=torch.asarray(record.screened_agent_mask, copy=copy),
        history_features=torch.asarray(record.history_features, copy=copy),
        history_mask=torch.asarray(record.history_mask, copy=copy),
        future_features=torch.asarray(record.future_features, copy=copy),
        future_mask=torch.asarray(record.future_mask, copy=copy),
        map_node_positions=torch.asarray(record.map_node_positions, copy=copy),
        map_edge_indices=torch.asarray(record.map_edge_indices, copy=copy),
        map_node_types=torch.asarray(record.map_node_types, copy=copy),
        map_edge_types=torch.asarray(record.map_edge_types, copy=copy),
    )


class IterableTorchSceneDataset(TorchSceneDataset[ReaderT], IterableDataset[TorchSceneRecord]):
    """Iterable Torch dataset wrapper over any `DatasetReader[SceneRecord]`.

    !!! tip "Iterable vs. Map-style"
        This dataset is identical to `TorchSceneDataset`, but also inherits from
        `torch.utils.data.IterableDataset` to support iterable-style usage with
        `torch.utils.data.DataLoader` (e.g. for streaming readers that are design
        around sequential access).
    """
