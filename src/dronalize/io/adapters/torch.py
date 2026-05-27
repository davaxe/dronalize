"""Torch dataset adapters built on top of framework-neutral scene readers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

from dronalize.core.optional import raise_missing_optional_dependency

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
    from typing_extensions import override
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(
        error, feature="The Torch scene dataset adapter", extra="torch"
    )


from dronalize.io.base import DatasetReader, IterableDatasetReader
from dronalize.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


ReaderT = TypeVar("ReaderT", bound=DatasetReader[SceneRecord])
IterableReaderT = TypeVar("IterableReaderT", bound=IterableDatasetReader[SceneRecord])
ObservationLength: TypeAlias = int | Callable[[SceneRecord], int]


@dataclass(slots=True)
class TorchSceneRecord:
    """Torch-backed equivalent of full-horizon `SceneRecord`."""

    scene_number: int
    """Scene identifier within the exported dataset."""
    dataset: str | None
    """Dataset label associated with this scene, if known."""
    default_observation_length: int | None
    """Default split point associated with this scene, if known."""
    position_offset: torch.Tensor
    """Global 2D translation offset with shape `(2,)`."""
    agent_types: torch.Tensor
    """Integer-encoded agent types with shape `(N,)`."""
    screened_agent_mask: torch.Tensor
    """Mask of agents that passed screening, shape `(N,)`."""
    features: torch.Tensor
    """Full-horizon agent features with shape `(N, T, F)`."""
    mask: torch.Tensor
    """Validity mask for `features`, shape `(N, T)`."""
    map_node_positions: torch.Tensor
    """2D map node coordinates with shape `(M, 2)`."""
    map_edge_indices: torch.Tensor
    """Directed map connectivity with shape `(2, E)`."""
    map_node_types: torch.Tensor
    """Integer-encoded map node types with shape `(M,)`."""
    map_edge_types: torch.Tensor
    """Integer-encoded map edge types with shape `(E,)`."""

    def split(self, observation_length: int) -> TorchSplitSceneRecord:
        """Split this full-horizon record into observation/prediction tensors."""
        total_length = int(self.features.size(1))
        if observation_length < 0 or observation_length > total_length:
            msg = (
                f"`observation_length` must be between 0 and {total_length}, "
                f"but got {observation_length}."
            )
            raise ValueError(msg)

        return TorchSplitSceneRecord(
            scene_number=self.scene_number,
            dataset=self.dataset,
            default_observation_length=self.default_observation_length,
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
        )


@dataclass(slots=True)
class TorchSplitSceneRecord:
    """Torch-backed equivalent of a split scene record."""

    scene_number: int
    dataset: str | None
    default_observation_length: int | None
    position_offset: torch.Tensor
    agent_types: torch.Tensor
    screened_agent_mask: torch.Tensor
    history_features: torch.Tensor
    history_mask: torch.Tensor
    future_features: torch.Tensor
    future_mask: torch.Tensor
    map_node_positions: torch.Tensor
    map_edge_indices: torch.Tensor
    map_node_types: torch.Tensor
    map_edge_types: torch.Tensor


class TorchSceneDataset(Dataset[TorchSceneRecord], Generic[ReaderT]):
    """Map-style Torch dataset over any `DatasetReader[SceneRecord]`."""

    def __init__(self, reader: ReaderT, *, copy: bool = True) -> None:
        super().__init__()
        self.reader: ReaderT = reader
        self._copy: bool = copy

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)

    def __iter__(self) -> Iterator[TorchSceneRecord]:
        """Iterate over full-horizon scene records converted to Torch tensors."""
        for record in self.reader:
            yield to_torch_scene_record(record, copy=self._copy)

    @override
    def __getitem__(self, index: int) -> TorchSceneRecord:
        """Return one full-horizon scene record converted to Torch tensors."""
        return to_torch_scene_record(self.reader[index], copy=self._copy)


class IterableTorchSceneDataset(IterableDataset[TorchSceneRecord], Generic[IterableReaderT]):
    """Iterable Torch dataset wrapper over any `DatasetReader[SceneRecord]`."""

    def __init__(self, reader: IterableReaderT, *, copy: bool = True) -> None:
        super().__init__()
        self.reader: IterableReaderT = reader
        self._copy: bool = copy

    @override
    def __iter__(self) -> Iterator[TorchSceneRecord]:
        """Iterate over full-horizon scene records converted to Torch tensors."""
        for record in self.reader:
            yield to_torch_scene_record(record, copy=self._copy)

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)


class TorchSplitSceneDataset(Dataset[TorchSplitSceneRecord], Generic[ReaderT]):
    """Map-style Torch dataset that splits full-horizon records on read."""

    def __init__(
        self,
        reader: ReaderT,
        *,
        observation_length: ObservationLength | None = None,
        copy: bool = True,
    ) -> None:
        super().__init__()
        self.reader: ReaderT = reader
        self.observation_length: ObservationLength | None = observation_length
        self._copy: bool = copy

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)

    def __iter__(self) -> Iterator[TorchSplitSceneRecord]:
        """Iterate over split scene records converted to Torch tensors."""
        for record in self.reader:
            observation_length = resolve_observation_length(self.observation_length, record)
            yield to_torch_scene_record(record, copy=self._copy).split(observation_length)

    @override
    def __getitem__(self, index: int) -> TorchSplitSceneRecord:
        """Return one split scene record converted to Torch tensors."""
        record = self.reader[index]
        observation_length = resolve_observation_length(self.observation_length, record)
        return to_torch_scene_record(record, copy=self._copy).split(observation_length)


class IterableTorchSplitSceneDataset(
    IterableDataset[TorchSplitSceneRecord], Generic[IterableReaderT]
):
    """Iterable Torch dataset wrapper that splits full-horizon records on read."""

    def __init__(
        self,
        reader: IterableReaderT,
        *,
        observation_length: ObservationLength | None = None,
        copy: bool = True,
    ) -> None:
        super().__init__()
        self.reader: IterableReaderT = reader
        self.observation_length: ObservationLength | None = observation_length
        self._copy: bool = copy

    @override
    def __iter__(self) -> Iterator[TorchSplitSceneRecord]:
        """Iterate over split scene records converted to Torch tensors."""
        for record in self.reader:
            observation_length = resolve_observation_length(self.observation_length, record)
            yield to_torch_scene_record(record, copy=self._copy).split(observation_length)

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)


def to_torch_scene_record(record: SceneRecord, *, copy: bool = True) -> TorchSceneRecord:
    """Convert a framework-neutral scene record into Torch tensors."""
    # Some readers can expose read-only NumPy views; copy once so Torch receives
    # writable, stable buffers without emitting warnings.
    return TorchSceneRecord(
        scene_number=record.scene_number,
        dataset=record.dataset,
        default_observation_length=record.default_observation_length,
        position_offset=torch.asarray(record.position_offset, copy=copy),
        agent_types=torch.asarray(record.agent_types, copy=copy),
        screened_agent_mask=torch.asarray(record.screened_agent_mask, copy=copy),
        features=torch.asarray(record.features, copy=copy),
        mask=torch.asarray(record.mask, copy=copy),
        map_node_positions=torch.asarray(record.map_node_positions, copy=copy),
        map_edge_indices=torch.asarray(record.map_edge_indices, copy=copy),
        map_node_types=torch.asarray(record.map_node_types, copy=copy),
        map_edge_types=torch.asarray(record.map_edge_types, copy=copy),
    )


def resolve_observation_length(
    observation_length: ObservationLength | None, record: SceneRecord
) -> int:
    """Resolve an explicit or record-local observation length for one sample."""
    if observation_length is None:
        if record.default_observation_length is None:
            msg = (
                "`observation_length` was not provided and the record does not "
                "define `default_observation_length`."
            )
            raise ValueError(msg)
        return record.default_observation_length
    if callable(observation_length):
        return int(observation_length(record))
    return int(observation_length)
