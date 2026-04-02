"""Lightweight in-memory sample containers returned by reader adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class RawSceneSample:
    """Raw, unprocessed scene sample as loaded from disk."""

    scene_number: int
    global_origin: torch.Tensor

    # Agent data
    agent_types: torch.Tensor
    input_features: torch.Tensor
    input_mask: torch.Tensor
    output_features: torch.Tensor
    output_mask: torch.Tensor

    # Map data
    map_node_positions: torch.Tensor
    map_edge_indices: torch.Tensor
    map_node_types: torch.Tensor
    map_edge_types: torch.Tensor
