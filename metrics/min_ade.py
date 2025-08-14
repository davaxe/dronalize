# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torchmetrics import Metric

from metrics.utils import filter_prediction


class MinADE(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        best_idx: Optional[torch.Tensor] = None,
        min_criterion: str = "FDE",
        mode_first: bool = False,
    ) -> None:
        """Update the metric state with predicted trajectories and ground-truth targets.

        This method computes the minimum Average Displacement Error (ADE) for each sample
        by first selecting the best trajectory mode (based on FDE, ADE, or maximum likelihood),
        then averaging the point-wise distance between predicted and ground-truth positions
        over all valid time steps.

        Args:
            pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2) for multimodal or (N, T, 2) if already filtered.
            trg (torch.Tensor): Ground-truth target trajectories, shape (N, T, 2).
            prob (Optional[torch.Tensor]): Mixture probabilities or logits, shape (N, M). Used for ML-based mode selection.
            mask (Optional[torch.Tensor]): Validity mask for time steps, shape (N, T). Indicates which time steps to evaluate.
            best_idx (Optional[torch.Tensor]): Precomputed indices of best prediction modes, shape (N,). Avoids recomputing.
            min_criterion (str): Criterion used to select the best mode. One of {"FDE", "ADE", "ML"}.
            mode_first (bool): If True, assumes the input is shaped (N, M, T, 2) and transposes to (N, T, M, 2).

        Returns:
            None

        Raises:
            ValueError: If `min_criterion` is not one of {"FDE", "ADE", "ML"}.

        """
        if pred.dim() == 4:
            pred, _ = filter_prediction(
                pred, trg, mask, prob, min_criterion, best_idx, mode_first=mode_first
            )

        norm = torch.linalg.norm(pred - trg, dim=-1)  # (N, T)
        if mask is not None:
            num_valid_steps = mask.sum(dim=-1)  # (N,)
            scored_agents = num_valid_steps > 0
            norm = norm * mask  # (N, T)
            norm = norm[scored_agents]
            num_valid_steps = num_valid_steps[scored_agents]
        else:
            num_valid_steps = torch.ones_like(norm).sum(dim=-1)  # (N,)

        ade = norm.sum(dim=-1) / num_valid_steps  # (N,)
        self.sum += ade.sum()
        self.count += ade.size(0)

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        return self.sum / self.count  # type: ignore
