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


class MinAPDE(Metric):
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

        This method computes the minimum Average Path Displacement Error (APDE), which
        measures how close each predicted point is to any point in the ground-truth trajectory.
        The closest point distance is averaged across time steps for each agent. For multimodal
        predictions, the best mode is selected according to the specified criterion before computing APDE.

        Args:
            pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2) for multimodal or (N, T, 2) for filtered.
            trg (torch.Tensor): Ground-truth trajectories, shape (N, T, 2).
            prob (Optional[torch.Tensor]): Mixture probabilities or logits, shape (N, M), used for "ML" mode selection.
            mask (Optional[torch.Tensor]): Validity mask over time steps, shape (N, T). Only valid steps are included.
            best_idx (Optional[torch.Tensor]): Precomputed indices of best modes, shape (N,). Avoids recomputing.
            min_criterion (str): Criterion for best mode selection, one of {"FDE", "ADE", "ML"}.
            mode_first (bool): If True, assumes input is (N, M, T, 2) and transposes to (N, T, M, 2).

        Returns:
            None

        Raises:
            ValueError: If `min_criterion` is not one of {"FDE", "ADE", "ML"}.

        """
        if pred.dim() == 4:
            pred, _ = filter_prediction(
                pred, trg, mask, prob, min_criterion, best_idx, mode_first=mode_first
            )

        batch_size, seq_len = pred.size()[:2]

        cdist = torch.cdist(pred, trg, p=2)  # (N, T, T)

        if mask is not None:
            mask_exp = mask.unsqueeze(1).repeat(1, seq_len, 1) & mask.unsqueeze(
                -1
            ).repeat(1, 1, seq_len)  # (N, T, T)
            cdist[~mask_exp] = float("1e9")
            path_dist, _ = cdist.min(dim=-1)  # (N, T)
            num_valid_steps = mask.sum(dim=-1)
            path_dist = path_dist * mask

            scored_agents = num_valid_steps > 0
            path_dist = path_dist[scored_agents]
            num_valid_steps = num_valid_steps[scored_agents]
        else:
            path_dist, _ = cdist.min(dim=-1)  # (N, T)
            num_valid_steps = torch.ones_like(path_dist).sum(-1)  # (N,)

        apde = path_dist.sum(-1) / num_valid_steps  # (N,)

        self.sum += apde.sum()
        self.count += apde.size(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.
        """
        return self.sum / self.count  # type: ignore
