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


import torch
from torchmetrics import Metric

from metrics.utils import filter_prediction


class MissRate(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
        mask: torch.Tensor | None = None,
        prob: torch.Tensor | None = None,
        best_idx: torch.Tensor | None = None,
        miss_criterion: str = "FDE",
        miss_threshold: float = 2.0,
        mode_first: bool = False,
    ) -> None:
        """Update the metric state with predicted and ground-truth trajectories.

        This method computes the Miss Rate, defined as the proportion of agents whose predicted
        final position deviates from the ground truth by more than `miss_threshold` meters.
        For multimodal predictions, the best mode is selected according to a criterion (FDE, ADE, or ML)
        before computing the distance to the target.

        Args:
            pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2) for multimodal or (N, T, 2) if already filtered.
            trg (torch.Tensor): Ground-truth trajectories, shape (N, T, 2).
            mask (Optional[torch.Tensor]): Validity mask over time steps, shape (N, T). If provided, selects the final valid time step.
            prob (Optional[torch.Tensor]): Mixture probabilities or logits, shape (N, M), used for "ML" mode selection.
            best_idx (Optional[torch.Tensor]): Precomputed best mode indices, shape (N,). Avoids recomputing.
            miss_criterion (str): Criterion for best mode selection, one of {"FDE", "ADE", "ML"}.
            miss_threshold (float): Distance threshold (in meters) beyond which a prediction is considered a miss.
            mode_first (bool): If True, assumes input is shaped (N, M, T, 2) and transposes to (N, T, M, 2).

        Returns:
            None

        Raises:
            ValueError: If `miss_criterion` is not one of {"FDE", "ADE", "ML"}.

        """
        if pred.dim() == 4:
            pred, _ = filter_prediction(
                pred,
                trg,
                mask,
                prob,
                miss_criterion,
                best_idx,
                mode_first=mode_first,
            )

        batch_size, seq_len = pred.size()[:2]

        if mask is not None:
            mask_reversed = 1 * mask.flip(dims=[-1])
            last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)

            pred = pred[torch.arange(batch_size), last_idx]  # (N, 2)
            trg = trg[torch.arange(batch_size), last_idx]  # (N, 2)

            scored_agents = mask.sum(dim=-1) > 0
            pred = pred[scored_agents]
            trg = trg[scored_agents]
        else:
            pred = pred[:, -1]  # (N, 2)
            trg = trg[:, -1]  # (N, 2)

        norm = torch.linalg.norm(pred - trg, dim=-1)  # (N,)

        mr = norm > miss_threshold  # (N,)

        self.sum += mr.sum()
        self.count += mr.size(0)

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        return self.sum / self.count  # type: ignore
