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


class MinBrier(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
        prob: torch.Tensor,
        mask: torch.Tensor | None = None,
        best_idx: torch.Tensor | None = None,
        logit: bool = False,
        min_criterion: str = "FDE",
        mode_first: bool = False,
    ) -> None:
        """Update the metric state with predicted trajectories, probabilities, and targets.

        This method computes a Brier-style score that penalizes confident but inaccurate
        predictions. The most likely mode is selected according to the specified criterion
        (FDE, ADE, or ML), and the distance to the ground-truth at the final timestep is
        weighted by (1 - predicted probability), reflecting a probabilistic confidence penalty.

        Args:
            pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2). Multimodal predictions are required.
            trg (torch.Tensor): Ground-truth trajectories, shape (N, T, 2).
            prob (torch.Tensor): Mixture probabilities or logits, shape (N, M).
            mask (Optional[torch.Tensor]): Validity mask for time steps, shape (N, T). If provided, last valid step is used.
            best_idx (Optional[torch.Tensor]): Precomputed best mode indices, shape (N,). Avoids recomputing.
            logit (bool): If True, applies sigmoid to convert logits to probabilities.
            min_criterion (str): Criterion used for best mode selection, one of {"FDE", "ADE", "ML"}.
            mode_first (bool): If True, assumes input is (N, M, T, 2) and transposes to (N, T, M, 2).

        Returns:
            None

        Raises:
            ValueError: If `prob` is None or if `pred` is not 4-dimensional.
            ValueError: If `min_criterion` is not one of {"FDE", "ADE", "ML"}.

        """
        if prob is None:
            msg = "Probabilistic criterion requires the probability of the predictions."
            raise ValueError(msg)
        if pred.dim() != 4:
            msg = f"The prediction tensor must be 4-dimensional, got shape {pred.shape}"
            raise ValueError(msg)

        pred, best_idx = filter_prediction(
            pred, trg, mask, prob, min_criterion, best_idx, mode_first=mode_first
        )

        batch_size, seq_len = pred.size()[:2]

        prob = prob[torch.arange(batch_size), best_idx]  # (N,)

        if mask is not None:
            mask_reversed = 1 * mask.flip(dims=[-1])
            last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)

            pred = pred[torch.arange(batch_size), last_idx]  # (N, 2)
            trg = trg[torch.arange(batch_size), last_idx]  # (N, 2)

            scored_agents = mask.sum(dim=-1) > 0
            pred = pred[scored_agents]
            trg = trg[scored_agents]
            prob = prob[scored_agents]
        else:
            pred = pred[:, -1]  # (N, 2)
            trg = trg[:, -1]  # (N, 2)

        if logit:
            prob = torch.sigmoid(prob)

        brier = (1.0 - prob) * torch.linalg.norm(pred - trg, dim=-1)  # (N,)

        self.sum += brier.sum()
        self.count += brier.size(0)

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        return self.sum / self.count  # type: ignore
