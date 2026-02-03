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

from math import comb

import torch
from torchmetrics import Metric

from metrics.utils import filter_prediction


class CollisionRate(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
        ptr: torch.Tensor,
        prob: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        best_idx: torch.Tensor | None = None,
        collision_criterion: str = "FDE",
        collision_threshold: float = 1.0,
        mode_first: bool = False,
    ) -> None:
        """Update internal state with predictions for a new batch of scenarios.

        For each scenario, the method computes the number of pairwise collisions
        between agents at each timestep, where a collision is defined as being closer
        than `collision_threshold` meters. The count is normalized across all time steps
        and agent pairs.

        Args:
            pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2) or (N, T, 2).
            trg (torch.Tensor): Ground-truth trajectories, shape (N, T, 2).
            ptr (torch.Tensor): Pointer tensor indicating the start indices of each scenario,
                shape (B + 1,).
            prob (Optional[torch.Tensor]): Mixture mode probabilities, shape (N, M).
            mask (Optional[torch.Tensor]): Validity mask for each timestep, shape (N, T).
            best_idx (Optional[torch.Tensor]): Precomputed best mode indices, shape (N,).
            collision_criterion (str): Strategy for selecting the best mode, one of {"FDE", "ADE", "ML"}.
            collision_threshold (float): Distance threshold in meters to define a collision.
            mode_first (bool): If True, assume the mode dimension comes first (M, N, T, 2).

        Returns:
            None

        """
        if pred.dim() <= 2:
            msg = "The prediction tensor must have at least 3 dimensions."
            raise ValueError(msg)

        if pred.dim() == 4:
            pred, _ = filter_prediction(
                pred,
                trg,
                mask,
                prob,
                collision_criterion,
                best_idx,
                mode_first=mode_first,
            )

        seq_len = pred.size(1)

        # Compute the collision rate for each scenario
        for i in range(len(ptr) - 1):
            ptr_from = ptr[i]
            ptr_to = ptr[i + 1]

            # Get the scenario
            scenario = pred[ptr_from:ptr_to]
            n = scenario.size(0)

            # Compute the number of possible collisions
            self.count += seq_len * comb(n, 2)  # type: ignore  # T * (n * (n - 1)) // 2
            for t in range(seq_len):
                dists = torch.cdist(scenario[:, t], scenario[:, t], p=2)  # (n, n)

                # Find the collisions and filter out the self-collisions
                collisions = (dists < collision_threshold) & (dists != 0.0)
                self.sum += collisions.sum().item() / 2  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        return self.sum / self.count  # type: ignore
