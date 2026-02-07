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

import warnings
from typing import Any

import torch
import torch.distributions as tdist
from torchmetrics import Metric


class NegativeLogLikelihood(Metric):
    dist: Any

    def __init__(self, dist: str = "mvn", **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.dist = self._get_distribution_initializer(dist)

    @staticmethod
    def _get_distribution_initializer(dist_name: str) -> Any:
        if dist_name == "mvn":
            return tdist.MultivariateNormal
        if dist_name == "normal":
            return tdist.Normal
        if dist_name == "laplace":
            return tdist.Laplace
        msg = f"Distribution '{dist_name}' is not supported. "
        raise ValueError(msg)

    @staticmethod
    def _handle_mode_first(
        pred: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.dim() == 4:
            return pred.transpose(1, 2), scale.transpose(1, 2)
        warnings.warn(
            "'mode_first' is set to True but the predictions"
            " are not multi-modal. Ignoring the flag.",
            stacklevel=2,
        )
        return pred, scale

    def _create_distribution(
        self, pred: torch.Tensor, scale: torch.Tensor, is_tril: bool
    ) -> Any:
        if self.dist.__name__ == "MultivariateNormal":
            assert scale.size(-1) == scale.size(-2), "Covariance matrix must be square."
            if not is_tril:
                scale = torch.linalg.cholesky(scale)
            return self.dist(loc=pred, scale_tril=scale)
        return self.dist(loc=pred, scale=scale)

    def update(
        self,
        pred: torch.Tensor,
        trg: torch.Tensor,
        scale: torch.Tensor,
        prob: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        logits: bool = False,
        is_tril: bool = False,
        mode_first: bool = False,
    ) -> None:
        """Update the metric state with a new batch of predicted distributions and targets.

        This method computes the negative log-likelihood (NLL) between predicted distributions
        and ground-truth targets, optionally aggregating over mixture modes using a
        `MixtureSameFamily` distribution. Supports both unimodal and multimodal prediction
        formats and allows for time-step masking and flexible distribution types.

        Args:
            pred (torch.Tensor): Predicted mean values, shape (N, T, M, D) for multimodal
                or (N, T, D) for unimodal predictions.
            trg (torch.Tensor): Ground-truth target trajectories, shape (N, T, D).
            scale (torch.Tensor): Scale parameter (e.g., std or covariance), shape (N, T, M, D[, D])
                or (N, T, D[, D]) depending on distribution type and modality.
            prob (Optional[torch.Tensor]): Mode probabilities or logits, shape (N, M).
                If None, a uniform distribution is assumed.
            mask (Optional[torch.Tensor]): Validity mask over time steps, shape (N, T).
                Only valid time steps are used in the loss.
            logits (bool): If True, interpret `prob` as logits and apply softmax.
            is_tril (bool): If True, assumes `scale` is a lower-triangular Cholesky factor.
                Required when using full covariance matrices with `MultivariateNormal`.
            mode_first (bool): If True, assumes input shape (N, M, T, D) and transposes to (N, T, M, D).

        Returns:
            None

        Raises:
            AssertionError: If using `MultivariateNormal` and the covariance matrix is not square.
            ValueError: If the distribution name is invalid when initializing the class.

        """
        if mode_first:
            # (N, M, T, 2) -> (N, T, M, 2)
            pred, scale = self._handle_mode_first(pred, scale)

        batch_size, seq_len = pred.size()[:2]

        distribution = self._create_distribution(pred, scale, is_tril)

        if pred.dim() == 4:
            if prob is None:
                prob = (
                    torch.ones(batch_size, pred.shape[2], device=pred.device)
                    / pred.shape[2]
                )
                if logits:
                    prob *= 0.0
            prob = prob.unsqueeze(1).expand(-1, seq_len, -1)  # (N, T, M)

            mix = (
                tdist.Categorical(logits=prob)
                if logits
                else tdist.Categorical(probs=prob)
            )
            if self.dist.__name__ != "MultivariateNormal":
                distribution = tdist.Independent(distribution, 1)
            distribution = tdist.MixtureSameFamily(mix, distribution)

        # Compute the negative log-likelihood
        neg_log_prob = distribution.log_prob(trg).neg()  # (N, T)

        if mask is not None:
            neg_log_prob = neg_log_prob * mask
            valid_time_steps = mask.sum(dim=-1)
            scored_agents = valid_time_steps > 0
            neg_log_prob = neg_log_prob[scored_agents]
            valid_time_steps = valid_time_steps[scored_agents]
        else:
            valid_time_steps = torch.ones_like(neg_log_prob).sum(-1)  # (N,)

        nll = neg_log_prob.sum(-1) / valid_time_steps  # (N,)

        self.sum += nll.sum()
        self.count += nll.size(0)

    def compute(self) -> torch.Tensor:
        """Compute the final metric."""
        return self.sum / self.count  # type: ignore
