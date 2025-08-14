# Copyright 2024-2025, Theodor Westny. All rights reserved.
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

from __future__ import annotations

import math

import torch
import torch.distributions as tdist
from torch import nn


class EWTALoss(nn.Module):
    """Compute the Evolving Winner-Takes-All (EWTA) loss for multimodal prediction.

    This loss selects the top-k best-matching mixture components for each sample and computes
    their average error against the ground truth, enforcing mode diversity and competition.

    """

    def __init__(self, metric: str = "huber") -> None:
        """Initialize the EWTA loss function with the selected regression metric.

        Args:
            metric (str): Choice of distance metric, one of ["huber", "mse", "norm"].

        Raises:
            ValueError: If an unsupported metric is provided.

        """
        super().__init__()
        if metric == "huber":
            self.reg = nn.HuberLoss(reduction="none")
        elif metric == "mse":
            self.reg = nn.MSELoss(reduction="none")
        elif metric == "norm":
            self.reg = lambda mu, x: torch.linalg.norm(mu - x, dim=-1, keepdim=True)
        else:
            msg = f"Invalid metric: {metric}"
            raise ValueError(msg)

    def forward(
        self,
        mu: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        w: int = 1,
        ret_modes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the EWTA loss over predicted and ground-truth trajectories.

        Args:
            mu (torch.Tensor): Predicted states, shape (N, T, M, D).
            x (torch.Tensor): Ground-truth trajectory, shape (N, T, D).
            mask (torch.Tensor): Validity mask for each time step, shape (N, T).
            w (int): Number of best mixture components to include in loss computation.
            ret_modes (bool): If True, also return per-mode losses for analysis.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The average EWTA loss over batch,
                optionally along with per-mode loss values (shape: [N, M]).

        """
        valid_time_steps = mask.sum(-1)
        scored_agents = valid_time_steps > 0

        x = x.unsqueeze(2).expand(-1, -1, mu.size(2), -1)  # (N, T, m, k)
        reg_loss = self.reg(mu, x).sum(dim=-1)  # (N, T, m)
        masked_loss = reg_loss * mask.unsqueeze(-1)  # (N, T, m)

        masked_time = masked_loss[scored_agents].sum(1)  # (N, m)
        vals, _ = torch.topk(masked_time, k=w, dim=-1, largest=False)
        loss = vals.mean()

        if ret_modes:
            return loss, masked_time

        return loss


class NLLMDNLossCustom(nn.Module):
    """Compute the negative log-likelihood loss for a Mixture Density Network (MDN) with temporal structure.

    This version explicitly computes the multivariate Gaussian log-probabilities using
    Mahalanobis distance and determinant.

    """

    def __init__(self) -> None:
        """Initialize constants used in the custom NLL computation."""
        super().__init__()
        self.eps = 1e-15
        self.log2pi = math.log(2 * math.pi)

    def gaussian_log_prob(
        self, mu: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log-likelihood of x under each mixture component.

        Args:
            mu (torch.Tensor): Mean of each Gaussian component, shape (N, T, M, D).
            sigma (torch.Tensor): Covariance matrices, shape (N, T, M, D, D).
            x (torch.Tensor): Ground-truth data points, shape (N, T, D).

        Returns:
            torch.Tensor: Log-likelihoods for each component, shape (N, T, M).

        """
        batch_size, seq_len, num_components, data_dim = mu.size()

        # Expand x to match the shape of mu for broadcasting
        x_expanded = x.unsqueeze(2).expand(
            batch_size, seq_len, num_components, data_dim
        )
        x_diff = x_expanded - mu  # Difference between x and mu
        x_diff = x_diff.unsqueeze(-1)  # Adding a dimension for matrix multiplication

        solve_result = torch.linalg.solve(sigma, x_diff)
        log_det_sigma = torch.log(
            torch.linalg.det(sigma).abs() + self.eps
        )  # Log determinant of the covariance matrix

        mahalanobis_distance = (
            torch.matmul(x_diff.transpose(-1, -2), solve_result).squeeze(-1).squeeze(-1)
        )
        return -0.5 * (data_dim * self.log2pi + log_det_sigma + mahalanobis_distance)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pi: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood loss for a Gaussian mixture model.

        Args:
            mu (torch.Tensor): Component means, shape (N, T, M, D).
            sigma (torch.Tensor): Component covariances, shape (N, T, M, D, D).
            pi (torch.Tensor): Mixture weights, shape (N, M).
            x (torch.Tensor): Ground-truth targets, shape (N, T, D).
            mask (torch.Tensor): Time-step validity mask, shape (N, T).

        Returns:
            torch.Tensor: Mean negative log-likelihood over valid agents.

        """
        log_component_prob = self.gaussian_log_prob(mu, sigma, x)
        log_mix_prob = torch.log_softmax(pi, dim=-1).unsqueeze(1)

        # Compute log-sum-exp in a numerically stable way over the component dimension
        neg_log_prob = torch.logsumexp(log_component_prob + log_mix_prob, dim=-1).neg()

        neg_log_prob = neg_log_prob * mask  # Mask out the loss for invalid time steps
        valid_time_steps = mask.sum(-1)
        scored_agents = valid_time_steps > 0
        neg_log_prob = neg_log_prob[scored_agents]
        valid_time_steps = valid_time_steps[scored_agents]
        summed_loss = neg_log_prob.sum(-1) / valid_time_steps
        return summed_loss.mean()


class NLLMDNLoss(nn.Module):
    """Compute the negative log-likelihood loss using PyTorch's MixtureSameFamily distribution.

    This implementation leverages PyTorch distributions to compute the NLL for a mixture of
    multivariate Gaussians over time and batches, with optional triangular Cholesky input.
    """

    def __init__(self) -> None:
        """Initialize the NLL-MDN loss module."""
        super().__init__()

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pi: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        is_tril: bool = False,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        """Compute the negative log-likelihood using MixtureSameFamily from PyTorch.

        Args:
            mu (torch.Tensor): Component means, shape (N, T, M, D).
            sigma (torch.Tensor): Component covariances, shape (N, T, M, D, D).
            pi (torch.Tensor): Mixture logits, shape (N, M).
            x (torch.Tensor): Ground-truth target values, shape (N, T, D).
            mask (torch.Tensor): Time-step validity mask, shape (N, T).
            is_tril (bool): Whether `sigma` is already lower-triangular.
            epsilon (float): Jitter added for numerical stability in Cholesky.

        Returns:
            torch.Tensor: Mean negative log-likelihood over valid agents.

        """
        pi_expanded = pi.unsqueeze(1).expand(-1, mu.shape[1], -1)
        mix = tdist.Categorical(logits=pi_expanded)  # Broadcast this over time dim.
        jitter = torch.ones_like(mu).diag_embed() * epsilon
        L = sigma if is_tril else torch.linalg.cholesky(sigma + jitter)
        mvn = tdist.MultivariateNormal(mu, scale_tril=L)
        gmm = tdist.MixtureSameFamily(mix, mvn)
        neg_log_prob = gmm.log_prob(x).neg()  # (N, T)

        valid_time_steps = mask.sum(-1)
        scored_agents = valid_time_steps > 0
        neg_log_prob = neg_log_prob[scored_agents]
        valid_time_steps = valid_time_steps[scored_agents]
        nll = neg_log_prob.sum(-1) / valid_time_steps
        return nll.mean()
