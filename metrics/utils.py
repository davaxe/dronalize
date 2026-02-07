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

import torch


def filter_prediction(
    pred: torch.Tensor,
    trg: torch.Tensor,
    mask: torch.Tensor | None = None,
    prob: torch.Tensor | None = None,
    min_criterion: str = "FDE",
    best_idx: torch.Tensor | None = None,
    mode_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the best trajectory prediction mode based on a given selection criterion.

    This function filters a set of multimodal predictions by selecting the most likely or
    most accurate trajectory mode per sample, according to a specified criterion.
    It returns a single-mode prediction tensor and the corresponding best mode indices.

    Args:
        pred (torch.Tensor): Predicted trajectories, shape (N, T, M, 2) or (N, M, T, 2) if `mode_first` is True.
        trg (torch.Tensor): Ground-truth trajectories, shape (N, T, 2).
        mask (Optional[torch.Tensor]): Validity mask for time steps, shape (N, T). Used to find last valid index for FDE.
        prob (Optional[torch.Tensor]): Mixture probabilities or logits, shape (N, M). Required for "ML" criterion.
        min_criterion (str): Criterion used to select the best mode, one of {"FDE", "ADE", "ML"}.
        best_idx (Optional[torch.Tensor]): Precomputed best mode indices, shape (N,). If provided, skips selection logic.
        mode_first (bool): If True, assumes input is shaped (N, M, T, 2) and transposes to (N, T, M, 2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Filtered single-mode predictions, shape (N, T, 2).
            - Selected mode indices, shape (N,).

    Raises:
        ValueError: If `min_criterion` is invalid or if `prob` is not provided for "ML".

    """
    if mode_first:
        # (N, M, T, 2) -> (N, T, M, 2)
        pred = pred.transpose(1, 2)

    if pred.size(-1) > 2 or trg.size(-1) > 2:
        warnings.warn(
            "The last dimension of the prediction or target tensors"
            " is greater than 2. Only the first two dimensions will be considered.",
            stacklevel=2,
        )
        pred = pred[..., :2]
        trg = trg[..., :2]

    batch_size, seq_len = pred.size()[:2]

    if best_idx is not None:
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)
        return pred, best_idx

    if min_criterion == "FDE":
        if mask is not None:
            mask_reversed = 1 * mask.flip(dims=[-1])  # (N, T)
            last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)  # (N,)

            last_pred = pred[torch.arange(batch_size), last_idx]  # (N, M, 2)
            last_trg = trg[torch.arange(batch_size), last_idx]  # (N, 2)
        else:
            last_pred = pred[:, -1]
            last_trg = trg[:, -1]

        best_idx = torch.linalg.norm(last_pred - last_trg.unsqueeze(1), dim=-1).argmin(
            dim=-1
        )  # (N,)

        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    elif min_criterion == "ADE":
        if mask is not None:
            multi_mask = mask.unsqueeze(-1).unsqueeze(-1)  # (N, T, 1, 1)
            masked_pred = pred * multi_mask  # (N, T, M, 2)
            masked_trg = trg.unsqueeze(2) * multi_mask  # (N, T, 1, 2)
        else:
            masked_pred = pred  # (N, T, M, 2)
            masked_trg = trg.unsqueeze(2)  # (N, T, 1, 2)

        norm = torch.linalg.norm(masked_pred - masked_trg, dim=-1)  # (N, T, M)

        best_idx = norm.sum(dim=1).argmin(dim=-1)  # (N,)
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    elif min_criterion == "ML":
        if prob is None:
            msg = "Probabilistic criterion requires the probability of the predictions."
            raise ValueError(prob)

        best_idx = prob.argmax(dim=-1)  # (N,)
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    else:
        msg = f"Invalid criterion: {min_criterion}"
        raise ValueError(msg)

    return pred, best_idx
