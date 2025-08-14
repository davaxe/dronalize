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

import torch
from torch import nn
from torch_geometric.data import HeteroData

from models.mtp_go.layers.decoder import GRUGNNDecoder
from models.mtp_go.layers.encoder import GRUGNNEncoder
from models.mtp_go.layers.motion_models import select_mm


class MTPGo(nn.Module):
    """MTP-GO model for multi-agent trajectory prediction with uncertainty estimation.

    This model combines a GRU-based Graph Neural Network (GNN) encoder and decoder with
    a parametric motion model (e.g., constant velocity or acceleration) to produce
    multimodal trajectory predictions along with associated uncertainty estimates.

    The encoder extracts interaction-aware latent representations from heterogeneous
    graph-structured input data. The decoder autoregressively predicts future states
    for each mixture component, using an Extended Kalman Filter (EKF) to propagate
    uncertainty through the motion model.

    Paper link: https://arxiv.org/abs/2302.00735
    """

    def __init__(self, config: dict) -> None:
        """Initialize the MTP-GO model and its submodules.

        This constructor sets up the encoder, decoder, and motion model components
        based on the provided configuration dictionary. It supports flexible GNN
        architectures and motion models, as well as multiple Gaussian mixture components
        for capturing multimodal behavior.

        Args:
            config (dict): Dictionary specifying model architecture parameters. Expected keys include:
                - "encoder": Settings for the GRU-GNN encoder (e.g., num_inputs, num_hidden, etc.)
                - "decoder": Settings for the GRU-GNN decoder
                - "motion_model": Settings for the dynamics model (class, solver, step size)
                - "num_mixtures": Number of mixture components used in prediction

        """
        super().__init__()
        encoder = config["encoder"]
        decoder = config["decoder"]
        motion_model = config["motion_model"]
        num_mixtures = config["num_mixtures"]

        self.initial_uncertainty = 1e-5

        mm = select_mm(motion_model["class"])
        self.motion_model = mm(
            solver=motion_model["solver"],
            dt=motion_model["step_size"],
            mixtures=num_mixtures,
        )

        self.num_mixtures = num_mixtures
        self.num_states = self.motion_model.n_states

        self.encoder = GRUGNNEncoder(
            num_inputs=encoder["num_inputs"],
            num_hidden=encoder["num_hidden"],
            num_heads=encoder["num_heads"],
            num_layers=encoder["num_layers"],
            num_mixtures=num_mixtures,
            dropout=encoder["dropout"],
            gnn_layer=encoder["gnn_layer"],
            use_edge_features=True,
        )

        self.decoder = GRUGNNDecoder(
            motion_model=self.motion_model,
            num_hidden=decoder["num_hidden"],
            num_heads=decoder["num_heads"],
            num_layers=decoder["num_layers"],
            dropout=decoder["dropout"],
            gnn_layer=decoder["gnn_layer"],
        )

    def _state_transition_matrix(
        self, next_states: torch.Tensor, model_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.motion_model.state_transition_matrix(
            next_states, model_input
        )

    def _input_transition_matrix(
        self, next_states: torch.Tensor, model_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.motion_model.input_transition_matrix(
            next_states, model_input
        )

    def _ekf(
        self,
        p_t: torch.Tensor,
        q_t: torch.Tensor,
        next_states: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        f_t, f_t_tp = self._state_transition_matrix(next_states, model_input)
        g_t, g_t_tp = self._input_transition_matrix(next_states, model_input)
        return f_t @ p_t @ f_t_tp + g_t @ q_t @ g_t_tp

    def forward(
        self, data: HeteroData, tf_prob: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the MTP-GO model forward pass to predict future trajectories and uncertainties.

        This function encodes past agent interactions using a GRU-GNN encoder, then autoregressively
        decodes future trajectories using a GRU-GNN decoder with optional teacher forcing.
        EKF-based covariance propagation is applied during decoding to estimate uncertainty.

        Args:
            data (HeteroData): Input graph data containing past/future positions and interaction edges.
            tf_prob (float): Probability of applying teacher forcing at each step (0 disables it).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - pred_states: Predicted future trajectories, shape (B, T, N, D).
                - cov_mats: Predicted state covariance matrices, shape (B, T, N, D, D).
                - mixture_weight: Mixture component probabilities for each mode, shape (B, N_modes).

        """
        trg_edge_idx = data["agent"]["trg_edge_index"]
        valid_mask = data["agent"]["valid_mask"]
        target = data["agent"]["trg_pos"]
        batch_size, trg_len = target.size()[0:2]

        use_teacher_forcing = (tf_prob > 0.0) and torch.rand(1).item() < tf_prob

        encoder_out, mixture_weight = self.encoder(data)

        p_t = (
            torch.diag_embed(
                torch.ones(
                    batch_size,
                    self.num_mixtures,
                    self.num_states,
                    device=encoder_out.device,
                )
            )
            * self.initial_uncertainty
        )  # (batch_size, mixtures, n_states, n_states)

        past_state = data["agent"]["inp_pos"][:, -1:].repeat(1, self.num_mixtures, 1)
        dec_hidden = encoder_out[:, -1]

        pred_states = []
        cov_mats = []
        for di in range(trg_len):
            decoder_input = (
                past_state.view(batch_size, -1),
                dec_hidden,
                encoder_out,
                trg_edge_idx[di],
                past_state,
            )
            next_states, model_input, q_t, dec_hidden = self.decoder(decoder_input)
            p_t = self._ekf(p_t, q_t, past_state, model_input)
            pred_states.append(next_states)
            cov_mats.append(p_t)

            pred_det = next_states.detach()
            if use_teacher_forcing:
                ri_mask = valid_mask[:, di].view(-1, 1, 1)  # (B, 1, 1)
                teacher_pred = target[:, di : di + 1, :].expand(
                    -1, self.num_mixtures, -1
                )  # (B, N_mixture, d)

                # Keep prediction when no teacher prediction exists
                past_state = ri_mask * teacher_pred + (~ri_mask) * pred_det
            else:
                past_state = pred_det

        pred_states = torch.stack(pred_states, dim=1)
        cov_mats = torch.stack(cov_mats, dim=1)
        return pred_states, cov_mats, mixture_weight
