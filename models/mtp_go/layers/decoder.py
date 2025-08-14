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

from typing import Optional

import torch
from torch import nn

from models.mtp_go.layers.gru_gnn_cell import GRUGNNCell


class GRUGNNDecoder(nn.Module):
    def __init__(
        self,
        motion_model: nn.Module,
        num_hidden: int = 64,
        num_heads: int = 3,
        num_layers: int = 1,
        dropout: float = 0.1,
        gnn_layer: str = "gat+",
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru_cell = GRUGNNCell(
            num_hidden, num_hidden, num_heads, num_layers, dropout, gnn_layer
        )

        # Model components
        self.motion_model = motion_model
        self.input_size = motion_model.n_states
        self.output_size = motion_model.n_inputs
        self.mixtures = motion_model.mixtures
        self.hidden_size = num_hidden

        # Activations
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Temporal attention weight calculations
        self.embedding = nn.Sequential(
            nn.Linear(self.input_size * self.mixtures, num_hidden),
            nn.Dropout(p=dropout),
        )
        self.attn_combine = nn.Linear(num_hidden * 2, num_hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=num_hidden, num_heads=1, dropout=dropout, batch_first=True
        )

        # Scale GRU outputs
        self.generator = nn.Sequential(
            nn.ELU(),
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

        # Motion model inputs
        self.controller = nn.Linear(num_hidden, self.output_size * self.mixtures)

        # Process noise generation
        self.sigma_1 = nn.Sequential(
            nn.Linear(num_hidden, self.mixtures), nn.Softplus()
        )
        self.sigma_2 = nn.Sequential(
            nn.Linear(num_hidden, self.mixtures), nn.Softplus()
        )
        self.rho = nn.Sequential(nn.Linear(num_hidden, self.mixtures), nn.Softsign())

    def _apply_attention(
        self, embedded: torch.Tensor, hidden: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        query = hidden.unsqueeze(1)  # (B, 1, H)
        key = encoder_out  # (B, T, H)
        value = encoder_out  # (B, T, H)

        attn_output = self.attn(query, key, value)[0].squeeze(1)  # (B, H)
        combined = torch.cat([embedded, attn_output], dim=-1)  # (B, 2H)
        output = self.attn_combine(combined)
        return self.leaky_relu(output)

    def process_noise_matrix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = x1.size(0)

        sigma_1 = self.sigma_1(x1)
        sigma_2 = self.sigma_2(x2)
        rho = self.rho(x3)

        q_t = torch.zeros(
            batch_size,
            self.mixtures,
            self.output_size,
            self.output_size,
            device=x1.device,
        )

        q_t[..., 0, 0] = sigma_1**2
        q_t[..., 1, 1] = sigma_2**2
        q_t[..., 0, 1] = q_t[..., 1, 0] = sigma_1 * sigma_2 * rho

        return q_t

    def forward(self, data: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x, hidden, encoder_out, edge_index, past_state = data
        batch_size = x.size(0)
        embedded = self.embedding(x)

        output = self._apply_attention(embedded, hidden, encoder_out)

        hidden = self.gru_cell(output, edge_index, hidden)

        output = self.generator(hidden)

        x1, x2, x3, x4 = torch.split(output, self.hidden_size, dim=-1)
        process_noise = self.process_noise_matrix(x1, x2, x3, batch_size)
        model_input = self.controller(x4).view(
            batch_size, self.mixtures, self.output_size
        )

        next_state, model_input = self.motion_model(past_state, model_input)

        return next_state, model_input, process_noise, hidden
