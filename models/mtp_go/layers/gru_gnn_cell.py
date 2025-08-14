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

from models.mtp_go.layers.gnn_layers import create_sequential_gnn


class GRUGNNCell(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        n_heads: int = 3,
        n_layers: int = 1,
        dropout: float = 0.1,
        gnn_layer: str = "gat+",
        edge_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size  # Need 3 hidden states for all GRU gates

        self.W_x = create_sequential_gnn(
            input_size=input_size,
            output_size=gate_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            layers=n_layers,
            activation="elu",
            gnn_layer=gnn_layer,
            edge_dim=edge_dim,
        )

        self.W_h = create_sequential_gnn(
            input_size=hidden_size,
            output_size=gate_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            layers=n_layers,
            activation="elu",
            gnn_layer=gnn_layer,
            edge_dim=edge_dim,
        )

        # Biases for GRU gates
        self.b_r = nn.Parameter(torch.empty(hidden_size).uniform_(-1e-2, 1e-2))
        self.b_z = nn.Parameter(torch.empty(hidden_size).uniform_(-1e-2, 1e-2))
        self.b_h = nn.Parameter(torch.empty(hidden_size).uniform_(-1e-2, 1e-2))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_size**0.5)

        # Exclude edge bws from initialization
        init_params = (
            p
            for name, p in self.named_parameters()
            if not str.endswith(name, "log_edge_bw")
        )
        for p in init_params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -std, std)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #  Implements GRUCell update:
        #  https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        #  Using GNNs as learnable functions
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        x_proj = self.W_x(x, edge_index, edge_attr)
        h_proj = self.W_h(h, edge_index, edge_attr)

        r_x, z_x, h_x = torch.chunk(x_proj, 3, dim=1)
        r_h, z_h, h_h = torch.chunk(h_proj, 3, dim=1)

        r = torch.sigmoid(r_x + r_h + self.b_r)
        z = torch.sigmoid(z_x + z_h + self.b_z)
        h_tilde = torch.tanh(h_x + r * h_h + self.b_h)

        h_new = (1 - z) * h_tilde + z * h
        return h_new
