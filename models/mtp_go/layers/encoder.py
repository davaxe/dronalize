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

from models.mtp_go.layers.gru_gnn_cell import GRUGNNCell


class GRUGNNEncoder(nn.Module):
    def __init__(
        self,
        num_inputs: int = 9,
        num_hidden: int = 64,
        num_heads: int = 3,
        num_layers: int = 1,
        num_mixtures: int = 6,
        dropout: float = 0.1,
        gnn_layer: str = "gat+",
        use_edge_features: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = num_hidden

        init_std = 1.0 / (num_hidden**0.5)
        self.init_state_param = nn.Parameter(
            torch.empty(num_hidden).uniform_(-init_std, init_std)
        )

        edge_dim = 1 if use_edge_features else None

        # GRU-GNN cell
        self.gru_cell = GRUGNNCell(
            num_inputs,
            num_hidden,
            num_heads,
            num_layers,
            dropout,
            gnn_layer,
            edge_dim=edge_dim,
        )

        # Mixture component logits
        self.mix_logits = nn.Sequential(
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_hidden, num_mixtures),
        )

    def _init_hidden(self, batch_size: int) -> torch.Tensor:
        return self.init_state_param.repeat(batch_size, 1)

    def _get_inputs(self, data: HeteroData) -> torch.Tensor:
        x = torch.cat(
            [
                data["agent"]["inp_pos"],
                data["agent"]["inp_vel"],
                data["agent"]["inp_yaw"],
            ],
            dim=-1,
        )

        if "inp_acc" in data["agent"]:
            x = torch.cat([x, data["agent"]["inp_acc"]], dim=-1)

        if "inp_r1" in data["agent"] and "inp_r2" in data["agent"]:
            """
            MTP-GO was designed to work with two additional inputs:
            - inp_r1: position relative to scenario center (urban) or relative position
            to the lane center (highway)
            - inp_r2: angle relative to scenario center (urban) or relative position to
            the road center (highway)

            These inputs complement the lack of map information in the model and can be
            computed using the optional arguments in the preprocessing step.
            If these inputs are not provided, the model will still work (assuming you
            have adjusted the input size accordingly),
            but it will not be as effective in scenarios where map information
            is crucial.
            """

            x = torch.cat([x, data["agent"]["inp_r1"]], dim=-1)
            x = torch.cat([x, data["agent"]["inp_r2"]], dim=-1)

        return x

    def forward(self, data: HeteroData):
        edge_index = data["agent"]["edge_index"]
        edge_attr = data["agent"]["edge_attr"]

        x = self._get_inputs(data)
        hidden = self._init_hidden(x.size(0))
        output = [hidden]

        for x_i, ei_i, ef_i in zip(x.transpose(0, 1), edge_index, edge_attr):
            hidden = self.gru_cell(x_i, ei_i, hidden, edge_attr=ef_i)
            output.append(hidden)

        output = torch.stack(output, dim=1)
        logits = self.mix_logits(hidden)
        return output, logits
