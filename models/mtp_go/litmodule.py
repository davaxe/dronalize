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

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius_graph
from torch_geometric.utils import subgraph, to_undirected

from metrics import MinADE, MinFDE, MissRate
from models.mtp_go.loss_fns import EWTALoss, NLLMDNLossCustom


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, config: dict, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.max_epochs = config["epochs"]
        self.decay_rate = config["decay_rate"]
        self.learning_rate = config["lr"]
        self.tf_init_p = config["teacher_forcing"]
        self.r = config["radius"]
        self.test_crit = config.get("criterion", "FDE")

        self.teacher_force_epochs = self.max_epochs // 2
        self.wta_epochs = self.max_epochs // 8
        self.annealing_epochs = int(self.max_epochs * 0.6)

        self.save_hyperparameters(ignore=["model"])

        self.wta = EWTALoss()
        self.nll = NLLMDNLossCustom()

        self.min_ade = MinADE()
        self.min_fde = MinFDE()
        self.mr = MissRate()

    def _create_edge_indices(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        num_graphs: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pos_s = pos.transpose(0, 1).reshape(-1, 2)
        batch_s = torch.cat(
            [batch + num_graphs * t for t in range(pos.shape[1])],
            dim=0,
        )
        mask_s = mask.transpose(0, 1).reshape(-1)

        # Compute edge indices and attributes for combined data
        edge_index = radius_graph(
            x=pos_s,
            r=self.r,
            batch=batch_s,
            loop=True,
            max_num_neighbors=10,
        )
        edge_index = subgraph(subset=mask_s, edge_index=edge_index)[0]
        edge_index = to_undirected(edge_index)
        edge_attrs = torch.linalg.norm(
            pos_s[edge_index[0]] - pos_s[edge_index[1]],
            dim=-1,
            keepdim=True,
        )

        # Map global edge indices back to per-time-step local indices
        num_agents = pos.shape[0]
        num_time_steps = pos.shape[1]

        # Initialize lists to store edge indices and attributes per time step
        edge_indices = []
        edge_attributes = []

        # Process the global edge indices to map back to local indices
        for t in range(num_time_steps):
            time_offset = t * num_agents
            mask = (edge_index >= time_offset) & (edge_index < time_offset + num_agents)
            local_edge_index = edge_index[:, mask[0]] - time_offset
            edge_indices.append(local_edge_index)

            # Filter edge attributes for the current time step
            local_edge_attr = edge_attrs[mask[0]]
            edge_attributes.append(local_edge_attr)

        return edge_indices, edge_attributes

    def _post_process(self, data: HeteroData) -> HeteroData:
        batch = data["agent"]["batch"]
        num_graphs = data.num_graphs
        inp_pos = data["agent"]["inp_pos"]
        inp_mask = data["agent"]["input_mask"]

        trg_pos = data["agent"]["trg_pos"]

        edge_indices, edge_attributes = self._create_edge_indices(
            inp_pos,
            batch,
            inp_mask,
            num_graphs,
        )

        seq_len = trg_pos.shape[1]
        edge_index = edge_indices[-1].clone()
        trg_edge_indices = [edge_index for _ in range(seq_len)]

        data["agent"]["edge_index"] = edge_indices
        data["agent"]["edge_attr"] = edge_attributes
        data["agent"]["trg_edge_index"] = trg_edge_indices

        return data

    def forward(
        self,
        data: HeteroData,
        tf_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._post_process(data)
        trg = data["agent"]["trg_pos"]

        pred_states, pred_cov, pred_mix = self.model(data, tf_prob)

        return trg, pred_states, pred_cov, pred_mix

    def training_step(self, data: HeteroData) -> torch.Tensor:
        valid_mask = data["agent"]["valid_mask"]
        tf_prob = max(
            0,
            self.tf_init_p
            * (
                (self.teacher_force_epochs - self.current_epoch)
                / self.teacher_force_epochs
            ),
        )

        trg, pred, covs, mix = self(data, tf_prob)

        if self.current_epoch < self.wta_epochs and self.wta_epochs > 1:
            # Only EWTA loss
            wta_weight = (self.wta_epochs - self.current_epoch) / self.wta_epochs

            #  The number of winners used for the WTA loss decreases with increasing epoch
            num_mixtures = mix.shape[1]
            num_winners = max(min(num_mixtures, int(wta_weight * num_mixtures)), 1)

            loss = self.wta(pred, trg, valid_mask, w=num_winners)

        else:
            # Compute NLL loss
            nll_loss = self.nll(pred, covs, mix, trg, valid_mask)
            # Compute WTA loss
            wta_loss = self.wta(pred, trg, valid_mask, w=1)

            # Compute the warm-up weight
            warm_weight = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * (self.current_epoch - self.wta_epochs)
                    / (self.max_epochs - self.wta_epochs)
                )
            )

            # Combine the losses
            loss = warm_weight * wta_loss + (1.0 - warm_weight) * nll_loss

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=trg.size(0),
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, data: HeteroData, *args) -> None:
        mask = data["agent"]["valid_mask"]

        trg, pred, covs, mix = self(data)

        val_loss = self.nll(pred, covs, mix, trg, mask)

        self.min_ade.update(pred, trg, mask=mask)
        self.min_fde.update(pred, trg, mask=mask)
        self.mr.update(pred, trg, mask=mask)

        metric_dict = {
            "val_loss": val_loss,
            "val_min_ade": self.min_ade,
            "val_min_fde": self.min_fde,
            "val_mr": self.mr,
        }

        self.log_dict(
            metric_dict,
            on_step=False,
            on_epoch=True,
            batch_size=trg.size(0),
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, data: HeteroData, *args) -> None:
        mask = data["agent"]["valid_mask"]

        trg, pred, _, mix = self(data)

        prob = nn.functional.softmax(mix, dim=-1)

        self.min_ade.update(
            pred,
            trg,
            prob=prob,
            mask=mask,
            min_criterion=self.test_crit,
        )
        self.min_fde.update(
            pred,
            trg,
            prob=prob,
            mask=mask,
            min_criterion=self.test_crit,
        )
        self.mr.update(pred, trg, prob=prob, mask=mask, miss_criterion=self.test_crit)

        metric_dict = {
            "test_min_ade": self.min_ade,
            "test_min_fde": self.min_fde,
            "test_mr": self.mr,
        }

        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0.5,
            total_iters=self.annealing_epochs,
        )
        return [optimizer], [scheduler]
