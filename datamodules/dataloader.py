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

from argparse import Namespace
from typing import TYPE_CHECKING

import torch_geometric.transforms as pyg_tf
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader

from datamodules.dataset import TrajDataset
from utils import import_from_module

if TYPE_CHECKING:
    from torch_geometric.data import Dataset


class TrajDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for trajectory datasets.

    This module handles the setup and loading of training, validation, and test datasets
    for trajectory prediction tasks using PyTorch Geometric and Lightning.

    Attributes:
        train (Dataset): Training dataset.
        val (Dataset): Validation dataset.
        test (Dataset): Test dataset.
        transform: Optional transform(s) applied to the datasets.

    """

    train: Dataset = None
    val: Dataset = None
    test: Dataset = None
    transform = None

    def __init__(self, config: dict, args: Namespace) -> None:
        """Initialize the TrajDataModule with configuration and command-line arguments.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
            args (Namespace): Command-line arguments namespace.

        """
        super().__init__()
        self.root = config["root"]
        self.dataset = config["name"]
        self.batch_size = config["batch_size"]

        self.small_data = args.small_ds
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers

        self.transform = self._get_transform(config.get("transform"))

    @staticmethod
    def _get_transform(transform_config: None | list | str) -> None | pyg_tf.Compose:
        if transform_config is None:
            return None
        if isinstance(transform_config, list):
            return pyg_tf.Compose(
                [
                    import_from_module("datamodules.transforms", t)()
                    for t in transform_config
                ],
            )
        return import_from_module("datamodules.transforms", transform_config)()

    def setup(self, stage: str | None = None) -> None:
        """Set up the training, validation, and test datasets."""
        if stage == "fit" or stage is None:
            self.train = TrajDataset(
                root=self.root,
                dataset=self.dataset,
                split="train",
                transform=self.transform,
                small_data=self.small_data,
            )

            self.val = TrajDataset(
                root=self.root,
                dataset=self.dataset,
                split="val",
                transform=self.transform,
                small_data=self.small_data,
            )

        if stage == "test" or stage is None:
            self.test = TrajDataset(
                root=self.root,
                dataset=self.dataset,
                split="test",
                transform=self.transform,
                small_data=self.small_data,
            )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training dataset."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation dataset."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test dataset."""
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    from lightning.pytorch import seed_everything

    from visualization import visualize_batch

    seed_everything(42)

    config = {"root": "./data", "name": "rounD", "batch_size": 32}
    args = Namespace(
        small_ds=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    dm = TrajDataModule(config, args)
    dm.setup()

    data = next(iter(dm.train_dataloader()))
    visualize_batch(
        data,
        batch_idx=24,
        use_ma_idx=True,
        save_fig=False,
    )  # 24 for Fig. 1 in toolbox paper
