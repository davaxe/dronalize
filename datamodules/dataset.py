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

import os
import pickle
from typing import Callable

from torch_geometric.data import Dataset, HeteroData


class TrajDataset(Dataset):
    """A dataset class for trajectory data using torch_geometric's Dataset.

    This class loads trajectory data from disk, supports train/val/test splits,
    and can optionally use a small subset for quick experiments.

    Attributes
    ----------
    root : str
        Root directory of the dataset.
    dataset : str
        Name of the dataset.
    split : str
        Split of the dataset, one of ['train', 'val', 'test'].
    files : list
        List of filenames in the dataset split.
    path : str
        Path to the dataset split.
    _num_samples : int
        Number of samples in the dataset.

    Methods
    -------
    len() -> int
        Returns the number of samples in the dataset.
    get(idx: int) -> HeteroData
        Returns a sample from the dataset by index.

    """

    def __init__(
        self,
        root: str,
        dataset: str,
        split: str,
        transform: Callable | None = None,
        *,
        small_data: bool = False,
    ) -> None:
        """Initialize the TrajDataset.

        Args:
            root (str): Root directory of the dataset.
            dataset (str): Name of the dataset.
            split (str): Split of the dataset, one of ['train', 'val', 'test'].
            transform (Callable | None, optional): Optional transform to be applied.
            small_data (bool, optional): If True, use only a small subset of the data.

        """
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=None,
            pre_filter=None,
        )
        if split not in ["train", "val", "test"]:
            msg = "Split must be one of [train, val, test]"
            raise ValueError(msg)

        self.root = root
        self.dataset = dataset
        self.split = split
        self.path = os.path.join(self.root, self.dataset, self.split)
        self.files = os.listdir(self.path)
        self.files = sorted(
            self.files,
        )  # sort files for consistency across operating systems

        if small_data:
            self.files = self.files[:100]

        self._num_samples = len(self.files)

        if self._num_samples == 0:
            msg = f"No files found in {self.path} for split '{self.split}'"
            raise ValueError(msg)

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        """Get a sample from the dataset by index."""
        with open(os.path.join(self.path, self.files[idx]), "rb") as f:
            return HeteroData(pickle.load(f))


if __name__ == "__main__":
    ds = TrajDataset(root="data", dataset="highD", split="test")
    print(len(ds))
