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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from preprocessing.utils import common

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class PedestrianLoaderConfig:
    """Configuration for pedestrian data loading."""

    data_root: Path
    """Root directory for the pedestrian data, should contain subdirectories for
    each dataset."""

    dataset: str | set[str]
    """Name of the dataset to load, e.g., 'eth', 'hotel', 'univ', 'zara1'."""

    split: Split = "train"
    """Data split to load, can be 'train', 'val', or 'test'."""

    org_obs_len: int = 8
    """Length of the observation sequence in frames."""

    org_pred_len: int = 12
    """Length of the prediction sequence in frames."""

    min_pedestrian: int = 2
    """Minimum number of pedestrians required in a sequence to be valid."""

    org_sample_time: float = 0.4
    """Time interval between frames in seconds."""

    require_all_valid: bool = True
    """If True, requires all pedestrians in a sequence to have valid positions."""

    interpolation_factor: int = 4
    """Factor by which to upsample the trajectory data after loading."""

    multiple_targets_per_window: bool = False
    """If True, allows multiple target pedestrians per sequence window."""

    window_skip: int = 1
    """Number of samples between sliding windows in the sequence."""

    sep: str = "\t"
    """Separator used in the csv-like data files"""


class PedestrianDataSample(TypedDict):
    """TypedDict for a single pedestrian data sample."""

    num_nodes: int
    ta_index: int
    type: torch.Tensor
    inp_pos: torch.Tensor
    inp_vel: torch.Tensor
    inp_acc: torch.Tensor
    inp_yaw: torch.Tensor
    trg_pos: torch.Tensor
    trg_vel: torch.Tensor
    trg_acc: torch.Tensor
    trg_yaw: torch.Tensor
    input_mask: torch.Tensor
    valid_mask: torch.Tensor
    ma_mask: torch.Tensor
    sa_mask: torch.Tensor


class PedestrianSampleLoader:
    """Class to load pedestrian data from various datasets."""

    def __init__(self, config: PedestrianLoaderConfig) -> None:
        """Initialize the pedestrian data loader.

        Args:
            config: Configuration data for loading pedestrian data.

        """
        self.config: PedestrianLoaderConfig = config
        # Each sample: (trajectory_index, valid_mask_index start_frame,
        # end_frame, target_pedestrian_index)
        self.sequences: list[tuple[int, int, int, int, int]] = []
        # Each sample: boolean mask indicating valid pedestrians for the sequence
        # (i.e., not all NaN values)
        self.valid_mask: list[npt.NDArray[np.bool]] = []
        self.trajectories: list[npt.NDArray[np.float32]] = []
        self._load_data()

    @property
    def sequence_length(self) -> int:
        """Get the length of the sequences."""
        return self.observation_length + self.prediction_length

    @property
    def observation_length(self) -> int:
        """Get the length of the observation sequences."""
        return self.config.org_obs_len

    @property
    def prediction_length(self) -> int:
        """Get the length of the prediction sequences."""
        return self.config.org_pred_len

    def _load_data(self) -> None:
        """Load pedestrian data from the configured dataset."""
        for _file, data in self._load_data_iter():
            self._compute_sequences_from_trajectory(
                data,
            )

    def _compute_sequences_from_trajectory(
        self,
        org_data: pd.DataFrame,
    ) -> None:
        """Compute sequences from a single trajectory.

        Args:
            org_data: Original data DataFrame containing pedestrian data.
            interp_data: Interpolated data DataFrame with upsampled trajectories.

        Returns:
            list[tuple[int, int, int]]: (trajectory_index, start_frame, end_frame)
                tuples.

        """
        trajectory = self._file_trajectory(org_data)
        frames = org_data["frame"].max() + 1

        index = len(self.trajectories)
        self.trajectories.append(trajectory)
        sequence_len = self.config.org_obs_len + self.config.org_pred_len
        for start in range(
            0,
            frames - sequence_len + 1,
            self.config.window_skip,
        ):
            end = start + sequence_len
            window = trajectory[:, start:end]
            res = self._is_valid_sequence(window)
            if res is None:
                continue

            valid_mask, target_ped = res
            valid_mask_index = len(self.valid_mask)
            self.valid_mask.append(valid_mask)
            for target_ped_index in target_ped:
                self.sequences.append((
                    index,
                    valid_mask_index,
                    start,
                    end,
                    target_ped_index,
                ))
                if not self.config.multiple_targets_per_window:
                    break

    def _is_valid_sequence(
        self,
        window: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.int64]] | None:
        """Check if a sequence window is valid to create a sample."""
        pred_index = self.config.org_obs_len - 1
        valid_mask: npt.NDArray[np.bool] = np.asarray(
            # If all values in the window are not all NaN
            ~np.isnan(window).all(axis=(1, 2))
            # If the pedestrian has a valid position at the prediction index
            & ~np.isnan(window[:, pred_index, :]).any(axis=1),
            dtype=bool,
        )

        if self.config.require_all_valid:
            # If all values in the window are not NaN for all pedestrians
            valid_mask &= ~np.isnan(window).any(axis=(1, 2))

        if valid_mask.sum() < self.config.min_pedestrian:
            return None

        # A target pedestrian is one with valid data for the full widow, i.e.,
        # not any NaN values in the window.
        target_ped = np.where(np.all(~np.isnan(window), axis=(1, 2)))[0]
        if target_ped.size == 0:
            return None

        return valid_mask, target_ped

    def _file_trajectory(self, data_file: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Process a single data file to extract trajectories.

        Trajectories are stored in a 3D numpy array with shape
        (num_pedestrians, time_steps, 2), where:
            - num_pedestrians: Number of unique pedestrians in the file.
            - time_steps: Number of time steps (frames) in the file.
            - 2: represent the x and y coordinates.

        Important: time instances where no data is available for a pedestrian
        are filled with NaN values!

        Args:
            data_file: DataFrame containing pedestrian data.

        Returns:
            npt.NDArray[np.float32]: 3D numpy array of trajectories.

        """
        num_pedestrians = data_file["id"].nunique()
        end_time = data_file["frame"].max() + 1
        start_time = data_file["frame"].min()
        time_steps = end_time - start_time

        trajectories = np.full(
            (num_pedestrians, time_steps, 2),
            fill_value=np.nan,
            dtype=np.float32,
        )
        ped_to_idx: dict[int, int] = {}
        for frame, obj_id, x, y in data_file.itertuples(index=False, name=None):
            ped_id = int(obj_id)
            if ped_id not in ped_to_idx:
                ped_to_idx[ped_id] = len(ped_to_idx)

            ped_idx = ped_to_idx[ped_id]
            frame_idx = int(frame) - start_time
            trajectories[ped_idx, frame_idx, :] = np.array(
                (x, y),
                dtype=np.float32,
            )

        return trajectories

    def _load_data_iter(self) -> Iterable[tuple[str, pd.DataFrame]]:
        """Load pedestrian data files from the dataset directory.

        Returns:
            Iterable[pd.DataFrame]: An iterable that yields DataFrames for each
            pedestrian data file in the dataset directory.

        Yields:
            pd.DataFrame: DataFrame containing the pedestrian data.

        """
        if not isinstance(self.config.dataset, set):
            datasets = {self.config.dataset}
        else:
            datasets = self.config.dataset

        for dataset in datasets:
            data_dir = self.config.data_root / dataset / self.config.split
            for data_file in data_dir.iterdir():
                data = _read_data(data_file, sep=self.config.sep)
                yield (
                    data_file.name,
                    data,
                )

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __iter__(self) -> Iterator[PedestrianDataSample]:
        """Return an iterator over the pedestrian data sequences."""
        return _PedestrianDataIterator(self)

    def __getitem__(
        self,
        index: int,
    ) -> PedestrianDataSample:
        """Get a sample from the dataset by index.

        Args:
            index: Index in interval [0, len(self))

        Returns:
            PedestrianDataSample: see PedestrianDataSample TypedDict for details.

        """
        seq_index, valid_mask_index, start_frame, end_frame, target_ped = (
            self.sequences[index]
        )

        trajectory = self.trajectories[seq_index]
        # Shift the trajectory sequence to account for the interpolation factor.
        # This is needed to ensure that the prediction starts at the correct
        # frame with respect to the uninterpolated trajectory, i.e, the prediction
        # should start at exactly the same position as the original trajectory.
        # Take account to the shift and reduce the end accordingly
        window = trajectory[:, start_frame:end_frame, :]
        valid_mask_ped = self.valid_mask[valid_mask_index]
        # The trajectory window is reduced to only the valid pedestrians
        sub_trajectory = window[valid_mask_ped]

        #  Map target_ped to the index in the extracted trajectory
        ped_indices = np.where(valid_mask_ped)[0]
        target_ped = int(np.where(ped_indices == target_ped)[0])

        return self._get_features(target_ped, sub_trajectory)

    def _get_features(
        self,
        ta_index: int,
        trajectory: npt.NDArray[np.float32],
    ) -> PedestrianDataSample:
        """Get features from the input position data."""
        dt: float = self.config.org_sample_time / self.config.interpolation_factor
        trajectory = _interpolate(trajectory, self.config.interpolation_factor)
        prediction_index = self.observation_length * self.config.interpolation_factor
        input_pos = trajectory[:, :prediction_index, :]
        target_pos = trajectory[:, prediction_index:, :]

        # Velocity and acceleration using np.gradient
        def _compute_derivatives(
            pos: npt.NDArray[np.float32],
            dt: float,
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            vel = np.gradient(pos, dt, axis=1)  # First derivative
            acc = np.gradient(vel, dt, axis=1)  # Second derivative
            return vel.astype(np.float32), acc.astype(np.float32)

        num_nodes: int = input_pos.shape[0]

        input_mask, target_mask, sa_mask, ma_mask = common.get_masks(
            torch.from_numpy(input_pos),
            torch.from_numpy(target_pos),
            ta_index=ta_index,
            ma_frames=self.prediction_length // 3,
        )

        input_vel, input_acc = _compute_derivatives(input_pos, dt)
        target_vel, target_acc = _compute_derivatives(target_pos, dt)

        input_yaw = np.atan2(input_vel[:, :, 1], input_vel[:, :, 0])[..., np.newaxis]
        target_yaw = np.atan2(target_vel[:, :, 1], target_vel[:, :, 0])[
            ...,
            np.newaxis,
        ]

        input_pos[np.isnan(input_pos)] = 0.0
        input_vel[np.isnan(input_vel)] = 0.0
        input_acc[np.isnan(input_acc)] = 0.0
        input_yaw[np.isnan(input_yaw)] = 0.0

        target_pos[np.isnan(target_pos)] = 0.0
        target_vel[np.isnan(target_vel)] = 0.0
        target_acc[np.isnan(target_acc)] = 0.0
        target_yaw[np.isnan(target_yaw)] = 0.0

        return {
            "num_nodes": num_nodes,
            "ta_index": ta_index,
            "type": torch.full((num_nodes,), fill_value=5).long(),
            "inp_pos": torch.from_numpy(input_pos).float(),
            "inp_vel": torch.from_numpy(input_vel).float(),
            "inp_acc": torch.from_numpy(input_acc).float(),
            "inp_yaw": torch.from_numpy(input_yaw).float(),
            "trg_pos": torch.from_numpy(target_pos).float(),
            "trg_vel": torch.from_numpy(target_vel).float(),
            "trg_acc": torch.from_numpy(target_acc).float(),
            "trg_yaw": torch.from_numpy(target_yaw).float(),
            "input_mask": input_mask,
            "valid_mask": target_mask,
            "sa_mask": sa_mask,
            "ma_mask": ma_mask,
        }


def _interpolate(
    data: npt.NDArray[np.float32],
    factor: int,
) -> npt.NDArray[np.float32]:
    """Interpolate data using linear interpolation.

    Args:
        data: 3D numpy array of shape (N, T, 2) where N is the number of
        factor: Interpolation factor.

    Returns:
        npt.NDArray[np.float32]: Interpolated data of shape (N, T * factor, 2).

    """
    if factor == 1:
        return data

    n, t, d = data.shape
    interp_data = np.full((n, t * factor, d), fill_value=np.nan, dtype=np.float32)

    # Minimum number of points needed for interpolation
    min_points_for_interp = 2

    # Process each pedestrian
    for ped_i in range(n):
        # Get valid sequences for this pedestrian (check x-axis for NaN patterns)
        ped_data_x = data[ped_i, :, 0]

        for start, end in _valid_sequences(ped_data_x):
            # Extract the valid segment for both x and y coordinates
            segment = data[ped_i, start:end, :]  # Shape: (seq_len, 2)
            seq_len = end - start

            if end - start < min_points_for_interp:
                # Can't interpolate with less than 2 points, replicate values
                # to fill the interpolated space
                interp_length = (end - start) * factor
                segment_repeated = np.repeat(segment, factor, axis=0)[:interp_length]
                interp_data[ped_i, start * factor : end * factor, :] = (
                    segment_repeated
                )
                continue

            # Interpolation
            original_indices = np.arange(seq_len)
            target_indices = (
                np.arange((seq_len - 1) * factor + 1, dtype=float) / factor
            )

            interp_segment = np.column_stack([
                np.interp(target_indices, original_indices, segment[:, 0]),
                np.interp(target_indices, original_indices, segment[:, 1]),
            ]).astype(np.float32)

            interp_segment = _extrapolate_back_linear(
                interp_segment,
                n=factor - 1,
            )

            # Store the interpolated segment
            interp_data[ped_i, start * factor : end * factor, :] = interp_segment

    return interp_data


def _valid_sequences(data: npt.NDArray[np.float32]) -> Iterable[tuple[int, int]]:
    valid_mask = ~np.isnan(data)

    # Find the change points: from NaN to valid or valid to NaN
    change = np.diff(valid_mask.astype(int))

    # Start indices of valid sequences: change from 0 → 1
    starts = np.where(change == 1)[0] + 1
    # End indices of valid sequences: change from 1 → 0
    ends = np.where(change == -1)[0] + 1

    # Handle edge cases: sequence starts at index 0 or ends at last element
    if valid_mask[0]:
        starts = np.insert(starts, 0, 0)
    if valid_mask[-1]:
        ends = np.append(ends, len(data))

    # Return list of (start, end) index tuples
    return zip(starts, ends, strict=False)


def _extrapolate_back_linear(
    data: npt.NDArray[np.float32],
    n: int,
) -> npt.NDArray[np.float32]:
    """Extrapolate linear backward `n` steps from the first two time steps.

    Args:
        data: 3D numpy array of shape (N, T, 2) where N is the number of
        n: Number of steps to extrapolate backward.

    Returns:
        npt.NDArray[np.float32]: Extrapolated data of shape (N, T + n, 2).

    """
    if data.ndim == 2:  # (T, 2)  →  promote to (1, T, 2)
        data_ext = data[None, ...]
        squeeze_back = True
    elif data.ndim == 3:  # (N, T, 2)  →  keep as is
        data_ext = data
        squeeze_back = False
    else:
        msg = f"Data must be 2D or 3D, got shape {data.shape}"
        raise ValueError(msg)

    # velocity from the first two frames
    v = data_ext[:, 1, :] - data_ext[:, 0, :]
    k = np.arange(n, 0, -1, dtype=np.float32).reshape(1, n, 1)
    extrapolated = data_ext[:, [0], :] - k * v[:, None, :]
    out = np.concatenate([extrapolated, data_ext], axis=1)
    return (out.squeeze(0) if squeeze_back else out).astype(np.float32)


def _read_data(data_path: Path, sep: str = "\t") -> pd.DataFrame:
    """Read pedestrian data from a file.

    Args:
        data_path: Path to the pedestrian data file.
        sep: Data separator. Defaults to tab.

    Returns:
        pd.DataFrame: DataFrame containing the pedestrian data.

    """

    def str_float_to_int(value: str, div: int = 1) -> int:
        """Convert a string representation of a float to an integer."""
        return int(float(value) // div)

    data_frame = pd.read_csv(
        data_path,
        header=None,
        sep=sep,
        names=["frame", "id", "x", "y"],
        converters={
            # Frames are 0.0, 10.0 etc., for convenience we convert them to integers
            # that can be used as indices.
            "frame": lambda x: str_float_to_int(x, div=10),
            "id": str_float_to_int,
            "x": float,
            "y": float,
        },
    )
    data_frame["frame"] -= data_frame["frame"].min()
    return data_frame


class _PedestrianDataIterator(Iterator):
    """Iterator for pedestrian data sequences.

    This class should not be used directly, but rather through the
    `PedestrianDataLoader` class, which provides an iterable interface.
    """

    def __init__(self, loader: PedestrianSampleLoader) -> None:
        """Initialize the iterator with a PedestrianDataLoader instance."""
        self.loader = loader
        self.index = 0

    def __next__(self) -> PedestrianDataSample:
        """Get the next sample from the iterator.

        Raises:
            StopIteration: If there are no more samples to return.

        Returns:
            PedestrianDataSample: The next sample from the dataset.

        """
        if self.index >= len(self.loader.sequences):
            raise StopIteration

        self.index += 1
        return self.loader[self.index - 1]
