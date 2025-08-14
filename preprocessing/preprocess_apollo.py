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

import logging
import os
import pickle
import shutil
import time
import warnings
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils import (
    # utils/common.py:
    class_list_to_int_list,
    create_directories,
    create_tensor_dict,
    dummy_map,
    erase_previous_line,
    get_features,
    get_meta_property,
    get_neighbors,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
printer = logging.getLogger(__name__)

object_types = {
    1: "car",  # small vehicle -> car
    2: "truck",  # large vehicle -> truck
    3: "pedestrian",  # pedestrian -> pedestrian
    4: "bicycle",  # motorcyclist and bicyclist -> bicycle
    5: "undefined",  # others -> undefined
}

worker_counter: Any = None
worker_lock: Any = None


def init_worker(counter, lock):
    # Attach the counter and lock to the worker
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    # Wrapper function to call extract_by_frame with multiple arguments
    return process_id(*arg)


def process_id(
    id0: int,
    rec_id: str,
    out_dir: str,
    tr: pd.DataFrame,
    current_set: str = "train",
    dataset: str = "ApolloScape",
    fz: int = 10,
    input_len: int = 2,
    output_len: int = 3,
    n_inputs: int = 7,
    n_outputs: int = 7,
    ds_factor: int = 1,
    filt_ord: int = 1,
    skip: int = 5,
    debug: bool = False,
) -> None:
    """Extract data for a given set of frames and saves it to a pickle file.

    Args:
        id0 (int): The ID of the target agent.
        rec_id (str): The ID of the recording.
        out_dir (str): Output directory.
        current_set (str): The current set (train, val, test).
        tr (pd.DataFrame): The trajectory data.
        fz (int): The orignal sampling frequency (Hz).
        input_len (int): The length of the input sequence.
        output_len (int): The length of the output sequence.
        n_inputs (int): The number of input features.
        n_outputs (int): The number of output features.
        ds_factor (int): The down-sampling factor.
        filt_ord (int): The filter order.
        skip (int): The number of frames to skip.
        dataset (str): The dataset name.
        debug (bool): If True, runs in debug mode without multiprocessing.

    Returns:
        None

    """
    df = tr[tr.track_id == id0]
    frames = df.frame.to_numpy()

    if current_set == "test":
        if len(frames) < fz * input_len:
            # If the current set is test, we need at least input_len
            return
    elif len(frames) < fz * (input_len + output_len):
        # If the current set is not test, we need at least input_len + output_len
        # to create ground truth
        return
    for frame in frames[::skip]:  # Skip every skip-th frame to reduce the overlap
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames and current_set != "test":
            # If the final frame is not in the frames, we cannot create a sample
            break

        sas = get_neighbors(tr, prediction_frame - 1, id0)
        sa_ids = pd.unique(sas.track_id)
        n_sas = len(sa_ids)

        agent_ids = [id0, *sa_ids]

        # Retrieve meta information
        agent_type = class_list_to_int_list(
            get_meta_property(tr, agent_ids, prop="agent_type"),
        )

        # Create the input and target arrays
        input_array = np.empty((n_sas + 1, fz * input_len, n_inputs))
        target_array = np.empty((n_sas + 1, fz * output_len, n_outputs))

        for j, v_id in enumerate(agent_ids):
            input_array[j] = get_features(
                tr,
                frame,
                prediction_frame - 1,
                n_inputs,
                v_id,
            )
            if current_set == "test":
                target_array[:] = 0.0
            else:
                target_array[j] = get_features(
                    tr,
                    prediction_frame,
                    final_frame - 1,
                    n_outputs,
                    v_id,
                )

        # Create the agent dictionary
        agent = create_tensor_dict(
            input_array,
            target_array,
            agent_ids,
            agent_type,
            fz,
            ds_factor,
            filt_ord,
        )

        # Add dummy map node type and edges (ApolloScape does not provide map data)
        data: dict[str, Any] = {"rec_id": rec_id, "agent": agent} | dummy_map()

        if not debug:
            with worker_lock:
                save_name = f"{dataset}_{current_set}_{worker_counter.value:05d}"
                worker_counter.value += 1

            file_path = Path(out_dir) / current_set / f"{save_name}.pkl"
            with file_path.open("wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def process_ids(
    current_set: str,
    rec_id: str,
    out_dir: str,
    tr: pd.DataFrame,
) -> None:
    """Extract the data for a given set of frames and saves it to a pickle file.

    Args:
        current_set (str): The current set (train, val, test)
        rec_id (str): The recording ID
        out_dir (str): Output directory
        tr (pd.DataFrame): The trajectory data

    Returns:
        None

    """
    if current_set not in ["train", "val", "test"]:
        msg = "current_set must be one of [train, val, test]"
        raise ValueError(msg)

    fz = config["sample_freq"]
    ds = config["dataset"]
    input_len = config["input_len"]
    output_len = config["output_len"]
    n_inputs = config["n_inputs"]
    n_outputs = config["n_outputs"]
    ds_factor = config["downsample"]
    skip_samples = config["skip_samples"]
    filt_ord = 1  # if ds.lower() in ("sind", "interaction") else 7
    debug = args.debug

    outer_args = (
        ds,
        fz,
        input_len,
        output_len,
        n_inputs,
        n_outputs,
        ds_factor,
        filt_ord,
        skip_samples,
        debug,
    )

    # Check if there are any saved samples in the current set directory
    set_dir = Path(output_dir) / current_set
    if any(set_dir.iterdir()):
        # get the highest save_id
        save_ids = [int(f.stem.split("_")[-1]) for f in set_dir.iterdir()]
        save_id = max(save_ids) + 1
    else:
        save_id = 0

    save_id_counter = Value("i", save_id)
    save_lock = Lock()

    ta_ids = list(tr.track_id.unique())

    arguments = [
        (ta_id, rec_id, out_dir, tr, current_set, *outer_args) for ta_id in ta_ids
    ]

    if args.debug:
        for arg in arguments:
            worker_function(arg)
        return

    n_workers = 1
    if args.use_threads:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            warnings.warn(
                "Could not determine the number of CPU cores. Using 1 thread.",
                stacklevel=2,
            )
        elif cpu_count <= 2:
            warnings.warn(
                "The number of CPU cores is too low. Using 1 thread.",
                stacklevel=2,
            )
        else:
            n_workers = cpu_count

        with (
            Pool(
                n_workers,
                initializer=init_worker,
                initargs=(save_id_counter, save_lock),
            ) as pool,
            tqdm(
                total=len(ta_ids),
                desc=f"{rec_id}",
                position=1,
                leave=False,
            ) as pbar,
        ):
            for _ in pool.imap_unordered(worker_function, arguments):
                pbar.update()

    else:
        for arg in tqdm(
            arguments,
            desc=f"{current_set.capitalize()}",
            position=1,
            leave=False,
        ):
            worker_function(arg)


if __name__ == "__main__":
    if args.debug:
        printer.info("DEBUG MODE: ON \n")

    if not args.use_threads:
        # Initialize the global variables for single-threaded (non-multiprocessing) mode
        worker_counter = Value("i", 0)
        worker_lock = Lock()

    config_file = args.config
    if not config_file.endswith(".yml"):
        config_file += ".yml"

    config_file_pth = Path("preprocessing") / "configs" / config_file

    if not config_file_pth.exists():
        msg = f"Config file {config_file} not found."
        raise FileNotFoundError(msg)

    printer.info(f"Using config file: {config_file} \n")

    with config_file_pth.open("r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]

    ori_dt = config["ori_dt"]  # Original time step in ApolloScape dataset
    dt = config["dt"]

    if dt > ori_dt:
        msg = f"dt ({dt}) must be less than or equal to the original dt ({ori_dt})."
        raise ValueError(msg)

    interpolation = False
    if dt < ori_dt:
        printer.info(
            f"Interpolation enabled: dt ({dt}) is less than original dt ({ori_dt}). \n",
        )
        interpolation = True

    config["sample_freq"] = int(1 / dt)  # Update sample frequency based on the new dt

    output_dir = create_directories(args, dataset)
    printer.info(f"Output directory: {output_dir} \n")

    random_seed = config["seed"]
    rng = np.random.default_rng(random_seed)

    # Get path to 'prediction_train'
    all_train_path = Path(
        args.path, dataset, "prediction_train",
    )  # path to the training data (.txt files)
    if not all_train_path.exists():
        msg = f"Path {all_train_path} does not exist."
        raise FileNotFoundError(msg)

    # Get all the files in the 'prediction_train' directory
    all_train_files = [
        f.name
        for f in all_train_path.iterdir()
        if f.is_file() and f.name.endswith(".txt")
    ]
    all_train_files = sorted(all_train_files)
    if len(all_train_files) == 0:
        msg = f"No training files found in {all_train_path}."
        raise ValueError(msg)

    # Create paths to train_split and val_split
    train_path = (
        Path(args.path) / dataset / "train_split"
    )  # path to the training data (.txt files)
    val_path = (
        Path(args.path) / dataset / "val_split"
    )  # path to the validation data (.txt files)

    # Create directories if they do not exist
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # Clean old files before copying new ones
    for folder in [train_path, val_path]:
        for f in folder.iterdir():
            if f.suffix == ".txt":
                f.unlink()

    # Split the files into train and val sets (80% train, 20% val) randomly
    rng.shuffle(all_train_files)
    split_index = int(len(all_train_files) * 0.8)
    train_files = all_train_files[:split_index]
    val_files = all_train_files[split_index:]

    # Copy files to respective split folders
    for f in train_files:
        shutil.copy(all_train_path / f, train_path / f)
    for f in val_files:
        shutil.copy(all_train_path / f, val_path / f)

    test_path = (
        Path(args.path) / dataset / "prediction_test"
    )  # path to the test data (single .txt file)

    try:
        for split, path in zip(
            ["train", "val", "test"],
            [train_path, val_path, test_path],
        ):
            if not Path(path).exists():
                msg = f"Path {path} does not exist."
                raise FileNotFoundError(msg)

            # get the list of files in the current split
            files = [f.stem for f in path.iterdir() if f.suffix == ".txt"]

            for r_id in tqdm(
                files,
                desc=f"{split.capitalize()} data creation process: ",
                position=0,
                leave=True,
            ):
                printer.info(f"Preprocessing started for recording {r_id}...")

                # get the path to the current trajectory file
                tracks_path = path / f"{r_id}.txt"

                # check if the trajectory file exists
                if not tracks_path.exists():
                    msg = f"Trajectory file {tracks_path} does not exist."
                    raise FileNotFoundError(msg)

                columns = [
                    "frame",
                    "track_id",
                    "object_type",
                    "x",
                    "y",
                    "z",
                    "length",
                    "width",
                    "height",
                    "heading",
                ]
                tracks = pd.read_csv(
                    tracks_path, sep=r"\s+", header=None, names=columns,
                )

                # Keep only relevant columns and convert frame to time
                #  (2Hz -> 0.5s intervals)
                tracks = tracks[["frame", "track_id", "object_type", "x", "y"]]
                tracks["time"] = tracks["frame"] * ori_dt

                # Interpolation target
                global_start = tracks["time"].min()
                global_end = tracks["time"].max()
                time_grid = np.arange(global_start, global_end + 1e-6, dt)

                # Interpolate per object and store synchronized rows
                interpolated_rows = []

                for obj_id, group_i in tracks.groupby("track_id"):
                    group = group_i.sort_values("time")
                    if group.shape[0] < 2:
                        continue  # Skip if not enough points

                    t_orig = group["time"].to_numpy()
                    x_orig = group["x"].to_numpy()
                    y_orig = group["y"].to_numpy()
                    obj_type = group["object_type"].iloc[0]

                    # Only interpolate within the object's observed time range
                    t_target = time_grid[
                        (time_grid >= t_orig[0]) & (time_grid <= t_orig[-1])
                    ]

                    if interpolation:
                        fx = interp1d(
                            t_orig,
                            x_orig,
                            kind="linear",
                            fill_value="extrapolate",
                        )
                        fy = interp1d(
                            t_orig,
                            y_orig,
                            kind="linear",
                            fill_value="extrapolate",
                        )

                        x_interp = fx(t_target)
                        y_interp = fy(t_target)
                    else:
                        x_interp = x_orig
                        y_interp = y_orig

                    vx = np.gradient(x_interp, dt)  # Velocity in x
                    vy = np.gradient(y_interp, dt)  # Velocity in y

                    heading = np.arctan2(vy, vx)  # Orientation based on velocity

                    ax = np.gradient(vx, dt)  # Acceleration in x
                    ay = np.gradient(vy, dt)  # Acceleration in y

                    temp_df = pd.DataFrame(
                        {
                            "time": t_target,
                            "track_id": obj_id,
                            "agent_type": object_types[obj_type],
                            "x": x_interp,
                            "y": y_interp,
                            "vx": vx,
                            "vy": vy,
                            "ax": ax,
                            "ay": ay,
                            "psi": heading,
                        },
                    )
                    interpolated_rows.append(temp_df)

                # Combine all into a synchronized interpolated DataFrame
                tracks_sync = pd.concat(interpolated_rows, ignore_index=True)

                # Assign new frame indices based on interpolated time grid
                tracks_sync["frame"] = (
                    ((tracks_sync["time"] - tracks_sync["time"].min()) / dt)
                    .round()
                    .astype(int)
                )

                # Erase preprocessing message
                erase_previous_line()

                # Print and immediately erase a "done" message (as an example)
                printer.info("Preprocessing completed.")
                time.sleep(1)  # Just to let the user see the message
                erase_previous_line(double_jump=True)

                process_ids(split, r_id, output_dir, tracks_sync)

    except KeyboardInterrupt:
        printer.info("\n\nProcess interrupted by user. Exiting... \n")

    finally:
        printer.info("Finished.")
