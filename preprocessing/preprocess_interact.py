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

import os
import pickle
import time
import warnings
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.urban import get_lane_graph
from preprocessing.utils import (
    # utils/common.py:
    class_list_to_int_list,
    compute_acceleration,
    create_directories,
    create_tensor_dict,
    erase_previous_line,
    # utils/interact_utils.py:
    # classify_ped_bike_robust,
    # classify_ped_bike_simple,
    find_interact_files,
    find_target_vehicle,
    get_features,
    get_meta_property,
)

worker_counter: Any
worker_lock: Any


def init_worker(counter, lock):
    # Attach the counter and lock to the worker
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    # Wrapper function to call extract_by_frame with multiple arguments
    return process_id(*arg)


def process_id(
    case_id: int,
    rec_id: str,
    out_dir: str,
    tr: pd.DataFrame,
    ln_graph: dict,
    current_set: str = "train",
    dataset: str = "INTERACTION",
    fz: int = 10,
    input_len: int = 1,
    output_len: int = 3,
    n_inputs: int = 7,
    n_outputs: int = 7,
    ds_factor: int = 2,
    filt_ord: int = 1,
    debug: bool = False,
) -> None:
    """Extract the data for a given case and save it to a pickle file.

    Args:
        case_id (int): The case ID
        rec_id (str): The ID of the recording
        out_dir (str): Output directory
        current_set (str): The current set (train, val, test)
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lane graph
        fz (int): The sampling frequency
        input_len (int): The length of the input sequence
        output_len (int): The length of the output sequence
        n_inputs (int): The number of input features
        n_outputs (int): The number of output features
        ds_factor (int): The down-sampling factor
        filt_ord (int): The filter order
        dataset (str): The dataset name
        debug (bool): Debug mode

    Returns:
        None

    """
    id0 = find_target_vehicle(tr, case_id)
    case_track = tr[tr["case_id"] == case_id]

    # get the first and final frames of the case
    frame = case_track["frame"].min()
    final_frame = case_track["frame"].max()

    tv_track = case_track[case_track["track_id"] == id0]

    start_time = tv_track["timestamp_ms"].min()
    prediction_time = start_time + 900  # 1 second = 1000 ms
    prediction_frame = tv_track[tv_track["timestamp_ms"] == prediction_time][
        "frame"
    ].to_numpy()[0]

    # Get all track_ids for case
    all_ids = case_track["track_id"].unique()

    # Remove ids that are only present after the prediction frame (impossible to predict)
    all_ids = all_ids[
        np.isin(
            all_ids,
            case_track[case_track["frame"] <= prediction_frame]["track_id"],
        )
    ]

    # All but the target vehicle
    sa_ids = all_ids[all_ids != id0]
    n_sas = len(sa_ids)

    agent_ids = [id0, *sa_ids.tolist()]

    # Retrieve meta information
    agent_type = class_list_to_int_list(
        get_meta_property(case_track, agent_ids, prop="agent_type"),
    )

    # Construct the input and target arrays
    input_array = np.empty((n_sas + 1, fz * input_len, n_inputs))
    target_array = np.empty((n_sas + 1, fz * output_len, n_outputs))

    for j, v_id in enumerate(agent_ids):
        input_array[j] = get_features(
            case_track,
            frame,
            prediction_frame,
            n_inputs,
            v_id,
        )
        if current_set == "test":
            target_array[:] = 0.0
        else:
            target_array[j] = get_features(
                case_track,
                prediction_frame + 1,
                final_frame,
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
        k_max=len(agent_ids),
    )

    data: dict[str, Any] = {"rec_id": rec_id, "agent": agent}
    data.update(ln_graph)

    if not debug:
        with worker_lock:
            save_name = f"{dataset}_{current_set}_{worker_counter.value:05d}"
            worker_counter.value += 1

        with open(f"{out_dir}/{current_set}/{save_name}.pkl", "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def process_ids(
    current_set: str, rec_id: str, out_dir: str, tr: pd.DataFrame, ln_graph: dict
) -> None:
    """Extract the data for a given set of case ids and save it to a pickle file.

    Args:
        current_set (str): The current set (train, val, test)
        rec_id (str): The recording ID
        out_dir (str): Output directory
        fr_dict (dict): The frames to extract
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lanelet graph

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
        debug,
    )

    # Check if there are any saved samples in the current set directory
    set_dir = f"{output_dir}/{current_set}"
    set_dir_path = Path(set_dir)
    if any(set_dir_path.iterdir()):
        # get the highest save_id
        save_ids = [
            int(f.name.split("_")[-1].split(".")[0])
            for f in set_dir_path.iterdir()
            if f.is_file()
        ]
        save_id = max(save_ids) + 1
    else:
        save_id = 0

    save_id_counter = Value("i", save_id)
    save_lock = Lock()

    case_ids = tr["case_id"].unique()

    arguments = [
        (case_id, rec_id, out_dir, tr, ln_graph, current_set, *outer_args)
        for case_id in case_ids
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
                total=len(case_ids),
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
        print("DEBUG MODE: ON \n")

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

    print(f"Using config file: {config_file} \n")

    with config_file_pth.open("r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]

    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir} \n")

    random_seed = config["seed"]
    np.random.seed(random_seed)

    train_path = (
        Path(args.path) / dataset / "train"
    )  # path to the training data (.csv files)
    val_path = (
        Path(args.path) / dataset / "val"
    )  # path to the validation data (.csv files)
    test_path = (
        Path(args.path) / dataset / "test_multi-agent"
    )  # path to the test data (.csv files)
    maps_path = Path(args.path) / dataset / "maps"  # path to the maps (.osm files)

    num_tracks = 0

    try:
        for split, path in zip(
            ["train", "val", "test"],
            [train_path, val_path, test_path],
        ):
            if not Path(path).exists():
                msg = f"Path {path} does not exist."
                raise FileNotFoundError(msg)

            # get the list of files in the current split
            files_wo_split = find_interact_files(path)

            print(f"Starting preprocessing of {split} data...")
            erase_previous_line()

            for r_id in tqdm(
                files_wo_split,
                desc=f"{split.capitalize()} data creation process: ",
                position=0,
                leave=True,
            ):
                print(f"Preprocessing started for recording {r_id}...")

                # get the path to the current trajectory file
                tracks_path = (
                    Path(path)
                    / f"{r_id}_{split if split in ('train', 'val') else 'obs'}.csv"
                )

                # check if the trajectory file exists
                if not Path(tracks_path).exists():
                    msg = f"Trajectory file {tracks_path} does not exist."
                    raise FileNotFoundError(msg)
                tracks = pd.read_csv(tracks_path, engine="pyarrow")

                # Rename columns to match the expected format
                tracks = tracks.rename(columns={"frame_id": "frame"})

                # add the psi column
                tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])

                # add empty columns 'ax', 'ay'
                tracks["ax"] = 0.0
                tracks["ay"] = 0.0
                if not args.debug:
                    # Compute acceleration
                    tracks = compute_acceleration(tracks, 0.1)

                # UNCOMMENT THIS BLOCK TO CLASSIFY AGENTS AS 'PEDESTRIAN' OR 'BICYCLE' (takes more time)
                # if not args.debug:
                #     # classify agents marked as 'pedestrian/bicycle' into separate classes
                #     tracks = classify_ped_bike_robust(tracks)
                # else:
                #     # change all agent types of 'pedestrian/bicycle' to 'pedestrian'
                #     tracks.loc[tracks['agent_type'].isin(['pedestrian/bicycle']), 'agent_type'] = 'pedestrian'

                tracks.loc[
                    tracks["agent_type"].isin(["pedestrian/bicycle"]),
                    "agent_type",
                ] = "pedestrian"

                # get the path to the current map file
                current_map = Path(maps_path) / f"{r_id}.osm"

                # check if the map file exists
                if not current_map.exists():
                    msg = f"Map file {current_map} does not exist."
                    raise FileNotFoundError(msg)

                # get the lane graph
                lane_graph = get_lane_graph(str(current_map), return_torch=True)

                # Erase preprocessing message
                erase_previous_line()

                # Print and immediately erase a "done" message (as an example)
                print("Preprocessing completed.")
                time.sleep(1)  # Just to let the user see the message
                erase_previous_line(double_jump=True)

                process_ids(split, r_id, output_dir, tracks, lane_graph)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting... \n")

    finally:
        print("Finished.")
