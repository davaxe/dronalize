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

import os  # noqa: I001
import pickle
import time
import warnings
from multiprocessing import Lock, Pool, Value
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.opendd import OpenDDMapGraphBuilder
from preprocessing.utils import (
    # utils/opendd_utils.py:
    get_opendd_recordings,
    # utils/common.py:
    class_list_to_int_list,
    compute_acceleration,
    compute_velocity,
    create_directories,
    create_tensor_dict,
    erase_previous_line,
    get_features,
    get_frame_split,
    get_meta_property,
    get_neighbors,
    get_other_sets,
    update_frames,
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
    id0: int,
    rec_id: str,
    out_dir: str,
    fr_dict: dict,
    tr: pd.DataFrame,
    ln_graph: dict,
    current_set: str = "train",
    dataset: str = "openDD",
    fz: int = 30,
    input_len: int = 2,
    output_len: int = 5,
    n_inputs: int = 7,
    n_outputs: int = 7,
    ds_factor: int = 3,
    filt_ord: int = 7,
    skip: int = 60,
    debug: bool = False,
) -> None:
    """Extract the data for a given set of frames and saves it to a pickle file.

    Args:
        id0 (int): The track_id of the target vehicle
        rec_id (str): The ID of the recording
        out_dir (str): Output directory
        current_set (str): The current set (train, val, test)
        fr_dict (dict): The frames to extract
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lane graph
        fz (int): The sampling frequency
        input_len (int): The length of the input sequence
        output_len (int): The length of the output sequence
        n_inputs (int): The number of input features
        n_outputs (int): The number of output features
        ds_factor (int): The down-sampling factor
        filt_ord (int): The filter order
        skip (int): The number of frames to skip
        dataset (str): The dataset name
        debug (bool): Debug mode

    Returns:
        None

    """
    # Check current split
    not_set = get_other_sets(current_set)

    if not_set is None:
        not_set = ["val", "test"]
    df = tr[tr.track_id == id0]
    frames = df.frame.to_numpy()

    # Remove frames that are not in the current set
    frames = update_frames(frames, fr_dict[not_set[0]], fr_dict[not_set[1]])

    if len(frames) < fz * (input_len + output_len) + 1:
        return
    for frame in frames[::skip]:  # Skip every 2 seconds
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames:
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
            determine_scored_ids=True,
        )

        if agent is None:
            # If the TA is a parked vehicle, skip the frame
            continue

        data: dict[str, Any] = {"rec_id": rec_id, "agent": agent}
        data.update(ln_graph)

        if not debug:
            with worker_lock:
                save_name = f"{dataset}_{current_set}_{worker_counter.value:06d}"
                worker_counter.value += 1

            with open(f"{out_dir}/{current_set}/{save_name}.pkl", "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def process_ids(
    current_set: str,
    rec_id: str,
    out_dir: str,
    fr_dict: dict,
    tr: pd.DataFrame,
    ln_graph: dict,
) -> None:
    """Extract the data for a given set of frames and saves it to a pickle file.

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
    filt_ord = 2 if "sind" in ds.lower() else 7
    skip = config["skip_samples"]
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
        skip,
        debug,
    )

    # Check if there are any saved samples in the current set directory
    set_dir = f"{output_dir}/{current_set}"
    if len(os.listdir(set_dir)) > 0:
        # get the highest save_id
        save_ids = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(set_dir)]
        save_id = max(save_ids) + 1
    else:
        save_id = 0

    save_id_counter = Value("i", save_id)
    save_lock = Lock()

    # Get all unique track IDs, excluding trailers
    frame_range = fr_dict[current_set]
    valid_tracks = tr[~tr.agent_type.isin(["trailer"]) & tr.frame.isin(frame_range)]
    ta_ids_set = set(valid_tracks.track_id.unique())

    # Filter out parked vehicles
    first_frame_ids = set(tr[tr.frame == 0].track_id)
    last_frame_ids = set(tr[tr.frame == tr.frame.max()].track_id)
    parked_vehicles = first_frame_ids & last_frame_ids

    ta_ids = list(ta_ids_set - parked_vehicles)

    arguments = [
        (ta_id, rec_id, out_dir, fr_dict, tr, ln_graph, current_set, *outer_args)
        for ta_id in ta_ids
    ]

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
                desc=f"{current_set.capitalize()}",
                position=1,
                leave=False,
            ) as pbar,
        ):
            for _ in pool.imap_unordered(worker_function, arguments):
                pbar.update()
    else:
        for arg in tqdm(
            arguments, desc=f"{current_set.capitalize()}", position=1, leave=False
        ):
            worker_function(arg)


if __name__ == "__main__":
    torch.set_num_threads(1)

    if args.debug:
        print("DEBUG MODE: ON\n")

    config_file = args.config
    if not config_file.endswith(".yml"):
        config_file += ".yml"
    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    if not os.path.exists(config_file_pth):
        msg = f"Config file {config_file} not found."
        raise FileNotFoundError(msg)

    with open(config_file_pth, "r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]
    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir}\n")

    random_seed = config["seed"]
    np.random.seed(random_seed)

    rec_ids = []  # rdb1, rdb2, ..., rdb7
    recordings = config["recordings"]
    for key, value in recordings.items():
        if value["include"]:
            rec_ids.append(key)

    try:
        for r_id in tqdm(rec_ids, desc="Main process: ", position=0, leave=True):
            root = os.path.join(args.path, dataset, f"opendd_v3-{r_id}", r_id)

            if not os.path.exists(root):
                msg = f"Path {root} does not exist."
                raise FileNotFoundError(msg)

            traj_path = os.path.join(root, f"trajectories_{r_id}_v3.sqlite")

            if not os.path.exists(traj_path):
                msg = f"Trajectory file {traj_path} does not exist."
                raise FileNotFoundError(msg)

            map_path = os.path.join(root, f"map_{r_id}", f"map_{r_id}.sqlite")
            if not os.path.exists(map_path):
                msg = f"Map file {map_path} does not exist."
                raise FileNotFoundError(msg)

            map_builder = OpenDDMapGraphBuilder.from_sqlite_file(map_path)
            map_graph = map_builder.build(interp_distance=3.0)

            map_reference = config.get("map_reference", "median")
            if map_reference == "median":
                reference = map_graph.node_positions.median(0).values
                reference = [float(coord) for coord in reference]
            elif map_reference == "world":
                world_ref_path = os.path.join(
                    root, f"geo-referenced_images_{r_id}", f"{r_id}.pgw"
                )
                if not os.path.exists(world_ref_path):
                    msg = f"World reference file {world_ref_path} does not exist."
                    raise FileNotFoundError(msg)

                # Use the world reference coordinates from the pgw file
                with open(world_ref_path) as f:
                    world_data = f.read()
                    # split by newline
                    world_data = world_data.split("\n")
                    # remove empty lines
                    world_data = [line for line in world_data if line.strip()]
                    # get the last two lines
                    reference = [float(coord) for coord in world_data[-2:]]
                    reference = torch.tensor(reference, dtype=torch.float32)
            else:
                msg = f"Unknown map_reference {map_reference}."
                raise ValueError(msg)

            # Normalize the lane graph coordinates
            map_graph.node_positions[:, 0] -= reference[0]
            map_graph.node_positions[:, 1] -= reference[1]

            lane_graph = map_graph.to_torch_graph()

            recording_iterator = get_opendd_recordings(traj_path)

            for tracks in recording_iterator:
                scene_id = tracks["recording"].iloc[0]

                print(f"Preprocessing started for recording {scene_id}...")

                if len(tracks) == 0:
                    continue

                # Normalize track coordinates
                tracks["x"] -= reference[0]
                tracks["y"] -= reference[1]

                # Make all 'category' values lowercase
                tracks["category"] = tracks["category"].str.lower()
                # Map categories 'medium vehicle' to 'car ' and 'heavy vehicle' to 'truck'
                tracks["category"] = tracks["category"].replace(
                    {"medium vehicle": "car", "heavy vehicle": "truck"},
                )
                tracks = tracks.rename(columns={"category": "agent_type"})
                tracks.loc[:, ["vx", "vy", "psi", "ax", "ay"]] = 0.0

                if not args.debug:
                    tracks = compute_velocity(tracks, 1 / 30)
                    tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])
                    tracks = compute_acceleration(tracks, 1 / 30)

                final_frame = tracks["frame"].max().astype(int)

                # Determine train, val, test split (by frames)
                train_frames, val_frames, test_frames = get_frame_split(
                    final_frame,
                    seed=random_seed,
                )

                frame_dict = {
                    "train": train_frames,
                    "val": val_frames,
                    "test": test_frames,
                }

                shared_args = (scene_id, output_dir, frame_dict, tracks, lane_graph)

                tasks = [
                    ("train", *shared_args),
                    ("val", *shared_args),
                    ("test", *shared_args),
                ]

                # Erase preprocessing message
                erase_previous_line()

                # Print and immediately erase a "done" message (as an example)
                print("Preprocessing completed.")
                time.sleep(1)  # Just to let the user see the message
                erase_previous_line(double_jump=True)

                for task in tasks:
                    process_ids(*task)

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        print("Finished.")
