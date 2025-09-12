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
import warnings
from multiprocessing import Lock, Pool, Value
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.argoverse1 import Argoverse1MapGraphBuilder
from preprocessing.utils import (
    # utils/common.py:
    class_list_to_int_list,
    compute_acceleration,
    compute_velocity,
    create_directories,
    create_tensor_dict,
    erase_previous_line,
    get_features,
    get_meta_property,
)

# Global variables for worker processes
worker_counter: Any
worker_lock: Any


def init_worker(counter, lock):
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def process_id(
    rec_id: str,
    out_dir: str,
    tr: pd.DataFrame,
    ln_graph: dict,
    current_set: str,
    dataset: str,
    fz: int,
    input_len: int,
    output_len: int,
    n_inputs: int,
    n_outputs: int,
    ds_factor: int,
    filt_ord: int,
    debug: bool,
) -> None:
    """Extract data for a given set of frames and saves it to a pickle file.

    Args:
        rec_id (str): The ID of the recording
        out_dir (str): Output directory
        current_set (str): The current set (train, val, test)
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lane graph
        fz (int): The original sampling frequency (Hz)
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
    # 20 frames for input, 30 frames for output
    frame = 0
    prediction_frame = 19
    final_frame = 49

    # The target agent is the 'AGENT' object type in the dataset
    id0 = tr[tr["object_type"] == "AGENT"]["track_id"].to_numpy()[0]
    sa_ids = tr[tr["object_type"] != "AGENT"]["track_id"].unique()

    # Group by track_id and keep those with at least one frame <= prediction_frame
    valid_sa_ids = (
        tr[tr["track_id"].isin(sa_ids)]
        .groupby("track_id")
        .filter(lambda df: (df["frame"] <= prediction_frame).any())["track_id"]
        .unique()
    )

    # We want to make sure the target agent is always first in the list
    agent_ids = [id0, *list(valid_sa_ids)]
    n_agents = len(agent_ids)

    # Retrieve meta information
    agent_type = class_list_to_int_list(
        get_meta_property(tr, agent_ids, prop="agent_type"),
    )

    # Construct the input and target arrays
    input_array = np.empty((n_agents, fz * input_len, n_inputs))
    target_array = np.empty((n_agents, fz * output_len, n_outputs))

    for j, v_id in enumerate(agent_ids):
        input_array[j] = get_features(tr, frame, prediction_frame, n_inputs, v_id)
        target_array[j] = (
            get_features(tr, prediction_frame + 1, final_frame, n_outputs, v_id)
            if current_set != "test"
            else 0.0
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

    data: dict[str, Any] = {"rec_id": rec_id, "agent": agent}
    data.update(ln_graph)

    if not debug:
        with worker_lock:
            save_name = f"{dataset}_{current_set}_{worker_counter.value:06d}"
            worker_counter.value += 1

        out_path = os.path.join(out_dir, current_set, f"{save_name}.pkl")
        with open(out_path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def worker_process_file(args_tuple: tuple[str, str, str, str, dict]):
    r_id, split, tracks_path, output_dir, config = args_tuple

    tracks = pd.read_csv(tracks_path, engine="pyarrow")
    tracks = tracks.rename(
        columns={
            "TIMESTAMP": "time",
            "TRACK_ID": "track_id",
            "X": "x",
            "Y": "y",
            "OBJECT_TYPE": "object_type",
        },
    )
    city = (
        tracks["CITY_NAME"].to_numpy()[0]
        if "CITY_NAME" in tracks.columns
        else "unknown"
    )

    # Assign frames based on sorted unique timestamps
    unique_times = np.sort(tracks["time"].unique())
    time_to_frame = {t: i for i, t in enumerate(unique_times)}
    tracks["frame"] = tracks["time"].map(time_to_frame)

    # Add agent_type based on object_type (we only know 'AV' and 'AGENT' as agents)
    tracks["agent_type"] = tracks["object_type"].apply(
        lambda x: "car" if x in ("AV", "AGENT") else "undefined",
    )

    # Add default values for velocity, acceleration, and heading
    tracks[["vx", "vy", "psi", "ax", "ay"]] = 0.0

    if not args.debug:
        # Compute velocity and acceleration using finite differences
        tracks = compute_velocity(tracks, 0.1)
        tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])
        tracks = compute_acceleration(tracks, 0.1)

    if city == "MIA":
        map_graph = mia_map_graph
    elif city == "PIT":
        map_graph = pit_map_graph
    else:
        msg = f"Unknown city {city}"
        raise ValueError(msg)

    # Get the (x, y) coordinates of the 'AV' object at prediction_frame (center map around AV)
    agent_coords = tracks[(tracks["object_type"] == "AV") & (tracks["frame"] == 19)][
        ["x", "y"]
    ].to_numpy()
    agent_coords = (
        torch.from_numpy(agent_coords).float()
        if agent_coords.size
        else torch.zeros((0, 2), dtype=torch.float)
    )

    lane_graph = map_graph.extract_radius(
        center=agent_coords,
        radius=config["lane_graph_radius"],
        return_as_dict=True,  # Return as a dictionary for compatibility
    )

    ## Return full lane graph if needed:
    # lane_graph = map_graph.to_torch_graph()

    process_id(
        rec_id=r_id.replace(".csv", ""),
        out_dir=output_dir,
        tr=tracks,
        ln_graph=lane_graph,
        current_set=split,
        dataset=config["dataset"],
        fz=config["sample_freq"],
        input_len=config["input_len"],
        output_len=config["output_len"],
        n_inputs=config["n_inputs"],
        n_outputs=config["n_outputs"],
        ds_factor=config["downsample"],
        filt_ord=1,
        debug=args.debug,
    )


if __name__ == "__main__":
    # Set PyTorch threading (solves issues with multiprocessing of lane graphs)
    torch.set_num_threads(1)

    if args.debug:
        print("DEBUG MODE: ON\n")

    config_file = (
        args.config if args.config.endswith(".yml") else args.config + ".yml"
    )
    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    if not os.path.exists(config_file_pth):
        msg = f"Config file {config_file} not found."
        raise FileNotFoundError(msg)

    with open(config_file_pth, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]
    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir} \n")

    train_path = os.path.join(
        args.path,
        dataset,
        "forecasting_train_v1.1",
        "train",
        "data",
    )
    val_path = os.path.join(
        args.path,
        dataset,
        "forecasting_val_v1.1",
        "val",
        "data",
    )
    test_path = os.path.join(
        args.path,
        dataset,
        "forecasting_test_v1.1",
        "test_obs",
        "data",
    )
    maps_path = os.path.join(args.path, dataset, "hd_maps", "map_files")

    # Get the map files for Miami and Pittsburgh
    miami_map = os.path.join(maps_path, "pruned_argoverse_MIA_10316_vector_map.xml")
    pit_map = os.path.join(maps_path, "pruned_argoverse_PIT_10314_vector_map.xml")

    if not os.path.exists(miami_map):
        msg = f"Miami map file {miami_map} not found."
        raise FileNotFoundError(msg)
    if not os.path.exists(pit_map):
        msg = f"Pittsburgh map file {pit_map} not found."
        raise FileNotFoundError(msg)

    # Initialize map graph builders
    mia_map_builder = Argoverse1MapGraphBuilder.from_xml_file(Path(miami_map))
    pit_map_builder = Argoverse1MapGraphBuilder.from_xml_file(Path(pit_map))

    # Build the map graphs
    print("Loading map graphs...")
    mia_map_graph = mia_map_builder.build()
    pit_map_graph = pit_map_builder.build()

    # Erase preprocessing message
    erase_previous_line()

    for split, path in zip(
        ["train", "val", "test"],
        [train_path, val_path, test_path],
        strict=True,
    ):
        if not os.path.exists(path):
            msg = f"Path {path} does not exist."
            raise FileNotFoundError(msg)

        files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

        tasks = [
            (f, split, os.path.join(path, f), output_dir, config) for f in files
        ]

        # Determine starting ID
        set_dir = os.path.join(output_dir, split)
        os.makedirs(set_dir, exist_ok=True)
        existing_files = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(set_dir)
            if f.endswith(".pkl")
        ]
        start_id = max(existing_files) + 1 if existing_files else 0

        save_id_counter = Value("i", start_id)
        save_lock = Lock()

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
            with Pool(
                n_workers,
                initializer=init_worker,
                initargs=(save_id_counter, save_lock),
            ) as pool:
                list(
                    tqdm(
                        pool.imap(worker_process_file, tasks),
                        total=len(tasks),
                        desc=f"{split.capitalize()}",
                    ),
                )
        else:
            init_worker(save_id_counter, save_lock)
            for task in tqdm(tasks, desc=f"{split.capitalize()}"):
                worker_process_file(task)

    print("Finished.")
