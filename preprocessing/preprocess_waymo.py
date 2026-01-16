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
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.waymo import get_waymo_scenarios_from_tfrecord
from preprocessing.utils import (
    # utils/common.py:
    class_list_to_int_list,
    compute_acceleration,
    create_directories,
    create_tensor_dict,
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
    """Extract the data for a given set of frames and save it to a pickle file.

    Args:
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
    # 10 + 1 frames for input, 80 frames for output
    frame = 0
    prediction_frame = 10  # The frame at which we want to predict the future
    final_frame = int(
        prediction_frame + fz * output_len,
    )  # 90 frames when output_len=8

    # The target agent will be the ego vehicle (id0)
    id0 = 0
    sa_ids = tr[tr["track_id"] != id0]["track_id"].unique()

    # Group by track_id and keep those with at least one frame <= prediction_frame
    valid_sa_ids = (
        tr[tr["track_id"].isin(sa_ids)]
        .groupby("track_id")
        .filter(lambda df: (df["frame"] <= prediction_frame).any())["track_id"]
        .unique()
    )

    # We want to make sure the target (focal) agent is
    # always first in the list (for scoring purposes)
    agent_ids = [id0, *list(valid_sa_ids)]
    n_agents = len(agent_ids)

    # Retrieve meta information
    agent_type = class_list_to_int_list(
        get_meta_property(tr, agent_ids, prop="category"),
    )

    categories = np.array(
        [
            3 if i == 0 or v else 0
            for i, v in enumerate(
                get_meta_property(tr, agent_ids, prop="of_interest"),
            )
        ],
    )

    # Construct the input and target arrays
    input_array = np.empty((n_agents, fz * input_len + 1, n_inputs))
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
        categories=None if categories.sum() <= 3 else categories,
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
    file, split, split_path, output_dir, config = args_tuple

    file_path = os.path.join(split_path, file)

    scene_iterator = get_waymo_scenarios_from_tfrecord(file_path)

    for scene in scene_iterator:
        rec_id = scene.id
        try:
            lane_graph = scene.map.build(
                interp_distance=3.0,
            ).to_torch_graph()
        except Exception as e:
            print(f"Error building lane graph for {rec_id}: {e}")
            continue
        tracks = scene.scenario_data

        tracks["track_id"] += 1
        tracks[["ax", "ay"]] = 0.0

        if not args.debug:
            # Compute velocity and acceleration using finite differences
            tracks = compute_acceleration(tracks, 0.1)

        process_id(
            rec_id=rec_id,
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
    if args.debug:
        print("DEBUG MODE: ON\n")

    config_file = args.config if args.config.endswith(".yml") else args.config + ".yml"
    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    if not os.path.exists(config_file_pth):
        msg = f"Config file {config_file} not found."
        raise FileNotFoundError(msg)

    with open(config_file_pth, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]
    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir} \n")

    train_path = os.path.join(args.path, dataset, "training")
    val_path = os.path.join(args.path, dataset, "validation")
    test_path = os.path.join(args.path, dataset, "testing")

    for split, path in zip(
        ["train", "val", "test"],
        [train_path, val_path, test_path],
        strict=False,
    ):
        if not os.path.exists(path):
            msg = f"Path {path} does not exist."
            raise FileNotFoundError(msg)

        files = sorted(os.listdir(path))

        tasks = [(f, split, path, output_dir, config) for f in files]

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
