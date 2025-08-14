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
from multiprocessing import Lock, Pool, Value
from typing import Any
import warnings  # noqa: UP035

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.vod import VODMapGraphBuilder
from preprocessing.utils import (
    # utils/vod_utils.py:
    VODData,
    # utils/vod_split.py:
    vod_split,
    # utils/common.py:
    class_list_to_int_list,
    compute_acceleration,
    compute_velocity,
    create_directories,
    create_tensor_dict,
    erase_previous_line,
    get_features,
    get_meta_property,
    get_neighbors,
)

worker_counter: Any = None
worker_lock: Any = None


def init_worker(counter, lock):
    # Attach the counter and lock to the worker
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def process_id(
    id0: int,
    rec_id: str,
    out_dir: str,
    tr: pd.DataFrame,
    ln_graph: dict,
    current_set: str = "train",
    dataset: str = "vod",
    fz: int = 10,
    input_len: float = 0.5,
    output_len: int = 3,
    n_inputs: int = 7,
    n_outputs: int = 7,
    ds_factor: int = 1,
    filt_ord: int = 1,
    skip: int = 5,
    debug: bool = False,
) -> None:
    """Extract the data for a given set of frames and save it to a pickle file.

    Args:
        id0 (int): The ID of the target agent
        rec_id (str): The ID of the recording
        out_dir (str): Output directory
        current_set (str): The current set (train, val, test)
        tr (pd.DataFrame): The trajectory data
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
    df = tr[tr.track_id == id0]
    frames = df.frame.to_numpy()

    if current_set == "test":
        if len(frames) < fz * input_len:
            # If the current set is test, we need at least input_len
            return
    elif len(frames) < fz * (input_len + output_len):
        # If the current set is not test, we need at least input_len + output_len
        return
    for frame in frames[::skip]:  # Skip every skip-th frame to reduce the overlap
        prediction_frame = int(frame + fz * input_len)
        final_frame = int(prediction_frame + fz * output_len)
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
        input_array = np.empty((n_sas + 1, int(fz * input_len), n_inputs))
        target_array = np.empty((n_sas + 1, int(fz * output_len), n_outputs))

        for j, v_id in enumerate(agent_ids):
            input_array[j] = get_features(
                tr,
                frame,
                prediction_frame - 1,
                n_inputs,
                v_id,
            )
            if current_set == "test":
                target_array[j] = 0.0
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
    tr: pd.DataFrame,
    ln_graph: dict,
    dataset: str,
    fz: int,
    input_len: float,
    output_len: int,
    n_inputs: int,
    n_outputs: int,
    ds_factor: int,
    filt_ord: int,
    debug: bool,
) -> None:
    global worker_counter, worker_lock

    ta_ids = list(tr.track_id.unique())

    # Filter track_ids
    ta_ids = [
        ta_id
        for ta_id in ta_ids
        if not tr[tr.track_id == ta_id]
        .agent_type.isin(["movable_object", "static_object", "undefined"])
        .any()
    ]
    ta_ids = [
        ta_id
        for ta_id in ta_ids
        if not tr[tr.track_id == ta_id].status.isin(["parked"]).any()
    ]

    for id0 in ta_ids:
        process_id(
            id0=id0,
            rec_id=rec_id,
            out_dir=out_dir,
            tr=tr,
            ln_graph=ln_graph,
            current_set=current_set,
            dataset=dataset,
            fz=fz,
            input_len=input_len,
            output_len=output_len,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            ds_factor=ds_factor,
            filt_ord=filt_ord,
            skip=config["skip_samples"],
            debug=debug,
        )


def worker_process_file(args_tuple: tuple[str, str, str, str, dict]):
    r_id, split, tracks_path, output_dir, config = args_tuple

    tracks = pd.read_csv(tracks_path, engine="pyarrow")
    location: str = tracks["map"].to_numpy()[0]

    map_graph = map_graphs.get(location)
    if map_graph is None:
        msg = f"Unknown location {location}"
        raise ValueError(msg)

    # Median frame
    median_frame = tracks["frame"].median().astype(int)
    agent_coords = tracks[tracks["frame"] == median_frame][["x", "y"]].mean().to_numpy()

    lane_graph = map_graph.extract_radius(
        center=agent_coords,
        radius=config["lane_graph_radius"],
        return_as_dict=True,
    )

    if args.debug:
        lane_graph_center = lane_graph["map_point"]["position"].mean(dim=0)
        dist = ((agent_coords - lane_graph_center.numpy()) ** 2).sum() ** 0.5
        if dist > config["lane_graph_radius"]:
            msg = (
                f"Agent coordinates {agent_coords} are outside the lane graph radius "
                f"({config['lane_graph_radius']}). Please check the configuration."
            )
            raise ValueError(msg)

    # remove all rows where agent_type is in ["movable_object", "static_object"]
    tracks = tracks[~tracks["agent_type"].isin(["movable_object", "static_object"])]

    dt = config.get("dt", 0.1)
    tracks[["vx", "vy", "psi", "ax", "ay"]] = 0.0

    if not args.debug:
        tracks = compute_velocity(tracks, dt)
        tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])
        tracks = compute_acceleration(tracks, dt)

    process_ids(
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

    if not args.use_threads:
        # Initialize globals for single-threaded mode
        worker_counter = Value("i", 0)
        worker_lock = Lock()

    # Resolve config file
    config_file = args.config
    if not config_file.endswith(".yml"):
        config_file += ".yml"
    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    if not os.path.exists(config_file_pth):
        msg = f"Config file {config_file} not found."
        raise ValueError(msg)

    print(f"Using config file: {config_file}\n")

    # Load config
    with open(config_file_pth, "r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]
    random_seed = config["seed"]
    np.random.seed(random_seed)

    # Prepare output
    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir}\n")

    # Path setup
    train_path = os.path.join(args.path, dataset, "train")
    val_path = os.path.join(args.path, dataset, "val")
    test_path = os.path.join(args.path, dataset, "test")

    raw_map_path = os.path.join(args.path, dataset, "maps", "expansion")

    # Check if train, val, and test directories exist and are populated
    if not all(
        os.path.exists(p) and os.listdir(p) for p in [train_path, val_path, test_path]
    ):
        print(
            "Train, val, or test directories are missing or empty."
            " Creating and populating them with CSVs...",
        )

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Load raw vod data
        raw_trainval_path = os.path.join(
            args.path,
            dataset,
            "v1.0-trainval",
        )
        raw_test_path = os.path.join(args.path, dataset, "v1.0-test")

        trainval_data = VODData(raw_trainval_path)
        trainval_scenes = trainval_data.get_scenes_as_pandas()

        test_data = VODData(raw_test_path)
        test_scenes = test_data.get_scenes_as_pandas()

        # Create train, val, and test splits based on nuscenes_split
        train_scenes = [
            scene
            for scene in trainval_scenes
            if scene.scene_name.iloc[0] in vod_split["train"]
        ]
        val_scenes = [
            scene
            for scene in trainval_scenes
            if scene.scene_name.iloc[0] in vod_split["val"]
        ]
        test_scenes = [
            scene
            for scene in test_scenes
            if scene.scene_name.iloc[0] in vod_split["test"]
        ]

        # Save to CSV
        def save_scenes(scenes, target_path, prefix) -> None:
            for scene_df in tqdm(
                scenes,
                desc=f"Saving {prefix} scenes as dataframes",
                position=0,
                leave=False,
            ):
                filename = f"{scene_df.scene_name.iloc[0]}.csv"
                scene_df.to_csv(os.path.join(target_path, filename), index=False)

        save_scenes(train_scenes, train_path, "train")
        save_scenes(val_scenes, val_path, "val")
        save_scenes(test_scenes, test_path, "test")

    # Load map graphs
    print("Loading map graphs...")

    delft_graph = VODMapGraphBuilder.from_json_file(
        os.path.join(raw_map_path, "delft.json"),
    ).build(
        interpolate=True,
        interp_distance=3.0,
        ignore_edge_types={"traffic_light"},
    )

    map_graphs = {
        "delft": delft_graph,
    }

    # Erase preprocessing message
    erase_previous_line()

    # Create processing tasks
    for split, path in zip(["train", "val", "test"], [train_path, val_path, test_path]):
        if not os.path.exists(path):
            msg = f"Path {path} does not exist."
            raise ValueError(msg)
        files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
        tasks = [(f, split, os.path.join(path, f), output_dir, config) for f in files]

        # Determine starting counter value
        set_dir = os.path.join(output_dir, split)
        os.makedirs(set_dir, exist_ok=True)
        existing = [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(set_dir)
            if f.endswith(".pkl")
        ]
        start_id = max(existing) + 1 if existing else 0

        save_id_counter = Value("i", start_id)
        save_lock = Lock()

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
                        desc=split.capitalize(),
                    ),
                )
        else:
            init_worker(save_id_counter, save_lock)
            for task in tqdm(tasks, desc=split.capitalize()):
                worker_process_file(task)

    print("Finished.")
