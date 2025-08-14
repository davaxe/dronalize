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
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.road_network.lyftlvl5 import LyftLVL5MapGraphBuilder
from preprocessing.utils import (
    # utils/lyft_utils.py:
    get_lyft_scenes_as_pandas_lazy,
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
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    return process_id(*arg)


def process_id(
    id0: int,
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
    skip: int,
    debug: bool,
) -> None:
    """Extract the data for a given scenario and save it to a pickle file.

    Args:
        id0 (int): The ID of the target agent.
        rec_id (str): The ID of the recording.
        out_dir (str): Output directory.
        tr (pd.DataFrame): The trajectory data.
        ln_graph (dict): The lane graph.
        current_set (str): The current set (train, val, test).
        dataset (str): The dataset name.
        fz (int): The original sampling frequency (Hz).
        input_len (int): The length of the input sequence.
        output_len (int): The length of the output sequence.
        n_inputs (int): The number of input features.
        n_outputs (int): The number of output features.
        ds_factor (int): The down-sampling factor.
        filt_ord (int): The filter order.
        skip (int): The number of frames to skip.
        debug (bool): Debug mode.

    Returns:
        None

    """
    df = tr[tr.track_id == id0]
    frames = df.frame.to_numpy()

    min_required_frames = (
        fz * input_len if current_set == "test" else fz * (input_len + output_len)
    )
    if len(frames) < min_required_frames:
        return

    for frame in frames[::skip]:
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames and current_set != "test":
            break

        sas = get_neighbors(tr, prediction_frame - 1, id0)
        sa_ids = pd.unique(sas.track_id)
        n_sas = len(sa_ids)
        agent_ids = [id0, *sa_ids]

        agent_type = class_list_to_int_list(
            get_meta_property(tr, agent_ids, prop="agent_type"),
        )
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
            target_array[j] = (
                0.0
                if current_set == "test"
                else get_features(
                    tr,
                    prediction_frame,
                    final_frame - 1,
                    n_outputs,
                    v_id,
                )
            )

        agent = create_tensor_dict(
            input_array,
            target_array,
            agent_ids,
            agent_type,
            fz,
            ds_factor,
            filt_ord,
        )

        data = {"rec_id": rec_id, "agent": agent}
        data.update(ln_graph)

        if not debug:
            with worker_lock:
                save_name = f"{dataset}_{current_set}_{worker_counter.value:06d}"
                worker_counter.value += 1
            with open(f"{out_dir}/{current_set}/{save_name}.pkl", "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


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

    train_path = os.path.join(args.path, dataset, "train", "train.zarr")
    val_path = os.path.join(args.path, dataset, "validate", "validate.zarr")

    proto_map = os.path.join(args.path, dataset, "semantic_map", "semantic_map.pb")
    meta_map = os.path.join(args.path, dataset, "semantic_map", "meta.json")

    print("Loading map graph...")
    map_graph = LyftLVL5MapGraphBuilder.from_files(proto_map, meta_map).build(
        interpolate=True,
        interp_distance=3.0,
    )
    erase_previous_line()

    for split, path in zip(["train", "val"], [train_path, val_path]):
        if not os.path.exists(path):
            msg = f"Path {path} does not exist."
            raise FileNotFoundError(msg)

        scenes_iterator = get_lyft_scenes_as_pandas_lazy(path, start=0, batch_size=1000)
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

        for batch_idx, tracks in enumerate(
            tqdm(
                scenes_iterator,
                desc=f"{split.capitalize()}",
                total=16265 if split == "train" else 16220,
            ),
        ):
            rec_id = f"{split}_scene_{batch_idx:06d}"
            if len(tracks) == 0:
                continue

            median_frame = tracks["frame"].median().astype(int)
            agent_coords = (
                tracks[tracks["frame"] == median_frame][["x", "y"]].mean().to_numpy()
            )

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

            tracks = tracks.rename(columns={"category": "agent_type"})
            tracks = tracks[~tracks["agent_type"].isin(["undefined"])].copy()
            tracks.loc[:, ["vx", "vy", "psi", "ax", "ay"]] = 0.0

            if not args.debug:
                tracks = compute_velocity(tracks, 0.1)
                tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])
                tracks = compute_acceleration(tracks, 0.1)

            tracks["track_id"] += 1
            if config.get("ego_only", True):
                ta_ids = [0]
            else:
                ta_ids = list(tracks.track_id.unique())

            task_args = [
                (
                    id0,
                    rec_id,
                    output_dir,
                    tracks,
                    lane_graph,
                    split,
                    dataset,
                    config["sample_freq"],
                    config["input_len"],
                    config["output_len"],
                    config["n_inputs"],
                    config["n_outputs"],
                    config["downsample"],
                    1,
                    config["skip_samples"],
                    args.debug,
                )
                for id0 in ta_ids
            ]

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
                    list(pool.imap_unordered(worker_function, task_args))
            else:
                init_worker(save_id_counter, save_lock)
                for arg in tqdm(task_args, desc=f"{rec_id}", position=1, leave=False):
                    worker_function(arg)

    print("Finished.")
