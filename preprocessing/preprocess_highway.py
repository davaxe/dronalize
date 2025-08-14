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

import math
import os
import pickle
import time
import warnings
from multiprocessing import Lock, Pool, Value
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils import (
    # utils/common.py:
    class_list_to_int_list,
    create_directories,
    create_tensor_dict,
    erase_previous_line,
    get_features,
    get_maneuver,
    get_meta_property,
    get_neighbors,
    get_other_sets,
    # utils/highway_utils.py:
    preprocess_ad4che,
    # utils/exit_utils.py:
    preprocess_exid,
    preprocess_highd,
    preprocess_isac,
    preprocess_ngsim,
    update_frames,
)

LANE_KEEP_INTENT = 3  # Intent for lane keeping

worker_counter: Any
worker_lock: Any


def init_worker(counter, lock) -> None:
    """Initialize the worker with a shared counter and lock."""
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    """Call process_id with multiple arguments."""
    return process_id(*arg)


def process_id(
    id0: int,
    rec_id: str,
    out_dir: str,
    fr_dict: dict,
    tr_meta: pd.DataFrame,
    tr: pd.DataFrame,
    ln_graph: dict,
    current_set: str = "train",
    dataset: str = "highD",
    fz: int = 25,
    input_len: int = 2,
    output_len: int = 5,
    n_inputs: int = 7,
    n_outputs: int = 7,
    ds_factor: int = 5,
    filt_ord: int = 7,
    skip: int = 12,
    add_supp: bool = False,
    debug: bool = False,
) -> None:
    """Extract the data for a given set of frames and save it to a pickle file.

    Args:
        id0 (int): The track_id of the target vehicle
        rec_id (str): The ID of the recording
        out_dir (str): Output directory
        current_set (str): The current set (train, val, test)
        fr_dict (dict): The frames to extract
        tr_meta (pd.DataFrame): The meta-data of the tracks
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lane graph
        fz (int): The original sampling frequency (Hz)
        input_len (int): The length of the input sequence
        output_len (int): The length of the output sequence
        n_inputs (int): The number of input features
        n_outputs (int): The number of output features
        ds_factor (int): The down-sampling factor
        filt_ord (int): The filter order
        skip (int): The number of frames to skip
        dataset (str): The dataset name
        add_supp (bool): Additional data
        debug (bool): Debug mode

    Returns:
        None

    """
    # Check if supplementary data should be added
    if add_supp:
        add_feats = ["laneDisplacement", "roadDisplacement"]
        n_inputs += 2
    else:
        add_feats = None

    not_set = get_other_sets(current_set)

    if not_set is None:
        not_set = ["val", "test"]
    df = tr[tr.track_id == id0]
    frames = df.frame.to_numpy()

    # Check if len(frames) > 1:
    if len(frames) <= 1:
        return

    # Remove frames that are not in the current set
    frames = update_frames(frames, fr_dict[not_set[0]], fr_dict[not_set[1]])

    if len(frames) < fz * (input_len + output_len) + 1:
        return

    driving_dir = int(tr_meta[tr_meta.track_id == id0].drivingDirection.iloc[0])

    # First, we filter out the frames where the target vehicle is performing a lane keep
    # that way we can sample more frames for lane changes
    lk_frames = []
    lc_frames = []
    for frame in frames[::skip]:
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames:
            break
        ta_intent = get_maneuver(tr, prediction_frame - 1, [id0], prop="maneuver")[0]
        if ta_intent == LANE_KEEP_INTENT:
            lk_frames.append(frame)
        else:
            lc_frames.append(frame)

    n_lc = len(lc_frames)
    n_lk = len(lk_frames)

    # Our goal is to not sample more lane keep frames than lane change frames
    keep_lk = min(n_lc, n_lk) if n_lc > 0 else min(n_lk, 5)

    # The stride is selected such that we retain 'keep_lk' lane keep frames
    k = max(math.ceil(n_lk / (keep_lk + 1)), 1)

    # 'Slicing' the lane keep frames assures that we sample
    # from all parts of the trajectory
    lk_frames = lk_frames[::k]

    # Combine the lane change and lane keep frames
    updated_frames = lc_frames + lk_frames

    for frame in updated_frames:
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len

        sas = get_neighbors(tr, prediction_frame - 1, id0, driving_dir)
        sa_ids = pd.unique(sas.track_id)
        n_sas = len(sa_ids)

        agent_ids = [id0, *sa_ids]

        # Retrieve meta information
        intentions = get_maneuver(tr, prediction_frame - 1, agent_ids, prop="maneuver")
        agent_type = class_list_to_int_list(
            get_meta_property(tr_meta, agent_ids, prop="class"),
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
                add_feats,
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
            additional_features=add_feats,
            intentions=intentions,
        )

        data: dict[str, Any] = {"rec_id": rec_id, "agent": agent}
        data.update(
            ln_graph["upper_map"] if driving_dir == 1 else ln_graph["lower_map"],
        )

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
    tr_meta: pd.DataFrame,
    tr: pd.DataFrame,
    ln_graph: dict,
) -> None:
    """Extract the data for a given set of frames and save it to a pickle file.

    Args:
        current_set (str): The current set (train, val, test)
        rec_id (str): The recording ID
        out_dir (str): Output directory
        fr_dict (dict): The frames to extract
        tr_meta (pd.DataFrame): The meta-data of the tracks
        tr (pd.DataFrame): The trajectory data
        ln_graph (dict): The lane graph

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
    filt_ord = 2 if ds.lower() in ("a43", "i80", "us101") else 7
    skip_lc = config["skip_lc_samples"]
    skip_lk = config["skip_lk_samples"]

    add_supp = args.add_supp
    debug = args.debug

    outer_lc_args = (
        ds,
        fz,
        input_len,
        output_len,
        n_inputs,
        n_outputs,
        ds_factor,
        filt_ord,
        skip_lc,
        add_supp,
        debug,
    )
    outer_lk_args = (
        ds,
        fz,
        input_len,
        output_len,
        n_inputs,
        n_outputs,
        ds_factor,
        filt_ord,
        skip_lk,
        add_supp,
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

    ta_ids = list(tr.track_id.unique())

    frame_range = fr_dict[current_set]
    ta_set = {
        ta_id
        for ta_id in ta_ids
        if any(tr[tr.track_id == ta_id].frame.isin(frame_range))
    }

    # Get the ids of all the TAs that perform lane changes
    lc_ta_ids = {
        ta_id
        for ta_id in ta_set
        if int(tr_meta[tr_meta.track_id == ta_id].numLaneChanges.iloc[0]) > 0
    }

    # Compute the ids of all the TAs that perform lane keeping
    lk_ta_ids = ta_set - lc_ta_ids

    frac = max(len(lc_ta_ids) // 10, 5)

    # Remove some of the lane keeping data
    lk_ta_ids = set(np.random.choice(list(lk_ta_ids), frac, replace=False))

    lc_arguments = [
        (
            ta_id,
            rec_id,
            out_dir,
            fr_dict,
            tr_meta,
            tr,
            ln_graph,
            current_set,
            *outer_lc_args,
        )
        for ta_id in lc_ta_ids
    ]
    lk_arguments = [
        (
            ta_id,
            rec_id,
            out_dir,
            fr_dict,
            tr_meta,
            tr,
            ln_graph,
            current_set,
            *outer_lk_args,
        )
        for ta_id in lk_ta_ids
    ]

    arguments = lc_arguments + lk_arguments

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
                total=len(arguments),
                desc=f"{current_set.capitalize()}",
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
            process_id(*arg)


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

    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    if not os.path.exists(config_file_pth):
        msg = f"Config file {config_file} not found."
        raise FileNotFoundError(msg)

    print(f"Using config file: {config_file} \n")

    with open(config_file_pth, encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]

    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir} \n")

    random_seed = config["seed"]
    np.random.seed(random_seed)

    rec_ids = []
    recordings = config["recordings"]
    for key, value in recordings.items():
        if value["include"]:
            rec_ids.append(key)

    if dataset == "exiD":
        temp_path = os.path.join(args.path, dataset, "maps")
        dirs = os.listdir(temp_path)

        # check if lanelet directory is named "lanelet" instead of "lanelets"
        if "lanelet2" in dirs:
            # update name in directory for consistency
            os.rename(
                os.path.join(temp_path, "lanelet2"),
                os.path.join(temp_path, "lanelets"),
            )

    try:
        for r_id in tqdm(rec_ids, desc="Main process: ", position=0, leave=True):
            print(f"Preprocessing started for recording {r_id}...")

            if dataset.lower() == "highd":
                shared_args = preprocess_highd(
                    args.path,
                    r_id,
                    config,
                    output_dir,
                    random_seed,
                    add_supp=args.add_supp,
                    debug=args.debug,
                )
            elif dataset.lower() == "exid":
                shared_args = preprocess_exid(
                    args.path,
                    r_id,
                    config,
                    output_dir,
                    random_seed,
                    add_supp=args.add_supp,
                    debug=args.debug,
                )
            elif dataset.lower() == "a43":
                shared_args = preprocess_isac(
                    args.path,
                    r_id,
                    config,
                    output_dir,
                    random_seed,
                    add_supp=args.add_supp,
                    debug=args.debug,
                )
            elif dataset.lower() in ("i80", "us101"):
                shared_args = preprocess_ngsim(
                    args.path,
                    r_id,
                    config,
                    output_dir,
                    random_seed,
                    dataset=dataset.lower(),
                    add_supp=args.add_supp,
                    debug=args.debug,
                )
            elif dataset.lower() == "ad4che":
                shared_args = preprocess_ad4che(
                    args.path,
                    r_id,
                    config,
                    output_dir,
                    random_seed,
                    add_supp=args.add_supp,
                    debug=args.debug,
                )
            else:
                msg = f"Unknown dataset: {dataset}"
                raise ValueError(msg)

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
