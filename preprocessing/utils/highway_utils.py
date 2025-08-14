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

import numpy as np
from pandas import DataFrame, isna, read_csv

from preprocessing.road_network.ad4che import img_to_map
from preprocessing.road_network.highway import get_highway_graph
from preprocessing.utils import (
    compute_acceleration,
    compute_velocity,
    get_frame_split,
    get_meta_property,
)


def align_origin_w_centroid(
    tracks_meta: DataFrame, tracks: DataFrame, debug: bool = False
) -> DataFrame:
    """
    The coordinates are given wrt the upper left corner of the bounding box
    this function modifies the dataframe such that the coordinates are align
    with the center of the bounding box
    """
    if debug:
        return tracks

    # Nested function to process each group (i.e., each track_id)
    def process_group(df):
        t_id = df["track_id"].iloc[0]
        driving_direction = tracks_meta[
            tracks_meta.track_id == t_id
        ].drivingDirection.values[0]

        # Update y-coordinate by adding half of the height
        df["y"] += df["height"] / 2

        # If driving direction is 2 (right to left), update the x-coordinate by adding half of the width
        if driving_direction == 2:
            df["x"] += df["width"] / 2

        else:
            # If driving direction is 1 (right to left), update the x-coordinate by subtracting half of the width
            df["x"] -= df["width"] / 2

        return df

    # Group by track_id and apply the processing function to each group
    result_df = tracks.groupby("track_id").apply(process_group)

    # Reset index to maintain DataFrame structure
    result_df = result_df.reset_index(drop=True)

    return result_df


def add_heading_feat(tracks: DataFrame, debug: bool = False) -> DataFrame:
    """
    Add heading as a feature to the tracks dataframe
    """
    tracks["psi"] = 0.0

    if debug:
        return tracks

    tracks["psi"] = np.arctan2(tracks["vy"], tracks["vx"])

    return tracks


def add_maneuver(
    tracks_meta: DataFrame, tracks: DataFrame, fz: int = 25, debug: bool = False
) -> DataFrame:
    """Add maneuver as a feature to the tracks dataframe.

    There are 7 different maneuvers:
    0: left lane change within the next (1) second
    1: left lane change within the next 3 seconds
    2: left lane change within the next 5 seconds
    3: no lane change (lane keep)
    4: right lane change within the next (1) second
    5: right lane change within the next 3 seconds
    6: right lane change within the next 5 seconds
    """
    lane_change_left = [0, 1, 2]
    lane_change_right = [4, 5, 6]

    tracks["maneuver"] = (
        np.ones(len(tracks), dtype=int) * 3
    )  # Initialize with 'no lane change'

    if debug:
        return tracks

    # Nested function to process each track_id group
    def process_group(df):
        t_id = df["track_id"].iloc[0]
        num_lane_changes = int(
            tracks_meta[tracks_meta.track_id == t_id].numLaneChanges.iloc[0],
        )

        if num_lane_changes == 0:
            # No lane changes, return the group as is with 'maneuver' already set to 3
            return df

        dr_dir = df.drivingDirection.to_numpy()[0]
        frames = df.frame.to_numpy()
        lanes = df.laneId.to_numpy()

        event_indices = [i for i in range(1, len(lanes)) if lanes[i] != lanes[i - 1]]

        for event_index in event_indices:
            five_seconds_prior = range(max(0, event_index - 5 * fz + 1), event_index)
            three_seconds_prior = range(max(0, event_index - 3 * fz + 1), event_index)
            one_second_prior = range(max(0, event_index - fz + 1), event_index)

            five_second_frames = {frames[i] for i in five_seconds_prior}
            three_second_frames = {frames[i] for i in three_seconds_prior}
            one_second_frames = {frames[i] for i in one_second_prior}

            delta_lane = lanes[event_index] - lanes[event_index - 1]
            if dr_dir == 1:
                maneuvers_type = (
                    lane_change_left if delta_lane > 0 else lane_change_right
                )
            else:
                maneuvers_type = (
                    lane_change_right if delta_lane > 0 else lane_change_left
                )

            df.loc[df.frame.isin(five_second_frames), "maneuver"] = maneuvers_type[2]
            df.loc[df.frame.isin(three_second_frames), "maneuver"] = maneuvers_type[1]
            df.loc[df.frame.isin(one_second_frames), "maneuver"] = maneuvers_type[0]

        return df

    # Group by track_id and apply the processing function to each group
    result_df = tracks.groupby("track_id").apply(process_group)

    # Reset index in case the grouping messes with the original DataFrame's structure
    result_df = result_df.reset_index(drop=True)

    return result_df


def add_driving_direction(
    tracks_meta: DataFrame, tracks: DataFrame, debug: bool = False
) -> DataFrame:
    """Add driving direction (1 or 2) as a feature to the tracks dataframe.

    If driving direction is 1, the vehicle is driving from right to left (negative x).
    If driving direction is 2, the vehicle is driving from left to right (positive x).
    """
    if debug:
        tracks["drivingDirection"] = 1
        return tracks

    # Nested function to process each track_id group
    def process_group(df):
        t_id = df["track_id"].iloc[0]
        driving_direction = tracks_meta[
            tracks_meta.track_id == t_id
        ].drivingDirection.values[0]
        df["drivingDirection"] = (
            driving_direction  # Set driving direction for the group
        )
        return df

    # Group by track_id and apply the processing function to each group
    result_df = tracks.groupby("track_id").apply(process_group)

    # Reset index in case the grouping affects the original DataFrame structure
    result_df = result_df.reset_index(drop=True)

    return result_df


def add_displacement_feat(
    rec_meta: DataFrame, tracks_meta: DataFrame, tracks: DataFrame, debug: bool = False
) -> DataFrame:
    """
    Add roadDisplacement and laneDisplacement as features to the tracks dataframe.
    These features represent the relative position of the vehicle
    with respect to the road and the lane.
    """

    ulm = np.array([float(l) for l in rec_meta["upperLaneMarkings"].iloc[0].split(";")])
    llm = np.array([float(l) for l in rec_meta["lowerLaneMarkings"].iloc[0].split(";")])

    def compute_road_w():
        return ulm[-1] - ulm[0], llm[-1] - llm[0]

    def compute_lane_w():
        return np.mean([np.mean(np.diff(ulm)), np.mean(np.diff(llm))])

    def get_road_edge_markings():
        return ulm[0], llm[0]

    def get_lane_markings():
        return np.concatenate((ulm, llm))

    def get_dyl(y, dd, lm, lw):
        dy = 2 * (y - lm) / lw - 1
        return dy * (-1) if dd == 2 else dy

    def get_dy(y, dd, curr_lane_id, lm, lw):
        dy = 2 * (y - lm[curr_lane_id - 2]) / lw - 1
        return dy * (-1) if dd == 2 else dy

    if debug:
        tracks["laneDisplacement"] = 0.0
        tracks["roadDisplacement"] = 0.0
        return tracks

    # Precompute road-related variables
    ur, lr = get_road_edge_markings()
    ruw, rlw = compute_road_w()
    lm = get_lane_markings()
    lw = compute_lane_w()

    def compute_displacement(df):
        driving_dir = int(
            tracks_meta.loc[tracks_meta.track_id == df.name, "drivingDirection"].iloc[0]
        )
        lane_ids = df["laneId"].values
        y = df["y"].values

        df["laneDisplacement"] = get_dy(y, driving_dir, lane_ids, lm, lw)
        marking, width = (ur, ruw) if driving_dir == 1 else (lr, rlw)
        df["roadDisplacement"] = get_dyl(y, driving_dir, marking, width)

        return df

    return tracks.groupby("track_id", group_keys=False).apply(compute_displacement)


def update_signs(
    rec_meta: DataFrame, tracks_meta: DataFrame, tracks: DataFrame, debug: bool = False
) -> DataFrame:
    """
    We are looking to unify the coordinate system under a
     FLU (frontward-leftward-upward) coordinate system (ISO standard):
    Forward motion = positive x
    Leftward motion = positive y
    (Upward motion = positive z)

    This requires updating the tracks differently depending
    on how the vehicles are moving (driving direction).
    To find the origin of the FLU coordinate system,
     we utilize the lower and upper lane markings.
    Longitudinal motion is updated based on the driving direction.

    """
    if debug:
        return tracks

    try:
        ulm = [float(x) for x in list(rec_meta["upperLaneMarkings"])[0].split(";")]
        llm = [float(x) for x in list(rec_meta["lowerLaneMarkings"])[0].split(";")]
    except KeyError:
        print(
            "Error: 'upperLaneMarkings' or 'lowerLaneMarkings' not found in rec_meta."
        )
        ulm = [0.0]
        llm = [0.0]

    # subtract x_min from all tracks (to make everything start/end at 0)
    x_min = tracks.x.min()
    tracks.x -= x_min

    x_max = tracks.x.max()

    # Define a function to process each track_id group
    def process_group(df):
        t_id = df["track_id"].iloc[0]
        driving_dir = tracks_meta[tracks_meta.track_id == t_id].drivingDirection.values[
            0
        ]

        if driving_dir == 1:
            df["y"] = df["y"] - ulm[0]
            df["x"] = -df["x"] + x_max
            df["vx"] = -df["vx"]
            df["ax"] = -df["ax"]
        else:
            df["y"] = llm[-1] - df["y"]
            df["vy"] = -df["vy"]
            df["ay"] = -df["ay"]

        return df

    # Group by track_id and apply the function to each group
    result_df = tracks.groupby("track_id").apply(process_group)

    # Reset index to maintain DataFrame structure
    result_df = result_df.reset_index(drop=True)

    return result_df


def lane_assignment(
    tr: DataFrame, neg_lanes: int = 0, num_lanes: int = 4, lane_width: float = 3.75
):
    # Function to apply the ffill or bfill logic based on the last value of laneId for each track_id
    def fill_lane_id(group):
        # Check if there are any NaNs in the 'laneId' column
        if group["laneId"].isna().any():
            # Check if the last value in 'laneId' is NaN
            if isna(group["laneId"].iloc[-1]):
                # If the last value is NaN, perform forward fill
                filled = group.ffill()
            else:
                # If the last value is not NaN, perform backward fill
                filled = group.bfill()

            # Explicitly infer objects to handle dtype conversion
            return filled.infer_objects(copy=False)
        else:
            # No NaNs in this group, return as is
            return group

    # Get lateral position of each track
    y = tr["y"].to_numpy()

    # Handle NaN values
    mask_valid = ~np.isnan(y)

    # Initialize the result array with None (default for invalid lanes)
    lane_id = np.full_like(y, None, dtype=object)

    # Calculate the ith_lane using vectorized operations
    ith_lane = np.ceil(y[mask_valid] / lane_width)

    # Mask for valid lane IDs (within the specified range)
    valid_mask = (ith_lane >= 1 - neg_lanes) & (ith_lane <= num_lanes - neg_lanes)

    # Assign valid lanes
    lane_id[mask_valid] = np.where(valid_mask, ith_lane, np.nan)

    # apply to dataframe
    tr["laneId"] = lane_id

    # Group by 'track_id' and apply the fill_lane_id function to each group
    tr["laneId"] = tr.groupby("track_id", group_keys=False).apply(fill_lane_id)[
        "laneId"
    ]

    return tr


def preprocess_highd(
    path: str,
    rec_id: str,
    config: dict,
    output_dir: str,
    seed: int = 42,
    dataset: str = "highD",
    add_supp: bool = False,
    debug: bool = False,
) -> tuple:
    # Construct the base directory path for your data
    base_dir = os.path.join(path, dataset, "data")

    # Use os.path.join for each specific file
    rec_meta_path = os.path.join(base_dir, f"{rec_id}_recordingMeta.csv")
    tracks_meta_path = os.path.join(base_dir, f"{rec_id}_tracksMeta.csv")
    tracks_path = os.path.join(base_dir, f"{rec_id}_tracks.csv")

    # Read the CSV files
    rec_meta = read_csv(rec_meta_path, engine="pyarrow")
    tracks_meta = read_csv(tracks_meta_path, engine="pyarrow")
    tracks = read_csv(tracks_path, engine="pyarrow")

    # For the lanelet file, construct the path similarly
    upper_map, lower_map, x_min, x_max = get_highway_graph(
        rec_meta,
        tracks,
        spacing=config["lane_graph"]["spacing"],
        buffer=config["lane_graph"]["buffer"],
    )
    lane_graph = {"upper_map": upper_map, "lower_map": lower_map}

    # Perform some initial renaming
    if "track_id" not in tracks_meta.columns:
        tracks_meta.rename(columns={"id": "track_id"}, inplace=True)
        tracks.rename(columns={"id": "track_id"}, inplace=True)
    if "vx" not in tracks.columns:
        tracks.rename(columns={"xVelocity": "vx"}, inplace=True)
        tracks.rename(columns={"yVelocity": "vy"}, inplace=True)
        tracks.rename(columns={"xAcceleration": "ax"}, inplace=True)
        tracks.rename(columns={"yAcceleration": "ay"}, inplace=True)
    if "x" not in tracks.columns:
        tracks.rename(columns={"xCenter": "x"}, inplace=True)
        tracks.rename(columns={"yCenter": "y"}, inplace=True)

    # Make class lowercase in tracks_meta
    tracks_meta["class"] = tracks_meta["class"].str.lower()

    tracks = align_origin_w_centroid(tracks_meta, tracks, debug=debug)
    tracks = add_driving_direction(tracks_meta, tracks, debug=debug)
    tracks = add_maneuver(tracks_meta, tracks, debug=debug)
    if add_supp:
        tracks = add_displacement_feat(rec_meta, tracks_meta, tracks, debug=debug)
    tracks = update_signs(rec_meta, tracks_meta, tracks, debug=debug)
    tracks = add_heading_feat(tracks, debug=debug)

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(
        tracks_meta.finalFrame.array[-1], seed=seed
    )
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args


def preprocess_isac(
    path: str,
    rec_id: str,
    config: dict,
    output_dir: str,
    seed: int = 42,
    dataset: str = "A43",
    add_supp: bool = False,
    debug: bool = False,
) -> tuple:
    if add_supp:
        raise NotImplementedError(
            "Support for additional data not implemented for A43 dataset."
        )

    # Construct the base directory path for your data
    base_dir = os.path.join(path, dataset)

    # Use os.path.join for each specific file
    tracks_path = os.path.join(base_dir, rec_id + ".csv")

    # Read the CSV files
    tracks = read_csv(tracks_path, engine="pyarrow")
    tracks["x"] = tracks["x"] - tracks["x"].min()

    # round tseconds to nearest 0.1
    tracks["tseconds"] = np.ceil(tracks["tseconds"] * 10) / 10

    # tracks['frame'] = (tracks['tseconds'] / 0.1).astype(int)
    tracks["frame"] = ((tracks["tseconds"] / 0.1).round()).astype(int)
    tracks["frame"] = tracks["frame"] - tracks["frame"].min()
    final_frame = tracks.frame.max()

    lane_markings = config["recordings"][rec_id]["lane_markings"]
    y0 = [float(l) for l in list(lane_markings)[0].split(";")][0]
    tracks["y"] = tracks["y"] - y0

    # Create a data frame for the rec_meta with two columns: 'upperLaneMarkings' and 'lowerLaneMarkings'
    rec_meta = DataFrame(
        {"upperLaneMarkings": lane_markings, "lowerLaneMarkings": lane_markings}
    )

    # For the lanelet file, construct the path similarly
    _, lower_map, x_min, x_max = get_highway_graph(
        rec_meta,
        tracks,
        spacing=config["lane_graph"]["spacing"],
        buffer=config["lane_graph"]["buffer"],
    )
    lane_graph = {"upper_map": lower_map, "lower_map": lower_map}

    # Perform some initial renaming
    if "track_id" not in tracks.columns:
        tracks.rename(columns={"ID": "track_id"}, inplace=True)
        tracks["track_id"] = tracks["track_id"] - tracks["track_id"].min()
    if "class" not in tracks.columns:
        tracks.rename(columns={"VehicleCategory": "class"}, inplace=True)

        # replace class names
        tracks["class"] = tracks["class"].replace(
            {
                "Passenger Car": "car",
                "Truck": "truck",
                "Van": "van",
                "Bus": "bus",
                "Motorcycle": "motorcycle",
                "Semi-trailer truck": "truck",
            }
        )

    # Assign lane IDs
    tracks = lane_assignment(tracks)

    # get the number of lane changes
    num_lcs = (
        tracks.groupby("track_id")
        .apply(lambda x: x["laneId"].diff().abs().sum())
        .values.astype(int)
    )
    classes = get_meta_property(tracks, tracks["track_id"].unique(), prop="class")
    tracks["drivingDirection"] = 1

    tracks_meta = DataFrame(
        {
            "track_id": tracks["track_id"].unique(),
            "numLaneChanges": num_lcs,
            "drivingDirection": 1,
            "class": classes,
        }
    )

    tracks = add_maneuver(tracks_meta, tracks, fz=10, debug=debug)
    tracks = add_heading_feat(tracks, debug=debug)

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(
        final_frame, seed=seed, test_size=0.25
    )
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args


def preprocess_ngsim(
    path: str,
    rec_id: str,
    config: dict,
    output_dir: str,
    seed: int = 42,
    dataset: str = "i-80",
    add_supp: bool = False,
    debug: bool = False,
) -> tuple:
    if add_supp:
        raise NotImplementedError(
            "Support for additional data not implemented for i-80 dataset."
        )

    def compute_lane_model(df: DataFrame):
        lane_ids = df.laneId.unique()
        lane_ids.sort()
        num_lanes = len(lane_ids)

        def lane_center(li):
            return th[0] + th[1] * np.array(li)

        lane_y = [np.median(df[df.laneId == i].y) for i in range(1, num_lanes)]
        th, _, _, _ = np.linalg.lstsq(
            np.column_stack((np.ones((num_lanes - 1, 1)), np.arange(1, num_lanes))),
            lane_y,
            rcond=None,
        )

        lane_width = lane_center(2) - lane_center(1)
        x1 = lane_center(1) - 1 / 2 * lane_width
        comp_lane_bounds = [x1 + li * lane_width for li in range(num_lanes)]
        return comp_lane_bounds

    assert rec_id in (
        # us-101
        "0750am-0805am",
        "0805am-0820am",
        "0820am-0835am",
        # i-80
        "0400pm-0415pm",
        "0500pm-0515pm",
        "0515pm-0530pm",
    ), (
        f"Recording ID {rec_id} is not supported for dataset {dataset}. Please check the available recordings."
    )

    # Construct the base directory path for your data
    base_dir = os.path.join(path, dataset, rec_id)

    # Find all CSV files in the directory that starts with 'trajectories'
    csv_files = [
        f
        for f in os.listdir(base_dir)
        if f.startswith("trajectories") and f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {base_dir}")
    file = csv_files[0]

    # Use os.path.join for each specific file
    tracks_path = os.path.join(base_dir, file)

    # Read the CSV files
    tracks = read_csv(tracks_path, engine="pyarrow")

    # Drop columns that are not needed
    columns_to_drop = [
        "total_frames",
        "global_time",
        "global_x",
        "global_y",
        "v_length",
        "v_width",
        "v_vel",
        "v_acc",
        "preceding",
        "following",
        "space_headway",
        "time_headway",
        "preceeding",
        "space_hdwy",
        "time_hdwy",
    ]

    # Rename columns
    columns_to_rename = {
        "vehicle_id": "track_id",
        "frame_id": "frame",
        "local_x": "y",  # switch x and y
        "local_y": "x",
        "v_class": "class",
        "lane_id": "laneId",
    }

    # Update class names (1: motorcycle, 2: car, 3: truck)
    class_mapping = {
        1: "motorcycle",
        2: "car",
        3: "truck",
    }

    # Apply the renaming and dropping of columns
    tracks.columns = [col.lower() for col in tracks.columns]
    tracks = tracks.rename(columns=columns_to_rename)
    tracks = tracks.drop(columns=columns_to_drop, errors="ignore")

    tracks["class"] = tracks["class"].replace(class_mapping)

    # Feet to meters conversion factor
    tracks["x"] = tracks["x"] * 0.3048
    tracks["y"] = tracks["y"] * 0.3048 * (-1)
    tracks["x"] = tracks["x"] - tracks["x"].min()

    # Add new columns with default values
    tracks[["vx", "vy", "psi", "ax", "ay"]] = 0.0

    # round tseconds to nearest 0.1
    final_frame = tracks.frame.max()

    # Compute lane markings and convert to string format
    lane_markings = sorted(compute_lane_model(tracks))
    y0 = lane_markings[0]
    lane_markings_str = [";".join([str(l) for l in lane_markings])]

    tracks["y"] = tracks["y"] - y0

    # Create a data frame for the rec_meta with two columns: 'upperLaneMarkings' and 'lowerLaneMarkings'
    rec_meta = DataFrame(
        {"upperLaneMarkings": lane_markings_str, "lowerLaneMarkings": lane_markings_str}
    )

    # For the lanelet file, construct the path similarly
    _, lower_map, x_min, x_max = get_highway_graph(
        rec_meta,
        tracks,
        spacing=config["lane_graph"]["spacing"],
        buffer=config["lane_graph"]["buffer"],
    )
    lane_graph = {"upper_map": lower_map, "lower_map": lower_map}

    # get the number of lane changes
    num_lcs = (
        tracks.groupby("track_id")
        .apply(lambda x: x["laneId"].diff().abs().sum())
        .to_numpy().astype(int)
    )
    classes = get_meta_property(tracks, tracks["track_id"].unique(), prop="class")
    tracks["drivingDirection"] = 1

    tracks_meta = DataFrame(
        {
            "track_id": tracks["track_id"].unique(),
            "numLaneChanges": num_lcs,
            "drivingDirection": 1,
            "class": classes,
        }
    )

    tracks = add_maneuver(tracks_meta, tracks, fz=10, debug=debug)
    if not debug:
        tracks = compute_velocity(tracks, 0.1)
        tracks = compute_acceleration(tracks, 0.1)
    tracks = add_heading_feat(tracks, debug=debug)

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(
        final_frame, seed=seed, test_size=0.25
    )
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args


def preprocess_ad4che(
    path: str,
    rec_id: str,
    config: dict,
    output_dir: str,
    seed: int = 42,
    dataset: str = "ad4che",
    add_supp: bool = False,
    debug: bool = False,
) -> tuple:
    def update_signs_for_tracks(
        tracks_meta: DataFrame, tracks: DataFrame, debug: bool = False
    ) -> DataFrame:
        """
        Update signs of position, velocity, acceleration for FLU convention:
        - x forward, y left (vehicle-centric)
        - vx, vy match those directions
        - psi aligned with vx
        """
        # Get max x for flipping
        x_max = tracks["x"].max()

        if debug:
            return tracks

        def process_group(df):
            t_id = df["track_id"].iloc[0]
            dd = tracks_meta[tracks_meta.track_id == t_id].drivingDirection.values[0]

            if dd == 1:
                # Driving right-to-left: flip x (to make forward positive)
                df["x"] = -df["x"] + x_max
                df["vx"] = -df["vx"]
                df["ax"] = -df["ax"]
            else:
                # In all cases: flip y to make left positive
                df["y"] = -df["y"]
                df["vy"] = -df["vy"]
                df["ay"] = -df["ay"]

            return df

        return (
            tracks.groupby("track_id", group_keys=False)
            .apply(process_group)
            .reset_index(drop=True)
        )

    # Construct the base directory path for your data
    base_dir = os.path.join(path, dataset, "AD4CHE_Data_V1.0")

    # Use os.path.join for each specific file
    rec_meta_path = os.path.join(base_dir, rec_id, f"{rec_id[-2:]}_recordingMeta.csv")
    tracks_meta_path = os.path.join(base_dir, rec_id, f"{rec_id[-2:]}_tracksMeta.csv")
    tracks_path = os.path.join(base_dir, rec_id, f"{rec_id[-2:]}_tracks.csv")
    img_path = os.path.join(base_dir, rec_id, f"{rec_id[-2:]}_lanePicture.png")

    # Read the CSV files
    rec_meta = read_csv(rec_meta_path, engine="pyarrow")
    tracks_meta = read_csv(tracks_meta_path, engine="pyarrow")
    tracks = read_csv(tracks_path, engine="pyarrow")

    # Perform some initial renaming
    if "track_id" not in tracks_meta.columns:
        tracks_meta.rename(columns={"id": "track_id"}, inplace=True)
        tracks.rename(columns={"id": "track_id"}, inplace=True)
    if "vx" not in tracks.columns:
        tracks.rename(columns={"xVelocity": "vx"}, inplace=True)
        tracks.rename(columns={"yVelocity": "vy"}, inplace=True)
        tracks.rename(columns={"xAcceleration": "ax"}, inplace=True)
        tracks.rename(columns={"yAcceleration": "ay"}, inplace=True)
    if "x" not in tracks.columns:
        tracks.rename(columns={"xCenter": "x"}, inplace=True)
        tracks.rename(columns={"yCenter": "y"}, inplace=True)

    # Make class lowercase in tracks_meta
    tracks_meta["class"] = tracks_meta["class"].str.lower()

    # flip the y-axis
    # tracks['y'] = -tracks['y']
    x_max = tracks.x.max()

    # For the lanelet file, construct the path similarly
    umd, lmd = img_to_map(img_path, x_max=x_max)
    lane_graph = {"upper_map": umd, "lower_map": lmd}

    tracks = add_driving_direction(tracks_meta, tracks, debug=debug)
    # tracks = add_maneuver(tracks_meta, tracks, debug=debug)
    tracks = add_maneuver(tracks_meta, tracks, debug=True)
    # tracks = update_signs(rec_meta, tracks_meta, tracks, debug=debug)

    tracks = update_signs_for_tracks(tracks_meta, tracks)
    tracks = add_heading_feat(tracks, debug=debug)

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(
        tracks_meta.finalFrame.array[-1], seed=seed
    )
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args
