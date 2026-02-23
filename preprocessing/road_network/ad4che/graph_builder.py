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

from copy import deepcopy

import numpy as np
import torch

from preprocessing.road_network.edge_type import EdgeType

try:
    import cv2
except ImportError:
    HAS_CV2 = False
else:
    HAS_CV2 = True

PIXEL_TO_METER = 0.0375


def get_black_border_pixels(gray_image):
    if not HAS_CV2:
        msg = (
            "OpenCV (cv2) is required to preprocess AD4Che dataset images. "
            "Please install it with `pip install opencv-python`."
        )
        raise ImportError(msg)
    black_mask = (gray_image == 0).astype(np.uint8)

    white_mask = (gray_image > 0).astype(np.uint8)
    dilated_white = cv2.dilate(white_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)

    border_mask = np.logical_and(black_mask == 1, dilated_white == 1).astype(np.uint8) * 255
    return border_mask


def extract_lane_borders_only(image_path, pixel_to_meter=PIXEL_TO_METER):
    if not HAS_CV2:
        msg = (
            "OpenCV (cv2) is required to preprocess AD4Che dataset images. "
            "Please install it with `pip install opencv-python`."
        )
        raise ImportError(msg)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to load image: {image_path}")

    border_mask = get_black_border_pixels(gray)
    ys, xs = np.where(border_mask == 255)
    coords_m = np.stack([xs, ys], axis=1) * pixel_to_meter
    return coords_m, border_mask


def downsample_polyline(polyline, step=5):
    return polyline[::step]


def extract_lane_polylines(border_mask, pixel_to_meter=PIXEL_TO_METER, spatial_ds=3.0):
    if not HAS_CV2:
        msg = (
            "OpenCV (cv2) is required to preprocess AD4Che dataset images. "
            "Please install it with `pip install opencv-python`."
        )
        raise ImportError(msg)
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polylines = []
    for cnt in contours:
        if len(cnt) > 1:
            coords = cnt[:, 0, :] * pixel_to_meter
            coords = spatial_downsample_polyline(coords, d_min=spatial_ds)
            polylines.append(coords)
    return polylines


def spatial_downsample_polyline(polyline, d_min=0.5):
    if len(polyline) < 2:
        return polyline
    filtered = [polyline[0]]
    for pt in polyline[1:]:
        if np.linalg.norm(pt - filtered[-1]) >= d_min:
            filtered.append(pt)
    return np.array(filtered)


def poly_to_map_data(polylines: list[np.ndarray], x_max: float = 0.0) -> dict:
    all_nodes = []
    all_types = []
    edge_index = []
    edge_attr = []

    node_counter = 0
    for poly in polylines:
        if len(poly) < 2:
            continue

        n = len(poly)
        all_nodes.append(poly)
        all_types.append(np.ones(n) * EdgeType.LINE_THIN_DASHED.value)

        # Create bidirectional edges
        edges = (
            np.array([[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)])
            + node_counter
        )
        edge_index.append(edges.T)
        edge_attr.append(np.ones((2 * (n - 1), 1)) * EdgeType.LINE_THIN_DASHED.value)

        node_counter += n

    pos_arr = np.concatenate(all_nodes, axis=0)
    pos_arr[:, 1] = -pos_arr[:, 1]  # flip y-axis for lower lane

    upper_pos = deepcopy(pos_arr)
    upper_pos[:, 1] = -pos_arr[:, 1]
    upper_pos[:, 0] = -upper_pos[:, 0] + x_max

    node_attr = np.concatenate(all_types, axis=0)
    edge_indices = np.concatenate(edge_index, axis=1)
    edge_attributes = np.concatenate(edge_attr, axis=0)

    lower_map_data = {
        "map_point": {
            "num_nodes": pos_arr.shape[0],
            "position": torch.from_numpy(pos_arr).float(),
            "type": torch.from_numpy(node_attr).long(),
        },
        ("map_point", "to", "map_point"): {
            "edge_index": torch.from_numpy(edge_indices).long(),
            "type": torch.from_numpy(edge_attributes).float(),
        },
    }

    upper_map_data = {
        "map_point": {
            "num_nodes": upper_pos.shape[0],
            "position": torch.from_numpy(upper_pos).float(),
            "type": torch.from_numpy(node_attr).long(),
        },
        ("map_point", "to", "map_point"): {
            "edge_index": torch.from_numpy(edge_indices).long(),
            "type": torch.from_numpy(edge_attributes).float(),
        },
    }

    return upper_map_data, lower_map_data


def img_to_map(img_path, pixel_to_meter=PIXEL_TO_METER, x_max=0.0):
    coords_m, filtered_mask = extract_lane_borders_only(img_path, pixel_to_meter)
    polylines_m = extract_lane_polylines(filtered_mask, pixel_to_meter)
    umd, lmd = poly_to_map_data(polylines_m, x_max=x_max)
    return umd, lmd
