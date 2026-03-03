from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.core import EdgeType
from dronalize.core.graph import GraphBuilder

if TYPE_CHECKING:
    from pathlib import Path

PIXEL_TO_METER: Final[float] = 0.0375


class AD4CHEGraphBuilder(GraphBuilder):
    """Graph builder for the AD4CHE dataset.

    This dataset do not contain explicit lane graph or similar data. Therefore
    this graph builder extracts lane borders from the provided segmented image
    of the map, by using OpenCV and some processing. While it is not perfect, it
    results in a reasonable generated map.

    """

    def __init__(
        self,
        map_image_path: Path,
        pixel_to_meter: float = PIXEL_TO_METER,
        spatial_ds: float = 3.0,
    ) -> None:
        """Initialize the graph builder.

        Parameters
        ----------
        map_image_path : Path
            Path to the segmented map image (e.g., "01_laneWidthColorAndID.png").
        pixel_to_meter : float, optional
            Conversion factor from pixels to meters, by default PIXEL_TO_METER.
        spatial_ds : float, optional
            Minimum distance between consecutive nodes in the generated graph,
            by default 3.0.

        """
        super().__init__()
        self.map_image_path = map_image_path
        self.pixel_to_meter = pixel_to_meter
        self.spatial_ds = spatial_ds

    @override
    def build_impl(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> None:
        # Load the image in grayscale
        gray = cv2.imread(str(self.map_image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            msg = f"Failed to load image: {self.map_image_path}"
            raise ValueError(msg)

        border_mask = _get_black_border_pixels(gray)

        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if len(cnt) > 1:
                coords = cnt[:, 0, :] * self.pixel_to_meter

                # Downsample
                coords = _spatial_downsample_polyline(coords, d_min=self.spatial_ds)

                if len(coords) < 2:
                    continue
                points = [(float(pt[0]), -float(pt[1])) for pt in coords]

                self.add_path_lazy(
                    points=points, edge_type=EdgeType.LINE_THICK_DASHED, is_polygon=False
                )


def _get_black_border_pixels(gray_image: npt.NDArray) -> npt.NDArray[np.uint8]:
    black_mask = (gray_image == 0).astype(np.uint8)
    white_mask = (gray_image > 0).astype(np.uint8)
    dilated_white = cv2.dilate(white_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(black_mask == 1, dilated_white == 1).astype(np.uint8) * 255


def _spatial_downsample_polyline(
    polyline: npt.NDArray[np.floating[Any]], d_min: float = 0.5
) -> npt.NDArray[np.floating[Any]]:
    if len(polyline) < 2:
        return polyline

    filtered = [polyline[0]]
    for pt in polyline[1:]:
        if np.linalg.norm(pt - filtered[-1]) >= d_min:
            filtered.append(pt)

    return np.array(filtered)
