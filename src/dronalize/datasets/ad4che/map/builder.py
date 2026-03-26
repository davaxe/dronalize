from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2  # Optional dependency, checked at import time (in __init__.py)
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.maps.builder import BaseMapBuilder
from dronalize.maps.edge_type import EdgeType

if TYPE_CHECKING:
    from pathlib import Path

    from cv2.typing import MatLike

PIXEL_TO_METER: Final[float] = 0.038


class AD4CHEMapBuilder(BaseMapBuilder):
    """Map builder for the AD4CHE dataset.

    This dataset does not contain explicit lane graph data. This map builder
    extracts lane borders from the provided segmented map image using OpenCV
    and related processing. While it is not perfect, it produces a reasonable
    generated map.

    """

    def __init__(
        self,
        map_image_path: Path,
        pixel_to_meter: float = PIXEL_TO_METER,
        spatial_ds: float = 3.0,
    ) -> None:
        """Initialize the map builder.

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
        if not map_image_path.is_file():
            msg = f"Map image file does not exist: {map_image_path}"
            raise ValueError(msg)

        super().__init__()
        self.map_image_path: Path = map_image_path

        self.pixel_to_meter: float = pixel_to_meter
        self.spatial_ds: float = spatial_ds

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # Load the image in grayscale
        gray = cv2.imread(str(self.map_image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            msg = f"Failed to load image: {self.map_image_path}"
            raise ValueError(msg)

        gray = cv2.flip(gray, 0)
        h_pixels, _w_pixels = gray.shape
        h_meters = h_pixels * self.pixel_to_meter

        border_mask = _get_black_border_pixels(gray)
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if len(cnt) > 1:
                coords = cnt[:, 0, :] * self.pixel_to_meter
                # Downsample
                coords = _spatial_downsample_polyline(coords, d_min=self.spatial_ds)

                if len(coords) < 2:
                    continue

                points = [(float(pt[0]), h_meters - float(pt[1])) for pt in coords]

                self.add_path_lazy(
                    points=points,
                    edge_type=EdgeType.LINE_THICK_DASHED,
                    is_polygon=False,
                )


def _get_black_border_pixels(gray_image: MatLike) -> npt.NDArray[np.uint8]:
    black_mask = (gray_image == 0).astype(np.uint8)
    white_mask = (gray_image > 0).astype(np.uint8)
    dilated_white = cv2.dilate(white_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(black_mask == 1, dilated_white == 1).astype(np.uint8) * 255


def _spatial_downsample_polyline(
    polyline: npt.NDArray[np.float64],
    d_min: float = 0.5,
) -> npt.NDArray[np.float64]:
    if len(polyline) < 2:
        return polyline

    filtered = [polyline[0]]
    for pt in polyline[1:]:
        if np.linalg.norm(pt - filtered[-1]) >= d_min:
            filtered.append(pt)

    return np.array(filtered)
