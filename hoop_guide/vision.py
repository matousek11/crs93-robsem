"""ArUco marker detection and mapping from the image to the table plane."""

import dataclasses

import cv2
import numpy as np

from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints

# Dictionary used in the course, IDs 0 to 49.
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


def _create_detector() -> cv2.aruco.ArucoDetector:
    parameters = cv2.aruco.DetectorParameters()
    # Sub-pixel corners, the plate pose is computed from them.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


_DETECTOR = _create_detector()


@dataclasses.dataclass(frozen=True)
class MarkerDetection:
    """One ArUco marker found in an image.

    Attributes:
        marker_id: ID of the marker in the ArUco 4x4 dictionary.
        corners: Corners of the marker, shape (4, 2), pixels, in the ArUco
            order: top left, top right, bottom right, bottom left of the
            marker itself, so they rotate with it.
    """

    marker_id: int
    corners: PixelPoints


@dataclasses.dataclass(frozen=True)
class Frame:
    """Camera image with the results computed from it.

    Attributes:
        image: Camera image.
        timestamp: time.monotonic() right after the image was grabbed [s].
        markers: Markers detected in the image.
    """

    image: Image
    timestamp: float
    markers: tuple[MarkerDetection, ...]


def process_image(image: Image, timestamp: float) -> Frame:
    """Runs the vision on one camera image.

    Args:
        image: Camera image.
        timestamp: time.monotonic() right after the image was grabbed [s].

    Returns:
        The image with its results.
    """
    return Frame(image, timestamp, detect_markers(image))


def detect_markers(image: Image) -> tuple[MarkerDetection, ...]:
    """Finds ArUco 4x4 markers in the image.

    Args:
        image: Camera image.

    Returns:
        Detected markers, empty if none is visible. Markers occluded so much
        that not all four corners are found are not returned.
    """
    corners, ids, _ = _DETECTOR.detectMarkers(image)
    if ids is None:
        return ()
    return tuple(
        MarkerDetection(
            int(marker_id),
            PixelPoints(marker_corners.reshape(4, 2).astype(np.float64)),
        )
        for marker_id, marker_corners in zip(
            ids.flatten(), corners, strict=True
        )
    )


MIN_HOOP_AREA = 100.0  # [px^2]
MAX_HOOP_AREA_RATIO = 0.8
_COLOR_CHANNELS = 3
_IMAGE_POINT_DIMENSION = 2
_HOMOGRAPHY_SHAPE = (3, 3)


def detect_hoop_center(image: Image) -> tuple[float, float] | None:
    """Finds the center of the contrast hoop in the image.

    Args:
        image: Camera image, shape (H, W, 3), BGR.

    Returns:
        Center coordinates (u, v) in pixels, or None if the hoop is not found.

    Raises:
        ValueError: If the image does not have shape (H, W, 3) or is empty.
    """
    if (
        image.ndim != _COLOR_CHANNELS
        or image.shape[2] != _COLOR_CHANNELS
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError(
            "Image must have shape (H, W, 3) with non-zero dimensions."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    total_area = float(image.shape[0] * image.shape[1])
    if area < MIN_HOOP_AREA or area >= total_area * MAX_HOOP_AREA_RATIO:
        return None

    (center_x, center_y), _ = cv2.minEnclosingCircle(largest_contour)
    return float(center_x), float(center_y)


def image_to_plane(points: PixelPoints, homography: Homography) -> PlanePoints:
    """Maps image points to the table plane in the robot frame.

    The result is valid only for points that lie in the plane the homography
    was calibrated for, points above it are shifted by parallax.

    Args:
        points: Points in the image.
        homography: Calibrated image to table plane homography.

    Returns:
        Points on the table plane, same order as the input.

    Raises:
        ValueError: Points are not shaped (N, 2), the homography is not
            3x3, or a value is not finite.
    """
    if (
        points.ndim != _IMAGE_POINT_DIMENSION
        or points.shape[1] != _IMAGE_POINT_DIMENSION
    ):
        raise ValueError("Image points must have shape (N, 2).")
    if homography.shape != _HOMOGRAPHY_SHAPE:
        raise ValueError("Homography must have shape (3, 3).")
    if not np.isfinite(points).all() or not np.isfinite(homography).all():
        raise ValueError("Image points and homography must be finite.")
    if len(points) == 0:
        return PlanePoints(np.zeros((0, 2), dtype=np.float64))
    transformed = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float64),
    )
    return PlanePoints(np.asarray(transformed, dtype=np.float64).reshape(-1, 2))
