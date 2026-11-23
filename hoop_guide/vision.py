"""ArUco marker detection and mapping from the image to the table plane."""

import dataclasses
import math

import cv2
import numpy as np
from numpy.typing import NDArray

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
        hoop_center: Hoop center (u, v) found by detect_hoop_center [px],
            None when the hoop was not found.
    """

    image: Image
    timestamp: float
    markers: tuple[MarkerDetection, ...]
    hoop_center: tuple[float, float] | None = None


def process_image(image: Image, timestamp: float) -> Frame:
    """Runs the vision on one camera image.

    Args:
        image: Camera image.
        timestamp: time.monotonic() right after the image was grabbed [s].

    Returns:
        The image with its results.
    """
    return Frame(
        image, timestamp, detect_markers(image), detect_hoop_center(image)
    )


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

MAX_HOOP_RADIAL_DEVIATION = 0.05
"""Largest RMS distance of a hoop contour from its fitted circle, as a part
of the circle radius. A square has about 0.1."""

MIN_HOOP_FILL = 0.8
"""Smallest area inside a hoop contour as a part of the area of its fitted
circle. Tells a closed contour from an arc."""

THRESHOLD_BLOCK_WIDTH_RATIO = 0.04
"""Side of the neighbourhood a pixel is compared with, as a part of the
image width. It must be several times wider than the hoop line."""

THRESHOLD_OFFSET = 8.0
"""How much darker than the mean of its neighbourhood a pixel of the hoop
line must be, in gray levels of 0 to 255."""

BLUR_SIGMA_WIDTH_RATIO = 0.001
"""Sigma of the blur that removes pixel noise before thresholding, as a part
of the image width."""

_COLOR_CHANNELS = 3
_IMAGE_POINT_DIMENSION = 2
_HOMOGRAPHY_SHAPE = (3, 3)
_MIN_THRESHOLD_BLOCK = 3  # [px], the smallest odd block with neighbours


def detect_hoop_center(image: Image) -> tuple[float, float] | None:
    """Finds the center of the hoop in the image.

    The hoop is the largest round closed contour that is darker than its
    neighbourhood. Every pixel is compared with the mean around it, so a
    thin hoop line is found under uneven light too. A contour is round when
    it stays close to the circle fitted to it, which rules out the gripper
    and the markers however large they are. The center is the center of
    the fitted circle.

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

    contours, _ = cv2.findContours(
        _threshold_darker_than_neighbourhood(image),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    max_area = image.shape[0] * image.shape[1] * MAX_HOOP_AREA_RATIO
    hoop_center = None
    hoop_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_HOOP_AREA or area >= max_area or area <= hoop_area:
            continue
        # A contour with an area never has all points on one line.
        center_x, center_y, radius, radial_deviation = _fit_circle(
            PixelPoints(contour.reshape(-1, 2).astype(float))
        )
        if (
            radial_deviation <= MAX_HOOP_RADIAL_DEVIATION * radius
            and area >= MIN_HOOP_FILL * math.pi * radius**2
        ):
            hoop_center = (center_x, center_y)
            hoop_area = area
    return hoop_center


def _threshold_darker_than_neighbourhood(image: Image) -> NDArray[np.uint8]:
    """Returns a mask of pixels darker than the mean around them, 255 or 0."""
    width = image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), width * BLUR_SIGMA_WIDTH_RATIO)
    block = max(
        _MIN_THRESHOLD_BLOCK, round(width * THRESHOLD_BLOCK_WIDTH_RATIO)
    )
    odd_block = block + 1 - block % 2
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        odd_block,
        THRESHOLD_OFFSET,
    )
    return mask.astype(np.uint8)


def _fit_circle(
    points: PixelPoints,
) -> tuple[float, float, float, float]:
    """Fits a circle to points by least squares.

    A circle with the center (a, b) and the radius r satisfies
    x^2 + y^2 = 2 a x + 2 b y + (r^2 - a^2 - b^2), which is linear in a, b
    and the last term.

    Args:
        points: Points on the circle, shape (N, 2) [px], not all on one
            line.

    Returns:
        Center x, center y, radius and the RMS distance of the points from
        the circle, all in pixels.
    """
    equation_coefficients = np.column_stack(
        [2.0 * points, np.ones(len(points))]
    )
    squared_distances = (points**2).sum(axis=1)
    solution = np.linalg.lstsq(
        equation_coefficients, squared_distances, rcond=None
    )[0]
    center_x, center_y, constant = (float(value) for value in solution)
    radius = math.sqrt(constant + center_x**2 + center_y**2)
    distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
    radial_deviation = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return center_x, center_y, radius, radial_deviation


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
