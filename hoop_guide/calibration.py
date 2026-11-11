"""Homography calibration from corresponding point pairs and persistence."""

import json
from pathlib import Path

import cv2
import numpy as np

from hoop_guide.types import Homography
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints
from hoop_guide.vision import image_to_plane

MIN_POINTS_FOR_HOMOGRAPHY = 4
_POINT_DIMENSION = 2
_HOMOGRAPHY_SHAPE = (3, 3)
_HOMOGRAPHY_DIMENSION = 3


def estimate_homography(
    image_points: PixelPoints, plane_points: PlanePoints
) -> Homography:
    """Estimates the homography mapping image points to the robot plane.

    Args:
        image_points: Points in the camera image [px], shape (N, 2).
        plane_points: Corresponding points on the table plane [m], shape (N, 2).

    Returns:
        Calibrated homography mapping image points to plane points,
        shape (3, 3).

    Raises:
        ValueError: If points do not have shape (N, 2), their lengths differ,
            values are not finite, N < 4, or estimation fails.
    """
    if (
        image_points.ndim != _POINT_DIMENSION
        or image_points.shape[1] != _POINT_DIMENSION
        or plane_points.ndim != _POINT_DIMENSION
        or plane_points.shape[1] != _POINT_DIMENSION
    ):
        raise ValueError(
            "Image points and plane points must have shape (N, 2)."
        )

    if len(image_points) != len(plane_points):
        raise ValueError(
            "Point collections must have the same length: "
            f"{len(image_points)} != {len(plane_points)}."
        )

    if (
        not np.isfinite(image_points).all()
        or not np.isfinite(plane_points).all()
    ):
        raise ValueError("Image points and plane points must be finite.")

    if len(image_points) < MIN_POINTS_FOR_HOMOGRAPHY:
        raise ValueError(
            f"At least {MIN_POINTS_FOR_HOMOGRAPHY} point pairs are required "
            f"for homography estimation, got {len(image_points)}."
        )

    homography_matrix, _ = cv2.findHomography(
        np.asarray(image_points, dtype=np.float64),
        np.asarray(plane_points, dtype=np.float64),
        method=0,
    )
    if (
        homography_matrix is None
        or not np.isfinite(homography_matrix).all()
        or np.linalg.matrix_rank(homography_matrix) < _HOMOGRAPHY_DIMENSION
    ):
        raise ValueError(
            "Homography estimation failed (points may be degenerate)."
        )

    scale = homography_matrix[2, 2]
    if scale != 0.0:
        homography_matrix = homography_matrix / scale

    return Homography(homography_matrix.astype(np.float64))


def compute_reprojection_error(
    image_points: PixelPoints,
    plane_points: PlanePoints,
    homography: Homography,
) -> float:
    """Computes the root-mean-square reprojection error in the robot plane.

    Args:
        image_points: Points in the camera image [px], shape (N, 2).
        plane_points: Ground truth points on the table plane [m], shape (N, 2).
        homography: Homography mapping image points to plane points,
            shape (3, 3).

    Returns:
        RMS error in meters [m].

    Raises:
        ValueError: If inputs do not have matching shapes, values are not
            finite, or collections are empty.
    """
    if (
        plane_points.ndim != _POINT_DIMENSION
        or plane_points.shape[1] != _POINT_DIMENSION
    ):
        raise ValueError("Plane points must have shape (N, 2).")

    if len(image_points) != len(plane_points):
        raise ValueError(
            "Point collections must have the same length: "
            f"{len(image_points)} != {len(plane_points)}."
        )

    if len(image_points) == 0:
        raise ValueError("Point collections must contain at least one point.")

    if not np.isfinite(plane_points).all():
        raise ValueError("Plane points must be finite.")

    transformed_points = image_to_plane(image_points, homography)
    residuals = np.linalg.norm(transformed_points - plane_points, axis=1)
    return float(np.sqrt(np.mean(residuals**2)))


def save_homography(path: Path | str, homography: Homography) -> None:
    """Saves a homography matrix to a JSON file.

    Args:
        path: Path to the target JSON file.
        homography: Homography matrix to save, shape (3, 3).

    Raises:
        ValueError: If homography is not shaped (3, 3) or not finite.
    """
    if homography.shape != _HOMOGRAPHY_SHAPE:
        raise ValueError("Homography must have shape (3, 3).")
    if not np.isfinite(homography).all():
        raise ValueError("Homography values must be finite.")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"homography": homography.tolist()}
    with target_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_homography(path: Path | str) -> Homography:
    """Loads a homography matrix from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Loaded homography matrix, shape (3, 3).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON, missing the 'homography'
            key, or the matrix is not (3, 3) and finite.
    """
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Homography file not found: {target_path}")

    try:
        with target_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file {target_path}: {error}") from error

    if not isinstance(payload, dict) or "homography" not in payload:
        raise ValueError(
            f"Missing 'homography' key in calibration file: {target_path}"
        )

    matrix = np.array(payload["homography"], dtype=np.float64)
    if matrix.shape != _HOMOGRAPHY_SHAPE:
        raise ValueError(
            f"Expected homography shape {_HOMOGRAPHY_SHAPE}, "
            f"got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Loaded homography values must be finite.")

    return Homography(matrix)
