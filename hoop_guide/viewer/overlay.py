"""Drawing of computed results into camera images.

All functions return a new image and leave the input unchanged.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from hoop_guide.plate import PlatePose
from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints
from hoop_guide.vision import MarkerDetection

# Colors are BGR.
MARKER_COLOR = (255, 255, 0)
AXIS_X_COLOR = (0, 0, 255)
AXIS_Y_COLOR = (0, 255, 0)
AXIS_LENGTH = 0.05  # [m]
LINE_THICKNESS = 2  # [px]


def draw_markers(image: Image, markers: Sequence[MarkerDetection]) -> Image:
    """Draws marker outlines and IDs.

    Args:
        image: Camera image.
        markers: Markers detected in the image.

    Returns:
        Copy of the image with the markers.
    """
    result = image.copy()
    if markers:
        # transform structure for OpenCV
        corners = [m.corners.astype(np.float32)[np.newaxis] for m in markers]
        ids = np.array([[m.marker_id] for m in markers], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(result, corners, ids, MARKER_COLOR)
    return Image(result)


def draw_plate_pose(
    image: Image, pose: PlatePose, homography: Homography
) -> Image:
    """Draws the x and y axes of the plate frame.

    Args:
        image: Camera image.
        pose: Pose of the plate in the robot frame.
        homography: Calibrated image to table plane homography.

    Returns:
        Copy of the image with the plate axes.
    """
    cos, sin = np.cos(pose.theta), np.sin(pose.theta)
    origin = np.array([pose.x, pose.y])
    axes_ends = origin + AXIS_LENGTH * np.array([[cos, sin], [-sin, cos]])
    points = _plane_to_image(
        PlanePoints(np.vstack([origin, axes_ends])), homography
    )
    origin_px, x_end_px, y_end_px = (
        tuple(int(c) for c in p) for p in points.round()
    )
    result = image.copy()
    cv2.arrowedLine(result, origin_px, x_end_px, AXIS_X_COLOR, LINE_THICKNESS)
    cv2.arrowedLine(result, origin_px, y_end_px, AXIS_Y_COLOR, LINE_THICKNESS)
    return Image(result)


def _plane_to_image(points: PlanePoints, homography: Homography) -> PixelPoints:
    image_points = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), np.linalg.inv(homography)
    )
    return PixelPoints(image_points.reshape(-1, 2).astype(np.float64))
