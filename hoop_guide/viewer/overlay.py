"""Drawing of computed results into camera images.

All functions return a new image and leave the input unchanged.
"""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from hoop_guide.estimation import TableMotion
from hoop_guide.plate import PlatePose
from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints
from hoop_guide.vision import Frame
from hoop_guide.vision import MarkerDetection

# Colors are BGR.
MARKER_COLOR = (255, 255, 0)
FIRST_CORNER_COLOR = (0, 0, 255)
HOOP_CENTER_COLOR = (0, 0, 255)
ROD_ORIGIN_COLOR = (0, 255, 0)
AXIS_X_COLOR = (0, 0, 255)
AXIS_Y_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE_COLOR = (0, 0, 0)
AXIS_LENGTH = 0.05  # [m]

REFERENCE_IMAGE_WIDTH = 640  # [px]
"""Image width the sizes below are given for.

In a wider image they grow with its width, so lines and text keep their
size on the screen when the window shrinks a large camera image.
"""

LINE_THICKNESS = 2  # [px]
FIRST_CORNER_RADIUS = 4  # [px]
HOOP_CENTER_RADIUS = 6  # [px]
ROD_ORIGIN_RADIUS = 6  # [px]
TEXT_ORIGIN = (10, 30)  # [px], bottom left of the text
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2  # [px]
TEXT_OUTLINE_THICKNESS = 5  # [px]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(
    frame: Frame,
    fps: float | None,
    plate_pose: PlatePose | None = None,
    homography: Homography | None = None,
    table_estimation: TableMotion | None = None,
) -> Image:
    """Draws everything computed from the frame for the camera window.

    Args:
        frame: Camera image with its results.
        fps: Frame rate of the stream, None while unknown.
        plate_pose: Pose of the plate in the robot frame, if computed.
        homography: Calibrated image to table plane homography, if available.
        table_estimation: Fitted table motion, if available.

    Returns:
        Copy of the image with the markers, detected hoop center, plate
        axes, rod origin and a status line.
    """
    result = draw_markers(frame.image, frame.markers)
    if frame.hoop_center is not None:
        result = draw_hoop_center(result, frame.hoop_center)
    if plate_pose is not None and homography is not None:
        result = draw_plate_pose(result, plate_pose, homography)
        result = draw_rod_origin(result, plate_pose, homography)
    is_rotating_leftwise = (
        table_estimation.is_rotating_leftwise
        if table_estimation is not None
        else None
    )
    return draw_text(result, status_line(frame, fps, is_rotating_leftwise))


def status_line(
    frame: Frame,
    fps: float | None,
    is_rotating_leftwise: bool | None = None,
) -> str:
    """Returns the status line with FPS, detected markers, and rotation.

    Args:
        frame: Camera image with its results.
        fps: Frame rate of the stream, None while unknown.
        is_rotating_leftwise: True for leftwise (CCW), False for rightwise (CW),
            None while unknown.
    """
    rate = "-- fps" if fps is None else f"{fps:.1f} fps"
    ids = sorted(marker.marker_id for marker in frame.markers)
    markers = ", ".join(str(i) for i in ids) if ids else "none"
    if is_rotating_leftwise is None:
        rotation = "--"
    else:
        rotation = "Left" if is_rotating_leftwise else "Right"

    return f"{rate} | markers: {markers} | rotation: {rotation}"


def draw_text(image: Image, text: str) -> Image:
    """Draws one line of text in the top left corner.

    The text has a dark outline, so it is readable on any background.

    Args:
        image: Camera image.
        text: Text to draw.

    Returns:
        Copy of the image with the text.
    """
    result = image.copy()
    size_scale = _get_size_scale(image)
    origin = (
        _scale_size(TEXT_ORIGIN[0], size_scale),
        _scale_size(TEXT_ORIGIN[1], size_scale),
    )
    _put_outlined_text(result, text, origin, TEXT_COLOR, size_scale)
    return Image(result)


def draw_markers(image: Image, markers: Sequence[MarkerDetection]) -> Image:
    """Draws marker outlines, IDs and a dot in the first corner.

    The first corner is the top left one of the marker itself, so the dot
    shows how the marker is turned.

    Args:
        image: Camera image.
        markers: Markers detected in the image.

    Returns:
        Copy of the image with the markers.
    """
    result = image.copy()
    size_scale = _get_size_scale(image)
    for marker in markers:
        corners = marker.corners.round().astype(np.int32)
        cv2.polylines(
            result,
            [corners],
            isClosed=True,
            color=MARKER_COLOR,
            thickness=_scale_size(LINE_THICKNESS, size_scale),
        )
        cv2.circle(
            result,
            (int(corners[0][0]), int(corners[0][1])),
            _scale_size(FIRST_CORNER_RADIUS, size_scale),
            FIRST_CORNER_COLOR,
            cv2.FILLED,
        )
        _put_centered_text(
            result,
            str(marker.marker_id),
            marker.corners.mean(axis=0),
            MARKER_COLOR,
            size_scale,
        )
    return Image(result)


def draw_hoop_center(image: Image, hoop_center: tuple[float, float]) -> Image:
    """Draws a dot where the hoop center was detected.

    Args:
        image: Camera image.
        hoop_center: Detected hoop center (u, v) [px].

    Returns:
        Copy of the image with the dot.
    """
    result = image.copy()
    cv2.circle(
        result,
        (round(hoop_center[0]), round(hoop_center[1])),
        _scale_size(HOOP_CENTER_RADIUS, _get_size_scale(image)),
        HOOP_CENTER_COLOR,
        cv2.FILLED,
    )
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
    cos, sin = np.cos(pose.heading), np.sin(pose.heading)
    origin = np.array([pose.x, pose.y])
    axes_ends = origin + AXIS_LENGTH * np.array([[cos, sin], [-sin, cos]])
    points = _plane_to_image(
        PlanePoints(np.vstack([origin, axes_ends])), homography
    )
    origin_px, x_end_px, y_end_px = (
        tuple(int(c) for c in p) for p in points.round()
    )
    result = image.copy()
    thickness = _scale_size(LINE_THICKNESS, _get_size_scale(image))
    cv2.arrowedLine(result, origin_px, x_end_px, AXIS_X_COLOR, thickness)
    cv2.arrowedLine(result, origin_px, y_end_px, AXIS_Y_COLOR, thickness)
    return Image(result)


def draw_rod_origin(
    image: Image, pose: PlatePose, homography: Homography
) -> Image:
    """Draws a dot where the rod starts on the plate.

    The rod is mounted in the origin of the plate frame, see plate.py.

    Args:
        image: Camera image.
        pose: Pose of the plate in the robot frame.
        homography: Calibrated image to table plane homography.

    Returns:
        Copy of the image with the dot.
    """
    rod_origin = _plane_to_image(
        PlanePoints(np.array([[pose.x, pose.y]])), homography
    )[0]
    result = image.copy()
    cv2.circle(
        result,
        (round(rod_origin[0]), round(rod_origin[1])),
        _scale_size(ROD_ORIGIN_RADIUS, _get_size_scale(image)),
        ROD_ORIGIN_COLOR,
        cv2.FILLED,
    )
    return Image(result)


def _plane_to_image(points: PlanePoints, homography: Homography) -> PixelPoints:
    image_points = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), np.linalg.inv(homography)
    )
    return PixelPoints(image_points.reshape(-1, 2).astype(np.float64))


def _get_size_scale(image: Image) -> float:
    """Returns how many times sizes grow in this image, never below 1."""
    return max(1.0, float(image.shape[1]) / REFERENCE_IMAGE_WIDTH)


def _scale_size(size: int, size_scale: float) -> int:
    return max(1, round(size * size_scale))


def _put_outlined_text(
    image: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    size_scale: float,
) -> None:
    """Draws text with a dark outline into the image, origin is bottom left."""
    for line_color, thickness in (
        (TEXT_OUTLINE_COLOR, TEXT_OUTLINE_THICKNESS),
        (color, TEXT_THICKNESS),
    ):
        cv2.putText(
            image,
            text,
            origin,
            _FONT,
            TEXT_SCALE * size_scale,
            line_color,
            _scale_size(thickness, size_scale),
            cv2.LINE_AA,
        )


def _put_centered_text(
    image: NDArray[np.uint8],
    text: str,
    center: NDArray[np.float64],
    color: tuple[int, int, int],
    size_scale: float,
) -> None:
    """Draws outlined text into the image with its middle in center [px]."""
    (width, height), _ = cv2.getTextSize(
        text,
        _FONT,
        TEXT_SCALE * size_scale,
        _scale_size(TEXT_THICKNESS, size_scale),
    )
    origin = (round(center[0] - width / 2), round(center[1] + height / 2))
    _put_outlined_text(image, text, origin, color, size_scale)
