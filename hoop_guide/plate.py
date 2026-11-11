"""Pose of the base plate on the table, dimensions from the drawing base 01-02.

Plate frame: origin in the center of the top surface, where the puzzle is
mounted. The x and y axes are parallel to the plate edges as in the drawing
(x right, y up), z points up.
"""

import dataclasses
import math
from collections.abc import Sequence

from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.types import Homography
from hoop_guide.vision import MarkerDetection
from hoop_guide.vision import image_to_plane

PLATE_SIZE = 0.125  # 12.5 cm
MARKER_SIZE = 0.040  # 4 cm

# Marker centers in the plate frame [m], top left and bottom right in the
# drawing. The IDs are placeholders, they must be verified in the lab.
MARKER_CENTERS: dict[int, tuple[float, float]] = {
    0: (-0.0375, 0.0375),
    1: (0.0375, -0.0375),
}

# Angle of each marker x axis from the plate x axis [rad]. The markers are
# mounted parallel to the plate, correct it in the lab if a marker is turned.
MARKER_HEADING_ON_PLATE = 0.0


@dataclasses.dataclass(frozen=True)
class PlatePose:
    """Planar pose of the plate frame in the robot frame.

    Attributes:
        x: Position of the plate center along the robot x axis [m].
        y: Position of the plate center along the robot y axis [m].
        heading: Angle of the plate x axis from the robot x axis about z,
            positive counterclockwise [rad].
    """

    x: float
    y: float
    heading: float


def estimate_plate_pose(
    markers: Sequence[MarkerDetection], homography: Homography
) -> PlatePose | None:
    """Estimates the plate pose from detected markers.

    Only markers with IDs from MARKER_CENTERS are used, others are ignored.
    One marker is enough. Its corners, mapped into the table plane, give
    the marker heading and center. The plate heading is that heading minus
    MARKER_HEADING_ON_PLATE, and the plate center follows from the known
    marker center on the plate. When several known markers are visible,
    their plate centers are averaged and their headings are combined by a
    circular mean.

    Args:
        markers: Markers detected in the image.
        homography: Calibrated image to table plane homography.

    Returns:
        Pose of the plate, None if no known marker is visible.
    """
    poses = [
        _pose_from_marker(marker, homography)
        for marker in markers
        if marker.marker_id in MARKER_CENTERS
    ]
    if not poses:
        return None
    headings = [pose.heading for pose in poses]
    return PlatePose(
        x=sum(pose.x for pose in poses) / len(poses),
        y=sum(pose.y for pose in poses) / len(poses),
        heading=math.atan2(
            sum(math.sin(heading) for heading in headings),
            sum(math.cos(heading) for heading in headings),
        ),
    )


def _pose_from_marker(
    marker: MarkerDetection, homography: Homography
) -> PlatePose:
    corners = image_to_plane(marker.corners, homography)
    top_left, top_right, bottom_right, bottom_left = corners
    x_axis = (top_right - top_left) + (bottom_right - bottom_left)
    marker_heading = math.atan2(float(x_axis[1]), float(x_axis[0]))
    heading = marker_heading - MARKER_HEADING_ON_PLATE
    center = corners.mean(axis=0)
    offset_x, offset_y = MARKER_CENTERS[marker.marker_id]
    cosine, sine = get_cosine_and_sine(heading)
    return PlatePose(
        x=float(center[0] - (cosine * offset_x - sine * offset_y)),
        y=float(center[1] - (sine * offset_x + cosine * offset_y)),
        heading=heading,
    )
