"""Pose of the base plate on the table, dimensions from the drawing base 01-02.

Plate frame: origin in the center of the top surface, where the puzzle is
mounted. The x and y axes are parallel to the plate edges as in the drawing
(x right, y up), z points up.
"""

import dataclasses
from collections.abc import Sequence

from hoop_guide.types import Homography
from hoop_guide.vision import MarkerDetection

PLATE_SIZE = 0.125
MARKER_SIZE = 0.040

# Marker centers in the plate frame [m], top left and bottom right in the
# drawing. The IDs are placeholders, they must be verified in the lab.
MARKER_CENTERS: dict[int, tuple[float, float]] = {
    0: (-0.0375, 0.0375),
    1: (0.0375, -0.0375),
}


@dataclasses.dataclass(frozen=True)
class PlatePose:
    """Planar pose of the plate frame in the robot frame.

    Attributes:
        x: Position of the plate center along the robot x axis [m].
        y: Position of the plate center along the robot y axis [m].
        theta: Rotation of the plate x axis from the robot x axis around z
            [rad].
    """

    x: float
    y: float
    theta: float


def estimate_plate_pose(
    markers: Sequence[MarkerDetection], homography: Homography
) -> PlatePose | None:
    """Estimates the plate pose from detected markers.

    Only markers with IDs from MARKER_CENTERS are used, others are ignored.
    One marker is enough, its four corners give both position and rotation.
    When both are visible, their estimates are combined.

    Args:
        markers: Markers detected in the image.
        homography: Calibrated image to table plane homography.

    Returns:
        Pose of the plate, None if no known marker is visible.
    """
    raise NotImplementedError
