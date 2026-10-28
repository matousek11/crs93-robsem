"""ArUco marker detection and mapping from the image to the table plane."""

import dataclasses

from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints


@dataclasses.dataclass(frozen=True)
class MarkerDetection:
    """One ArUco marker found in an image.

    Attributes:
        marker_id: ID of the marker in the ArUco 4x4 dictionary.
        corners: Corners of the marker, shape (4, 2), pixels, in the ArUco
            order: top left, top right, bottom right, bottom left.
    """

    marker_id: int
    corners: PixelPoints


def detect_markers(image: Image) -> list[MarkerDetection]:
    """Finds ArUco 4x4 markers in the image.

    Args:
        image: Camera image.

    Returns:
        Detected markers, empty if none is visible. Markers occluded so much
        that not all four corners are found are not returned.
    """
    raise NotImplementedError


def image_to_plane(points: PixelPoints, homography: Homography) -> PlanePoints:
    """Maps image points to the table plane in the robot frame.

    The result is valid only for points that lie in the plane the homography
    was calibrated for, points above it are shifted by parallax.

    Args:
        points: Points in the image.
        homography: Calibrated image to table plane homography.

    Returns:
        Points on the table plane, same order as the input.
    """
    raise NotImplementedError
