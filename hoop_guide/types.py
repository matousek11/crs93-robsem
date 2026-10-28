"""Domain types shared by several modules.

Lengths are in meters, angles in radians. The robot frame is the base frame
of the robot.
"""

from typing import NewType

import numpy as np
from numpy.typing import NDArray

Image = NewType("Image", NDArray[np.uint8])
"""Camera image, shape (H, W, 3), BGR."""

PixelPoints = NewType("PixelPoints", NDArray[np.float64])
"""Points in the image, shape (N, 2), pixels."""

PlanePoints = NewType("PlanePoints", NDArray[np.float64])
"""Points on the table plane in the robot frame, shape (N, 2), meters."""

Points3D = NewType("Points3D", NDArray[np.float64])
"""Points in the robot frame, shape (N, 3), meters."""

Directions3D = NewType("Directions3D", NDArray[np.float64])
"""Unit vectors in the robot frame, shape (N, 3)."""

Homography = NewType("Homography", NDArray[np.float64])
"""Maps image points to the table plane in the robot frame, shape (3, 3)."""
