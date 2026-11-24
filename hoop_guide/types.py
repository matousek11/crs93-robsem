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

PlatePoints3D = NewType("PlatePoints3D", NDArray[np.float64])
"""Points in the plate frame (see plate.py), shape (N, 3), meters."""

PlateDirections3D = NewType("PlateDirections3D", NDArray[np.float64])
"""Unit vectors in the plate frame (see plate.py), shape (N, 3)."""

Homography = NewType("Homography", NDArray[np.float64])
"""Maps image points to the table plane in the robot frame, shape (3, 3)."""

Pose = NewType("Pose", NDArray[np.float64])
"""Homogeneous transform to the robot frame, shape (4, 4)."""

JointConfig = NewType("JointConfig", NDArray[np.float64])
"""Joint angles of the robot from the base to the flange, shape (6,), rad."""

JointSpeeds = NewType("JointSpeeds", NDArray[np.float64])
"""Speeds of the robot joints from the base to the flange, shape (6,),
rad/s."""

Point3D = NewType("Point3D", NDArray[np.float64])
"""One point in the robot frame, shape (3,), meters."""

RotationMatrix = NewType("RotationMatrix", NDArray[np.float64])
"""Orientation in the robot frame, shape (3, 3), the rotation part of a Pose."""

TurnAngles = NewType("TurnAngles", NDArray[np.float64])
"""Turns about the x, y and z axis of the robot frame, shape (3,), rad.

Positive angles are counterclockwise when viewed against the axis. Turns
about different axes do not commute, so the three angles alone do not give
an orientation.
"""
