"""Rotations about the axes of the robot frame.

Positive angles are counterclockwise when viewed from above, the same
sense as a plate heading. Angle 0 leaves the x axis in place. A positive
quarter turn maps it onto the y axis.

A point (x, y) rotated about the origin by angle is
(cosine * x - sine * y, sine * x + cosine * y).

The same holds for the x and y axis when viewed against the axis: a positive
quarter turn about x maps y onto z, about y it maps z onto x.
"""

import math

import numpy as np

from hoop_guide.types import RotationAngles
from hoop_guide.types import RotationMatrix

AXIS_COUNT = 3

GIMBAL_LOCK_TOLERANCE = 1e-9
"""Below this length of the horizontal part of the rotated x axis the angles
about x and z cannot be told apart."""

_ROTATION_SHAPE = (AXIS_COUNT, AXIS_COUNT)


def get_cosine_and_sine(angle: float) -> tuple[float, float]:
    """Returns the cosine and sine of a counterclockwise angle.

    Args:
        angle: Rotation about z, positive counterclockwise [rad].

    Returns:
        Cosine and sine of angle.
    """
    return math.cos(angle), math.sin(angle)


def rotation_about_axis(axis_index: int, angle: float) -> RotationMatrix:
    """Returns the rotation about one axis of the robot frame.

    Args:
        axis_index: 0 for x, 1 for y, 2 for z.
        angle: Positive counterclockwise when viewed against the axis [rad].

    Raises:
        ValueError: axis_index is not 0, 1 or 2.
    """
    if axis_index not in range(AXIS_COUNT):
        raise ValueError(f"Axis index must be 0, 1 or 2, got {axis_index}.")
    cosine, sine = get_cosine_and_sine(angle)
    # The two axes that turn, in the order that makes the angle positive.
    first_axis = (axis_index + 1) % AXIS_COUNT
    second_axis = (axis_index + 2) % AXIS_COUNT
    rotation = np.eye(AXIS_COUNT)
    rotation[first_axis, first_axis] = cosine
    rotation[first_axis, second_axis] = -sine
    rotation[second_axis, first_axis] = sine
    rotation[second_axis, second_axis] = cosine
    return RotationMatrix(rotation)


def rotation_from_angles(rotation_angles: RotationAngles) -> RotationMatrix:
    """Returns the rotation matrix Rz @ Ry @ Rx of the angles.

    Raises:
        ValueError: rotation_angles is not shaped (3,).
    """
    if rotation_angles.shape != (AXIS_COUNT,):
        raise ValueError("Rotation angles must have shape (3,).")
    angle_x, angle_y, angle_z = (float(angle) for angle in rotation_angles)
    return RotationMatrix(
        rotation_about_axis(2, angle_z)
        @ rotation_about_axis(1, angle_y)
        @ rotation_about_axis(0, angle_x)
    )


def angles_from_rotation(rotation: RotationMatrix) -> RotationAngles:
    """Returns angles that give the rotation matrix.

    The angle about y is between -pi/2 and pi/2. When it is a quarter turn,
    the rotated x axis is vertical and only the sum or the difference of the
    other two angles is given. The angle about x is then 0 and the whole
    turn is in the angle about z.

    Args:
        rotation: Orthonormal matrix with determinant 1, not checked.

    Raises:
        ValueError: rotation is not shaped (3, 3).
    """
    if rotation.shape != _ROTATION_SHAPE:
        raise ValueError("Rotation matrix must have shape (3, 3).")
    # The first column is the rotated x axis:
    # (cos y * cos z, cos y * sin z, -sin y).
    horizontal_length = math.hypot(rotation[0, 0], rotation[1, 0])
    angle_y = math.atan2(-rotation[2, 0], horizontal_length)
    if horizontal_length > GIMBAL_LOCK_TOLERANCE:
        angle_x = math.atan2(rotation[2, 1], rotation[2, 2])
        angle_z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        angle_x = 0.0
        angle_z = math.atan2(-rotation[0, 1], rotation[1, 1])
    return RotationAngles(np.array([angle_x, angle_y, angle_z]))
