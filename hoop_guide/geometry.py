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

from hoop_guide.types import RotationMatrix

AXIS_COUNT = 3


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

    Multiplied from the left it turns an orientation about the axis of the
    robot, whatever the orientation is.

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
