"""Planar rotations about the vertical axis.

Positive angles are counterclockwise when viewed from above, the same
sense as a plate heading. Angle 0 leaves the x axis in place. A positive
quarter turn maps it onto the y axis.

A point (x, y) rotated about the origin by angle is
(cosine * x - sine * y, sine * x + cosine * y).
"""

import math


def get_cosine_and_sine(angle: float) -> tuple[float, float]:
    """Returns the cosine and sine of a counterclockwise angle.

    Args:
        angle: Rotation about z, positive counterclockwise [rad].

    Returns:
        Cosine and sine of angle.
    """
    return math.cos(angle), math.sin(angle)
