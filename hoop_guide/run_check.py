"""How far a run is from what was predicted, for the log.

The values compare the real robot and the camera with the predictions, so
the step time model and the table motion can be tuned from the log.
"""

import math

import numpy as np

from hoop_guide.plate import PlatePose
from hoop_guide.types import Pose

_TINY_SPEED = 1e-9  # [m/s]


def get_hoop_offset(
    hoop_pose: Pose, rod_pose: Pose, rod_velocity: tuple[float, float, float]
) -> tuple[float, float]:
    """Returns how far the hoop center is from the rod axis.

    Only the part across the rod counts, a hoop higher or lower on the rod
    is not off it.

    Args:
        hoop_pose: Measured pose of the hoop in the robot frame.
        rod_pose: Predicted hoop pose on the rod at the same time, its z axis
            is the rod axis.
        rod_velocity: Velocity of the rod at that point [m/s].

    Returns:
        The distance of the hoop center from the rod axis [m] and its part
        along the motion of the rod across its axis [m], positive when the
        hoop is ahead of the rod, 0 when the rod stands.
    """
    rod_axis = rod_pose[:3, 2]
    offset = hoop_pose[:3, 3] - rod_pose[:3, 3]
    across = offset - float(offset @ rod_axis) * rod_axis
    velocity = np.asarray(rod_velocity, dtype=np.float64)
    velocity_across = velocity - float(velocity @ rod_axis) * rod_axis
    speed = float(np.linalg.norm(velocity_across))
    ahead = (
        float(across @ velocity_across) / speed if speed > _TINY_SPEED else 0.0
    )
    return float(np.linalg.norm(across)), ahead


def get_plate_error(
    measured: PlatePose, predicted: PlatePose
) -> tuple[float, float]:
    """Returns how far the plate seen by the camera is from its prediction.

    Returns:
        The distance of the plate centers [m] and the measured minus the
        predicted heading, between -pi and pi [rad].
    """
    distance = math.hypot(measured.x - predicted.x, measured.y - predicted.y)
    heading = math.remainder(measured.heading - predicted.heading, 2 * math.pi)
    return distance, heading
