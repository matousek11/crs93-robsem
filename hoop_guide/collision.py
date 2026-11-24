"""Check that the robot with the hoop stays above the table."""

import math
from typing import Protocol

import numpy as np

from hoop_guide.models.hoop_mount import HOOP_INNER_RADIUS
from hoop_guide.models.hoop_mount import HOOP_RING_WIDTH
from hoop_guide.models.hoop_mount import HOOP_THICKNESS
from hoop_guide.types import JointConfig
from hoop_guide.types import Points3D
from hoop_guide.types import Pose

MIN_HEIGHT = 0.07
"""Nothing of the robot or the hoop may get below this height above the
robot base frame origin, the plate top [m]."""

# TODO: measure in the lab.
ARM_RADIUS = 0.04
"""Distance of the surface of the arm from the axis through its joints [m]."""

# TODO: measure in the lab.
FLANGE_RADIUS = 0.03
"""Distance of the edge of the flange with the gripper from its center [m]."""

HOOP_OUTER_RADIUS = HOOP_INNER_RADIUS + HOOP_RING_WIDTH
"""Radius of the outer edge of the hoop [m]."""


class LinkPoints(Protocol):
    """Where the joints of the robot are."""

    def get_link_points(self, joint_config: JointConfig) -> Points3D:
        """Returns the origins of the joint frames, the last one the flange.

        Shape (6, 3) in the robot frame [m], the base is not included.
        """


class CollisionCheck:
    """Tells whether the robot and the hoop stay at least at MIN_HEIGHT.

    The arm is checked at its joints with ARM_RADIUS around them, a straight
    link between two joints above the height stays above it too. The flange
    is checked with FLANGE_RADIUS, the hoop at the lowest point of its ring.

    Args:
        robot: Where the joints of the robot are.
    """

    def __init__(self, robot: LinkPoints) -> None:
        self._robot = robot

    def is_free(self, joint_config: JointConfig, hoop_pose: Pose) -> bool:
        """Returns True when nothing gets below MIN_HEIGHT.

        Args:
            joint_config: Joint angles of the robot [rad].
            hoop_pose: Pose of the hoop the joint angles put it into, in the
                robot frame.
        """
        link_points = self._robot.get_link_points(joint_config)
        arm_heights = link_points[:-1, 2] - ARM_RADIUS
        flange_height = float(link_points[-1, 2]) - FLANGE_RADIUS
        lowest_height = min(
            float(np.min(arm_heights)),
            flange_height,
            get_lowest_hoop_height(hoop_pose),
        )
        return lowest_height >= MIN_HEIGHT


def get_lowest_hoop_height(hoop_pose: Pose) -> float:
    """Returns the height of the lowest point of the hoop [m].

    The hoop is a ring of HOOP_OUTER_RADIUS and HOOP_THICKNESS about its z
    axis. Tilted by an angle from level, its outer edge reaches
    HOOP_OUTER_RADIUS times the sine of the angle below its center, its side
    half of HOOP_THICKNESS times the cosine.
    """
    center_height = float(hoop_pose[2, 3])
    axis_vertical_part = abs(float(hoop_pose[2, 2]))
    level_part = math.sqrt(max(0.0, 1.0 - axis_vertical_part**2))
    return (
        center_height
        - HOOP_OUTER_RADIUS * level_part
        - HOOP_THICKNESS / 2 * axis_vertical_part
    )
