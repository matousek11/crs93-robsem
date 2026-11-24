import math

import numpy as np
import pytest

from hoop_guide.collision import ARM_RADIUS
from hoop_guide.collision import FLANGE_RADIUS
from hoop_guide.collision import HOOP_OUTER_RADIUS
from hoop_guide.collision import MIN_HEIGHT
from hoop_guide.collision import CollisionCheck
from hoop_guide.collision import get_lowest_hoop_height
from hoop_guide.models.hoop_mount import HOOP_THICKNESS
from hoop_guide.types import JointConfig
from hoop_guide.types import Points3D
from hoop_guide.types import Pose

JOINT_CONFIG = JointConfig(np.zeros(6))
HIGH = 0.5  # [m]


class FakeRobot:
    """Joints high above the table, one of them at a chosen height."""

    def __init__(self, low_point_index=None, low_point_height=HIGH):
        self._low_point_index = low_point_index
        self._low_point_height = low_point_height
        self.joint_configs = []

    def get_link_points(self, joint_config):
        self.joint_configs.append(joint_config)
        points = np.full((6, 3), HIGH)
        if self._low_point_index is not None:
            points[self._low_point_index, 2] = self._low_point_height
        return Points3D(points)


def hoop_pose(height, tilt=0.0):
    """Hoop at a height, its axis tilted from vertical about the x axis."""
    pose = np.eye(4)
    cosine, sine = math.cos(tilt), math.sin(tilt)
    pose[:3, :3] = [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    pose[2, 3] = height
    return Pose(pose)


def test_robot_and_hoop_high_above_the_table_are_free():
    check = CollisionCheck(FakeRobot())

    assert check.is_free(JOINT_CONFIG, hoop_pose(HIGH))


def test_check_asks_for_the_joints_of_the_given_config():
    robot = FakeRobot()

    CollisionCheck(robot).is_free(JOINT_CONFIG, hoop_pose(HIGH))

    np.testing.assert_allclose(robot.joint_configs[0], JOINT_CONFIG)


@pytest.mark.parametrize(
    ("low_point_index", "radius"),
    [(2, ARM_RADIUS), (5, FLANGE_RADIUS)],
    ids=["arm_joint", "flange"],
)
@pytest.mark.parametrize(
    ("below", "expected"),
    [(0.001, False), (-0.001, True)],
    ids=["just_below", "just_above"],
)
def test_point_of_the_robot_with_its_radius_must_stay_above_the_table(
    low_point_index, radius, below, expected
):
    robot = FakeRobot(low_point_index, MIN_HEIGHT + radius - below)

    assert CollisionCheck(robot).is_free(JOINT_CONFIG, hoop_pose(HIGH)) is (
        expected
    )


def test_hoop_below_the_table_is_not_free():
    check = CollisionCheck(FakeRobot())

    assert not check.is_free(JOINT_CONFIG, hoop_pose(MIN_HEIGHT))


@pytest.mark.parametrize(
    ("tilt", "below_center"),
    [
        (0.0, HOOP_THICKNESS / 2),
        (math.pi / 2, HOOP_OUTER_RADIUS),
        (
            math.pi / 6,
            HOOP_OUTER_RADIUS * 0.5 + HOOP_THICKNESS / 2 * math.sqrt(3) / 2,
        ),
    ],
    ids=["level", "upright", "tilted"],
)
def test_lowest_point_of_the_hoop_depends_on_its_tilt(tilt, below_center):
    lowest = get_lowest_hoop_height(hoop_pose(0.2, tilt))

    assert lowest == pytest.approx(0.2 - below_center)
