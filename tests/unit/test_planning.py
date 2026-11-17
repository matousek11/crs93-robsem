import numpy as np
import pytest

from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import Planning
from hoop_guide.types import Pose

JOINT_LIMIT = 1.0  # [rad]
START_JOINT_ANGLES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
HOOP_POSE = Pose(np.eye(4))
FLANGE_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 1.0, 0.0, 0.1],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class FakeRobot:
    def __init__(self, ik_options=(), is_moving=False):
        self.is_moving = is_moving
        self.commands = []
        self.ik_flange_poses = []
        self._ik_options = list(ik_options)

    def get_joint_angles(self):
        return np.array(START_JOINT_ANGLES)

    def set_joint_angles(self, joint_config):
        self.commands.append(joint_config)

    def get_ik_options(self, flange_pose):
        self.ik_flange_poses.append(flange_pose)
        return self._ik_options

    def is_in_limits(self, joint_config):
        return bool(np.all(np.abs(joint_config) <= JOINT_LIMIT))

    def get_fk_pos(self):
        return FLANGE_POSE

    def is_in_motion(self):
        return self.is_moving


def test_move_to_ik_asks_ik_for_the_flange_pose_behind_the_hoop():
    robot = FakeRobot([np.zeros(6)])

    Planning(robot).move_to_ik(HOOP_POSE)

    assert len(robot.ik_flange_poses) == 1
    np.testing.assert_allclose(
        robot.ik_flange_poses[0] @ FLANGE_TO_HOOP, HOOP_POSE, atol=1e-12
    )


def test_move_to_ik_sends_the_option_nearest_to_current_joint_angles():
    far = np.full(6, -0.9)
    near = np.array(START_JOINT_ANGLES) + 0.05
    robot = FakeRobot([far, near])

    Planning(robot).move_to_ik(HOOP_POSE)

    assert len(robot.commands) == 1
    np.testing.assert_allclose(robot.commands[0], near)


def test_move_to_ik_counts_moves_in_both_directions():
    # Signed differences of this option cancel out to zero.
    opposite_moves = np.array(START_JOINT_ANGLES) + [0.4, -0.4] * 3
    small_moves = np.array(START_JOINT_ANGLES) + 0.1
    robot = FakeRobot([opposite_moves, small_moves])

    Planning(robot).move_to_ik(HOOP_POSE)

    np.testing.assert_allclose(robot.commands[0], small_moves)


def test_move_to_ik_skips_a_nearer_option_beyond_joint_limits():
    beyond_limits = np.array(START_JOINT_ANGLES)
    beyond_limits[5] = JOINT_LIMIT + 0.01
    within_limits = np.full(6, -0.9)
    robot = FakeRobot([beyond_limits, within_limits])

    Planning(robot).move_to_ik(HOOP_POSE)

    assert len(robot.commands) == 1
    np.testing.assert_allclose(robot.commands[0], within_limits)


@pytest.mark.parametrize(
    "ik_options",
    [[], [np.full(6, 2 * JOINT_LIMIT)]],
    ids=["out_of_reach", "only_beyond_joint_limits"],
)
def test_move_to_ik_unreachable_pose_raises_and_sends_nothing(ik_options):
    robot = FakeRobot(ik_options)

    with pytest.raises(NoJointConfigError):
        Planning(robot).move_to_ik(HOOP_POSE)

    assert robot.commands == []


def test_get_hoop_pose_is_the_measured_flange_pose_moved_to_the_hoop():
    hoop_pose = Planning(FakeRobot()).get_hoop_pose()

    np.testing.assert_allclose(hoop_pose, FLANGE_POSE @ FLANGE_TO_HOOP)


@pytest.mark.parametrize("is_moving", [True, False], ids=["moving", "still"])
def test_is_in_motion_is_what_the_robot_reports(is_moving):
    assert Planning(FakeRobot(is_moving=is_moving)).is_in_motion() is is_moving
