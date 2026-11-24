import numpy as np
import pytest

from hoop_guide.control import MotionController
from hoop_guide.control import get_move_time
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds

JOINT_LIMIT = 1.0  # [rad]
START_JOINT_ANGLES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


class FakeRobot:
    def __init__(self):
        self.joint_angles = np.array(START_JOINT_ANGLES)
        self.commands = []

    @property
    def joint_speeds(self):
        return JointSpeeds(np.ones(6))

    def get_joint_angles(self):
        return self.joint_angles

    def set_joint_angles(self, joint_config):
        if np.any(np.abs(joint_config) > JOINT_LIMIT):
            raise ValueError("Joint angles out of limits.")
        self.commands.append(joint_config)

    def get_ik_options(self, flange_pose):
        return []

    def is_in_limits(self, joint_config):
        return bool(np.all(np.abs(joint_config) <= JOINT_LIMIT))

    def get_fk_pos(self):
        return np.eye(4)

    def is_in_motion(self):
        return False


def test_move_joint_changes_only_that_joint():
    robot = FakeRobot()

    MotionController(robot).move_joint(2, -0.7)

    assert len(robot.commands) == 1
    np.testing.assert_allclose(
        robot.commands[0], [0.1, 0.2, -0.7, 0.4, 0.5, 0.6]
    )


def test_move_joint_keeps_measured_joint_angles_unchanged():
    robot = FakeRobot()

    MotionController(robot).move_joint(2, -0.7)

    np.testing.assert_allclose(robot.joint_angles, START_JOINT_ANGLES)


@pytest.mark.parametrize(
    "joint_index", [-1, 6], ids=["negative", "past_last_joint"]
)
def test_move_joint_with_unknown_joint_raises_and_sends_nothing(joint_index):
    robot = FakeRobot()

    with pytest.raises(ValueError, match="Joint index"):
        MotionController(robot).move_joint(joint_index, 0.0)

    assert robot.commands == []


def test_move_joint_out_of_limits_raises_and_sends_nothing():
    robot = FakeRobot()

    with pytest.raises(ValueError, match="out of limits"):
        MotionController(robot).move_joint(0, 2 * JOINT_LIMIT)

    assert robot.commands == []


# get_move_time

JOINT_SPEEDS = JointSpeeds(np.array([0.5, 0.1, 0.3, 1.0, 1.0, 2.0]))  # [rad/s]
START = JointConfig(np.array(START_JOINT_ANGLES))


@pytest.mark.parametrize(
    ("change", "expected_time"),
    [
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.4], 0.2),
        ([0.0, -0.05, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.2, 0.05, 0.3, 0.5, 0.5, 0.4], 1.0),
    ],
    ids=["no_change", "one_joint", "backwards", "slowest_joint_wins"],
)
def test_move_time_is_set_by_the_slowest_joint_for_its_change(
    change, expected_time
):
    target = JointConfig(START + np.array(change))

    move_time = get_move_time(START, target, JOINT_SPEEDS)

    assert move_time == pytest.approx(expected_time)


def test_move_time_with_different_shapes_raises():
    with pytest.raises(ValueError, match="same shape"):
        get_move_time(START, JointConfig(START[:5]), JOINT_SPEEDS)


@pytest.mark.parametrize("speed", [0.0, -1.0], ids=["zero", "negative"])
def test_move_time_with_a_speed_that_is_not_positive_raises(speed):
    joint_speeds = JointSpeeds(JOINT_SPEEDS.copy())
    joint_speeds[2] = speed

    with pytest.raises(ValueError, match="positive"):
        get_move_time(START, START, joint_speeds)
