import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from hoop_guide.estimation import TableMotion
from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.planning import PATH_STEP
from hoop_guide.planning import SYNC_DISTANCE
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import Planning
from hoop_guide.planning import RodPlanner
from hoop_guide.plate import PlatePose
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
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


# RodPlanner

PLATE_HEIGHT = 0.08  # [m]
PLATE_CENTER = (0.45, 0.0)  # in front of the robot [m]
ROTATION_CENTER = (0.40, 0.03)  # [m]
ANGULAR_VELOCITY = 0.3  # [rad/s]
DOWN = (0.0, 0.0, -1.0)
TOWARD_BASE = (-1.0, 0.0, 0.0)  # level, from in front of the robot
# Second segment of rod B, it slopes toward the robot when the plate is not
# turned.
ROD_B_SLOPE = (-math.sqrt(0.5), 0.0, -math.sqrt(0.5))
ROD_B_BEND = (0.07, 0.0, 0.15)  # in the plate frame [m]


def table(center=PLATE_CENTER, angular_velocity=0.0, plate_center=PLATE_CENTER):
    return TableMotion(
        center_x=center[0],
        center_y=center[1],
        angular_velocity=angular_velocity,
        t_ref=0.0,
        pose_ref=PlatePose(plate_center[0], plate_center[1], 0.0),
    )


TURNING_TABLE = table(ROTATION_CENTER, ANGULAR_VELOCITY)


def make_planner(puzzle=Puzzle.A1, table_motion=None):
    return RodPlanner(puzzle, table_motion or table(), PLATE_HEIGHT)


def handle(hoop_pose):
    return hoop_pose[:3, 0]


def hoop_axis(hoop_pose):
    return hoop_pose[:3, 2]


def position(hoop_pose):
    return hoop_pose[:3, 3]


def angle_between(first, second):
    return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))


def index_nearest(hoop_poses, plate_point):
    robot_point = np.add((*PLATE_CENTER, PLATE_HEIGHT), plate_point)
    return int(
        np.argmin(
            [np.linalg.norm(position(p) - robot_point) for p in hoop_poses]
        )
    )


def test_path_starts_at_the_sync_point_beyond_the_free_end():
    hoop_poses = make_planner().update(0.0, 0)

    np.testing.assert_allclose(
        position(hoop_poses[0]),
        (*PLATE_CENTER, PLATE_HEIGHT + 0.2 + SYNC_DISTANCE),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        position(hoop_poses[1]), (*PLATE_CENTER, PLATE_HEIGHT + 0.2), atol=1e-12
    )


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C], ids=lambda p: p.value
)
def test_path_ends_at_the_stop_height_above_the_plate_center(puzzle):
    hoop_poses = make_planner(puzzle).update(0.0, 0)

    np.testing.assert_allclose(
        position(hoop_poses[-1]),
        (*PLATE_CENTER, PLATE_HEIGHT + STOP_HEIGHT),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C], ids=lambda p: p.value
)
def test_points_on_the_rod_are_at_most_one_step_apart(puzzle):
    hoop_poses = make_planner(puzzle).update(0.0, 0)

    on_rod = np.array([position(hoop_pose) for hoop_pose in hoop_poses[1:]])
    distances = np.linalg.norm(np.diff(on_rod, axis=0), axis=1)

    assert distances.max() <= PATH_STEP + 1e-12


def test_hoop_axis_points_along_a_straight_rod_toward_the_plate():
    hoop_poses = make_planner().update(0.0, 0)

    for hoop_pose in hoop_poses:
        np.testing.assert_allclose(hoop_axis(hoop_pose), DOWN, atol=1e-12)


def test_hoop_turns_in_a_bend_over_two_steps():
    hoop_poses = make_planner(Puzzle.B).update(0.0, 0)
    bend_index = index_nearest(hoop_poses, ROD_B_BEND)

    axes = [hoop_axis(hoop_poses[bend_index + k]) for k in (-2, -1, 0, 1)]

    np.testing.assert_allclose(axes[0], DOWN, atol=1e-12)
    np.testing.assert_allclose(axes[3], ROD_B_SLOPE, atol=1e-12)
    for axis in axes[1:3]:
        assert 0.0 < angle_between(axis, DOWN) < math.pi / 4
        assert 0.0 < angle_between(axis, ROD_B_SLOPE) < math.pi / 4


def test_handle_points_level_toward_the_base_on_a_vertical_rod():
    hoop_poses = make_planner().update(0.0, 0)

    for hoop_pose in hoop_poses:
        np.testing.assert_allclose(handle(hoop_pose), TOWARD_BASE, atol=1e-12)


def test_handle_on_a_sloped_rod_is_perpendicular_to_it_and_toward_the_base():
    hoop_poses = make_planner(Puzzle.B).update(0.0, 0)
    on_slope = hoop_poses[index_nearest(hoop_poses, ROD_B_BEND) + 3]

    # The rod slopes down toward the robot, so the handle points up.
    np.testing.assert_allclose(
        handle(on_slope), (-math.sqrt(0.5), 0.0, math.sqrt(0.5)), atol=1e-12
    )


def test_handle_keeps_pointing_toward_the_base_while_the_table_turns():
    target_time = 4.0

    hoop_poses = make_planner(table_motion=TURNING_TABLE).update(target_time, 0)

    for hoop_pose in hoop_poses:
        point = position(hoop_pose)
        toward_base = -np.array([point[0], point[1], 0.0])
        np.testing.assert_allclose(
            handle(hoop_pose),
            toward_base / np.linalg.norm(toward_base),
            atol=1e-12,
        )


def test_poses_follow_the_plate_predicted_for_the_target_time():
    target_time = 2.0

    hoop_poses = make_planner(table_motion=TURNING_TABLE).update(target_time, 0)

    plate = TURNING_TABLE.predict(target_time)
    np.testing.assert_allclose(
        position(hoop_poses[-1]),
        (plate.x, plate.y, PLATE_HEIGHT + STOP_HEIGHT),
        atol=1e-12,
    )


@settings(max_examples=25, deadline=None)
@given(
    puzzle=st.sampled_from([Puzzle.A1, Puzzle.B, Puzzle.C]),
    target_time=st.floats(-30.0, 30.0),
)
def test_hoop_poses_are_rotations(puzzle, target_time):
    hoop_poses = make_planner(puzzle, TURNING_TABLE).update(target_time, 0)

    for hoop_pose in hoop_poses:
        rotation = hoop_pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(hoop_pose[3], (0.0, 0.0, 0.0, 1.0))


def test_start_index_drops_the_passed_points():
    planner = make_planner(Puzzle.B)
    whole_path = planner.update(0.0, 0)

    rest = planner.update(0.0, 5)

    assert len(rest) == len(whole_path) - 5
    for rest_pose, path_pose in zip(rest, whole_path[5:], strict=True):
        np.testing.assert_allclose(rest_pose, path_pose, atol=1e-12)


def test_start_index_outside_the_path_raises():
    planner = make_planner()
    point_count = len(planner.update(0.0, 0))

    for start_index in (-1, point_count):
        with pytest.raises(ValueError, match="Start index"):
            planner.update(0.0, start_index)


def test_rod_above_the_robot_base_raises_with_the_point_index():
    # The hoop axis is vertical on the robot z axis, every handle is level.
    planner = make_planner(table_motion=table(plate_center=(0.0, 0.0)))

    with pytest.raises(ValueError, match="point 3"):
        planner.update(0.0, 3)


@pytest.mark.parametrize(
    "plate_height", [math.nan, math.inf], ids=["nan", "inf"]
)
def test_planner_with_an_invalid_plate_height_raises(plate_height):
    with pytest.raises(ValueError, match="Plate height"):
        RodPlanner(Puzzle.A1, table(), plate_height)
