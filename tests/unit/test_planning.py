import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from hoop_guide.control import MOVE_OVERHEAD
from hoop_guide.control import get_lead_time
from hoop_guide.control import get_move_time
from hoop_guide.estimation import TableMotion
from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.planning import ARRIVAL_TOLERANCE
from hoop_guide.planning import HANDLE_ANGLE_STEP
from hoop_guide.planning import MAX_JOINT_STEP
from hoop_guide.planning import MAX_TILT
from hoop_guide.planning import MIN_WRIST_BEND
from hoop_guide.planning import PATH_STEP
from hoop_guide.planning import SYNC_DISTANCE
from hoop_guide.planning import TILT_DIRECTION_STEP
from hoop_guide.planning import WRIST_BEND_JOINT
from hoop_guide.planning import HoopTurn
from hoop_guide.planning import JointFlipError
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import Planning
from hoop_guide.planning import RodPlanner
from hoop_guide.planning import WristSingularityError
from hoop_guide.planning import check_joint_step
from hoop_guide.planning import check_wrist_bend
from hoop_guide.planning import choose_hoop_turn
from hoop_guide.planning import plan_step
from hoop_guide.planning import turn_hoop
from hoop_guide.plate import PlatePose
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
from hoop_guide.types import Points3D
from hoop_guide.types import Pose

JOINT_LIMIT = 1.0  # [rad]
START_JOINT_ANGLES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# High above the table, so only the fake joints can collide.
HOOP_POSE = Pose(
    np.array(
        [
            [1.0, 0.0, 0.0, 0.4],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
)
FLANGE_POSE = np.array(
    [
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 1.0, 0.0, 0.1],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class FakeRobot:
    """Robot whose joints are high above the table.

    Args:
        ik_options: What get_ik_options returns.
        is_moving: What is_in_motion returns.
        low_configs: Joint configurations whose joints touch the table.
    """

    def __init__(self, ik_options=(), is_moving=False, low_configs=()):
        self.is_moving = is_moving
        self.commands = []
        self.ik_flange_poses = []
        self._ik_options = list(ik_options)
        self._low_configs = list(low_configs)

    def get_link_points(self, joint_config):
        is_low = any(
            np.allclose(joint_config, low) for low in self._low_configs
        )
        return Points3D(np.full((6, 3), 0.0 if is_low else 0.5))

    @property
    def joint_speeds(self):
        return JointSpeeds(np.ones(6))

    @property
    def joint_limits(self):
        return JointConfig(np.full(6, -10.0)), JointConfig(np.full(6, 10.0))

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


def test_move_to_ik_skips_a_nearer_option_that_goes_below_the_table():
    hits_the_table = np.array(START_JOINT_ANGLES) + 0.05
    stays_above = np.full(6, -0.9)
    robot = FakeRobot(
        [hits_the_table, stays_above], low_configs=[hits_the_table]
    )

    Planning(robot).move_to_ik(HOOP_POSE)

    np.testing.assert_allclose(robot.commands[0], stays_above)


def test_move_to_ik_with_the_hoop_below_the_table_raises():
    hoop_pose_on_the_table = Pose(HOOP_POSE.copy())
    hoop_pose_on_the_table[2, 3] = 0.0
    robot = FakeRobot([np.zeros(6)])

    with pytest.raises(NoJointConfigError):
        Planning(robot).move_to_ik(hoop_pose_on_the_table)

    assert robot.commands == []


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


def test_chosen_config_is_the_option_nearest_to_the_given_angles():
    near_start = np.array(START_JOINT_ANGLES) + 0.05
    near_given = np.full(6, -0.8)
    robot = FakeRobot([near_start, near_given])

    config = Planning(robot).choose_joint_config(
        HOOP_POSE, JointConfig(np.full(6, -0.9))
    )

    np.testing.assert_allclose(config, near_given)


def test_joint_speeds_limits_and_angles_are_the_robot_ones():
    robot = FakeRobot()
    planning = Planning(robot)

    np.testing.assert_allclose(planning.joint_speeds, robot.joint_speeds)
    for planning_limit, robot_limit in zip(
        planning.joint_limits, robot.joint_limits, strict=True
    ):
        np.testing.assert_allclose(planning_limit, robot_limit)
    np.testing.assert_allclose(planning.get_joint_angles(), START_JOINT_ANGLES)


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
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C, Puzzle.D], ids=lambda p: p.value
)
def test_path_ends_at_the_stop_height_above_the_plate_center(puzzle):
    hoop_poses = make_planner(puzzle).update(0.0, 0)

    np.testing.assert_allclose(
        position(hoop_poses[-1]),
        (*PLATE_CENTER, PLATE_HEIGHT + STOP_HEIGHT),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C, Puzzle.D], ids=lambda p: p.value
)
def test_points_on_the_rod_are_at_most_one_step_apart(puzzle):
    hoop_poses = make_planner(puzzle).update(0.0, 0)

    on_rod = np.array([position(hoop_pose) for hoop_pose in hoop_poses[1:]])
    distances = np.linalg.norm(np.diff(on_rod, axis=0), axis=1)

    assert distances.max() <= PATH_STEP + 1e-12


def test_rod_d_sync_point_is_before_the_horizontal_free_tip():
    hoop_poses = make_planner(Puzzle.D).update(0.0, 0)

    np.testing.assert_allclose(
        position(hoop_poses[0]),
        (PLATE_CENTER[0] + 0.185, PLATE_CENTER[1], PLATE_HEIGHT + 0.155),
        atol=1e-12,
    )
    np.testing.assert_allclose(hoop_axis(hoop_poses[0]), (-1.0, 0.0, 0.0))
    np.testing.assert_allclose(handle(hoop_poses[0]), (0.0, 1.0, 0.0))


def test_rod_d_handle_rotates_with_plate_without_flipping():
    target_time = 4.0
    plate_heading = TURNING_TABLE.predict(target_time).heading
    hoop_poses = make_planner(Puzzle.D, TURNING_TABLE).update(target_time, 0)
    expected_handle = (-math.sin(plate_heading), math.cos(plate_heading), 0.0)

    for hoop_pose in hoop_poses:
        np.testing.assert_allclose(handle(hoop_pose), expected_handle, atol=1e-12)
        assert float(handle(hoop_pose) @ hoop_axis(hoop_pose)) == pytest.approx(0.0, abs=1e-12)


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
    puzzle=st.sampled_from([Puzzle.A1, Puzzle.B, Puzzle.C, Puzzle.D]),
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


def test_pose_of_one_point_is_that_point_of_the_rest_of_the_path():
    planner = make_planner(Puzzle.B, TURNING_TABLE)

    hoop_pose = planner.get_hoop_pose(1.5, 4)

    np.testing.assert_allclose(hoop_pose, planner.update(1.5, 4)[0])


def test_point_count_is_the_length_of_the_whole_path():
    planner = make_planner(Puzzle.B)

    assert planner.point_count == len(planner.update(0.0, 0))


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


# check_joint_step


def test_step_within_the_largest_joint_change_passes():
    start = JointConfig(np.zeros(6))
    target = JointConfig(np.full(6, MAX_JOINT_STEP))

    check_joint_step(start, target)


def test_step_beyond_the_largest_joint_change_names_the_joint():
    start = JointConfig(np.zeros(6))
    target = JointConfig(np.array([0.1, 0.0, 0.0, 0.0, 0.0, -math.pi]))

    with pytest.raises(JointFlipError, match="Joint 6 would turn by 180 deg"):
        check_joint_step(start, target)


# handle angle

LEFT = (0.0, 1.0, 0.0)  # level, to the left of the robot


class FakeKinematics:
    """Joint configuration is the handle direction, joints are fast.

    Args:
        is_reachable: Rule on the hoop pose.
        wrist_bend: Angle of the wrist bend joint for a hoop pose [rad].
    """

    def __init__(
        self, is_reachable=lambda pose: True, wrist_bend=lambda pose: 1.0
    ):
        self._is_reachable = is_reachable
        self._wrist_bend = wrist_bend
        self.tried_handles = []

    @property
    def joint_speeds(self):
        return JointSpeeds(np.full(6, 1000.0))

    @property
    def joint_limits(self):
        return JointConfig(np.full(6, -10.0)), JointConfig(np.full(6, 10.0))

    def choose_joint_config(self, hoop_pose, joint_angles):
        self.tried_handles.append(handle(hoop_pose).copy())
        if not self._is_reachable(hoop_pose):
            raise NoJointConfigError("No joint configuration.")
        return JointConfig(
            np.r_[handle(hoop_pose), 0.0, self._wrist_bend(hoop_pose), 0.0]
        )


def angles_toward_base():
    """Joint angles of FakeKinematics with the handle toward the base."""
    return JointConfig(np.r_[TOWARD_BASE, 0.0, 1.0, 0.0])


def test_handle_angle_turns_the_handle_about_the_rod():
    planner = make_planner()

    hoop_pose = planner.get_hoop_pose(
        0.0, 3, HoopTurn(handle_angle=math.pi / 2)
    )

    # The rod points down, a quarter turn moves the handle from the robot
    # to its left.
    np.testing.assert_allclose(handle(hoop_pose), LEFT, atol=1e-12)
    np.testing.assert_allclose(hoop_axis(hoop_pose), DOWN, atol=1e-12)
    np.testing.assert_allclose(
        hoop_pose[:3, :3].T @ hoop_pose[:3, :3], np.eye(3), atol=1e-12
    )


def test_no_hoop_turn_is_the_pose_of_update():
    planner = make_planner(Puzzle.B, TURNING_TABLE)

    np.testing.assert_allclose(
        planner.get_hoop_pose(1.5, 4, HoopTurn()), planner.update(1.5, 4)[0]
    )


def test_previous_handle_angle_is_kept_while_it_goes():
    turn, _ = choose_hoop_turn(
        make_planner(),
        FakeKinematics(),
        3,
        0.0,
        angles_toward_base(),
        previous_turn=HoopTurn(handle_angle=0.3),
    )

    assert turn.handle_angle == pytest.approx(0.3)
    assert turn.tilt == 0.0


def test_nearest_handle_angle_that_goes_is_chosen():
    # Handles toward the robot do not go, the nearest that does is two
    # steps away.
    def is_reachable(pose):
        return handle(pose) @ TOWARD_BASE < math.cos(1.5 * HANDLE_ANGLE_STEP)

    turn, joint_angles = choose_hoop_turn(
        make_planner(),
        FakeKinematics(is_reachable),
        3,
        0.0,
        angles_toward_base(),
        previous_turn=HoopTurn(),
    )

    # A turn of the handle is tried before a tilt.
    assert abs(turn.handle_angle) == pytest.approx(2 * HANDLE_ANGLE_STEP)
    assert turn.tilt == 0.0
    assert joint_angles[0] > -1.0


def test_handle_angle_that_flips_a_joint_is_skipped():
    # Joint angles far from every handle direction: every angle flips.
    kinematics = FakeKinematics()
    far_angles = JointConfig(np.full(6, 5.0))

    with pytest.raises(NoJointConfigError, match="would turn by"):
        choose_hoop_turn(
            make_planner(),
            kinematics,
            3,
            0.0,
            far_angles,
            previous_turn=HoopTurn(handle_angle=0.0),
        )

    handle_angle_count = round(2 * math.pi / HANDLE_ANGLE_STEP)
    tilt_count = 1 + round(2 * math.pi / TILT_DIRECTION_STEP)
    assert len(kinematics.tried_handles) == handle_angle_count * tilt_count


def test_flip_is_allowed_when_asked():
    far_angles = JointConfig(np.full(6, 5.0))

    turn, _ = choose_hoop_turn(
        make_planner(),
        FakeKinematics(),
        3,
        0.0,
        far_angles,
        previous_turn=HoopTurn(),
        can_flip=True,
    )

    assert turn == HoopTurn()


def test_no_handle_angle_that_goes_raises_with_the_last_reason():
    with pytest.raises(NoJointConfigError, match="No hoop turn.*point 3"):
        choose_hoop_turn(
            make_planner(),
            FakeKinematics(lambda pose: False),
            3,
            0.0,
            angles_toward_base(),
            previous_turn=HoopTurn(handle_angle=0.0),
        )


def test_step_is_aimed_at_the_plate_when_the_robot_arrives():
    planner = make_planner(table_motion=TURNING_TABLE)

    step = plan_step(
        planner,
        FakeKinematics(),
        0,
        5.0,
        angles_toward_base(),
        hoop_turn=HoopTurn(),
    )

    # Fast fake joints: the arrival is MOVE_OVERHEAD after sending.
    assert step.arrival_time == pytest.approx(5.0 + MOVE_OVERHEAD, abs=1e-3)
    np.testing.assert_allclose(
        step.hoop_pose,
        planner.get_hoop_pose(step.arrival_time, 0, step.hoop_turn),
    )


# wrist singularity


def joint_angles_with_wrist_bend(wrist_bend):
    joint_angles = np.full(6, 0.5)
    joint_angles[WRIST_BEND_JOINT] = wrist_bend
    return JointConfig(joint_angles)


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_wrist_bent_enough_passes(sign):
    check_wrist_bend(joint_angles_with_wrist_bend(sign * MIN_WRIST_BEND * 1.1))


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_wrist_too_straight_raises_with_the_joint(sign):
    joint_angles = joint_angles_with_wrist_bend(sign * MIN_WRIST_BEND * 0.5)

    with pytest.raises(WristSingularityError, match="Joint 5"):
        check_wrist_bend(joint_angles)


def test_handle_angle_with_the_wrist_near_its_singularity_is_skipped():
    # The wrist is straight with the handle toward the robot.
    def wrist_bend(pose):
        is_toward_base = handle(pose) @ TOWARD_BASE > math.cos(
            0.5 * HANDLE_ANGLE_STEP
        )
        return 0.0 if is_toward_base else 1.0

    turn, joint_angles = choose_hoop_turn(
        make_planner(),
        FakeKinematics(wrist_bend=wrist_bend),
        3,
        0.0,
        angles_toward_base(),
        previous_turn=HoopTurn(),
    )

    assert abs(turn.handle_angle) == pytest.approx(HANDLE_ANGLE_STEP)
    assert joint_angles[WRIST_BEND_JOINT] == pytest.approx(1.0)


class SlowKinematics(FakeKinematics):
    """Joint configuration is a multiple of the hoop position, joints are slow.

    A step then takes seconds and its target moves with the plate meanwhile.

    Args:
        scale: Joint angle per meter of the hoop position [rad/m].
    """

    def __init__(self, scale=10.0):
        super().__init__()
        self._scale = scale

    @property
    def joint_speeds(self):
        return JointSpeeds(np.full(6, 1.0))

    def choose_joint_config(self, hoop_pose, joint_angles):
        position = self._scale * hoop_pose[:3, 3]
        return JointConfig(np.r_[position, 0.0, 1.0, 0.0])


def plan_slow_step(scale):
    planner = make_planner(table_motion=TURNING_TABLE)
    kinematics = SlowKinematics(scale)
    start_angles = kinematics.choose_joint_config(
        planner.get_hoop_pose(0.0, 0), JointConfig(np.zeros(6))
    )
    step = plan_step(
        planner,
        kinematics,
        1,
        0.0,
        start_angles,
        hoop_turn=HoopTurn(),
        can_flip=True,
    )
    return step, start_angles, kinematics


def test_long_step_is_aimed_at_where_the_target_is_when_it_arrives():
    step, start_angles, kinematics = plan_slow_step(scale=10.0)

    lead_time = get_lead_time(
        get_move_time(start_angles, step.joint_angles, kinematics.joint_speeds)
    )
    assert step.arrival_time > 1.0
    assert step.arrival_time == pytest.approx(lead_time, abs=ARRIVAL_TOLERANCE)


def test_step_whose_arrival_does_not_settle_uses_the_last_estimate():
    # The joints are so slow that every estimate moves the arrival by more
    # than the tolerance.
    step, _, _ = plan_slow_step(scale=1000.0)

    np.testing.assert_allclose(
        step.hoop_pose,
        make_planner(table_motion=TURNING_TABLE).get_hoop_pose(
            step.arrival_time, 1, step.hoop_turn
        ),
    )


# hoop tilt


def test_tilt_turns_the_hoop_axis_toward_the_direction_about_its_center():
    hoop_pose = make_planner().get_hoop_pose(0.0, 3)

    tilted = turn_hoop(hoop_pose, HoopTurn(tilt=MAX_TILT, tilt_direction=0.0))

    # Toward the handle, which points to the robot base.
    expected_axis = math.cos(MAX_TILT) * np.array(DOWN) + math.sin(
        MAX_TILT
    ) * np.array(TOWARD_BASE)
    np.testing.assert_allclose(hoop_axis(tilted), expected_axis, atol=1e-12)
    np.testing.assert_allclose(position(tilted), position(hoop_pose))
    np.testing.assert_allclose(
        tilted[:3, :3].T @ tilted[:3, :3], np.eye(3), atol=1e-12
    )


def test_tilt_is_chosen_when_no_handle_angle_goes():
    # Only a hoop axis tilted from the rod is reachable.
    def is_reachable(pose):
        return hoop_axis(pose) @ DOWN < math.cos(0.5 * MAX_TILT)

    turn, _ = choose_hoop_turn(
        make_planner(),
        FakeKinematics(is_reachable),
        3,
        0.0,
        angles_toward_base(),
        previous_turn=HoopTurn(),
    )

    assert turn.tilt == pytest.approx(MAX_TILT)
    assert turn.handle_angle == pytest.approx(0.0)


def test_tilt_and_its_direction_are_kept_while_they_go():
    previous_turn = HoopTurn(
        handle_angle=0.2, tilt=MAX_TILT, tilt_direction=TILT_DIRECTION_STEP
    )

    turn, _ = choose_hoop_turn(
        make_planner(),
        FakeKinematics(),
        3,
        0.0,
        angles_toward_base(),
        previous_turn=previous_turn,
    )

    assert turn.tilt == pytest.approx(MAX_TILT)
    assert turn.tilt_direction == pytest.approx(TILT_DIRECTION_STEP)
    assert turn.handle_angle == pytest.approx(0.2)
