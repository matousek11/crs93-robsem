import math

import numpy as np
import pytest

from hoop_guide.estimation import TableMotion
from hoop_guide.planning import MIN_WRIST_BEND
from hoop_guide.planning import WRIST_BEND_JOINT
from hoop_guide.planning import HoopTurn
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.plate import PlatePose
from hoop_guide.rods import Puzzle
from hoop_guide.run_schedule import SEARCH_WINDOW
from hoop_guide.run_schedule import RunPhase
from hoop_guide.run_schedule import find_run_schedule
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds

PLATE_HEIGHT = 0.08  # [m]
LOW = PLATE_HEIGHT + 0.1  # points below this are "low" in the rules [m]
ROTATION_CENTER = (0.40, 0.0)  # [m]
ROTATION_RADIUS = 0.06  # [m]
ANGULAR_VELOCITY = 0.2  # [rad/s]
START_DELAY = 3.0  # time the search is given before the earliest start [s]
NEAR_X = 0.40  # the rules make points nearer to the robot unreachable [m]


class FakeKinematics:
    """Joint configuration is the hoop position, so moves are short.

    Args:
        is_reachable: Rule on the hoop pose.
        flip_below: Below this height a joint turns by half a turn [m].
        lowest_x: Lower limit of the joint that is the hoop x [m], the
            joints have no other limits.
        wrist_bend: Angle of the wrist bend joint for a hoop pose [rad].
    """

    def __init__(
        self,
        is_reachable=lambda pose: True,
        flip_below=None,
        lowest_x=-math.inf,
        wrist_bend=lambda pose: 1.0,
    ):
        self._is_reachable = is_reachable
        self._wrist_bend = wrist_bend
        self._flip_below = flip_below
        self._lowest_x = lowest_x

    @property
    def joint_speeds(self):
        return JointSpeeds(np.full(6, 1.0))

    @property
    def joint_limits(self):
        lowest = np.full(6, -math.inf)
        lowest[0] = self._lowest_x
        return JointConfig(lowest), JointConfig(np.full(6, math.inf))

    def choose_joint_config(self, hoop_pose, joint_angles):
        if not self._is_reachable(hoop_pose):
            raise NoJointConfigError("No joint configuration.")
        config = np.zeros(6)
        config[:3] = hoop_pose[:3, 3]
        config[WRIST_BEND_JOINT] = self._wrist_bend(hoop_pose)
        if self._flip_below is not None and hoop_pose[2, 3] < self._flip_below:
            config[3] = math.pi
        return JointConfig(config)


def planner_with_plate_at(angle):
    """Planner of rod A1 on a plate turning about ROTATION_CENTER."""
    motion = TableMotion(
        center_x=ROTATION_CENTER[0],
        center_y=ROTATION_CENTER[1],
        angular_velocity=ANGULAR_VELOCITY,
        t_ref=0.0,
        pose_ref=PlatePose(
            ROTATION_CENTER[0] + ROTATION_RADIUS * math.cos(angle),
            ROTATION_CENTER[1] + ROTATION_RADIUS * math.sin(angle),
            angle,
        ),
    )
    return RodPlanner(Puzzle.A1, motion, PLATE_HEIGHT)


def find_schedule(planner, kinematics, now=0.0):
    sync_pose = planner.update(now, 0)[0]
    joint_angles = JointConfig(np.r_[sync_pose[:3, 3], 0.0, 1.0, 0.0])
    return find_run_schedule(
        planner,
        kinematics,
        joint_angles,
        HoopTurn(),
        now=now,
        earliest_start=now + START_DELAY,
    )


def search(planner, kinematics, now=0.0):
    """Start time of the chosen run, None when there is none."""
    schedule = find_schedule(planner, kinematics, now)
    return None if schedule is None else schedule.start_time


def is_far_or_high(pose):
    return pose[0, 3] >= NEAR_X or pose[2, 3] >= LOW


def test_run_that_goes_through_starts_as_early_as_allowed():
    start_time = search(planner_with_plate_at(0.0), FakeKinematics(), now=3.0)

    assert start_time == pytest.approx(3.0 + START_DELAY)


def test_start_keeps_the_run_farthest_from_a_joint_limit():
    # The limit is on the hoop x, so the best run stays far from the robot.
    # Now the plate turns towards the robot, the run would pass x = 0.34.
    planner = planner_with_plate_at(math.pi / 2)
    lowest_x = 0.30

    start_time = search(planner, FakeKinematics(lowest_x=lowest_x))

    assert start_time is not None
    assert start_time > 5.0
    # At the start the plate is about as near as the table axis and turns
    # away from the robot.
    plate_x_at_start = planner.update(start_time, 0)[0][0, 3]
    plate_x_later = planner.update(start_time + 1.0, 0)[0][0, 3]
    assert plate_x_at_start > ROTATION_CENTER[0] - ROTATION_RADIUS / 4
    assert plate_x_later > plate_x_at_start


def test_equally_good_starts_give_the_earliest():
    start_time = search(planner_with_plate_at(math.pi / 2), FakeKinematics())

    assert start_time == pytest.approx(START_DELAY)


def test_start_waits_until_the_plate_turns_away_from_the_robot():
    # The plate starts nearest to the robot, low points are out of reach
    # there.
    planner = planner_with_plate_at(math.pi)

    start_time = search(planner, FakeKinematics(is_far_or_high))

    assert start_time is not None
    assert START_DELAY < start_time <= SEARCH_WINDOW
    plate_x_at_start = planner.update(start_time, 0)[0][0, 3]
    assert plate_x_at_start > NEAR_X - ROTATION_RADIUS


def test_low_points_never_reachable_give_no_start():
    kinematics = FakeKinematics(lambda pose: pose[2, 3] >= LOW)

    assert search(planner_with_plate_at(0.0), kinematics) is None


def test_no_start_once_the_sync_point_gets_out_of_reach():
    # The plate is just far enough now and turns towards the robot, so the
    # run fails and soon the hoop cannot wait at the sync point either.
    def is_far(pose):
        return pose[0, 3] >= NEAR_X

    planner = planner_with_plate_at(1.4)

    assert search(planner, FakeKinematics(is_far)) is None


def test_flip_of_the_joint_configuration_gives_no_start():
    kinematics = FakeKinematics(flip_below=LOW)

    assert search(planner_with_plate_at(0.0), kinematics) is None


def test_start_is_found_by_turning_the_handle_away_from_the_robot():
    # Low points are reachable only with the handle not toward the robot.
    def is_reachable(pose):
        return pose[2, 3] >= LOW or pose[0, 0] > -0.5

    start_time = search(
        planner_with_plate_at(0.0), FakeKinematics(is_reachable)
    )

    assert start_time == pytest.approx(START_DELAY)


def test_wrist_near_its_singularity_on_the_rod_gives_no_start():
    def wrist_bend(pose):
        return MIN_WRIST_BEND / 2 if pose[2, 3] < LOW else 1.0

    kinematics = FakeKinematics(wrist_bend=wrist_bend)

    assert search(planner_with_plate_at(0.0), kinematics) is None


def test_start_keeps_the_wrist_farthest_from_its_singularity():
    # The wrist is the straighter the nearer the plate is to the robot.
    def wrist_bend(pose):
        return MIN_WRIST_BEND + 0.05 + 2.0 * (pose[0, 3] - 0.33)

    planner = planner_with_plate_at(math.pi / 2)

    start_time = search(planner, FakeKinematics(wrist_bend=wrist_bend))

    assert start_time is not None
    assert start_time > 5.0


def test_schedule_goes_down_waits_at_the_bottom_and_goes_up_in_time():
    planner = planner_with_plate_at(0.0)

    schedule = find_schedule(planner, FakeKinematics())

    phases = [step.phase for step in schedule.steps]
    last_index = planner.point_count - 1
    down = [s.index for s in schedule.steps if s.phase is RunPhase.DOWN]
    up = [s.index for s in schedule.steps if s.phase is RunPhase.UP]
    assert down == list(range(1, last_index + 1))
    assert up == list(range(last_index - 1, -1, -1))
    assert phases.index(RunPhase.BOTTOM) == len(down)
    assert schedule.steps[0].sent_time == pytest.approx(schedule.start_time)
    for earlier, later in zip(schedule.steps, schedule.steps[1:], strict=False):
        assert earlier.sent_time < earlier.arrival_time <= later.sent_time
