"""Start time of a run down the rod, found by simulating the whole run.

The simulation follows the steps of the real run with the predicted plate
poses and the step times of control.py. The real run then starts at the
found time and plans every step again from the real time and joint angles,
so small differences from the simulation do not add up.

Of the starts that go through, the one that keeps the joints farthest from
their limits is chosen. Both failures seen in the lab on 2026-11-24 were
the elbow joint reaching its limit, a run along a limit fails with a small
error of the prediction or of the step times.
"""

import math
from typing import Protocol

import numpy as np

from hoop_guide.control import MOVE_OVERHEAD
from hoop_guide.control import get_lead_time
from hoop_guide.control import get_move_time
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
from hoop_guide.types import Pose

BOTTOM_WAIT = 1.0
"""How long the hoop follows the last path point before going back [s]."""

SEARCH_WINDOW = 20.0
"""How far after now a start is searched for [s]."""

EARLIEST_START = 2.0
"""How soon after now a start may be. The search itself takes up to about
1 s on the robot, a start in that time would be gone before it is found
[s]."""

SEARCH_STEP = 0.5
"""Time between two tried starts [s]."""

TIME_MARGIN = 0.2
"""Every path point must be reachable this long before and after the time
the simulation reaches it [s]."""

# From the log of 2026-11-24: the steps took about 0.05 s longer than they
# were aimed. TODO: refine with a log of a whole run down the rod.
READY_FLAG_DELAY = 0.05
"""Time from the robot arriving to the controller reporting it ready [s]."""

JOINT_MARGIN_TOLERANCE = math.radians(2.0)
"""Starts whose joint limit margin is at most this much smaller than the
best one count as equally good, the earliest of them is chosen, because
the prediction gets worse with time [rad]."""

MAX_JOINT_STEP = math.radians(45.0)
"""Largest change of one joint in one step. A larger one means the joint
configuration flips and the arm would swing with the hoop on the rod."""

_SYNC_INDEX = 0


class Kinematics(Protocol):
    """Joint configurations and speeds of the robot."""

    @property
    def joint_speeds(self) -> JointSpeeds:
        """Speeds of the joints in a move [rad/s]."""

    @property
    def joint_limits(self) -> tuple[JointConfig, JointConfig]:
        """Lowest and highest angle of every joint [rad]."""

    def choose_joint_config(
        self, hoop_pose: Pose, joint_angles: JointConfig
    ) -> JointConfig:
        """Returns the reaching configuration nearest to the joint angles.

        Raises:
            NoJointConfigError: The pose is out of reach or only beyond the
                joint limits.
        """


class _JointFlipError(Exception):
    """A step changes a joint by more than MAX_JOINT_STEP."""


def find_start_time(
    planner: RodPlanner,
    kinematics: Kinematics,
    joint_angles: JointConfig,
    now: float,
) -> float | None:
    """Returns the start of a run down the rod with the most joint margin.

    Tried every SEARCH_STEP from now + EARLIEST_START to now +
    SEARCH_WINDOW. Until the start the hoop follows the synchronization
    point, which must stay reachable from now on,
    later starts are not tried once it is not. From the start the run goes
    down the rod, follows the last point for BOTTOM_WAIT and goes back up to
    the synchronization point, as PuzzleMode does. Every point must be
    reachable within TIME_MARGIN of the time it is reached and no step may
    flip the joint configuration.

    The margin of a run is the smallest distance of a joint from its limit
    on the rod, waiting at the synchronization point does not count: a
    failure there only stops the run. The earliest start within
    JOINT_MARGIN_TOLERANCE of the largest margin wins.

    Args:
        planner: Path of the hoop.
        kinematics: Joint configurations and speeds of the robot.
        joint_angles: Joint angles now, with the hoop at the synchronization
            point [rad].
        now: time.monotonic now [s].

    Returns:
        time.monotonic of the start [s], None when no start goes through.
    """
    last_index = planner.point_count - 1
    waiting_angles = joint_angles
    margin_by_start: dict[float, float] = {}
    for search_index in range(round(SEARCH_WINDOW / SEARCH_STEP) + 1):
        start_time = now + search_index * SEARCH_STEP
        try:
            # Waiting at the synchronization point until the start.
            waiting_angles = _choose_checked(
                planner, kinematics, _SYNC_INDEX, start_time, waiting_angles
            )
        except (NoJointConfigError, ValueError, _JointFlipError):
            break
        if start_time < now + EARLIEST_START:
            continue
        margin = _simulate_run(
            planner, kinematics, waiting_angles, start_time, last_index
        )
        if margin is not None:
            margin_by_start[start_time] = margin
    if not margin_by_start:
        return None
    good_enough = max(margin_by_start.values()) - JOINT_MARGIN_TOLERANCE
    return min(
        start_time
        for start_time, margin in margin_by_start.items()
        if margin >= good_enough
    )


def _simulate_run(
    planner: RodPlanner,
    kinematics: Kinematics,
    joint_angles: JointConfig,
    start_time: float,
    last_index: int,
) -> float | None:
    """Simulates the run from start_time.

    Returns:
        The smallest distance of a joint from its limit in the run [rad],
        None when a step does not go.
    """
    time, angles = start_time, joint_angles
    margin = math.inf
    try:
        for index in range(_SYNC_INDEX + 1, last_index + 1):
            time, angles, step_margin = _simulate_step(
                planner, kinematics, index, time, angles
            )
            margin = min(margin, step_margin)
        wait_end = time + BOTTOM_WAIT
        while time < wait_end:
            time, angles, step_margin = _simulate_step(
                planner, kinematics, last_index, time, angles
            )
            margin = min(margin, step_margin)
        for index in range(last_index - 1, _SYNC_INDEX - 1, -1):
            time, angles, step_margin = _simulate_step(
                planner, kinematics, index, time, angles
            )
            margin = min(margin, step_margin)
    except (NoJointConfigError, ValueError, _JointFlipError):
        return None
    return margin


def _simulate_step(
    planner: RodPlanner,
    kinematics: Kinematics,
    index: int,
    sent_time: float,
    joint_angles: JointConfig,
) -> tuple[float, JointConfig, float]:
    """Simulates PuzzleMode._step.

    Returns:
        When the next step is sent [s], the joint angles then [rad] and the
        smallest distance of a joint from its limit around the arrival
        [rad].

    Raises:
        NoJointConfigError: The point is not reachable around the arrival.
        ValueError: The planner cannot give a pose for the point.
        _JointFlipError: The step flips the joint configuration.
    """
    estimate = planner.get_hoop_pose(sent_time + MOVE_OVERHEAD, index)
    move_time = get_move_time(
        joint_angles,
        kinematics.choose_joint_config(estimate, joint_angles),
        kinematics.joint_speeds,
    )
    arrival_time = sent_time + get_lead_time(move_time)
    angles = _choose_checked(
        planner, kinematics, index, arrival_time, joint_angles
    )
    limit_margin = _get_limit_margin(angles, kinematics.joint_limits)
    for time_offset in (-TIME_MARGIN, TIME_MARGIN):
        nearby_angles = kinematics.choose_joint_config(
            planner.get_hoop_pose(arrival_time + time_offset, index),
            joint_angles,
        )
        limit_margin = min(
            limit_margin,
            _get_limit_margin(nearby_angles, kinematics.joint_limits),
        )
    return arrival_time + READY_FLAG_DELAY, angles, limit_margin


def _get_limit_margin(
    joint_angles: JointConfig, joint_limits: tuple[JointConfig, JointConfig]
) -> float:
    """Returns the smallest distance of a joint from its limit [rad]."""
    lowest, highest = joint_limits
    return float(
        np.min(np.minimum(joint_angles - lowest, highest - joint_angles))
    )


def _choose_checked(
    planner: RodPlanner,
    kinematics: Kinematics,
    index: int,
    time: float,
    joint_angles: JointConfig,
) -> JointConfig:
    """Returns the configuration for a point at a time, from joint_angles.

    Raises:
        NoJointConfigError: The point is not reachable then.
        ValueError: The planner cannot give a pose for the point.
        _JointFlipError: The configuration flips from joint_angles.
    """
    angles = kinematics.choose_joint_config(
        planner.get_hoop_pose(time, index), joint_angles
    )
    if float(np.max(np.abs(angles - joint_angles))) > MAX_JOINT_STEP:
        raise _JointFlipError
    return angles
