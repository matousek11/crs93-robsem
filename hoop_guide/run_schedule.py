"""Start time of a run down the rod, found by simulating the whole run.

The simulation follows the steps of the real run with the predicted plate
poses and the step times of control.py. The real run then starts at the
found time and plans every step again from the real time and joint angles,
so small differences from the simulation do not add up.

Of the starts that go through, the one that keeps the joints farthest from
their limits and the wrist farthest from its singularity is chosen. The
failures seen in the lab on 2026-11-24 were the elbow joint reaching its
limit and the wrist getting straight, a run along either fails with a small
error of the prediction or of the step times.
"""

import math

import numpy as np

from hoop_guide.planning import Kinematics
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.planning import check_wrist_bend
from hoop_guide.planning import choose_handle_angle
from hoop_guide.planning import get_wrist_margin
from hoop_guide.planning import plan_step
from hoop_guide.types import JointConfig

BOTTOM_WAIT = 1.0
"""How long the hoop follows the last path point before going back [s]."""

SEARCH_WINDOW = 20.0
"""How far after now a start is searched for [s]."""

EARLIEST_START = 3.0
"""How soon after now a start may be. The search took 2.4 s on the robot on
2026-11-24, a start in that time would be gone before it is found [s]."""

SEARCH_STEP = 0.5
"""Time between two tried starts [s]."""

TIME_MARGIN = 0.2
"""Every path point must be reachable this long before and after the time
the simulation reaches it [s]."""

# From the third test of 2026-11-24, 55 steps: real time = move time +
# 0.24 s, MOVE_OVERHEAD already covers it.
READY_FLAG_DELAY = 0.0
"""Time from the robot arriving to the controller reporting it ready [s]."""

JOINT_MARGIN_TOLERANCE = math.radians(2.0)
"""Starts whose joint limit margin is at most this much smaller than the
best one count as equally good, the earliest of them is chosen, because
the prediction gets worse with time [rad]."""

_SYNC_INDEX = 0


def find_start_time(
    planner: RodPlanner,
    kinematics: Kinematics,
    joint_angles: JointConfig,
    handle_angle: float,
    now: float,
) -> float | None:
    """Returns the start of a run down the rod with the most joint margin.

    Tried every SEARCH_STEP from now + EARLIEST_START to now +
    SEARCH_WINDOW. Until the start the hoop follows the synchronization
    point, which must stay reachable from now on, later starts are not tried
    once it is not. From the start the run goes down the rod, follows the
    last point for BOTTOM_WAIT and goes back up to the synchronization
    point, every step planned by plan_step as PuzzleMode does. Every point
    must also be reachable with its handle angle within TIME_MARGIN of the
    time it is reached.

    The margin of a run is the smallest distance of a joint from its limit
    or of the wrist from its singularity on the rod. Waiting at the
    synchronization point does not count, a failure there only stops the
    run. The earliest start within JOINT_MARGIN_TOLERANCE of the largest
    margin wins.

    Args:
        planner: Path of the hoop.
        kinematics: Joint configurations and speeds of the robot.
        joint_angles: Joint angles now, with the hoop at the synchronization
            point [rad].
        handle_angle: Handle angle now, see RodPlanner.get_hoop_pose [rad].
        now: time.monotonic now [s].

    Returns:
        time.monotonic of the start [s], None when no start goes through.
    """
    margin_by_start: dict[float, float] = {}
    for search_index in range(round(SEARCH_WINDOW / SEARCH_STEP) + 1):
        start_time = now + search_index * SEARCH_STEP
        try:
            # Waiting at the synchronization point until the start.
            handle_angle, joint_angles = choose_handle_angle(
                planner,
                kinematics,
                _SYNC_INDEX,
                start_time,
                joint_angles,
                previous_angle=handle_angle,
            )
        except (NoJointConfigError, ValueError):
            break
        if start_time < now + EARLIEST_START:
            continue
        margin = _simulate_run(
            planner, kinematics, joint_angles, handle_angle, start_time
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
    handle_angle: float,
    start_time: float,
) -> float | None:
    """Simulates the run from start_time.

    Returns:
        The smallest joint margin in the run, see _get_joint_margin [rad],
        None when a step does not go.
    """
    last_index = planner.point_count - 1
    way_down = list(range(_SYNC_INDEX + 1, last_index + 1))
    way_up = list(range(last_index - 1, _SYNC_INDEX - 1, -1))
    time, angles, angle = start_time, joint_angles, handle_angle
    margin = math.inf
    try:
        for index in way_down:
            time, angles, angle, step_margin = _simulate_step(
                planner, kinematics, index, time, angles, angle
            )
            margin = min(margin, step_margin)
        wait_end = time + BOTTOM_WAIT
        while time < wait_end:
            time, angles, angle, step_margin = _simulate_step(
                planner, kinematics, last_index, time, angles, angle
            )
            margin = min(margin, step_margin)
        for index in way_up:
            time, angles, angle, step_margin = _simulate_step(
                planner, kinematics, index, time, angles, angle
            )
            margin = min(margin, step_margin)
    except (NoJointConfigError, ValueError):
        return None
    return margin


def _simulate_step(  # noqa: PLR0913, PLR0917
    planner: RodPlanner,
    kinematics: Kinematics,
    index: int,
    sent_time: float,
    joint_angles: JointConfig,
    handle_angle: float,
) -> tuple[float, JointConfig, float, float]:
    """Simulates PuzzleMode._step.

    Returns:
        When the next step is sent [s], the joint angles then [rad], the
        handle angle then [rad] and the smallest joint margin around the
        arrival, see _get_joint_margin [rad].

    Raises:
        NoJointConfigError: The point is not reachable around the arrival.
        ValueError: The planner cannot give a pose for the point.
    """
    step = plan_step(
        planner,
        kinematics,
        index,
        sent_time,
        joint_angles,
        handle_angle=handle_angle,
    )
    joint_margin = _get_joint_margin(step.joint_angles, kinematics.joint_limits)
    for time_offset in (-TIME_MARGIN, TIME_MARGIN):
        nearby_pose = planner.get_hoop_pose(
            step.arrival_time + time_offset, index, step.handle_angle
        )
        nearby_angles = kinematics.choose_joint_config(
            nearby_pose, joint_angles
        )
        check_wrist_bend(nearby_angles)
        joint_margin = min(
            joint_margin,
            _get_joint_margin(nearby_angles, kinematics.joint_limits),
        )
    return (
        step.arrival_time + READY_FLAG_DELAY,
        step.joint_angles,
        step.handle_angle,
        joint_margin,
    )


def _get_joint_margin(
    joint_angles: JointConfig, joint_limits: tuple[JointConfig, JointConfig]
) -> float:
    """Returns how far the joints are from where they must not get [rad].

    The smaller of the smallest distance of a joint from its limit and of
    the distance of the wrist from its singularity, see get_wrist_margin.
    """
    lowest, highest = joint_limits
    limit_margin = float(
        np.min(np.minimum(joint_angles - lowest, highest - joint_angles))
    )
    return min(limit_margin, get_wrist_margin(joint_angles))
