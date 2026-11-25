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

import dataclasses
import enum
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

SEARCH_WINDOW = 30.0
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


class RunPhase(enum.Enum):
    """Part of a run down the rod."""

    DOWN = "down"
    BOTTOM = "bottom"
    UP = "up"


@dataclasses.dataclass(frozen=True, eq=False)
class SimulatedStep:
    """One step of a simulated run.

    Attributes:
        phase: Part of the run the step belongs to.
        index: Path point of the step.
        sent_time: time.monotonic when the step is sent [s].
        arrival_time: time.monotonic when the robot arrives [s].
        handle_angle: Handle angle of the step [rad].
        joint_angles: Joint angles the robot arrives with [rad].
    """

    phase: RunPhase
    index: int
    sent_time: float
    arrival_time: float
    handle_angle: float
    joint_angles: JointConfig


@dataclasses.dataclass(frozen=True, eq=False)
class RunSchedule:
    """The start of a run and its simulated steps.

    Attributes:
        start_time: time.monotonic when the hoop leaves the synchronization
            point [s].
        joint_margin: Smallest joint margin of the run, see
            find_run_schedule [rad].
        steps: Simulated steps from the start back to the synchronization
            point.
    """

    start_time: float
    joint_margin: float
    steps: tuple[SimulatedStep, ...]


def find_run_schedule(
    planner: RodPlanner,
    kinematics: Kinematics,
    joint_angles: JointConfig,
    handle_angle: float,
    now: float,
) -> RunSchedule | None:
    """Returns the run down the rod with the most joint margin.

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
        The chosen run, None when no start goes through.
    """
    schedules: list[RunSchedule] = []
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
        schedule = _simulate_run(
            planner, kinematics, joint_angles, handle_angle, start_time
        )
        if schedule is not None:
            schedules.append(schedule)
    if not schedules:
        return None
    good_enough = (
        max(schedule.joint_margin for schedule in schedules)
        - JOINT_MARGIN_TOLERANCE
    )
    return next(
        schedule
        for schedule in schedules
        if schedule.joint_margin >= good_enough
    )


def _simulate_run(
    planner: RodPlanner,
    kinematics: Kinematics,
    joint_angles: JointConfig,
    handle_angle: float,
    start_time: float,
) -> RunSchedule | None:
    """Simulates the run from start_time, None when a step does not go."""
    last_index = planner.point_count - 1
    time, angles, angle = start_time, joint_angles, handle_angle
    steps: list[SimulatedStep] = []
    margin = math.inf

    def simulate(phase: RunPhase, index: int) -> None:
        nonlocal time, angles, angle, margin
        step, step_margin = _simulate_step(
            planner, kinematics, phase, index, time, angles, angle
        )
        steps.append(step)
        margin = min(margin, step_margin)
        time = step.arrival_time + READY_FLAG_DELAY
        angles, angle = step.joint_angles, step.handle_angle

    try:
        for index in range(_SYNC_INDEX + 1, last_index + 1):
            simulate(RunPhase.DOWN, index)
        wait_end = time + BOTTOM_WAIT
        while time < wait_end:
            simulate(RunPhase.BOTTOM, last_index)
        for index in range(last_index - 1, _SYNC_INDEX - 1, -1):
            simulate(RunPhase.UP, index)
    except (NoJointConfigError, ValueError):
        return None
    return RunSchedule(start_time, margin, tuple(steps))


def _simulate_step(  # noqa: PLR0913, PLR0917
    planner: RodPlanner,
    kinematics: Kinematics,
    phase: RunPhase,
    index: int,
    sent_time: float,
    joint_angles: JointConfig,
    handle_angle: float,
) -> tuple[SimulatedStep, float]:
    """Simulates PuzzleMode._step.

    Returns:
        The step and the smallest joint margin around its arrival, see
        _get_joint_margin [rad].

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
    simulated_step = SimulatedStep(
        phase=phase,
        index=index,
        sent_time=sent_time,
        arrival_time=step.arrival_time,
        handle_angle=step.handle_angle,
        joint_angles=step.joint_angles,
    )
    return simulated_step, joint_margin


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
