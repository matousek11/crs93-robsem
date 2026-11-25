"""Run of one puzzle: the hoop goes through the rod of the rotating plate."""

import logging
import math
import time
from types import TracebackType
from typing import Protocol
from typing import Self

from hoop_guide.control import MOVE_OVERHEAD
from hoop_guide.estimation import TableMotion
from hoop_guide.keyboard import Key
from hoop_guide.planning import HoopTurn
from hoop_guide.planning import Kinematics
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.planning import plan_step
from hoop_guide.plate import PLATE_HEIGHT
from hoop_guide.plate import estimate_plate_pose
from hoop_guide.plate_tracker import CAMERA_LATENCY
from hoop_guide.plate_tracker import PLATE_SAMPLE_COUNT
from hoop_guide.plate_tracker import PlateTracker
from hoop_guide.rods import Puzzle
from hoop_guide.run_check import get_hoop_offset
from hoop_guide.run_check import get_plate_error
from hoop_guide.run_schedule import BOTTOM_WAIT
from hoop_guide.run_schedule import SEARCH_WINDOW
from hoop_guide.run_schedule import RunPhase
from hoop_guide.run_schedule import RunSchedule
from hoop_guide.run_schedule import find_run_schedule
from hoop_guide.types import Homography
from hoop_guide.types import JointConfig
from hoop_guide.types import Pose
from hoop_guide.vision import Frame

logger = logging.getLogger(__name__)


MEASUREMENT_TIMEOUT = 45.0
"""Longest wait for the frames the table motion is measured from [s].

PLATE_SAMPLE_COUNT frames take about 30 s, the rest is for frames where
the markers are hidden.
"""

FRAME_POLL_PERIOD = 0.02
"""Time between two looks for a new camera frame [s]."""

MOTION_POLL_PERIOD = 0.02
"""Time between two questions whether the robot still moves [s]."""

# Measured on 2026-11-25: with tilts the search took up to 8.3 s on a
# laptop, the lab computer was about twice slower.
INITIAL_SEARCH_TIME = 10.0
"""Expected time of the first start search [s]. Later searches expect the
time the previous one took."""

SEARCH_TIME_MARGIN = 1.5
"""How many times the expected search time the earliest tried start is
after the search begins."""

MIN_START_DELAY = 1.0
"""Shortest time from the search to the earliest tried start [s]."""

ROD_VELOCITY_TIME = 0.1
"""Time over which the velocity of the rod is taken for the log [s]."""

_SYNC_INDEX = 0
_MILLIMETERS_PER_METER = 1000.0


class PuzzleMotion(Kinematics, Protocol):
    """Moving the hoop."""

    def get_joint_angles(self) -> JointConfig:
        """Returns the joint angles measured by the controller."""

    def get_hoop_pose(self) -> Pose:
        """Returns the hoop pose computed from the measured joint angles."""

    def move_to_ik(self, hoop_pose: Pose) -> None:
        """Starts a move that puts the hoop into the pose.

        Raises:
            NoJointConfigError: The pose cannot be reached, nothing is sent
                to the robot.
        """

    def is_in_motion(self) -> bool:
        """Returns True while the robot is still moving."""


class HomeRobot(Protocol):
    """Robot that can return to its home position."""

    def soft_home(self) -> None:
        """Moves the robot to the home position."""


class FrameSource(Protocol):
    """Source of camera frames."""

    @property
    def latest(self) -> Frame | None:
        """The newest Frame, None before the first one arrives."""


class KeyPoller(Protocol):
    """Key presses of the operator, read while used as a context."""

    def __enter__(self) -> Self:
        """Starts reading single key presses."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stops reading single key presses."""

    def poll_key(self) -> Key | None:
        """Returns a waiting key press at once, None when there is none."""

    def discard_pending(self) -> None:
        """Drops key presses that arrived while nobody was reading."""


class Reporter(Protocol):
    """Messages for the operator."""

    def info(self, message: str) -> None:
        """Shows an information message."""

    def warning(self, message: str) -> None:
        """Shows a warning."""


class Clock(Protocol):
    """Time on the clock of the camera frames."""

    def now(self) -> float:
        """Returns time.monotonic [s]."""

    def sleep(self, seconds: float) -> None:
        """Waits for the time [s]."""


class SystemClock:
    """time.monotonic and time.sleep."""

    def now(self) -> float:
        """Returns time.monotonic [s]."""
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Waits for the time [s]."""
        time.sleep(seconds)


class PuzzleMode:
    """Guides the hoop through the rod of the chosen puzzle.

    A run measures the table motion from the first PLATE_SAMPLE_COUNT frames
    with the plate, moves the hoop to the synchronization point above the
    rod and follows the rotation there. Enter searches for a start whose
    whole run goes through, see find_run_schedule, the hoop keeps following
    the point until then. Then it goes down the rod point by point, follows
    the last point for BOTTOM_WAIT, goes back out to the synchronization
    point and returns home. Every step is aimed where the rod is when the
    robot arrives, the time of the move at full joint speeds times
    MOVE_TIME_FACTOR plus MOVE_OVERHEAD, and waits until the robot stands
    still.

    The c key stops the run where it is before the hoop goes down. On the
    rod it and an unreachable point both turn the hoop back out of the rod,
    because the rotating rod would push a standing hoop. Going back out is
    not stopped by c.

    Args:
        motion: Moves the hoop.
        robot: Returns the robot home at the end.
        frames: Camera frames, None without a camera.
        keyboard: Key presses of the operator.
        reporter: Messages for the operator.
        homography: Calibrated image to table plane homography, None when
            none is saved.
        plate_height: Height of the plate top surface above the robot base
            [m], None when not measured.
        clock: Time of the camera frames.
    """

    def __init__(  # noqa: PLR0913
        self,
        motion: PuzzleMotion,
        robot: HomeRobot,
        frames: FrameSource | None,
        keyboard: KeyPoller,
        reporter: Reporter,
        *,
        homography: Homography | None,
        plate_height: float | None = PLATE_HEIGHT,
        clock: Clock | None = None,
    ) -> None:
        self._motion = motion
        self._robot = robot
        self._frames = frames
        self._keyboard = keyboard
        self._reporter = reporter
        self._homography = homography
        self._plate_height = plate_height
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._hoop_turn = HoopTurn()
        self._expected_search_time = INITIAL_SEARCH_TIME
        self._last_step_time = MOVE_OVERHEAD
        self._table_motion: TableMotion | None = None
        self._planned_sent_times: dict[tuple[RunPhase, int], float] = {}

    def run(self, puzzle: Puzzle) -> None:
        """Goes through the rod of the puzzle, until done or stopped.

        Args:
            puzzle: Puzzle whose rod the hoop goes through.
        """
        logger.info("Puzzle %s started.", puzzle.value)
        frames, homography = self._frames, self._homography
        plate_height = self._plate_height
        if frames is None or homography is None or plate_height is None:
            self._warn(self._missing_requirement())
            return
        self._hoop_turn = HoopTurn()
        with self._keyboard:
            self._keyboard.discard_pending()
            table_motion = self._measure_table_motion(frames, homography)
            if table_motion is None:
                return
            self._table_motion = table_motion
            self._planned_sent_times = {}
            planner = RodPlanner(puzzle, table_motion, plate_height)
            if not self._synchronize(planner):
                return
            last_index = planner.point_count - 1
            reached_index = self._go_down(planner, last_index)
            if not self._go_back_out(planner, reached_index):
                return
        self._info("Puzzle done, returning home.")
        self._robot.soft_home()

    def _missing_requirement(self) -> str:
        """Returns what a run misses, called when something is missing."""
        if self._frames is None:
            return "Puzzle modes need the camera of the real robot."
        if self._homography is None:
            return "No homography is saved, calibrate in Manual move first."
        return "Plate height is not set, fill PLATE_HEIGHT in."

    def _measure_table_motion(
        self, frames: FrameSource, homography: Homography
    ) -> TableMotion | None:
        """Collects plate poses from new frames, None when stopped or late."""
        self._info(
            f"Measuring the table motion from {PLATE_SAMPLE_COUNT} frames..."
        )
        tracker = PlateTracker(homography)
        deadline = self._clock.now() + MEASUREMENT_TIMEOUT
        last_timestamp: float | None = None
        while tracker.motion is None:
            if self._keyboard.poll_key() is Key.STOP:
                self._info("Stopped.")
                return None
            if self._clock.now() > deadline:
                self._warn(
                    f"The plate was not seen in {PLATE_SAMPLE_COUNT} frames"
                    f" within {MEASUREMENT_TIMEOUT:.0f} s, check the markers."
                )
                return None
            frame = frames.latest
            if frame is not None and frame.timestamp != last_timestamp:
                tracker.update(frame)
                last_timestamp = frame.timestamp
            self._clock.sleep(FRAME_POLL_PERIOD)
        direction = "left" if tracker.motion.is_rotating_leftwise else "right"
        self._info(
            f"The table turns {direction} at"
            f" {abs(tracker.motion.angular_velocity):.3f} rad/s."
        )
        return tracker.motion

    def _synchronize(self, planner: RodPlanner) -> bool:
        """Follows the synchronization point until the start, False if stopped.

        Enter searches for a start whose whole run goes through, see
        find_run_schedule. The hoop keeps following the point until then. A
        step that would end after the start is not sent, the hoop waits for
        the start instead: in the lab on 2026-11-25 the last step made the
        run start 0.28 s after the simulated run.
        """
        self._info("Moving to the synchronization point.")
        try:
            self._step(planner, _SYNC_INDEX, can_flip=True)
            self._info(
                "At the synchronization point. Press Enter to go down the rod,"
                " c to stop."
            )
            start_time: float | None = None
            while True:
                key = self._keyboard.poll_key()
                if key is Key.STOP:
                    self._info("Stopped.")
                    return False
                if key is Key.ENTER and start_time is None:
                    start_time = self._find_start_time(planner)
                if (
                    start_time is not None
                    and self._clock.now() + self._last_step_time > start_time
                ):
                    self._clock.sleep(max(0.0, start_time - self._clock.now()))
                    return True
                self._step(planner, _SYNC_INDEX)
        except (NoJointConfigError, ValueError) as error:
            self._warn(f"Synchronization point is out of reach: {error}")
            return False

    def _find_start_time(self, planner: RodPlanner) -> float | None:
        """Searches for a safe start and reports it, None when there is none.

        The earliest tried start is SEARCH_TIME_MARGIN times the time of the
        previous search after now, so the search ends before it. The
        simulated steps of the chosen run go to the log, and their send
        times are kept to compare the real steps with them.
        """
        search_start = self._clock.now()
        start_delay = max(
            MIN_START_DELAY, SEARCH_TIME_MARGIN * self._expected_search_time
        )
        schedule = find_run_schedule(
            planner,
            self._motion,
            self._motion.get_joint_angles(),
            self._hoop_turn,
            now=search_start,
            earliest_start=search_start + start_delay,
        )
        search_time = self._clock.now() - search_start
        self._expected_search_time = search_time
        if schedule is None:
            self._warn(
                f"No run down the rod goes through within {SEARCH_WINDOW:.0f}"
                " s, the hoop stays at the synchronization point. Press Enter"
                " to search again, c to stop."
            )
            return None
        if schedule.start_time < self._clock.now():
            self._warn(
                f"The search took {search_time:.1f} s, longer than the"
                f" {start_delay:.1f} s expected, and the start has passed."
                " Press Enter to search again, c to stop."
            )
            return None
        self._log_schedule(schedule, search_time)
        self._planned_sent_times = {}
        for step in schedule.steps:
            self._planned_sent_times.setdefault(
                (step.phase, step.index), step.sent_time
            )
        remaining_time = schedule.start_time - self._clock.now()
        self._info(f"Going down the rod in {remaining_time:.1f} s.")
        return schedule.start_time

    def _log_schedule(self, schedule: RunSchedule, search_time: float) -> None:
        """Writes the simulated steps of a run to the log."""
        for step in schedule.steps:
            logger.info(
                "Simulated step %s to path point %d: sent at %+.3f s, arrives"
                " at %+.3f s, %s, joints %s deg.",
                step.phase.value,
                step.index,
                step.sent_time - schedule.start_time,
                step.arrival_time - schedule.start_time,
                _describe_turn(step.hoop_turn),
                [round(math.degrees(angle), 1) for angle in step.joint_angles],
            )
        logger.info(
            "Simulated run: %d steps, %.2f s from the start back to the"
            " synchronization point, joint margin %.1f deg, search took"
            " %.2f s.",
            len(schedule.steps),
            schedule.steps[-1].arrival_time - schedule.start_time,
            math.degrees(schedule.joint_margin),
            search_time,
        )

    def _go_down(self, planner: RodPlanner, last_index: int) -> int:
        """Goes down the rod, returns the last path point reached."""
        self._info("Going down the rod, press c to go back out.")
        reached_index = _SYNC_INDEX
        try:
            for index in range(_SYNC_INDEX + 1, last_index + 1):
                if self._keyboard.poll_key() is Key.STOP:
                    self._info("Stopped, going back out.")
                    return reached_index
                self._step(planner, index, phase=RunPhase.DOWN)
                reached_index = index
            wait_end = self._clock.now() + BOTTOM_WAIT
            while self._clock.now() < wait_end:
                if self._keyboard.poll_key() is Key.STOP:
                    self._info("Stopped, going back out.")
                    return reached_index
                self._step(planner, last_index, phase=RunPhase.BOTTOM)
        except (NoJointConfigError, ValueError) as error:
            self._warn(f"Path point is out of reach, going back out: {error}")
        return reached_index

    def _go_back_out(self, planner: RodPlanner, reached_index: int) -> bool:
        """Goes up the rod to the synchronization point, False when it fails.

        Starts one point above reached_index, the hoop is already there. The c
        key is ignored, the hoop must leave the rod.
        """
        self._info("Going back out of the rod.")
        try:
            for index in range(reached_index - 1, _SYNC_INDEX - 1, -1):
                self._step(planner, index, phase=RunPhase.UP)
        except (NoJointConfigError, ValueError) as error:
            self._warn(
                f"Going back out failed, the hoop may still be on the rod: "
                f"{error} Use the Emergency Stop button if the rod pushes it."
            )
            return False
        self._keyboard.discard_pending()
        return True

    def _step(
        self,
        planner: RodPlanner,
        index: int,
        can_flip: bool = False,
        phase: RunPhase | None = None,
    ) -> None:
        """Moves the hoop to a path point predicted for its arrival.

        Args:
            planner: Path of the hoop.
            index: Path point to move to.
            can_flip: True when a joint may change by more than
                MAX_JOINT_STEP, for the move from home to the path.
            phase: Part of the run down the rod, None at the synchronization
                point. On the rod every step is checked in the log, see
                _log_check.

        Raises:
            NoJointConfigError: No handle angle reaches the point, nothing
                is sent.
            ValueError: The planner cannot give a pose for the point.
        """
        sent_time = self._clock.now()
        step = plan_step(
            planner,
            self._motion,
            index,
            sent_time,
            self._motion.get_joint_angles(),
            hoop_turn=self._hoop_turn,
            can_flip=can_flip,
        )
        self._hoop_turn = step.hoop_turn
        self._motion.move_to_ik(step.hoop_pose)
        while self._motion.is_in_motion():
            self._clock.sleep(MOTION_POLL_PERIOD)
        self._last_step_time = self._clock.now() - sent_time
        logger.info(
            "Step to path point %d: expected %.3f s, took %.3f s, %s, joints"
            " %s deg.",
            index,
            step.arrival_time - sent_time,
            self._clock.now() - sent_time,
            _describe_turn(step.hoop_turn),
            [round(math.degrees(angle), 1) for angle in step.joint_angles],
        )
        if phase is not None:
            self._log_check(planner, index, phase, sent_time)

    def _log_check(
        self,
        planner: RodPlanner,
        index: int,
        phase: RunPhase,
        sent_time: float,
    ) -> None:
        """Writes how far the arrived step is from the predictions.

        The measured hoop against the predicted rod at the same moment, the
        send time against the simulated run and the plate seen by the camera
        against its prediction.
        """
        now = self._clock.now()
        rod_pose = planner.get_hoop_pose(now, index)
        later_rod_pose = planner.get_hoop_pose(now + ROD_VELOCITY_TIME, index)
        rod_velocity = (later_rod_pose[:3, 3] - rod_pose[:3, 3]) / (
            ROD_VELOCITY_TIME
        )
        offset, ahead = get_hoop_offset(
            self._motion.get_hoop_pose(),
            rod_pose,
            (
                float(rod_velocity[0]),
                float(rod_velocity[1]),
                float(rod_velocity[2]),
            ),
        )
        planned_sent_time = self._planned_sent_times.pop((phase, index), None)
        timing = (
            "no simulated step"
            if planned_sent_time is None
            else f"sent {sent_time - planned_sent_time:+.2f} s against the"
            " simulation"
        )
        logger.info(
            "Check %s at path point %d: hoop %.1f mm off the rod, %+.1f mm"
            " ahead of it; %s; %s.",
            phase.value,
            index,
            offset * _MILLIMETERS_PER_METER,
            ahead * _MILLIMETERS_PER_METER,
            timing,
            self._describe_plate_error(),
        )

    def _describe_plate_error(self) -> str:
        """Compares the plate in the newest frame with its prediction."""
        frame = self._frames.latest if self._frames is not None else None
        if frame is None or self._homography is None:
            return "no camera frame"
        marker_ids = sorted(marker.marker_id for marker in frame.markers)
        markers = ", ".join(str(marker_id) for marker_id in marker_ids)
        measured = estimate_plate_pose(frame.markers, self._homography)
        if measured is None or self._table_motion is None:
            return f"plate not seen (markers: {markers or 'none'})"
        predicted = self._table_motion.predict(frame.timestamp - CAMERA_LATENCY)
        distance, heading = get_plate_error(measured, predicted)
        return (
            f"plate {distance * _MILLIMETERS_PER_METER:.1f} mm and"
            f" {math.degrees(heading):+.1f} deg off the prediction (markers:"
            f" {markers})"
        )

    def _info(self, message: str) -> None:
        logger.info(message)
        self._reporter.info(message)

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self._reporter.warning(message)


def _describe_turn(hoop_turn: HoopTurn) -> str:
    """Describes a hoop turn for the log, angles in degrees."""
    description = f"handle {math.degrees(hoop_turn.handle_angle):.0f} deg"
    if hoop_turn.tilt:
        description += (
            f", tilt {math.degrees(hoop_turn.tilt):.0f} deg toward"
            f" {math.degrees(hoop_turn.tilt_direction):.0f} deg"
        )
    return description
