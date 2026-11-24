"""Run of one puzzle: the hoop goes through the rod of the rotating plate."""

import logging
import time
from types import TracebackType
from typing import Protocol
from typing import Self

from hoop_guide.estimation import TableMotion
from hoop_guide.keyboard import Key
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.plate_tracker import PLATE_SAMPLE_COUNT
from hoop_guide.plate_tracker import PlateTracker
from hoop_guide.rods import Puzzle
from hoop_guide.types import Homography
from hoop_guide.types import Pose
from hoop_guide.vision import Frame

logger = logging.getLogger(__name__)


PLATE_HEIGHT: float | None = 0.07
"""Height of the plate top surface above the robot base frame origin, the
surface the robot stands on [m]."""

# Measured in the lab on 2026-11-24, puzzle A2, from the expected and the
# real step times in app.log. With an overhead of 0.03 s and no factor the
# hoop lagged visibly behind the rod center at the synchronization point.
# The smallest steps, 0.01 s at full joint speeds, took 0.29 to 0.30 s.
# The move from home to the synchronization point, 3.94 s at full joint
# speeds, took 5.32 s: 1.27 times longer after the overhead. Steps of 2 cm
# down the rod were not in that log, refine both when they are.
# In the second test that day, with 0.3 s, the hoop was slightly ahead of the
# rod, about the same all the time: the ready flag of the controller comes
# later than the robot stops. Lowered to 0.25 s, tune further.
MOVE_OVERHEAD = 0.25
"""Time every step takes on top of its move, however small [s].

Communication with the controller, waiting for its ready flag and
settling.
"""

MOVE_TIME_FACTOR = 1.3
"""How many times longer a move takes than at full joint speeds.

The joints speed up and slow down, see MOVE_OVERHEAD for the measurement.
"""

MEASUREMENT_TIMEOUT = 15.0
"""Longest wait for the frames the table motion is measured from [s]."""

BOTTOM_WAIT = 1.0
"""How long the hoop follows the last path point before going back [s]."""

FRAME_POLL_PERIOD = 0.02
"""Time between two looks for a new camera frame [s]."""

MOTION_POLL_PERIOD = 0.02
"""Time between two questions whether the robot still moves [s]."""

_SYNC_INDEX = 0


class PuzzleMotion(Protocol):
    """Moving the hoop."""

    def move_to_ik(self, hoop_pose: Pose) -> None:
        """Starts a move that puts the hoop into the pose.

        Raises:
            NoJointConfigError: The pose cannot be reached, nothing is sent
                to the robot.
        """

    def get_move_time(self, hoop_pose: Pose) -> float:
        """Returns how long move_to_ik to the pose takes at full joint speeds.

        Raises:
            NoJointConfigError: The pose cannot be reached.
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
    rod and follows the rotation there until the operator presses Enter.
    Then it goes down the rod point by point, follows the last point for
    BOTTOM_WAIT, goes back out to the synchronization point and returns
    home. Every step is aimed where the rod is when the robot arrives, the
    time of the move at full joint speeds times MOVE_TIME_FACTOR plus
    MOVE_OVERHEAD, and waits until the robot stands still.

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
        with self._keyboard:
            self._keyboard.discard_pending()
            table_motion = self._measure_table_motion(frames, homography)
            if table_motion is None:
                return
            planner = RodPlanner(puzzle, table_motion, plate_height)
            if not self._synchronize(planner):
                return
            last_index = len(planner.update(self._clock.now(), _SYNC_INDEX)) - 1
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
        """Follows the synchronization point until Enter, False when stopped."""
        self._info("Moving to the synchronization point.")
        try:
            self._step(planner, _SYNC_INDEX)
            self._info(
                "At the synchronization point. Press Enter to go down the rod,"
                " c to stop."
            )
            while True:
                key = self._keyboard.poll_key()
                if key is Key.ENTER:
                    return True
                if key is Key.STOP:
                    self._info("Stopped.")
                    return False
                self._step(planner, _SYNC_INDEX)
        except (NoJointConfigError, ValueError) as error:
            self._warn(f"Synchronization point is out of reach: {error}")
            return False

    def _go_down(self, planner: RodPlanner, last_index: int) -> int:
        """Goes down the rod, returns the last path point reached."""
        self._info("Going down the rod, press c to go back out.")
        reached_index = _SYNC_INDEX
        try:
            for index in range(_SYNC_INDEX + 1, last_index + 1):
                if self._keyboard.poll_key() is Key.STOP:
                    self._info("Stopped, going back out.")
                    return reached_index
                self._step(planner, index)
                reached_index = index
            wait_end = self._clock.now() + BOTTOM_WAIT
            while self._clock.now() < wait_end:
                if self._keyboard.poll_key() is Key.STOP:
                    self._info("Stopped, going back out.")
                    return reached_index
                self._step(planner, last_index)
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
                self._step(planner, index)
        except (NoJointConfigError, ValueError) as error:
            self._warn(
                f"Going back out failed, the hoop may still be on the rod: "
                f"{error} Use the Emergency Stop button if the rod pushes it."
            )
            return False
        self._keyboard.discard_pending()
        return True

    def _step(self, planner: RodPlanner, index: int) -> None:
        """Moves the hoop to a path point predicted for its arrival.

        Raises:
            NoJointConfigError: The point cannot be reached.
            ValueError: The planner cannot give a pose for the point.
        """
        sent_time = self._clock.now()
        # The arrival time depends on the target, one estimate is enough
        # because the table hardly moves in the difference.
        estimate = planner.update(sent_time + MOVE_OVERHEAD, index)[0]
        lead_time = (
            MOVE_TIME_FACTOR * self._motion.get_move_time(estimate)
            + MOVE_OVERHEAD
        )
        self._motion.move_to_ik(planner.update(sent_time + lead_time, index)[0])
        while self._motion.is_in_motion():
            self._clock.sleep(MOTION_POLL_PERIOD)
        logger.info(
            "Step to path point %d: expected %.3f s, took %.3f s.",
            index,
            lead_time,
            self._clock.now() - sent_time,
        )

    def _info(self, message: str) -> None:
        logger.info(message)
        self._reporter.info(message)

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self._reporter.warning(message)
