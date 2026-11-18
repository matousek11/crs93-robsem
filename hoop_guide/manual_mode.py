"""Manual move of the hoop and recording of calibration points."""

import dataclasses
import enum
import logging
import math
import time
from pathlib import Path
from types import TracebackType
from typing import Protocol
from typing import Self

import numpy as np

from hoop_guide.calibration import CALIBRATION_POINT_COUNT
from hoop_guide.calibration import MAX_REPROJECTION_ERROR
from hoop_guide.calibration import CalibrationSession
from hoop_guide.calibration import save_homography
from hoop_guide.geometry import angles_from_rotation
from hoop_guide.geometry import rotation_from_angles
from hoop_guide.keyboard import Key
from hoop_guide.planning import NoJointConfigError
from hoop_guide.types import Point3D
from hoop_guide.types import Pose
from hoop_guide.types import RotationAngles
from hoop_guide.types import RotationMatrix
from hoop_guide.vision import Frame
from hoop_guide.vision import detect_hoop_center

logger = logging.getLogger(__name__)

MANUAL_STEP = 0.01
"""Change of the target position by one arrow key press [m]."""

MANUAL_ROTATION_STEP = math.radians(5.0)
"""Change of a target rotation angle by one arrow key press [rad]."""

MOTION_POLL_PERIOD = 0.05
"""Time between two questions whether the robot still moves [s]."""

_MILLIMETERS_PER_METER = 1000.0


class ManualRow(enum.Enum):
    """Rows of the manual move table, from the top to the bottom."""

    X = "x"
    Y = "y"
    Z = "z"
    ROTATION_X = "rotation x"
    ROTATION_Y = "rotation y"
    ROTATION_Z = "rotation z"
    EXIT = "exit"


class ManualCommand(enum.Enum):
    """What a key press asks the manual mode to do."""

    MOVE = "move"
    RECORD = "record"
    EXIT = "exit"


POSITION_ROWS = (ManualRow.X, ManualRow.Y, ManualRow.Z)
"""Rows of the target position, in the order of its axes."""

ROTATION_ROWS = (
    ManualRow.ROTATION_X,
    ManualRow.ROTATION_Y,
    ManualRow.ROTATION_Z,
)
"""Rows of the target rotation angles, in the order of their axes."""

_ROWS = tuple(ManualRow)
_ROW_OFFSET_BY_KEY = {Key.UP: -1, Key.DOWN: 1}
_TARGET_CHANGE_SIGN_BY_KEY = {Key.LEFT: -1.0, Key.RIGHT: 1.0}


@dataclasses.dataclass(frozen=True, eq=False)
class ManualState:
    """What the operator has edited in the manual move table.

    Attributes:
        selected_row: Row the arrow keys and Enter act on.
        target_position: Position of the hoop center the robot moves to on
            Enter, not sent to the robot yet.
        target_rotation_angles: Orientation of the hoop the robot moves to
            on Enter, not sent to the robot yet.
    """

    selected_row: ManualRow
    target_position: Point3D
    target_rotation_angles: RotationAngles

    def handle_key(
        self, key: Key
    ) -> tuple["ManualState", ManualCommand | None]:
        """Returns the state after a key press and what has to be done.

        Up and down select the neighbouring row and stop at the first and
        the last one. Left and right change the selected axis of the target
        position by MANUAL_STEP or the selected target rotation angle by
        MANUAL_ROTATION_STEP, right increases it. Enter asks for a move to
        the whole target, on the Exit row it asks to leave. Space asks to
        record a calibration point.

        Args:
            key: Pressed key.

        Returns:
            The new state and the command, None when the key only edits the
            state.
        """
        if key in _ROW_OFFSET_BY_KEY:
            return self._select_neighbour(_ROW_OFFSET_BY_KEY[key]), None
        if key in _TARGET_CHANGE_SIGN_BY_KEY:
            return self._change_target(_TARGET_CHANGE_SIGN_BY_KEY[key]), None
        if key is Key.SPACE:
            return self, ManualCommand.RECORD
        if self.selected_row is ManualRow.EXIT:
            return self, ManualCommand.EXIT
        return self, ManualCommand.MOVE

    def _select_neighbour(self, offset: int) -> "ManualState":
        row_index = _ROWS.index(self.selected_row) + offset
        row_index = min(max(row_index, 0), len(_ROWS) - 1)
        return dataclasses.replace(self, selected_row=_ROWS[row_index])

    def _change_target(self, sign: float) -> "ManualState":
        if self.selected_row in POSITION_ROWS:
            target_position = self.target_position.copy()
            axis_index = POSITION_ROWS.index(self.selected_row)
            target_position[axis_index] += sign * MANUAL_STEP
            return dataclasses.replace(
                self, target_position=Point3D(target_position)
            )
        if self.selected_row in ROTATION_ROWS:
            target_rotation_angles = self.target_rotation_angles.copy()
            axis_index = ROTATION_ROWS.index(self.selected_row)
            target_rotation_angles[axis_index] += sign * MANUAL_ROTATION_STEP
            return dataclasses.replace(
                self,
                target_rotation_angles=RotationAngles(target_rotation_angles),
            )
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class ManualView:
    """Everything the manual move table shows.

    Attributes:
        state: Selected row, target position and target rotation angles.
        current_position: Position of the hoop center measured after the
            last move.
        current_rotation_angles: Orientation of the hoop measured after the
            last move. The same orientation can have other angles than the
            target, see angles_from_rotation.
        recorded_points: Number of calibration points recorded so far.
        message: What happened last, empty when nothing did.
    """

    state: ManualState
    current_position: Point3D
    current_rotation_angles: RotationAngles
    recorded_points: int
    message: str


class HoopMotion(Protocol):
    """Moving the hoop and reading where it is."""

    def move_to_ik(self, hoop_pose: Pose) -> None:
        """Starts a move that puts the hoop into the pose.

        Raises:
            NoJointConfigError: The pose cannot be reached, nothing is sent
                to the robot.
        """

    def get_hoop_pose(self) -> Pose:
        """Returns the hoop pose computed from the measured joint angles."""

    def is_in_motion(self) -> bool:
        """Returns True while the robot is still moving."""


class FrameSource(Protocol):
    """Source of camera frames."""

    @property
    def latest(self) -> Frame | None:
        """The newest Frame, None before the first one arrives."""


class KeyReader(Protocol):
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

    def read_key(self) -> Key:
        """Waits for the next key press."""

    def discard_pending(self) -> None:
        """Drops key presses that arrived while nobody was reading."""


class ManualDisplay(Protocol):
    """Manual move table, drawn while used as a context."""

    def __enter__(self) -> Self:
        """Starts drawing the table."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stops drawing the table."""

    def show(self, view: ManualView) -> None:
        """Redraws the table."""


class ManualMode:
    """Moves the hoop from the keyboard and records calibration points.

    The target starts at the measured hoop pose, so a move without an edit
    leaves the hoop where it is. After CALIBRATION_POINT_COUNT recorded
    points the homography is computed and saved when its reprojection error
    is at most MAX_REPROJECTION_ERROR. Points are never kept between two runs.

    Args:
        motion: Moves the hoop.
        frames: Camera frames the hoop is detected in.
        keyboard: Key presses of the operator.
        display: Table shown to the operator.
        homography_path: File the homography is saved to.
    """

    def __init__(
        self,
        motion: HoopMotion,
        frames: FrameSource,
        keyboard: KeyReader,
        display: ManualDisplay,
        homography_path: Path,
    ) -> None:
        self._motion = motion
        self._frames = frames
        self._keyboard = keyboard
        self._display = display
        self._homography_path = homography_path

    def run(self) -> None:
        """Runs until the operator leaves the mode."""
        logger.info("Manual move started.")
        with self._keyboard, self._display:
            current_hoop_pose = self._motion.get_hoop_pose()
            state = ManualState(
                ManualRow.X,
                _get_position(current_hoop_pose),
                _get_rotation_angles(current_hoop_pose),
            )
            session = CalibrationSession()
            message = ""
            while True:
                view = ManualView(
                    state,
                    _get_position(current_hoop_pose),
                    _get_rotation_angles(current_hoop_pose),
                    session.point_count,
                    message,
                )
                self._display.show(view)
                state, command = state.handle_key(self._keyboard.read_key())
                if command is ManualCommand.EXIT:
                    break
                if command is ManualCommand.MOVE:
                    self._display.show(
                        dataclasses.replace(view, message="Moving...")
                    )
                    current_hoop_pose, message = self._move(
                        state, current_hoop_pose
                    )
                elif command is ManualCommand.RECORD:
                    session, message = self._record(
                        session, _get_position(current_hoop_pose)
                    )
        logger.info("Manual move finished.")

    def _move(
        self, state: ManualState, current_hoop_pose: Pose
    ) -> tuple[Pose, str]:
        """Moves the hoop, returns its measured pose and a message."""
        try:
            self._motion.move_to_ik(_get_target_hoop_pose(state))
        except NoJointConfigError:
            return current_hoop_pose, _log(
                "Target is out of reach, robot did not move."
            )
        while self._motion.is_in_motion():
            time.sleep(MOTION_POLL_PERIOD)
        self._keyboard.discard_pending()
        return self._motion.get_hoop_pose(), _log("Move done.")

    def _record(
        self, session: CalibrationSession, hoop_position: Point3D
    ) -> tuple[CalibrationSession, str]:
        """Records one point, returns the session to continue with."""
        frame = self._frames.latest
        if frame is None:
            return session, _log("No camera image yet, point not recorded.")
        hoop_center = detect_hoop_center(frame.image)
        if hoop_center is None:
            return session, _log(
                "Hoop not found in the image, point not recorded."
            )
        try:
            session.record(hoop_center, hoop_position)
        except ValueError as error:
            return session, _log(str(error))
        if not session.is_complete:
            return session, _log(
                f"Point {session.point_count}/{CALIBRATION_POINT_COUNT}"
                " recorded."
            )
        return CalibrationSession(), self._save_homography(session)

    def _save_homography(self, session: CalibrationSession) -> str:
        """Saves the homography of a complete session if it is good enough."""
        try:
            homography, reprojection_error = session.solve()
        except ValueError as error:
            return _log(f"{error} Homography not saved.")
        error_millimeters = reprojection_error * _MILLIMETERS_PER_METER
        if reprojection_error > MAX_REPROJECTION_ERROR:
            limit_millimeters = MAX_REPROJECTION_ERROR * _MILLIMETERS_PER_METER
            return _log(
                f"Error {error_millimeters:.1f} mm is above"
                f" {limit_millimeters:.1f} mm, homography not saved."
            )
        save_homography(self._homography_path, homography)
        return _log(
            f"Homography saved, error {error_millimeters:.1f} mm."
            " Restart the application to use it."
        )


def _get_position(hoop_pose: Pose) -> Point3D:
    return Point3D(hoop_pose[:3, 3].copy())


def _get_rotation_angles(hoop_pose: Pose) -> RotationAngles:
    return angles_from_rotation(RotationMatrix(hoop_pose[:3, :3]))


def _get_target_hoop_pose(state: ManualState) -> Pose:
    target_hoop_pose = Pose(np.eye(4))
    target_hoop_pose[:3, :3] = rotation_from_angles(
        state.target_rotation_angles
    )
    target_hoop_pose[:3, 3] = state.target_position
    return target_hoop_pose


def _log(message: str) -> str:
    logger.info(message)
    return message
