"""Thin wrapper over the ctu_crs robots."""

from ctu_crs import CRS93
from ctu_crs import CRS97

from hoop_guide.types import Image

ROBOT_CLASSES = {"crs93": CRS93, "crs97": CRS97}


class VirtualRobotError(RuntimeError):
    """Raised when an operation needs the controller of the real robot."""


class Robot:
    """Real or virtual CRS robot.

    The virtual robot has no connection to the controller, only computations
    like fk and ik work. Operations that need the controller raise
    VirtualRobotError.

    Args:
        name: Robot name, a key of ROBOT_CLASSES.
        is_real: Connect to the real robot.
    """

    def __init__(self, name: str, is_real: bool) -> None:
        robot_class = ROBOT_CLASSES[name]
        self._crs = robot_class() if is_real else robot_class(tty_dev=None)
        self._name = name.upper()
        self._is_real = is_real
        self._is_initialized = False

    @property
    def name(self) -> str:
        """Display name, e.g. CRS93."""
        return self._name

    @property
    def is_real(self) -> bool:
        """True for the real robot, False for the virtual one."""
        return self._is_real

    def initialize(self, should_home: bool) -> None:
        """Initializes the controller, optionally with hard and soft home."""
        self._require_real()
        self._crs.initialize(home=should_home)
        self._is_initialized = True

    def soft_home(self) -> None:
        """Moves the robot to the home position."""
        self._require_real()
        self._crs.soft_home()

    def release(self) -> None:
        """Engages the brakes and disconnects feedback."""
        self._require_real()
        self._crs.release()

    def reset_motors(self) -> None:
        """Resets motors in an error state."""
        self._require_real()
        self._crs.reset_motors()

    def grab_image(self) -> Image:
        """Returns the current image from the station camera.

        ctu_crs connects to the camera on the first call. Raises
        VirtualRobotError for the virtual robot and RuntimeError when the
        camera returns no image.
        """
        self._require_real()
        image = self._crs.grab_image()
        if image is None:
            raise RuntimeError("Camera returned no image.")
        return Image(image)

    def shutdown(self) -> None:
        """Moves the robot home if initialized and closes the connection.

        Follows the end of session procedure from the station manual. Does
        nothing for the virtual robot.
        """
        if not self._is_real:
            return
        try:
            if self._is_initialized:
                self._crs.wait_for_motion_stop()
                self._crs.soft_home()
        finally:
            self._crs.close()

    def _require_real(self) -> None:
        if not self._is_real:
            raise VirtualRobotError(f"{self._name} is virtual.")
