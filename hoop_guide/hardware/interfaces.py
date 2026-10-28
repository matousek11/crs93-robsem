"""Interfaces of the hardware used by the application."""

from typing import Protocol

from hoop_guide.types import Image


class Camera(Protocol):
    """Source of camera images."""

    def grab_image(self) -> Image:
        """Returns the current image from the camera."""


class RobotControl(Protocol):
    """Robot operations used by the application."""

    @property
    def name(self) -> str:
        """Display name, e.g. CRS93."""

    @property
    def is_real(self) -> bool:
        """True for the real robot, False for the virtual one."""

    def initialize(self, should_home: bool) -> None:
        """Initializes the controller, optionally with hard and soft home."""

    def soft_home(self) -> None:
        """Moves the robot to the home position."""

    def release(self) -> None:
        """Engages the brakes and disconnects feedback."""

    def reset_motors(self) -> None:
        """Resets motors in an error state."""

    def shutdown(self) -> None:
        """Moves the robot home if initialized and closes the connection."""
