"""Interfaces of the hardware used by the application."""

from typing import Protocol

from hoop_guide.types import Image
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
from hoop_guide.types import Points3D
from hoop_guide.types import Pose


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


class JointMotion(Protocol):
    """Reading and commanding the joint angles of the robot."""

    @property
    def joint_speeds(self) -> JointSpeeds:
        """Speeds of the joints in a move [rad/s]."""

    def get_joint_angles(self) -> JointConfig:
        """Returns the joint angles measured by the controller."""

    def set_joint_angles(self, joint_config: JointConfig) -> None:
        """Starts a move to the joint angles and returns at once.

        Raises:
            ValueError: The joint angles are outside the joint limits, nothing
                is sent to the controller.
        """

    def get_ik_options(self, flange_pose: Pose) -> list[JointConfig]:
        """Returns every joint configuration that reaches the flange pose.

        Joint limits are not checked, the list is empty when the pose is out
        of reach.
        """

    @property
    def joint_limits(self) -> tuple[JointConfig, JointConfig]:
        """Lowest and highest angle of every joint [rad]."""

    def is_in_limits(self, joint_config: JointConfig) -> bool:
        """Returns True if every joint angle is within its limits."""

    def get_link_points(self, joint_config: JointConfig) -> Points3D:
        """Returns the origins of the joint frames, the last one the flange.

        Shape (6, 3) in the robot frame [m], the base is not included.
        """

    def get_fk_pos(self) -> Pose:
        """Returns the flange pose computed from the measured joint angles."""

    def is_in_motion(self) -> bool:
        """Returns True while the controller is still executing a move."""
