"""Thin wrapper over the ctu_crs robots."""

import numpy as np
from ctu_crs import CRS93
from ctu_crs import CRS97
from ctu_crs.crs_robot import CRSRobot

from hoop_guide.types import Image
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
from hoop_guide.types import Points3D
from hoop_guide.types import Pose

ROBOT_CLASSES: dict[str, type[CRSRobot]] = {"crs93": CRS93, "crs97": CRS97}

_IRC_PER_IRC256 = 1 / 256
_MILLISECONDS_PER_SECOND = 1000.0


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
        self._crs: CRSRobot = (
            robot_class() if is_real else robot_class(tty_dev=None)
        )
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

    @property
    def joint_speeds(self) -> JointSpeeds:
        """Speeds of the joints in a move, the defaults initialize() sets.

        ctu_crs keeps them only in private attributes, in IRC * 256 per
        millisecond of the motor, and the IRC per degree of each joint.
        """
        motor_irc_per_second = (
            self._crs._default_speed_irc256_per_ms
            * _IRC_PER_IRC256
            * _MILLISECONDS_PER_SECOND
        )
        degrees_per_second = motor_irc_per_second / np.abs(
            self._crs._deg_to_irc
        )
        return JointSpeeds(np.deg2rad(degrees_per_second))

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

    def get_joint_angles(self) -> JointConfig:
        """Returns the joint angles measured by the controller."""
        self._require_real()
        return JointConfig(np.asarray(self._crs.get_q(), dtype=np.float64))

    def set_joint_angles(self, joint_config: JointConfig) -> None:
        """Starts a coordinated move to the joint angles and returns at once.

        Poll is_in_motion() to learn when the robot arrives.

        Raises:
            VirtualRobotError: The robot is virtual.
            RuntimeError: The robot was not initialized.
            ValueError: The joint angles are outside the joint limits, nothing
                is sent to the controller.
        """
        self._require_real()
        if not self._is_initialized:
            raise RuntimeError(f"{self._name} is not initialized.")
        if not self.is_in_limits(joint_config):
            raise ValueError(f"Joint angles out of limits: {joint_config}.")
        self._crs.move_to_q(joint_config)

    def get_ik_options(self, flange_pose: Pose) -> list[JointConfig]:
        """Returns every joint configuration that reaches the flange pose.

        Joint limits are not checked, the list is empty when the pose is out
        of reach. Options are calculated without hoop in mind.
        """
        return [
            JointConfig(np.asarray(joint_config, dtype=np.float64))
            for joint_config in self._crs.ik(flange_pose)
        ]

    def get_link_points(self, joint_config: JointConfig) -> Points3D:
        """Returns the origins of the joint frames of a joint configuration.

        From the DH model of ctu_crs, the same chain as its fk. Shape (6, 3)
        in the robot frame [m]: the frames after joints 1 to 6, the last one
        is the flange. The base, which stands on the table, is not included.
        """
        pose = np.eye(4)
        points = []
        for d, a, alpha, offset, joint_angle in zip(
            self._crs.dh_d,
            self._crs.dh_a,
            self._crs.dh_alpha,
            self._crs.dh_offset,
            joint_config,
            strict=True,
        ):
            pose = pose @ self._crs.dh_to_se3(d, joint_angle + offset, a, alpha)
            points.append(pose[:3, 3].copy())
        return Points3D(np.array(points))

    def get_fk_pos(self) -> Pose:
        """Returns the flange pose computed from the measured joint angles.

        The pose is of the flange, the gripper and the hoop are not included.
        """
        joint_config = self.get_joint_angles()
        return Pose(np.asarray(self._crs.fk(joint_config), dtype=np.float64))

    @property
    def joint_limits(self) -> tuple[JointConfig, JointConfig]:
        """Lowest and highest angle of every joint [rad]."""
        return (
            JointConfig(np.asarray(self._crs.q_min, dtype=np.float64)),
            JointConfig(np.asarray(self._crs.q_max, dtype=np.float64)),
        )

    def is_in_limits(self, joint_config: JointConfig) -> bool:
        """Returns True if every joint angle is within its limits."""
        return bool(self._crs.in_limits(joint_config))

    def is_in_motion(self) -> bool:
        """Returns True while the controller is still executing a move."""
        self._require_real()
        return bool(self._crs.in_motion())

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
        """Checks whether current instance uses real robot."""
        if not self._is_real:
            raise VirtualRobotError(f"{self._name} is virtual.")
