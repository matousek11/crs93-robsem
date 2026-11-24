"""Commands sent to the robot and the timing of robot moves."""

import numpy as np

from hoop_guide.hardware.interfaces import JointMotion
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds


class MotionController:
    """Moves the robot, the only place that sends motion commands.

    Args:
        robot: Robot to move.
    """

    def __init__(self, robot: JointMotion) -> None:
        self._robot = robot

    def move_joint(self, joint_index: int, joint_angle: float) -> None:
        """Starts a move of one joint, the other joints keep their angles.

        Used for manual control of the robot joint by joint. Returns at once,
        before the robot arrives.

        Args:
            joint_index: Joint to move, 0 is the base joint.
            joint_angle: Target angle of that joint [rad].

        Raises:
            ValueError: The robot has no such joint, or the angle is outside
                the joint limits. Nothing is sent to the robot.
        """
        joint_config = self._robot.get_joint_angles().copy()
        if not 0 <= joint_index < len(joint_config):
            raise ValueError(
                f"Joint index must be between 0 to {len(joint_config) - 1}, "
                f"got {joint_index}."
            )
        joint_config[joint_index] = joint_angle

        # TODO: check that the move does not hit an object in the work area.

        self._robot.set_joint_angles(joint_config)


def get_move_time(
    start: JointConfig, target: JointConfig, joint_speeds: JointSpeeds
) -> float:
    """Returns how long a coordinated move takes at full joint speeds.

    In a coordinated move all joints arrive together, so the time is set by
    the joint whose change takes longest at its speed. Speeding up and
    slowing down are not included.

    Args:
        start: Joint angles at the start [rad].
        target: Joint angles at the end [rad].
        joint_speeds: Speed of every joint [rad/s].

    Returns:
        Time of the move [s], 0 when no joint changes.

    Raises:
        ValueError: The three do not have the same shape, or a speed is not
            positive.
    """
    if not start.shape == target.shape == joint_speeds.shape:
        raise ValueError(
            "Start, target and joint speeds must have the same shape."
        )
    if np.any(joint_speeds <= 0.0):
        raise ValueError("Joint speeds must be positive.")
    return float(np.max(np.abs(target - start) / joint_speeds))
