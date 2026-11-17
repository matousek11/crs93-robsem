"""Commands sent to the robot."""

from hoop_guide.hardware.interfaces import JointMotion


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
