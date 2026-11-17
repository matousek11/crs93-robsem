"""Moves of the hoop to a target pose."""

from hoop_guide.hardware.interfaces import JointMotion
from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.models.hoop_mount import HoopMount
from hoop_guide.types import Pose


class NoJointConfigError(RuntimeError):
    """Raised when no joint configuration is left to reach a pose."""


class Planning:
    """Chooses the joint configuration for a hoop pose and moves there.

    Args:
        robot: Robot to move.
    """

    def __init__(self, robot: JointMotion) -> None:
        self.robot: JointMotion = robot
        self.hoop_mount: HoopMount = HoopMount(FLANGE_TO_HOOP)

    def move_to_ik(self, hoop_pose: Pose) -> None:
        """Starts a move that puts the hoop into the pose.

        Of the joint configurations within the joint limits, the one with
        the smallest mean absolute change of the joint angles is used.
        Returns at once, before the robot arrives.

        Args:
            hoop_pose: Target pose of the hoop in the robot frame.

        Raises:
            NoJointConfigError: The pose is out of reach or only beyond the
                joint limits. Nothing is sent to the robot.
        """
        # 1. get available configurations from crs library
        flange_pose = self.hoop_mount.to_flange_pose(hoop_pose)
        possible_configs = self.robot.get_ik_options(flange_pose)

        # 2. cut-off configurations beyond the joint limits
        # TODO: cut-off configurations that hit an obstacle too
        possible_configs = [
            config
            for config in possible_configs
            if self.robot.is_in_limits(config)
        ]

        # 3. stop when no config remained
        if len(possible_configs) == 0:
            raise NoJointConfigError(
                "No joint configuration reaches the hoop pose."
            )

        # 4. find the configuration with the smallest average change per joint
        curr_config = self.robot.get_joint_angles()
        avg_change = []
        for possible_config in possible_configs:
            new_conf_change_sum = 0.0
            for new_joint, curr_joint in zip(
                possible_config, curr_config, strict=True
            ):
                new_conf_change_sum += abs(new_joint - curr_joint)

            avg_change.append(new_conf_change_sum / len(curr_config))

        picked_config = possible_configs[avg_change.index(min(avg_change))]

        # 5. move robot to new configurations
        self.robot.set_joint_angles(picked_config)

    def get_hoop_pose(self) -> Pose:
        """Returns the hoop pose computed from the measured joint angles."""
        return self.hoop_mount.to_hoop_pose(self.robot.get_fk_pos())

    def is_in_motion(self) -> bool:
        """Returns True while the robot is still moving."""
        return self.robot.is_in_motion()
