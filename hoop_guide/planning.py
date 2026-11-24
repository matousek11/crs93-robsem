"""Moves of the hoop to a target pose and the hoop path along the rod."""

import math

import numpy as np
from numpy.typing import NDArray

from hoop_guide.estimation import TableMotion
from hoop_guide.hardware.interfaces import JointMotion
from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.models.hoop_mount import HoopMount
from hoop_guide.plate import plate_to_robot_directions
from hoop_guide.plate import plate_to_robot_points
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
from hoop_guide.rods import rod_centerline
from hoop_guide.types import PlateDirections3D
from hoop_guide.types import PlatePoints3D
from hoop_guide.types import Pose

PATH_STEP = 0.02
"""Distance between neighbouring points of the hoop path [m].

Tangents are averaged over three neighbouring points, so this also sets how
long the hoop takes to turn in a bend of the rod: two steps.
"""

SYNC_DISTANCE = 0.10
"""Distance of the synchronization point from the free end of the rod, along
the first segment of the rod [m]."""

_MIN_HANDLE_LENGTH = 1e-9
"""Below this the direction to the robot base is along the hoop axis."""


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


class RodPlanner:
    """Hoop path along a rod of the rotating plate.

    The path is computed once in the plate frame: points every PATH_STEP from
    the synchronization point down to STOP_HEIGHT, with tangents averaged
    over three neighbouring points. An update moves the path with the
    predicted plate pose. The planner keeps no state between updates.

    In a hoop pose the z axis is the tangent of the rod, pointing toward the
    plate. The x axis points along the handle, as close to the robot base as
    the z axis allows: the level direction from the hoop center to the robot
    z axis with its part along the z axis removed.

    Args:
        puzzle: Rod to go through.
        table_motion: Measured motion of the table.
        plate_height: Height of the plate top surface above the robot base
            [m].

    Raises:
        ValueError: plate_height is not finite.
        NotImplementedError: The rod of the puzzle has no model.
    """

    def __init__(
        self, puzzle: Puzzle, table_motion: TableMotion, plate_height: float
    ) -> None:
        if not math.isfinite(plate_height):
            raise ValueError("Plate height must be finite.")
        self._table_motion = table_motion
        self._plate_height = plate_height
        rod_points, rod_tangents = rod_centerline(puzzle, PATH_STEP)
        points, tangents = _cut_at_stop_height(
            rod_points, _average_neighbours(rod_tangents)
        )
        sync_point = points[0] - SYNC_DISTANCE * tangents[0]
        self._points = PlatePoints3D(np.vstack([sync_point, points]))
        self._tangents = PlateDirections3D(np.vstack([tangents[0], tangents]))

    def update(self, target_time: float, start_index: int) -> tuple[Pose, ...]:
        """Returns the hoop poses from a path point on, for a moment.

        Args:
            target_time: time.monotonic at which the poses hold [s].
            start_index: First path point not passed yet, 0 is the
                synchronization point.

        Returns:
            Poses of the hoop in the robot frame, from start_index to the end
            of the path.

        Raises:
            ValueError: start_index is not a point of the path, or the hoop
                axis in a point is level and points at the robot z axis, so
                the handle cannot point toward the robot base.
        """
        if not 0 <= start_index < len(self._points):
            raise ValueError(
                f"Start index must be between 0 and {len(self._points) - 1},"
                f" got {start_index}."
            )
        plate = self._table_motion.predict(target_time)
        points = plate_to_robot_points(
            PlatePoints3D(self._points[start_index:]), plate, self._plate_height
        )
        tangents = plate_to_robot_directions(
            PlateDirections3D(self._tangents[start_index:]), plate
        )
        hoop_poses = []
        for offset, (point, tangent) in enumerate(
            zip(points, tangents, strict=True)
        ):
            # TODO: when the rod slopes to the robot, the handle points
            # down and near the plate the flange can go below the plate top.
            handle = _handle_toward_base(point, tangent)
            if handle is None:
                raise ValueError(
                    "Handle cannot point toward the robot base at path point"
                    f" {start_index + offset}."
                )
            hoop_pose = np.eye(4)
            hoop_pose[:3, 0] = handle
            hoop_pose[:3, 1] = np.cross(tangent, handle)
            hoop_pose[:3, 2] = tangent
            hoop_pose[:3, 3] = point
            hoop_poses.append(Pose(hoop_pose))
        return tuple(hoop_poses)


def _average_neighbours(tangents: PlateDirections3D) -> PlateDirections3D:
    """Averages every tangent with its neighbours, ends with one neighbour."""
    sums = tangents.copy()
    sums[1:] += tangents[:-1]
    sums[:-1] += tangents[1:]
    return PlateDirections3D(sums / np.linalg.norm(sums, axis=1, keepdims=True))


def _cut_at_stop_height(
    points: PlatePoints3D, tangents: PlateDirections3D
) -> tuple[PlatePoints3D, PlateDirections3D]:
    """Ends the path where it first gets down to STOP_HEIGHT above the plate.

    The last point lies exactly at STOP_HEIGHT, between the last point above
    it and the next one, with the tangent of the point above. Every rod
    model starts above STOP_HEIGHT and ends on the plate.
    """
    first_down = int(np.flatnonzero(points[:, 2] <= STOP_HEIGHT)[0])
    above_point, down_point = points[first_down - 1], points[first_down]
    fraction = (above_point[2] - STOP_HEIGHT) / (above_point[2] - down_point[2])
    stop_point = above_point + fraction * (down_point - above_point)
    return (
        PlatePoints3D(np.vstack([points[:first_down], stop_point])),
        PlateDirections3D(
            np.vstack([tangents[:first_down], tangents[first_down - 1]])
        ),
    )


def _handle_toward_base(
    point: NDArray[np.float64], tangent: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Returns the unit handle direction perpendicular to the unit tangent.

    None when the level direction to the robot z axis lies along the tangent
    or the point is on the robot z axis.
    """
    from_hoop_center_toward_base = np.array([-point[0], -point[1], 0.0])
    handle = (
        from_hoop_center_toward_base
        - float(from_hoop_center_toward_base @ tangent) * tangent
    )
    length = float(np.linalg.norm(handle))
    if length < _MIN_HANDLE_LENGTH:
        return None
    return handle / length
