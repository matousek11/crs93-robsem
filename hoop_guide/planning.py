"""Moves of the hoop to a target pose and the hoop path along the rod."""

import dataclasses
import functools
import math
from collections.abc import Iterator
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hoop_guide import control
from hoop_guide.collision import CollisionCheck
from hoop_guide.estimation import TableMotion
from hoop_guide.hardware.interfaces import JointMotion
from hoop_guide.models.hoop_mount import FLANGE_TO_HOOP
from hoop_guide.models.hoop_mount import HoopMount
from hoop_guide.plate import plate_to_robot_directions
from hoop_guide.plate import plate_to_robot_points
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
from hoop_guide.rods import rod_centerline
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
from hoop_guide.types import PlateDirections3D
from hoop_guide.types import PlatePoints3D
from hoop_guide.types import Pose

PATH_STEP = 0.015
"""Distance between neighbouring points of the hoop path [m].

Tangents are averaged over three neighbouring points, so this also sets how
long the hoop takes to turn in a bend of the rod: two steps.
"""

SYNC_DISTANCE = 0.10
"""Distance of the synchronization point from the free end of the rod, along
the first segment of the rod [m]."""

MAX_JOINT_STEP = math.radians(45.0)
"""Largest change of one joint in one step. A larger one means the joint
configuration flips and the arm would swing with the hoop on the rod."""

HANDLE_ANGLE_STEP = math.radians(10.0)
"""Spacing of the handle angles tried when the previous one does not go
[rad]."""

# Decided on 2026-11-25: rod C at the lab table position went from no
# heading out of 8 with no tilt to 7 with tilts up to 10 deg in the start
# search, 20 deg added none. A tilt of 10 deg costs about 1.4 mm of the
# about 23 mm the rod has in the hoop.
MAX_TILT = math.radians(10.0)
"""Largest tilt of the hoop axis away from the rod [rad]."""

TILT_DIRECTION_STEP = math.radians(45.0)
"""Spacing of the directions the hoop axis may tilt to [rad]."""

TILT_CHANGE_COST = math.radians(15.0)
"""How much a change of the tilt or its direction counts against keeping
it, in the order the turns are tried, as a turn of the handle by this angle
[rad]."""

WRIST_BEND_JOINT = 4
"""Index of the joint that bends the wrist, joint 5 of the CRS robots."""

# Seen in the lab on 2026-11-24 on rod B: joint 5 went from -65 to -20 deg,
# joints 4 and 6 turned 15 deg a step, then no handle angle went and on the
# way back joint 4 ran to the end of its range with the hoop on the rod.
MIN_WRIST_BEND = math.radians(10.0)
"""Smallest angle of the wrist bend joint from zero [rad].

At zero the axes of joints 4 and 6 line up, the wrist singularity. Near it
those two joints turn far for a small move of the hoop.
"""

ARRIVAL_TOLERANCE = 0.02
"""Two estimates of the arrival of a step closer than this agree [s]."""

MAX_ARRIVAL_ESTIMATES = 4
"""Most estimates of the arrival of one step, the last one is used."""

_MIN_HANDLE_LENGTH = 1e-9
"""Below this the direction to the robot base is along the hoop axis."""


class NoJointConfigError(RuntimeError):
    """Raised when no joint configuration is left to reach a pose."""


@dataclasses.dataclass(frozen=True)
class HoopTurn:
    """How the hoop is turned from its pose on the rod.

    With no turn the hoop axis is the rod tangent and the handle points
    toward the robot base, see RodPlanner.

    Attributes:
        handle_angle: Turn about the rod, counterclockwise looking against
            the hoop z axis [rad].
        tilt: Tilt of the hoop axis away from the rod, 0 or MAX_TILT [rad].
        tilt_direction: Where the hoop z axis tilts to, from the handle
            counterclockwise looking against the hoop z axis [rad].
    """

    handle_angle: float = 0.0
    tilt: float = 0.0
    tilt_direction: float = 0.0


class Kinematics(Protocol):
    """Joint configurations and speeds of the robot."""

    @property
    def joint_speeds(self) -> JointSpeeds:
        """Speeds of the joints in a move [rad/s]."""

    @property
    def joint_limits(self) -> tuple[JointConfig, JointConfig]:
        """Lowest and highest angle of every joint [rad]."""

    def choose_joint_config(
        self, hoop_pose: Pose, joint_angles: JointConfig
    ) -> JointConfig:
        """Returns the reaching configuration nearest to the joint angles.

        Raises:
            NoJointConfigError: The pose is out of reach or only beyond the
                joint limits.
        """


class JointFlipError(RuntimeError):
    """Raised when a step changes a joint by more than MAX_JOINT_STEP."""


class WristSingularityError(NoJointConfigError):
    """Raised when the wrist bend joint is closer than MIN_WRIST_BEND to 0."""


def get_wrist_margin(joint_angles: JointConfig) -> float:
    """Returns how far the wrist bend is from being too straight [rad].

    The angle of the wrist bend joint from zero minus MIN_WRIST_BEND,
    negative when the wrist is too near its singularity.
    """
    return abs(float(joint_angles[WRIST_BEND_JOINT])) - MIN_WRIST_BEND


def check_wrist_bend(joint_angles: JointConfig) -> None:
    """Checks that the wrist is not near its singularity.

    Raises:
        WristSingularityError: The wrist bend joint is closer than
            MIN_WRIST_BEND to zero.
    """
    if get_wrist_margin(joint_angles) < 0.0:
        bend = math.degrees(float(joint_angles[WRIST_BEND_JOINT]))
        raise WristSingularityError(
            f"Joint {WRIST_BEND_JOINT + 1} would be at {bend:.0f} deg, too"
            " near the wrist singularity."
        )


def check_joint_step(start: JointConfig, target: JointConfig) -> None:
    """Checks that no joint changes by more than MAX_JOINT_STEP.

    Args:
        start: Joint angles before the step [rad].
        target: Joint angles after the step [rad].

    Raises:
        JointFlipError: A joint changes by more, the message names it.
    """
    changes = np.abs(target - start)
    joint_index = int(np.argmax(changes))
    if float(changes[joint_index]) > MAX_JOINT_STEP:
        raise JointFlipError(
            f"Joint {joint_index + 1} would turn by"
            f" {math.degrees(float(changes[joint_index])):.0f} deg in one"
            " step."
        )


class Planning:
    """Chooses the joint configuration for a hoop pose and moves there.

    Args:
        robot: Robot to move.
    """

    def __init__(self, robot: JointMotion) -> None:
        self.robot: JointMotion = robot
        self.hoop_mount: HoopMount = HoopMount(FLANGE_TO_HOOP)
        self.collision_check: CollisionCheck = CollisionCheck(robot)

    def move_to_ik(self, hoop_pose: Pose) -> None:
        """Starts a move that puts the hoop into the pose.

        Of the joint configurations within the joint limits that keep the
        robot and the hoop above the table (CollisionCheck), the one with the
        smallest mean absolute change of the joint angles is used. Returns at
        once, before the robot arrives.

        Args:
            hoop_pose: Target pose of the hoop in the robot frame.

        Raises:
            NoJointConfigError: The pose is out of reach, or only beyond the
                joint limits or below the table. Nothing is sent to the
                robot.
        """
        picked_config = self.choose_joint_config(
            hoop_pose, self.robot.get_joint_angles()
        )

        # 5. move robot to new configurations
        self.robot.set_joint_angles(picked_config)

    @property
    def joint_speeds(self) -> JointSpeeds:
        """Speeds of the joints in a move [rad/s]."""
        return self.robot.joint_speeds

    @property
    def joint_limits(self) -> tuple[JointConfig, JointConfig]:
        """Lowest and highest angle of every joint [rad]."""
        return self.robot.joint_limits

    def get_joint_angles(self) -> JointConfig:
        """Returns the joint angles measured by the controller."""
        return self.robot.get_joint_angles()

    def choose_joint_config(
        self, hoop_pose: Pose, joint_angles: JointConfig
    ) -> JointConfig:
        """Returns the reaching configuration nearest to the joint angles.

        Of the joint configurations within the joint limits that keep the
        robot and the hoop above the table (CollisionCheck), the one with the
        smallest mean absolute change from joint_angles.

        Args:
            hoop_pose: Target pose of the hoop in the robot frame.
            joint_angles: Joint angles the move starts from [rad].

        Raises:
            NoJointConfigError: The pose is out of reach, or only beyond the
                joint limits or below the table.
        """
        # 1. get available configurations from crs library
        flange_pose = self.hoop_mount.to_flange_pose(hoop_pose)
        possible_configs = self.robot.get_ik_options(flange_pose)

        # 2. cut-off configurations beyond the joint limits and below the
        # table, see CollisionCheck
        # TODO: cut-off configurations that hit the rod too
        possible_configs = [
            config
            for config in possible_configs
            if self.robot.is_in_limits(config)
            and self.collision_check.is_free(config, hoop_pose)
        ]

        # 3. stop when no config remained
        if len(possible_configs) == 0:
            raise NoJointConfigError(
                "No joint configuration reaches the hoop pose."
            )

        # 4. find the configuration with the smallest average change per joint
        curr_config = joint_angles
        avg_change = []
        for possible_config in possible_configs:
            new_conf_change_sum = 0.0
            for new_joint, curr_joint in zip(
                possible_config, curr_config, strict=True
            ):
                new_conf_change_sum += abs(new_joint - curr_joint)

            avg_change.append(new_conf_change_sum / len(curr_config))

        return possible_configs[avg_change.index(min(avg_change))]

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
    z axis with its part along the z axis removed. For rod D the handle
    points along the plate y axis, avoiding the undefined base direction
    at the horizontal free tip.

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
        self._puzzle = puzzle
        rod_points, rod_tangents = rod_centerline(puzzle, PATH_STEP)
        points, tangents = _cut_at_stop_height(
            rod_points, _average_neighbours(rod_tangents)
        )
        sync_point = points[0] - SYNC_DISTANCE * tangents[0]
        self._points = PlatePoints3D(np.vstack([sync_point, points]))
        self._tangents = PlateDirections3D(np.vstack([tangents[0], tangents]))

    @property
    def point_count(self) -> int:
        """Number of path points, the synchronization point included."""
        return len(self._points)

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
        return self._get_hoop_poses(target_time, start_index, self.point_count)

    def get_hoop_pose(
        self, target_time: float, index: int, hoop_turn: HoopTurn | None = None
    ) -> Pose:
        """Returns the hoop pose of one path point, for a moment.

        With no hoop_turn the same as update(target_time, index)[0], without
        computing the rest of the path.

        Args:
            target_time: time.monotonic at which the pose holds [s].
            index: Path point, 0 is the synchronization point.
            hoop_turn: How the hoop is turned from its pose on the rod, the
                hoop center stays on the rod.

        Raises:
            ValueError: As update.
        """
        hoop_pose = self._get_hoop_poses(target_time, index, index + 1)[0]
        if hoop_turn is None:
            return hoop_pose
        return turn_hoop(hoop_pose, hoop_turn)

    def _get_hoop_poses(
        self, target_time: float, start_index: int, stop_index: int
    ) -> tuple[Pose, ...]:
        """Returns the hoop poses of the points start_index to stop_index - 1.

        See update.
        """
        if not 0 <= start_index < self.point_count:
            raise ValueError(
                f"Start index must be between 0 and {self.point_count - 1},"
                f" got {start_index}."
            )
        plate = self._table_motion.predict(target_time)
        points = plate_to_robot_points(
            PlatePoints3D(self._points[start_index:stop_index]),
            plate,
            self._plate_height,
        )
        tangents = plate_to_robot_directions(
            PlateDirections3D(self._tangents[start_index:stop_index]), plate
        )
        hoop_poses = []
        plate_y = (
            np.array([-math.sin(plate.heading), math.cos(plate.heading), 0.0])
            if self._puzzle is Puzzle.D
            else None
        )
        for offset, (point, tangent) in enumerate(
            zip(points, tangents, strict=True)
        ):
            handle = (
                plate_y
                if plate_y is not None
                else _handle_toward_base(point, tangent)
            )
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


def turn_hoop(hoop_pose: Pose, hoop_turn: HoopTurn) -> Pose:
    """Returns the hoop pose turned about the hoop center.

    The handle is turned about the hoop z axis first, then the hoop is
    tilted toward tilt_direction, measured from the turned handle.
    """
    rotation = hoop_pose[:3, :3] @ _rotation_about_z(hoop_turn.handle_angle)
    if hoop_turn.tilt:
        tilt_axis = rotation @ np.array(
            [
                -math.sin(hoop_turn.tilt_direction),
                math.cos(hoop_turn.tilt_direction),
                0.0,
            ]
        )
        rotation = _rotation_about(tilt_axis, hoop_turn.tilt) @ rotation
    turned_pose = hoop_pose.copy()
    turned_pose[:3, :3] = rotation
    return Pose(turned_pose)


@dataclasses.dataclass(frozen=True, eq=False)
class StepPlan:
    """One step of the hoop to a path point, planned for its arrival.

    Attributes:
        arrival_time: time.monotonic when the robot arrives [s].
        hoop_turn: How the hoop is turned in the step, see HoopTurn.
        hoop_pose: Pose of the hoop to send, in the robot frame.
        joint_angles: Joint angles the robot reaches the pose with [rad].
    """

    arrival_time: float
    hoop_turn: HoopTurn
    hoop_pose: Pose
    joint_angles: JointConfig


def plan_step(  # noqa: PLR0913
    planner: RodPlanner,
    kinematics: Kinematics,
    index: int,
    sent_time: float,
    joint_angles: JointConfig,
    *,
    hoop_turn: HoopTurn,
    can_flip: bool = False,
) -> StepPlan:
    """Plans a step to a path point, sent now and aimed at the arrival.

    How long the move takes depends on where it goes, and where it goes on
    when it arrives. Starting from MOVE_OVERHEAD ahead, the target at the
    estimated arrival gives the move time and from it a new arrival, until
    two estimates agree within ARRIVAL_TOLERANCE or MAX_ARRIVAL_ESTIMATES
    are made. Short steps agree at the second estimate.

    One estimate was not enough: in the lab on 2026-11-25 the steps in the
    bend of rod B took 2 to 3 s, the plate turned 0.5 rad meanwhile, and the
    run fell 3.2 s behind the start search.

    Args:
        planner: Path of the hoop.
        kinematics: Joint configurations and speeds of the robot.
        index: Path point to move to.
        sent_time: time.monotonic when the step is sent [s].
        joint_angles: Joint angles before the step [rad].
        hoop_turn: How the hoop was turned in the previous step.
        can_flip: True when a joint may change by more than MAX_JOINT_STEP,
            for the move from home to the path.

    Raises:
        NoJointConfigError: No hoop turn reaches the point.
        ValueError: The planner cannot give a pose for the point.
    """
    arrival_time = sent_time + control.MOVE_OVERHEAD
    for _ in range(MAX_ARRIVAL_ESTIMATES):
        chosen_turn, target_angles = choose_hoop_turn(
            planner,
            kinematics,
            index,
            arrival_time,
            joint_angles,
            previous_turn=hoop_turn,
            can_flip=can_flip,
        )
        move_time = control.get_move_time(
            joint_angles, target_angles, kinematics.joint_speeds
        )
        next_arrival_time = sent_time + control.get_lead_time(move_time)
        if abs(next_arrival_time - arrival_time) < ARRIVAL_TOLERANCE:
            break
        arrival_time = next_arrival_time
    return StepPlan(
        arrival_time=arrival_time,
        hoop_turn=chosen_turn,
        hoop_pose=planner.get_hoop_pose(arrival_time, index, chosen_turn),
        joint_angles=target_angles,
    )


def choose_hoop_turn(  # noqa: PLR0913
    planner: RodPlanner,
    kinematics: Kinematics,
    index: int,
    time: float,
    joint_angles: JointConfig,
    *,
    previous_turn: HoopTurn,
    can_flip: bool = False,
) -> tuple[HoopTurn, JointConfig]:
    """Returns the hoop turn nearest to the previous one that goes.

    The previous turn is tried first. Then the handle angles HANDLE_ANGLE_STEP
    apart on both sides, the tilts of 0 and MAX_TILT and the tilt directions
    TILT_DIRECTION_STEP apart, ordered by how far they are from the previous
    turn: the change of the handle angle plus the change of the tilt, plus
    TILT_CHANGE_COST when the tilt or its direction changes. A turn goes
    when a joint configuration within the limits and above the table
    reaches the pose, the wrist is not near its singularity (MIN_WRIST_BEND)
    and no joint changes by more than MAX_JOINT_STEP from joint_angles.

    Args:
        planner: Path of the hoop.
        kinematics: Joint configurations of the robot.
        index: Path point.
        time: time.monotonic of the pose [s].
        joint_angles: Joint angles the robot moves from [rad].
        previous_turn: How the hoop was turned in the previous step.
        can_flip: True when the joints may change by more than
            MAX_JOINT_STEP.

    Returns:
        The hoop turn and the joint angles for it [rad].

    Raises:
        NoJointConfigError: No hoop turn goes, the message gives the reason
            of the last one tried.
        ValueError: The planner cannot give a pose for the point.
    """
    hoop_pose = planner.get_hoop_pose(time, index)
    last_error: Exception | None = None
    for hoop_turn in _nearest_turns_first(previous_turn):
        try:
            target_angles = kinematics.choose_joint_config(
                turn_hoop(hoop_pose, hoop_turn), joint_angles
            )
            check_wrist_bend(target_angles)
            if not can_flip:
                check_joint_step(joint_angles, target_angles)
        except (NoJointConfigError, JointFlipError) as error:
            last_error = error
            continue
        return hoop_turn, target_angles
    raise NoJointConfigError(
        f"No hoop turn reaches path point {index}: {last_error}"
    )


def _nearest_turns_first(previous_turn: HoopTurn) -> Iterator[HoopTurn]:
    """Yields all hoop turns, nearest to previous_turn first."""
    is_tilted = previous_turn.tilt > 0.0
    for handle_steps, tilt, direction_steps in _turn_changes(is_tilted):
        tilt_direction = (
            previous_turn.tilt_direction if is_tilted else 0.0
        ) + direction_steps * TILT_DIRECTION_STEP
        yield HoopTurn(
            handle_angle=math.remainder(
                previous_turn.handle_angle + handle_steps * HANDLE_ANGLE_STEP,
                2 * math.pi,
            ),
            tilt=tilt,
            tilt_direction=math.remainder(tilt_direction, 2 * math.pi)
            if tilt
            else 0.0,
        )


@functools.cache
def _turn_changes(is_tilted: bool) -> tuple[tuple[int, float, int], ...]:
    """Returns the changes from a turn, the nearest first.

    Each change is the number of HANDLE_ANGLE_STEP to add to the handle
    angle, the new tilt and the number of TILT_DIRECTION_STEP to add to the
    tilt direction. Sorted by the cost described in choose_hoop_turn.
    """
    half_turn_steps = round(math.pi / HANDLE_ANGLE_STEP)
    handle_steps = range(-half_turn_steps + 1, half_turn_steps + 1)
    direction_count = round(2 * math.pi / TILT_DIRECTION_STEP)
    previous_tilt = MAX_TILT if is_tilted else 0.0
    tilts = [(0.0, 0)] + [
        (MAX_TILT, direction_steps)
        for direction_steps in range(direction_count)
    ]
    changes = []
    for tilt, direction_steps in tilts:
        is_same_tilt = tilt == previous_tilt and (
            tilt == 0.0 or direction_steps == 0
        )
        tilt_cost = (
            0.0
            if is_same_tilt
            else abs(tilt - previous_tilt) + TILT_CHANGE_COST
        )
        for steps in handle_steps:
            cost = abs(steps) * HANDLE_ANGLE_STEP + tilt_cost
            changes.append((cost, steps, tilt, direction_steps))
    changes.sort(key=lambda change: change[0])
    return tuple(
        (steps, tilt, direction_steps)
        for _, steps, tilt, direction_steps in changes
    )


def _rotation_about_z(angle: float) -> NDArray[np.float64]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def _rotation_about(
    axis: NDArray[np.float64], angle: float
) -> NDArray[np.float64]:
    """Returns the rotation by angle about the unit axis (Rodrigues)."""
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation: NDArray[np.float64] = (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * cross @ cross
    )
    return rotation


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
