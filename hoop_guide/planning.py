"""Offline target poses of a hoop following the nominal rod centerline.

No IK, collision checking, timing, robot hardware or camera is used here.
The result is geometry for further planning, NOT an executable trajectory.
"""

import dataclasses
import math

import numpy as np
from numpy.typing import NDArray

from hoop_guide.geometry import plate_to_robot
from hoop_guide.geometry import transform_directions
from hoop_guide.geometry import transform_points
from hoop_guide.plate import PlatePose
from hoop_guide.rods import model_for_puzzle
from hoop_guide.rods import Puzzle
from hoop_guide.rods import RodModel
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D
from hoop_guide.types import Pose
from hoop_guide.tool_setup import ToolCalibration

# BEST GUESS / OFFLINE PLANNING THRESHOLD, NOT A PHYSICAL SAFETY MARGIN.
# At true corners the rod tangent jumps, so a hoop cannot instantly rotate
# between the two normals. Even with a smooth reference pose the FULL RING
# AND ADAPTER geometry needs collision checks before robot execution.
CORNER_ANGLE_THRESHOLD_RAD = math.radians(2.0)
HEADING_EPS = 1e-9
XY_COORDINATE_COUNT = 2
# BEST GUESS / OFFLINE DIAGNOSTIC ONLY - not an actuator velocity limit.
MAX_PREVIEW_ROTATION_STEP_RAD = math.radians(30.0)


@dataclasses.dataclass(frozen=True)
class HoopPosePath:
    """Nominal center poses with diagnostics, NOT collision-checked.

    Attributes:
        poses_robot: Array of N rigid (4, 4) transforms, hoop frame H to R.
        points_robot: N center points in meters, ordered tip-to-plate.
        tangents_robot: N unit tangents pointing tip-to-plate.
        sharp_corner_indices: Indices where rod tangent jumps at a junction.
        fallback_heading_indices: Indices where the direction to the robot
            base cannot uniquely define the holder heading in hoop plane.
        large_rotation_step_indices: Indices where consecutive hoop poses
            differ by more than the offline preview orientation threshold.
    """

    poses_robot: NDArray[np.float64]
    points_robot: Points3D
    tangents_robot: Directions3D
    sharp_corner_indices: tuple[int, ...]
    fallback_heading_indices: tuple[int, ...]
    large_rotation_step_indices: tuple[int, ...]

    def flange_poses(
        self, calibration: ToolCalibration, *, preview_only: bool = False
    ) -> NDArray[np.float64]:
        """Computes nominal flange poses from hoop poses and mounting data.

        By default refuses unverified mounting and ambiguous/discontinuous
        geometry. ``preview_only=True`` permits OFFLINE inspection of such
        approximations, but does not make a collision-free robot trajectory.
        This method NEVER sends robot commands.
        """
        if not preview_only:
            if not calibration.is_verified:
                raise ValueError(
                    "Tool mounting is UNVERIFIED. Use preview_only=True "
                    "for offline calculations; calibrate before robot use."
                )
            if self.sharp_corner_indices:
                raise ValueError(
                    "Rod has SHARP CORNERS. A centerline pose sequence is "
                    "not a collision-free hoop trajectory."
                )
            if self.fallback_heading_indices:
                raise ValueError(
                    "Hoop holder heading is AMBIGUOUS at some points."
                )
            if self.large_rotation_step_indices:
                raise ValueError(
                    "Large ORIENTATION STEPS need a smooth motion plan."
                )
        return np.stack(
            [
                calibration.mount.flange_for_hoop(Pose(pose))
                for pose in self.poses_robot
            ]
        )


def _corner_indices(model: RodModel, step: float) -> tuple[int, ...]:
    """Locates true tangent discontinuities at sampled segment junctions."""
    boundaries: list[int] = []
    index = 0
    for previous, following in zip(model.segments, model.segments[1:]):
        index += math.ceil(previous.length / step)
        incoming = previous.tangent_at(previous.length)
        outgoing = following.tangent_at(0.0)
        cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        if math.acos(cosine) > CORNER_ANGLE_THRESHOLD_RAD:
            boundaries.append(index)
    return tuple(boundaries)


def _holder_y_axis(
    point: NDArray[np.float64],
    normal: NDArray[np.float64],
    base_xy: NDArray[np.float64],
    previous_y: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], bool]:
    """Projects horizontal direction toward base into the hoop plane.

    Local +Y_H points from ring center toward its holder in the Hoop.step
    orientation (CAD center lies at negative Y from the attachment). If this
    direction is parallel to the hoop normal, choose a stable fallback and
    report the ambiguity rather than claiming an unambiguous physical pose.
    """
    direction = np.array(
        [base_xy[0] - point[0], base_xy[1] - point[1], 0.0]
    )
    projected = direction - np.dot(direction, normal) * normal
    magnitude = float(np.linalg.norm(projected))
    if magnitude > HEADING_EPS:
        return projected / magnitude, False
    if previous_y is not None:
        projected = previous_y - np.dot(previous_y, normal) * normal
        magnitude = float(np.linalg.norm(projected))
        if magnitude > HEADING_EPS:
            return projected / magnitude, True
    # Deterministic fallback: choose the horizontal axis most perpendicular
    # to the normal; neither choice is a calibrated holder direction.
    candidates = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    axis = min(candidates, key=lambda v: abs(float(np.dot(v, normal))))
    projected = axis - np.dot(axis, normal) * normal
    return projected / np.linalg.norm(projected), True


def hoop_pose_path(
    puzzle: Puzzle,
    plate: PlatePose,
    plate_height: float,
    step: float,
    *,
    robot_base_xy: tuple[float, float] = (0.0, 0.0),
) -> HoopPosePath:
    """Samples nominal hoop poses with local +Z_H opposite travel tangent.

    All positions are in the CRS93 base frame R. The model follows the rod
    centerline from its free end toward the plate, at a fixed plate pose.
    +Z_H = -tangent; +Y_H points toward the robot-base XY reference after
    projection into the hoop plane; +X_H = +Y_H cross +Z_H.

    Args:
        puzzle: Rod geometry (A1/A2/B/C/D).
        plate: Known plate pose relative to R, NOT estimated here.
        plate_height: Top surface height relative to R [m], MUST be measured.
        step: Maximum sampling interval along the rod [m].
        robot_base_xy: Preferred holder direction target in robot XY [m].
            Defaults to R origin (0,0), a geometrical assumption, NOT a
            verified safe mounting direction.

    Returns:
        Center poses and explicit sharp-corner/heading diagnostics. The
        end at 40 mm center height is a nominal model, not a confirmed
        safe/acceptable final state. Neither approach nor exit is planned.
    """
    if len(robot_base_xy) != XY_COORDINATE_COUNT or not np.isfinite(robot_base_xy).all():
        raise ValueError("robot_base_xy must contain two finite coordinates.")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Sampling step must be finite and positive.")
    model = model_for_puzzle(puzzle)
    points_plate, tangents_plate = model.sample(step)
    plate_robot = plate_to_robot(plate, plate_height)
    points_robot = transform_points(points_plate, plate_robot)
    tangents_robot = transform_directions(tangents_plate, plate_robot)

    transforms: list[NDArray[np.float64]] = []
    ambiguous: list[int] = []
    large_steps: list[int] = []
    previous_y: NDArray[np.float64] | None = None
    target = np.asarray(robot_base_xy, dtype=np.float64)
    for index, (point, tangent) in enumerate(zip(points_robot, tangents_robot)):
        normal = -np.asarray(tangent, dtype=np.float64)
        holder_y, has_fallback = _holder_y_axis(
            point, normal, target, previous_y
        )
        if has_fallback:
            ambiguous.append(index)
        axis_x = np.cross(holder_y, normal)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(normal, axis_x)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.column_stack((axis_x, axis_y, normal))
        if transforms:
            relative = transforms[-1][:3, :3].T @ pose[:3, :3]
            cosine = float(np.clip((np.trace(relative) - 1) / 2, -1, 1))
            if math.acos(cosine) > MAX_PREVIEW_ROTATION_STEP_RAD:
                large_steps.append(index)
        pose[:3, 3] = point
        transforms.append(pose)
        previous_y = axis_y

    return HoopPosePath(
        poses_robot=np.stack(transforms),
        points_robot=points_robot,
        tangents_robot=tangents_robot,
        sharp_corner_indices=_corner_indices(model, step),
        fallback_heading_indices=tuple(ambiguous),
        large_rotation_step_indices=tuple(large_steps),
    )
