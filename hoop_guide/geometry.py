"""Rigid transforms used for the plate, hoop, and robot flange.

Pose matrices map homogeneous coordinates from a local frame to a parent
frame. Distances are in meters, angles in radians.
"""

import dataclasses
import math

import numpy as np

from hoop_guide.plate import PlatePose
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D
from hoop_guide.types import Pose


def plate_to_robot(plate: PlatePose, plate_height: float) -> Pose:
    """Returns the plate-to-robot transform for a horizontal plate.

    The plate origin is at the center of its top face, with the local z
    axis pointing upward. Only a rotation about the robot z axis is allowed.

    Args:
        plate: Position (x, y) and orientation (theta) of the plate.
        plate_height: Height of the plate top above robot frame origin [m].

    Returns:
        The 4x4 plate-to-robot pose.

    Raises:
        ValueError: If a pose parameter is non-finite.
    """
    if not all(
        math.isfinite(value)
        for value in (plate.x, plate.y, plate.theta, plate_height)
    ):
        raise ValueError("Plate pose and height must be finite.")
    cosine = math.cos(plate.theta)
    sine = math.sin(plate.theta)
    return Pose(
        np.array(
            [
                [cosine, -sine, 0.0, plate.x],
                [sine, cosine, 0.0, plate.y],
                [0.0, 0.0, 1.0, plate_height],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )


def _checked_pose(transform: Pose) -> np.ndarray:
    """Checks that a pose is a proper rigid 4x4 transformation."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Pose must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("Pose must have homogeneous last row [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("Pose rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("Pose rotation must have determinant +1.")
    return matrix


def transform_points(points: Points3D, transform: Pose) -> Points3D:
    """Transforms Nx3 points using rotation and translation of a pose."""
    matrix = _checked_pose(transform)
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError("Points must be a finite array with shape (N, 3).")
    return Points3D(data @ matrix[:3, :3].T + matrix[:3, 3])


def transform_directions(
    directions: Directions3D, transform: Pose
) -> Directions3D:
    """Rotates Nx3 unit directions; translation does not affect them."""
    matrix = _checked_pose(transform)
    data = np.asarray(directions, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError("Directions must have finite shape (N, 3).")
    if not np.allclose(np.linalg.norm(data, axis=1), 1.0, atol=1e-8):
        raise ValueError("Directions must be unit vectors.")
    return Directions3D(data @ matrix[:3, :3].T)


def inverse_pose(transform: Pose) -> Pose:
    """Returns the inverse of a rigid transform without a general inverse."""
    matrix = _checked_pose(transform)
    rotation_transpose = matrix[:3, :3].T
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation_transpose
    inverse[:3, 3] = -rotation_transpose @ matrix[:3, 3]
    return Pose(inverse)


def flange_pose_for_hoop(hoop_in_robot: Pose, hoop_in_flange: Pose) -> Pose:
    """Returns robot-frame flange pose needed for a desired hoop pose.

    Args:
        hoop_in_robot: Desired pose of hoop-center frame in robot frame.
        hoop_in_flange: Fixed transform from robot flange to hoop-center
            frame, including the adapter. It must be measured from the real
            assembled tool before using the result on physical hardware.

    Returns:
        Required flange pose in robot frame, for use with robot IK.
    """
    robot_hoop = _checked_pose(hoop_in_robot)
    flange_hoop_inverse = inverse_pose(hoop_in_flange)
    return Pose(robot_hoop @ flange_hoop_inverse)


# Exact center of the ring opening in the ROB1-2025-hw Hoop.step CAD frame.
# STEP coordinates are millimeters. The ring lies in the CAD XY plane, with
# flat faces at z=0 and z=-6.6 mm. The ring center is midway between them.
HOOP_CAD_CENTER_M = (0.0, -0.135, -0.0033)


def hoop_cad_to_center() -> Pose:
    """Returns the fixed hoop-CAD-to-hoop-center pose, in meters.

    The center frame has the same orientation as Hoop.step: the opening is
    parallel to the CAD XY plane, and its normal is CAD +Z. This does not
    locate the hoop relative to the flange or the physical CRS93 adapter.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = HOOP_CAD_CENTER_M
    return Pose(transform)


@dataclasses.dataclass(frozen=True)
class AdapterMount:
    """Explicit mounting calibration of the adapter and hoop.

    Both transforms MUST be measured/verified for the assembled hardware:

    - flange_to_adapter maps adapter-frame coordinates into the frame used
      by CRS93 FK/IK (the actual flange/tool frame must also be verified).
    - adapter_to_hoop_cad maps the Hoop.step CAD frame into adapter frame.

    For the final ring-center frame, the CAD-derived -135 mm offset and
    -3.3 mm z-offset are applied automatically. No default identity poses
    are supplied, because an uncalibrated mounting must not look valid.
    """

    flange_to_adapter: Pose
    adapter_to_hoop_cad: Pose

    def __post_init__(self) -> None:
        # Copies prevent the caller from later mutating the array in place.
        object.__setattr__(
            self,
            "flange_to_adapter",
            Pose(_checked_pose(self.flange_to_adapter).copy()),
        )
        object.__setattr__(
            self,
            "adapter_to_hoop_cad",
            Pose(_checked_pose(self.adapter_to_hoop_cad).copy()),
        )

    def flange_to_hoop_center(self) -> Pose:
        """Returns the full flange-to-ring-center transform incl. adapter."""
        return Pose(
            self.flange_to_adapter
            @ self.adapter_to_hoop_cad
            @ hoop_cad_to_center()
        )

    def flange_for_hoop(self, desired_hoop_in_robot: Pose) -> Pose:
        """Returns required robot-frame flange pose for target ring pose."""
        return flange_pose_for_hoop(
            desired_hoop_in_robot, self.flange_to_hoop_center()
        )
