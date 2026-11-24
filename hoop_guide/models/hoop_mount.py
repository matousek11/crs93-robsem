"""How the hoop is mounted on the robot flange."""

import dataclasses

import numpy as np

from hoop_guide.types import Pose

_POSE_SHAPE = (4, 4)

GRIP_TO_HOOP_CENTER = 0.135
"""Distance from the gripped hole of the handle to the hoop center [m]."""

# TODO: measure in the lab, the gripper length is not in the ctu_crs model.
FLANGE_TO_GRIP = 0.0
"""Distance from the flange face to the gripped hole of the handle [m]."""

HOOP_THICKNESS = 0.0066
"""Size of the hoop along its axis, which is the flange z axis [m]."""

ADAPTER = 0.01

# Assumes the hoop center lies on the flange z axis and the hoop frame is not
# rotated against the flange frame. TODO: verify both in the lab.
FLANGE_TO_HOOP = Pose(
    np.array(
        [
            [1.0, 0.0, 0.0, -GRIP_TO_HOOP_CENTER],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, (ADAPTER + (HOOP_THICKNESS / 2))],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
)
"""Pose of the hoop in the flange frame, see HoopMount."""


@dataclasses.dataclass(frozen=True)
class HoopMount:
    """Fixed transform between the robot flange and the hoop.

    Attributes:
        flange_to_hoop: Pose of the hoop in the flange frame. The origin is
            the hoop center, so a point p in the hoop frame lies at
            flange_to_hoop @ p in the flange frame.

    Raises:
        ValueError: flange_to_hoop is not shaped (4, 4), is not finite or
            cannot be inverted.
    """

    flange_to_hoop: Pose

    def __post_init__(self) -> None:
        """Validates the transform."""
        if self.flange_to_hoop.shape != _POSE_SHAPE:
            raise ValueError("Flange to hoop transform must have shape (4, 4).")
        if not np.isfinite(self.flange_to_hoop).all():
            raise ValueError("Flange to hoop transform must be finite.")
        if np.linalg.matrix_rank(self.flange_to_hoop) < _POSE_SHAPE[0]:
            raise ValueError("Flange to hoop transform must be invertible.")

    def to_hoop_pose(self, flange_pose: Pose) -> Pose:
        """Returns the hoop pose in the robot frame for a flange pose."""
        return Pose(flange_pose @ self.flange_to_hoop)

    def to_flange_pose(self, hoop_pose: Pose) -> Pose:
        """Returns the flange pose in the robot frame that puts the hoop there.

        This is the pose to pass to the inverse kinematics.
        """
        return Pose(hoop_pose @ np.linalg.inv(self.flange_to_hoop))
