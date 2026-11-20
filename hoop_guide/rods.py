"""Rod centerlines sampled from the free tip to the plate mounting point.

The rod vertices are nominal dimensions from the ROB1-2025-hw drawings.
In the plate frame, x points right, y up and z above the plate, with the
mounting triangle pointing down in the top view. Verify the physical mounting
orientation before using these models for robot motion.
"""

import enum
import math

import numpy as np
from numpy.typing import NDArray

from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.plate import PlatePose
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D

STOP_HEIGHT = 0.040
"""Target height of the hoop center above the plate [m], used by planning."""


class Puzzle(enum.Enum):
    """Puzzle variants from the easiest to the hardest, see README."""

    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"
    D = "D"


_STRAIGHT_ROD = (
    (0.0, 0.0, 0.200),
    (0.0, 0.0, 0.000),
)

_ROD_VERTICES: dict[Puzzle, tuple[tuple[float, float, float], ...]] = {
    Puzzle.A1: _STRAIGHT_ROD,
    Puzzle.A2: _STRAIGHT_ROD,
    Puzzle.B: (
        (0.070, 0.0, 0.200),
        (0.070, 0.0, 0.150),
        (0.000, 0.0, 0.080),
        (0.000, 0.0, 0.000),
    ),
    Puzzle.C: (
        (-0.050, -0.050, 0.200),
        (-0.050, -0.050, 0.150),
        (-0.050, 0.000, 0.100),
        (0.000, 0.000, 0.050),
        (0.000, 0.000, 0.000),
    ),
}


def rod_centerline(
    puzzle: Puzzle, plate: PlatePose, plate_height: float, step: float
) -> tuple[Points3D, Directions3D]:
    """Samples the complete rod and its unit tangents in the robot frame.

    Vertices are first sampled in the plate frame, then transformed into the
    robot frame. Samples run from the free tip down to the plate (z = 0 in
    the plate frame). The stop height is applied later by the planner.
    Tangents follow the outgoing segment at bends and point toward the plate.

    Args:
        puzzle: Rod variant. D is not implemented yet.
        plate: Planar position and heading of the plate in the robot frame.
        plate_height: Height of the plate top above the robot base [m].
        step: Maximum distance between neighboring rod points [m].

    Returns:
        Points and unit tangent vectors, both shape (N, 3) in the robot
        frame. Point coordinates are in meters.

    Raises:
        ValueError: Sampling step or plate pose is invalid.
        NotImplementedError: The requested rod is not implemented.
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Sampling step must be finite and positive.")
    if not all(
        math.isfinite(value)
        for value in (plate.x, plate.y, plate.heading, plate_height)
    ):
        raise ValueError("Plate pose and height must be finite.")
    if puzzle not in _ROD_VERTICES:
        raise NotImplementedError(f"Rod model for {puzzle.value} is missing.")

    points_plate: list[NDArray[np.float64]] = []
    tangents_plate: list[NDArray[np.float64]] = []
    vertices = _ROD_VERTICES[puzzle]

    for start, end in zip(vertices, vertices[1:]):
        start_point = np.asarray(start, dtype=float)
        displacement = np.asarray(end, dtype=float) - start_point
        length = float(np.linalg.norm(displacement))
        tangent = displacement / length
        sample_count = math.ceil(length / step)
        for fraction in np.linspace(0.0, 1.0, sample_count, endpoint=False):
            points_plate.append(start_point + fraction * displacement)
            tangents_plate.append(tangent)

    points_plate.append(np.asarray(vertices[-1], dtype=float))
    tangents_plate.append(tangents_plate[-1])

    cosine, sine = get_cosine_and_sine(plate.heading)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([plate.x, plate.y, plate_height])

    return (
        Points3D(np.asarray(points_plate) @ rotation.T + translation),
        Directions3D(np.asarray(tangents_plate) @ rotation.T),
    )
