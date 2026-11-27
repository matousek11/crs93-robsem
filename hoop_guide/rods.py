"""Rod centerlines sampled from the free tip to the plate mounting point.

The rod vertices are nominal dimensions from the ROB1-2025-hw drawings.
In the plate frame, x points right, y up and z above the plate, with the
mounting triangle pointing down in the top view. Verify the physical mounting
orientation before using these models for robot motion.
"""

import enum
import itertools
import math

import numpy as np
from numpy.typing import NDArray

from hoop_guide.types import PlateDirections3D
from hoop_guide.types import PlatePoints3D

STOP_HEIGHT = 0.04
"""Target height of the hoop center above the plate [m], used by planning."""


class Puzzle(enum.Enum):
    """Puzzle variants from the easiest to the hardest, see README."""

    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"
    D = "D"


# Puzzle D: drawing dimensions, in the plate X/Z plane [m].
_D_BEND_RADIUS = 0.050
_D_TOP_STRAIGHT = 0.050
_D_UPPER_VERTICAL = 0.060
_D_LOWER_VERTICAL = 0.030
_D_DIAGONAL_OFFSET = 0.015
_D_ARC_SEGMENTS = 18  # Five-degree chords, at most 0.05 mm arc deviation.


def _rod_d_vertices() -> tuple[tuple[float, float, float], ...]:
    """Returns the 90-degree curved rod D from its free tip to the plate."""
    arc_center_x = _D_BEND_RADIUS - _D_DIAGONAL_OFFSET
    arc_center_z = _D_LOWER_VERTICAL + _D_DIAGONAL_OFFSET + _D_UPPER_VERTICAL
    arc_angles = np.linspace(math.pi / 2, math.pi, _D_ARC_SEGMENTS + 1)
    arc = tuple(
        (
            arc_center_x + _D_BEND_RADIUS * math.cos(angle),
            0.0,
            arc_center_z + _D_BEND_RADIUS * math.sin(angle),
        )
        for angle in arc_angles[1:]
    )
    return (
        (arc_center_x + _D_TOP_STRAIGHT, 0.0, arc_center_z + _D_BEND_RADIUS),
        (arc_center_x, 0.0, arc_center_z + _D_BEND_RADIUS),
        *arc,
        (-_D_DIAGONAL_OFFSET, 0.0, _D_LOWER_VERTICAL + _D_DIAGONAL_OFFSET),
        (0.0, 0.0, _D_LOWER_VERTICAL),
        (0.0, 0.0, 0.0),
    )


_STRAIGHT_ROD = (
    (0.0, 0.0, 0.200),
    (0.0, 0.0, 0.000),
)

"""Rod trajectories from the free tip toward the plate."""
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
    Puzzle.D: _rod_d_vertices(),
}


def rod_centerline(
    puzzle: Puzzle, step: float
) -> tuple[PlatePoints3D, PlateDirections3D]:
    """Samples the complete rod and its unit tangents in the plate frame.

    Samples run from the free tip down to the plate (z = 0). The stop height
    is applied later by the planner. Tangents follow the outgoing segment at
    bends and point toward the plate.

    Args:
        puzzle: Rod variant.
        step: Maximum distance between neighboring rod points [m].

    Returns:
        Points and unit tangent vectors, both shape (N, 3) in the plate
        frame. Point coordinates are in meters.

    Raises:
        ValueError: Sampling step is not finite and positive.
        NotImplementedError: The requested rod is not implemented.
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Sampling step must be finite and positive.")
    if puzzle not in _ROD_VERTICES:
        raise NotImplementedError(f"Rod model for {puzzle.value} is missing.")

    points_plate: list[NDArray[np.float64]] = []
    tangents_plate: list[NDArray[np.float64]] = []
    vertices = _ROD_VERTICES[puzzle]

    for start, end in itertools.pairwise(vertices):
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

    return (
        PlatePoints3D(np.asarray(points_plate)),
        PlateDirections3D(np.asarray(tangents_plate)),
    )
