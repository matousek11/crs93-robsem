"""Rod models of the puzzles and sampling of their centerlines.

Rods are modelled in the plate frame (see plate.py) as a chain of straight
and arc segments of the centerline, taken from the drawings in ROB1-2025-hw.
"""

import enum

from hoop_guide.plate import PlatePose
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D

STOP_HEIGHT = 0.040
"""Height above the plate where the hoop center stops [m]."""


class Puzzle(enum.Enum):
    """Puzzle variants from the easiest to the hardest, see README."""

    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"
    D = "D"


def rod_centerline(
    puzzle: Puzzle, plate: PlatePose, plate_height: float, step: float
) -> tuple[Points3D, Directions3D]:
    """Samples the rod centerline in the robot frame.

    The points go from the free end of the rod down to STOP_HEIGHT above the
    plate, so they are the path of the hoop center.

    Args:
        puzzle: Which rod.
        plate: Pose of the plate.
        plate_height: Height of the plate top surface above the robot base
            [m].
        step: Distance between neighbouring points along the rod [m].

    Returns:
        Points, shape (N, 3), and unit tangents in them, shape (N, 3), both in
        the robot frame. Tangents point along the path, from the free end
        down.
    """
    raise NotImplementedError
