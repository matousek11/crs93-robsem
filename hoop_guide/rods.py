"""Rod centerline models, in the coordinate frame of the base plate.

A model consists of straight and circular-arc segments ordered from the
free tip toward the plate. The model and its unit tangents can be sampled
without a connected robot or camera.
"""

import dataclasses
import enum
import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hoop_guide.geometry import plate_to_robot
from hoop_guide.geometry import transform_directions
from hoop_guide.geometry import transform_points
from hoop_guide.plate import PlatePose
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D

STOP_HEIGHT = 0.040
"""Height of the hoop center above the plate at the end of the path [m]."""

ROD_A_NOMINAL_HEIGHT = 0.200
"""Nominal A-rod tip height [m], from the ROB1-2025-hw drawing.

The drawing dimensions the rod from its mounting foot. Verify the actual
height relative to the plate top with the assembled part before a real run.
"""


class Puzzle(enum.Enum):
    """Puzzle variants from the easiest to the hardest, see README."""

    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"
    D = "D"


class RodSegment(Protocol):
    """Continuous centerline segment parameterized by travel distance."""

    @property
    def length(self) -> float:
        """Length of the segment [m]."""

    def point_at(self, distance: float) -> NDArray[np.float64]:
        """Returns a 3D centerline point at a distance from segment start."""

    def tangent_at(self, distance: float) -> NDArray[np.float64]:
        """Returns a unit vector in the direction of travel."""


@dataclasses.dataclass(frozen=True)
class LineSegment:
    """Straight rod centerline segment between two plate-frame points."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not np.isfinite([*self.start, *self.end]).all() or self.length == 0:
            raise ValueError("Line segment needs distinct finite endpoints.")

    @property
    def length(self) -> float:
        """Length in meters."""
        return float(np.linalg.norm(np.subtract(self.end, self.start)))

    def point_at(self, distance: float) -> NDArray[np.float64]:
        """Returns the interpolated point at the given distance in meters."""
        _check_distance(distance, self.length)
        start = np.asarray(self.start, dtype=np.float64)
        return start + distance * self.tangent_at(distance)

    def tangent_at(self, distance: float) -> NDArray[np.float64]:
        """Returns the unit tangent from start to end."""
        _check_distance(distance, self.length)
        return np.subtract(self.end, self.start) / self.length


@dataclasses.dataclass(frozen=True)
class ArcSegment:
    """Circular 3D arc specified by center, radial start, axis and sweep.

    Axis is a unit vector perpendicular to the start radius. A positive
    sweep follows the right-hand rule around this axis.
    """

    center: tuple[float, float, float]
    start_radius: tuple[float, float, float]
    axis: tuple[float, float, float]
    sweep: float

    def __post_init__(self) -> None:
        if not np.isfinite(
            [*self.center, *self.start_radius, *self.axis, self.sweep]
        ).all():
            raise ValueError("Arc parameters must be finite.")
        if self.radius == 0 or self.sweep == 0:
            raise ValueError("Arc radius and sweep must be nonzero.")
        if not np.isclose(np.linalg.norm(self.axis), 1.0, atol=1e-8):
            raise ValueError("Arc axis must have unit length.")
        if not np.isclose(
            np.dot(self.axis, self.start_radius), 0.0, atol=1e-8
        ):
            raise ValueError("Arc axis must be perpendicular to radius.")

    @property
    def radius(self) -> float:
        """Arc radius [m]."""
        return float(np.linalg.norm(self.start_radius))

    @property
    def length(self) -> float:
        """Arc length [m]."""
        return self.radius * abs(self.sweep)

    def _radial_at(self, distance: float) -> NDArray[np.float64]:
        _check_distance(distance, self.length)
        angle = distance / self.radius * math.copysign(1.0, self.sweep)
        radial = np.asarray(self.start_radius, dtype=np.float64)
        axis = np.asarray(self.axis, dtype=np.float64)
        return (
            radial * math.cos(angle)
            + np.cross(axis, radial) * math.sin(angle)
        )

    def point_at(self, distance: float) -> NDArray[np.float64]:
        """Returns a point on the circular arc."""
        center = np.asarray(self.center, dtype=np.float64)
        return center + self._radial_at(distance)

    def tangent_at(self, distance: float) -> NDArray[np.float64]:
        """Returns unit tangent in the direction of the arc sweep."""
        axis = np.asarray(self.axis, dtype=np.float64)
        tangent = np.cross(axis, self._radial_at(distance)) / self.radius
        return tangent * math.copysign(1.0, self.sweep)


def _check_distance(distance: float, length: float) -> None:
    if not math.isfinite(distance) or not 0.0 <= distance <= length:
        raise ValueError("Distance must be finite and within segment length.")


@dataclasses.dataclass(frozen=True)
class RodModel:
    """Rod path as contiguous segments, from its free tip to the stop point.

    The path starts at the free end. At a sharp corner, the tangent in the
    sampled result is that of the incoming segment; an instantaneous change
    of hoop orientation is not safe and must be handled by the planner.
    """

    segments: tuple[RodSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("Rod model needs at least one segment.")
        for previous, following in zip(self.segments, self.segments[1:]):
            if not np.allclose(
                previous.point_at(previous.length), following.point_at(0.0),
                atol=1e-9, rtol=0.0,
            ):
                raise ValueError("Adjacent rod segments must be continuous.")

    def sample(self, step: float) -> tuple[Points3D, Directions3D]:
        """Samples each segment with spacing no larger than step [m].

        Both endpoints and all segment boundaries are included only once.
        """
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Sampling step must be finite and positive.")
        point_parts: list[NDArray[np.float64]] = []
        direction_parts: list[NDArray[np.float64]] = []
        for index, segment in enumerate(self.segments):
            distances = np.linspace(
                0.0, segment.length, math.ceil(segment.length / step) + 1
            )
            if index:
                distances = distances[1:]
            point_parts.append(
                np.stack([segment.point_at(d) for d in distances])
            )
            direction_parts.append(
                np.stack([segment.tangent_at(d) for d in distances])
            )
        return (
            Points3D(np.concatenate(point_parts)),
            Directions3D(np.concatenate(direction_parts)),
        )


def model_for_puzzle(puzzle: Puzzle) -> RodModel:
    """Returns the CAD-derived centerline in the local plate frame.

    The puzzle letters B, C, D correspond to the like-named models in
    ROB1-2025-hw-main/puzzles/*.step and drawings/puzzles.pdf, pages 2-4.
    Coordinates below are in meters; they were checked against the solid
    model cross-sections. All paths run from the free tip toward the plate.

    The nominal CAD mounting origin is (0, 0, 0) on the plate centerline.
    The height of the mounted physical rod relative to the plate top, the
    actual rod-to-plate yaw and the final safe clearance require calibration.
    These models describe centerlines, NOT collision-free hoop trajectories.

    A1 and A2 share a rod shape. A2 table motion must be supplied through
    the time-dependent PlatePose, not encoded in a different rod model.
    """
    if puzzle in (Puzzle.A1, Puzzle.A2):
        return RodModel(
            (
                LineSegment(
                    (0.0, 0.0, ROD_A_NOMINAL_HEIGHT),
                    (0.0, 0.0, STOP_HEIGHT),
                ),
            )
        )
    if puzzle is Puzzle.B:
        # B.step: tip (70, 0, 200) mm; diagonal connects z=150 to 80 mm.
        return RodModel(
            (
                LineSegment((0.070, 0.0, 0.200), (0.070, 0.0, 0.150)),
                LineSegment((0.070, 0.0, 0.150), (0.0, 0.0, 0.080)),
                LineSegment((0.0, 0.0, 0.080), (0.0, 0.0, STOP_HEIGHT)),
            )
        )
    if puzzle is Puzzle.C:
        # C.step: two 45-degree offsets, in the x-z and y-z planes.
        return RodModel(
            (
                LineSegment((-0.050, -0.050, 0.200), (-0.050, -0.050, 0.150)),
                LineSegment((-0.050, -0.050, 0.150), (-0.050, 0.0, 0.100)),
                LineSegment((-0.050, 0.0, 0.100), (0.0, 0.0, 0.050)),
                LineSegment((0.0, 0.0, 0.050), (0.0, 0.0, STOP_HEIGHT)),
            )
        )
    if puzzle is Puzzle.D:
        # D.step: horizontal tip at z=155 mm; R50 quarter circle centered
        # at (35, 0, 105) mm; short bottom 45-degree offset.
        # At the 40 mm stop height the centerline is on the lower diagonal:
        # (-10, 0, 40) mm, not at the mounting axis (0, 0, 40) mm.
        return RodModel(
            (
                LineSegment((0.085, 0.0, 0.155), (0.035, 0.0, 0.155)),
                ArcSegment(
                    center=(0.035, 0.0, 0.105),
                    start_radius=(0.0, 0.0, 0.050),
                    axis=(0.0, 1.0, 0.0),
                    sweep=-math.pi / 2,
                ),
                LineSegment((-0.015, 0.0, 0.105), (-0.015, 0.0, 0.045)),
                LineSegment(
                    (-0.015, 0.0, 0.045), (-0.010, 0.0, STOP_HEIGHT)
                ),
            )
        )
    raise NotImplementedError(f"Unknown puzzle: {puzzle!r}")


def rod_centerline(
    puzzle: Puzzle, plate: PlatePose, plate_height: float, step: float
) -> tuple[Points3D, Directions3D]:
    """Samples the rod centerline in the robot frame.

    The points go from the free end of the rod to STOP_HEIGHT above the
    plate, and represent the nominal path of the hoop center. This is only
    centerline geometry, not a collision-checked robot trajectory.

    Args:
        puzzle: Which rod.
        plate: Pose of the plate relative to the robot.
        plate_height: Height of plate top above the robot origin [m].
        step: Maximum arc-length spacing of sampled points [m].

    Returns:
        Points and unit tangents, each shape (N, 3), in the robot frame.
    """
    plate_points, plate_tangents = model_for_puzzle(puzzle).sample(step)
    transform = plate_to_robot(plate, plate_height)
    return (
        transform_points(plate_points, transform),
        transform_directions(plate_tangents, transform),
    )
