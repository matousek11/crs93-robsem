import math

import numpy as np
import pytest

from hoop_guide.rods import Puzzle
from hoop_guide.rods import rod_centerline

STEP = 0.01  # [m]
DOWN = (0.0, 0.0, -1.0)
# Second segment of rod B, from (0.07, 0, 0.15) to (0, 0, 0.08).
ROD_B_SLOPE = (-math.sqrt(0.5), 0.0, -math.sqrt(0.5))


@pytest.mark.parametrize(
    ("puzzle", "top_height"),
    [
        (Puzzle.A1, 0.2),
        (Puzzle.A2, 0.2),
        (Puzzle.B, 0.2),
        (Puzzle.C, 0.2),
        (Puzzle.D, 0.155),
    ],
    ids=["A1", "A2", "B", "C", "D"],
)
def test_rod_runs_from_its_free_end_to_the_plate_center(puzzle, top_height):
    points, _ = rod_centerline(puzzle, STEP)

    assert points[0][2] == pytest.approx(top_height)
    np.testing.assert_allclose(points[-1], (0.0, 0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C, Puzzle.D], ids=lambda p: p.value
)
def test_neighbouring_points_are_at_most_one_step_apart(puzzle):
    points, _ = rod_centerline(puzzle, STEP)

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    assert distances.max() <= STEP + 1e-12


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C], ids=lambda p: p.value
)
def test_tangents_are_unit_vectors_toward_the_plate(puzzle):
    _, tangents = rod_centerline(puzzle, STEP)

    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.all(tangents[:, 2] < 0.0)


def test_tangent_in_a_bend_follows_the_outgoing_segment():
    points, tangents = rod_centerline(Puzzle.B, STEP)

    bend_index = int(
        np.argmin(np.linalg.norm(points - (0.07, 0, 0.15), axis=1))
    )

    np.testing.assert_allclose(tangents[bend_index - 1], DOWN, atol=1e-12)
    np.testing.assert_allclose(tangents[bend_index], ROD_B_SLOPE, atol=1e-12)


def test_rod_d_has_horizontal_tip_and_a_quarter_circle():
    points, tangents = rod_centerline(Puzzle.D, STEP)
    np.testing.assert_allclose(points[0], (0.085, 0.0, 0.155), atol=1e-12)
    np.testing.assert_allclose(tangents[0], (-1.0, 0.0, 0.0), atol=1e-12)

    center = np.array((0.035, 0.0, 0.105))
    arc_points = points[(points[:, 0] <= 0.035) & (points[:, 2] >= 0.105)]
    radii = np.linalg.norm(arc_points - center, axis=1)
    assert np.max(np.abs(radii - 0.050)) < 0.00005


def test_rod_d_tangents_turn_smoothly_from_horizontal_to_down():
    points, tangents = rod_centerline(Puzzle.D, STEP)
    midpoint = np.array(
        (0.035 - 0.050 / math.sqrt(2), 0.0, 0.105 + 0.050 / math.sqrt(2))
    )
    midpoint_index = int(np.argmin(np.linalg.norm(points - midpoint, axis=1)))
    np.testing.assert_allclose(
        tangents[midpoint_index],
        (-math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
        atol=0.035,
    )
    assert np.max(tangents[:, 2]) <= 1e-12


@pytest.mark.parametrize(
    "step",
    [0.0, -0.01, math.nan, math.inf],
    ids=["zero", "negative", "nan", "inf"],
)
def test_invalid_step_raises(step):
    with pytest.raises(ValueError, match="step"):
        rod_centerline(Puzzle.A1, step)
