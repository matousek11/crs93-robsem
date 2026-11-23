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
    "puzzle", [Puzzle.A1, Puzzle.A2, Puzzle.B, Puzzle.C], ids=lambda p: p.value
)
def test_rod_runs_from_its_free_end_to_the_plate_center(puzzle):
    points, _ = rod_centerline(puzzle, STEP)

    assert points[0][2] == pytest.approx(0.2)
    np.testing.assert_allclose(points[-1], (0.0, 0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize(
    "puzzle", [Puzzle.A1, Puzzle.B, Puzzle.C], ids=lambda p: p.value
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


def test_rod_without_a_model_raises():
    with pytest.raises(NotImplementedError, match="D"):
        rod_centerline(Puzzle.D, STEP)


@pytest.mark.parametrize(
    "step",
    [0.0, -0.01, math.nan, math.inf],
    ids=["zero", "negative", "nan", "inf"],
)
def test_invalid_step_raises(step):
    with pytest.raises(ValueError, match="step"):
        rod_centerline(Puzzle.A1, step)
