import math

import numpy as np
import pytest

from hoop_guide.plate import PlatePose
from hoop_guide.rods import Puzzle
from hoop_guide.rods import rod_centerline


PLATE = PlatePose(0.0, 0.0, 0.0)


@pytest.mark.parametrize("puzzle", [Puzzle.A1, Puzzle.A2, Puzzle.B, Puzzle.C])
def test_rod_points_and_tangents_are_matching_arrays(puzzle):
    points, tangents = rod_centerline(puzzle, PLATE, 0.0, 0.005)

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape == tangents.shape
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)


def test_a1_is_vertical_from_free_tip_to_plate():
    points, tangents = rod_centerline(Puzzle.A1, PLATE, 0.0, 0.005)

    assert points.shape == (41, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.2])
    np.testing.assert_allclose(points[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(points[:, :2], 0.0)
    np.testing.assert_allclose(tangents, [[0.0, 0.0, -1.0]] * 41)


def test_a1_and_a2_have_the_same_geometry():
    points_a1, tangents_a1 = rod_centerline(Puzzle.A1, PLATE, 0.0, 0.005)
    points_a2, tangents_a2 = rod_centerline(Puzzle.A2, PLATE, 0.0, 0.005)

    np.testing.assert_allclose(points_a1, points_a2)
    np.testing.assert_allclose(tangents_a1, tangents_a2)


@pytest.mark.parametrize(
    ("puzzle", "vertices"),
    [
        (
            Puzzle.B,
            [[0.07, 0.0, 0.2], [0.07, 0.0, 0.15], [0, 0, 0.08], [0, 0, 0]],
        ),
        (
            Puzzle.C,
            [
                [-0.05, -0.05, 0.2],
                [-0.05, -0.05, 0.15],
                [-0.05, 0.0, 0.1],
                [0.0, 0.0, 0.05],
                [0.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_each_bend_vertex_is_present_once(puzzle, vertices):
    points, _ = rod_centerline(puzzle, PLATE, 0.0, 0.005)

    for vertex in vertices:
        is_vertex = np.all(np.isclose(points, vertex, atol=1e-12), axis=1)
        assert np.count_nonzero(is_vertex) == 1


@pytest.mark.parametrize("puzzle", [Puzzle.A1, Puzzle.A2, Puzzle.B, Puzzle.C])
def test_all_sample_distances_are_positive_and_bounded(puzzle):
    points, _ = rod_centerline(puzzle, PLATE, 0.0, 0.007)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    assert np.all(distances > 0)
    assert np.all(distances <= 0.007 + 1e-12)


@pytest.mark.parametrize("puzzle", [Puzzle.A1, Puzzle.A2, Puzzle.B, Puzzle.C])
def test_tangents_follow_the_next_sample(puzzle):
    points, tangents = rod_centerline(puzzle, PLATE, 0.0, 0.005)
    displacements = np.diff(points, axis=0)
    expected = displacements / np.linalg.norm(
        displacements, axis=1, keepdims=True
    )

    np.testing.assert_allclose(tangents[:-1], expected, atol=1e-12)
    np.testing.assert_allclose(tangents[-1], expected[-1], atol=1e-12)


def test_b_has_outgoing_tangent_at_the_first_bend():
    points, tangents = rod_centerline(Puzzle.B, PLATE, 0.0, 0.005)
    bend_index = np.flatnonzero(np.isclose(points[:, 2], 0.15))[0]

    np.testing.assert_allclose(tangents[bend_index - 1], [0, 0, -1])
    np.testing.assert_allclose(
        tangents[bend_index], [-1 / math.sqrt(2), 0, -1 / math.sqrt(2)]
    )


def test_translation_only_affects_points():
    points, tangents = rod_centerline(
        Puzzle.A1, PlatePose(0.45, 0.10, 0.0), 0.08, 0.005
    )

    np.testing.assert_allclose(points[0], [0.45, 0.1, 0.28])
    np.testing.assert_allclose(points[-1], [0.45, 0.1, 0.08])
    np.testing.assert_allclose(tangents, [[0, 0, -1]] * len(tangents))


def test_plate_rotation_rotates_points_and_unit_tangents():
    points, tangents = rod_centerline(
        Puzzle.B, PlatePose(0.45, 0.10, math.pi / 2), 0.08, 0.005
    )

    np.testing.assert_allclose(points[0], [0.45, 0.17, 0.28], atol=1e-12)
    np.testing.assert_allclose(points[-1], [0.45, 0.10, 0.08], atol=1e-12)
    np.testing.assert_allclose(
        tangents[11], [0, -1 / math.sqrt(2), -1 / math.sqrt(2)], atol=1e-12
    )


@pytest.mark.parametrize("step", [0, -0.01, float("nan"), float("inf")])
def test_invalid_sampling_step_raises_value_error(step):
    with pytest.raises(ValueError, match="Sampling step"):
        rod_centerline(Puzzle.A1, PLATE, 0.0, step)


@pytest.mark.parametrize("height", [float("nan"), float("inf")])
def test_invalid_plate_height_raises_value_error(height):
    with pytest.raises(ValueError, match="Plate pose"):
        rod_centerline(Puzzle.A1, PLATE, height, 0.005)


def test_d_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="D"):
        rod_centerline(Puzzle.D, PLATE, 0.0, 0.005)
