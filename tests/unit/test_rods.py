import math

import numpy as np
import pytest

from hoop_guide.plate import PlatePose
from hoop_guide.rods import ArcSegment
from hoop_guide.rods import LineSegment
from hoop_guide.rods import model_for_puzzle
from hoop_guide.rods import Puzzle
from hoop_guide.rods import ROD_A_NOMINAL_HEIGHT
from hoop_guide.rods import RodModel
from hoop_guide.rods import rod_centerline
from hoop_guide.rods import STOP_HEIGHT


def test_a1_starts_at_rod_tip_ends_at_stop_height_and_goes_down():
    points, tangents = rod_centerline(
        Puzzle.A1, PlatePose(0.0, 0.0, 0.0), 0.0, 0.020
    )

    np.testing.assert_allclose(points[0], [0.0, 0.0, ROD_A_NOMINAL_HEIGHT])
    np.testing.assert_allclose(points[-1], [0.0, 0.0, STOP_HEIGHT])
    np.testing.assert_allclose(tangents, np.tile([0, 0, -1], (len(points), 1)))
    assert np.all(np.diff(points[:, 2]) < 0)


def test_a2_has_the_same_local_centerline_as_a1():
    pose = PlatePose(0.0, 0.0, 0.0)

    a1 = rod_centerline(Puzzle.A1, pose, 0.0, 0.015)
    a2 = rod_centerline(Puzzle.A2, pose, 0.0, 0.015)

    for left, right in zip(a1, a2):
        np.testing.assert_allclose(left, right)


def test_a1_transforms_to_robot_frame_including_plate_height():
    points, tangents = rod_centerline(
        Puzzle.A1, PlatePose(0.45, -0.1, math.pi / 2), 0.08, 0.02
    )

    np.testing.assert_allclose(points[0], [0.45, -0.1, 0.28], atol=1e-12)
    np.testing.assert_allclose(points[-1], [0.45, -0.1, 0.12], atol=1e-12)
    np.testing.assert_allclose(tangents[0], [0.0, 0.0, -1.0])


def test_a1_includes_endpoints_and_never_exceeds_step():
    step = 0.017

    points, tangents = model_for_puzzle(Puzzle.A1).sample(step)

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(lengths <= step + 1e-12)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert len(points) == len(tangents)


@pytest.mark.parametrize("step", [0.0, -0.01, float("nan"), float("inf")])
def test_sampling_rejects_invalid_step(step):
    with pytest.raises(ValueError, match="step"):
        model_for_puzzle(Puzzle.A1).sample(step)


@pytest.mark.parametrize(
    ("puzzle", "tip", "end", "segments"),
    [
        (Puzzle.B, (0.070, 0, 0.200), (0, 0, STOP_HEIGHT), 3),
        (Puzzle.C, (-0.050, -0.050, 0.200), (0, 0, STOP_HEIGHT), 4),
        (Puzzle.D, (0.085, 0, 0.155), (-0.010, 0, STOP_HEIGHT), 4),
    ],
)
def test_cad_puzzle_tip_stop_and_segments(puzzle, tip, end, segments):
    model = model_for_puzzle(puzzle)
    points, tangents = model.sample(0.007)

    assert len(model.segments) == segments
    np.testing.assert_allclose(points[0], tip, atol=1e-12)
    np.testing.assert_allclose(points[-1], end, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.all(np.diff(points[:, 2]) <= 1e-12)
    assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) < 0.0071)


def test_b_and_c_have_correct_cad_corner_locations():
    model_b = model_for_puzzle(Puzzle.B)
    model_c = model_for_puzzle(Puzzle.C)

    expected_b = ((0.070, 0, 0.150), (0, 0, 0.080))
    expected_c = (
        (-0.050, -0.050, 0.150),
        (-0.050, 0, 0.100),
        (0, 0, 0.050),
    )
    for model, expected in ((model_b, expected_b), (model_c, expected_c)):
        for segment, corner in zip(model.segments, expected):
            np.testing.assert_allclose(
                segment.point_at(segment.length), corner, atol=1e-12
            )


def test_d_arc_has_cad_radius_center_and_smooth_junctions():
    model = model_for_puzzle(Puzzle.D)
    first, arc, vertical, lower_diagonal = model.segments

    assert isinstance(arc, ArcSegment)
    assert arc.radius == pytest.approx(0.050)
    assert arc.length == pytest.approx(math.pi * 0.050 / 2)
    np.testing.assert_allclose(arc.point_at(0), [0.035, 0, 0.155])
    np.testing.assert_allclose(arc.point_at(arc.length), [-0.015, 0, 0.105])
    np.testing.assert_allclose(arc.tangent_at(0), first.tangent_at(0))
    np.testing.assert_allclose(
        arc.tangent_at(arc.length), vertical.tangent_at(0), atol=1e-12
    )
    np.testing.assert_allclose(
        lower_diagonal.point_at(lower_diagonal.length),
        [-0.010, 0, STOP_HEIGHT],
    )


def test_cad_puzzles_transform_as_rigid_objects():
    plate = PlatePose(0.4, -0.1, math.pi / 2)
    for puzzle in (Puzzle.B, Puzzle.C, Puzzle.D):
        local_points, local_tangents = model_for_puzzle(puzzle).sample(0.025)
        points, tangents = rod_centerline(puzzle, plate, 0.08, 0.025)
        np.testing.assert_allclose(
            points[:, 0], 0.4 - local_points[:, 1], atol=1e-12
        )
        np.testing.assert_allclose(
            points[:, 1], -0.1 + local_points[:, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            points[:, 2], 0.08 + local_points[:, 2], atol=1e-12
        )
        np.testing.assert_allclose(
            tangents[:, 0], -local_tangents[:, 1], atol=1e-12
        )
        np.testing.assert_allclose(
            tangents[:, 1], local_tangents[:, 0], atol=1e-12
        )


def test_line_interpolation_uses_travel_distance_and_unit_tangent():
    line = LineSegment((0.0, 0.0, 0.0), (0.03, 0.04, 0.0))

    assert line.length == pytest.approx(0.05)
    np.testing.assert_allclose(line.point_at(0.025), [0.015, 0.02, 0.0])
    np.testing.assert_allclose(line.tangent_at(0.0), [0.6, 0.8, 0.0])


@pytest.mark.parametrize("distance", [-0.01, 0.2, float("nan")])
def test_line_rejects_distance_outside_segment(distance):
    line = LineSegment((0.0, 0.0, 0.0), (0.0, 0.0, 0.1))

    with pytest.raises(ValueError, match="Distance"):
        line.point_at(distance)


@pytest.mark.parametrize(
    "start,end",
    [
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, float("nan")), (0, 0, 1)),
    ],
)
def test_line_rejects_invalid_endpoints(start, end):
    with pytest.raises(ValueError, match="distinct finite"):
        LineSegment(start, end)


def test_arc_endpoints_and_tangents_follow_right_hand_rule():
    arc = ArcSegment(
        center=(0.0, 0.0, 0.0),
        start_radius=(0.05, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        sweep=math.pi / 2,
    )

    assert arc.length == pytest.approx(0.05 * math.pi / 2)
    np.testing.assert_allclose(arc.point_at(0), [0.05, 0, 0])
    np.testing.assert_allclose(
        arc.point_at(arc.length), [0, 0.05, 0], atol=1e-12
    )
    np.testing.assert_allclose(arc.tangent_at(0), [0, 1, 0])
    np.testing.assert_allclose(
        arc.tangent_at(arc.length), [-1, 0, 0], atol=1e-12
    )


def test_negative_arc_sweep_reverses_tangent():
    arc = ArcSegment((0, 0, 0), (0.05, 0, 0), (0, 0, 1), -math.pi / 2)

    np.testing.assert_allclose(
        arc.point_at(arc.length), [0, -0.05, 0], atol=1e-12
    )
    np.testing.assert_allclose(arc.tangent_at(0), [0, -1, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            center=(float("nan"), 0, 0),
            start_radius=(1, 0, 0),
            axis=(0, 0, 1),
            sweep=1,
        ),
        dict(center=(0, 0, 0), start_radius=(0, 0, 0), axis=(0, 0, 1), sweep=1),
        dict(center=(0, 0, 0), start_radius=(1, 0, 0), axis=(0, 0, 1), sweep=0),
        dict(center=(0, 0, 0), start_radius=(1, 0, 0), axis=(0, 0, 2), sweep=1),
        dict(center=(0, 0, 0), start_radius=(1, 0, 1), axis=(0, 0, 1), sweep=1),
    ],
)
def test_arc_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError, match="Arc"):
        ArcSegment(**kwargs)


def test_arc_rejects_distance_outside_segment():
    arc = ArcSegment((0, 0, 0), (0.05, 0, 0), (0, 0, 1), math.pi / 2)

    with pytest.raises(ValueError, match="Distance"):
        arc.tangent_at(arc.length + 0.01)


def test_composed_line_and_arc_have_no_duplicate_joint():
    line = LineSegment((0.05, 0, 0.05), (0.05, 0, 0))
    arc = ArcSegment((0, 0, 0), (0.05, 0, 0), (0, 0, 1), math.pi / 2)
    rod = RodModel((line, arc))

    points, tangents = rod.sample(0.02)

    assert np.sum(np.linalg.norm(points - [0.05, 0, 0], axis=1) < 1e-12) == 1
    np.testing.assert_allclose(points[-1], [0.0, 0.05, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)


def test_rod_model_rejects_no_segments_or_disconnected_segments():
    with pytest.raises(ValueError, match="at least one"):
        RodModel(())
    with pytest.raises(ValueError, match="continuous"):
        RodModel(
            (
                LineSegment((0, 0, 0), (0, 0, 0.1)),
                LineSegment((0, 0, 0.2), (0, 0, 0.3)),
            )
        )
