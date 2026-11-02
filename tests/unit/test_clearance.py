"""Offline geometry screening only: no robot connection or movement."""

import math
from functools import lru_cache

import numpy as np
import pytest

from hoop_guide.clearance import _bend_windows
from hoop_guide.clearance import _distance_to_segments
from hoop_guide.clearance import _frame_for_tangent
from hoop_guide.clearance import _interpolate_pose
from hoop_guide.clearance import _interval_approved
from hoop_guide.clearance import _point_on_model
from hoop_guide.clearance import _pose_clearance
from hoop_guide.clearance import _rod_segments
from hoop_guide.clearance import _rotation_angle
from hoop_guide.clearance import _smooth_geometry
from hoop_guide.clearance import ClearanceSettings
from hoop_guide.clearance import screen_hoop_path
from hoop_guide.geometry import plate_to_robot
from hoop_guide.plate import PlatePose
from hoop_guide.rods import model_for_puzzle
from hoop_guide.rods import Puzzle
from hoop_guide.tool_setup import initial_offline_tool_estimate

PLATE = PlatePose(0.45, 0.10, 0)
HEIGHT = 0.08


@lru_cache(maxsize=5)
def nominal_result(puzzle: Puzzle):
    return screen_hoop_path(puzzle, PLATE, HEIGHT)


@pytest.mark.parametrize("puzzle", list(Puzzle))
def test_nominal_hoop_and_handle_trajectory_has_continuous_clearance(puzzle):
    """CAD-derived conservative surrogate passes at the example plate pose."""
    report = nominal_result(puzzle)
    assert report.geometric_screen_passed
    assert report.unresolved_intervals == ()
    assert len(report.progress_m) == len(report.poses_robot)
    assert len(report.clearances_m) == len(report.progress_m)
    assert len(report.heading_offsets_rad) == len(report.progress_m)
    assert np.all(np.diff(report.progress_m) > 0)
    assert report.progress_m[0] == 0.0
    assert report.clearances_m.min() >= report.settings.target_clearance_m
    assert np.allclose(report.poses_robot[:, 3, :], [0, 0, 0, 1])
    for matrix in report.poses_robot[:, :3, :3]:
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_model_bends_are_blended_and_d_last_bend_has_limited_space():
    straight = nominal_result(Puzzle.A1)
    assert straight.blended_corner_indices == ()
    assert straight.restricted_corner_indices == ()
    for puzzle, expected in ((Puzzle.B, 2), (Puzzle.C, 3), (Puzzle.D, 1)):
        report = nominal_result(puzzle)
        assert len(report.blended_corner_indices) == expected
        for index in report.blended_corner_indices:
            assert 0 < index < len(report.poses_robot) - 1
    assert nominal_result(Puzzle.D).restricted_corner_indices


def test_unverified_mount_is_preview_only_and_transform_composition_matches():
    result = nominal_result(Puzzle.A1)
    estimate = initial_offline_tool_estimate()
    assert not estimate.is_verified
    flange_poses = result.flange_preview(estimate)
    fixed_tool = estimate.mount.flange_to_hoop_center()
    for flange, ring in zip(flange_poses, result.poses_robot):
        np.testing.assert_allclose(flange @ fixed_tool, ring, atol=1e-12)


def test_oversized_rod_is_not_reported_as_clear():
    """Lack of clearance must result in an explicitly unresolved report."""
    settings = ClearanceSettings(
        rod_radius_bound_m=0.040, max_subdivision_depth=1
    )
    result = screen_hoop_path(Puzzle.A1, PLATE, HEIGHT, settings=settings)
    assert not result.geometric_screen_passed
    assert result.unresolved_intervals
    assert result.clearances_m.min() < settings.target_clearance_m


@pytest.mark.parametrize("params", [
    {"path_step_m": 0.0},
    {"blend_half_length_m": -0.01},
    {"rod_sample_step_m": math.nan},
    {"rod_radius_bound_m": -0.01},
    {"target_clearance_m": math.inf},
    {"ring_points": 8},
    {"max_subdivision_depth": -1},
    {"heading_samples": 4},
])
def test_invalid_settings_are_rejected(params):
    with pytest.raises(ValueError):
        ClearanceSettings(**params)


@pytest.mark.parametrize("base", [(1.0,), (math.inf, 0.0)])
def test_invalid_robot_base_xy_is_rejected(base):
    with pytest.raises(ValueError, match="robot_base_xy"):
        screen_hoop_path(Puzzle.A1, PLATE, HEIGHT, robot_base_xy=base)


def test_cubic_blend_is_continuous_at_both_ends_and_offsets_corner():
    model = model_for_puzzle(Puzzle.B)
    windows = _bend_windows(model, 0.018)
    assert len(windows) == 2
    corner, half, incoming, outgoing = windows[0]
    assert 0 < half <= 0.018
    point_before, tangent_before = _smooth_geometry(model, corner-half, windows)
    point_after, tangent_after = _smooth_geometry(model, corner+half, windows)
    np.testing.assert_allclose(
        point_before, _point_on_model(model, corner-half)[0]
    )
    np.testing.assert_allclose(
        point_after, _point_on_model(model, corner+half)[0]
    )
    np.testing.assert_allclose(tangent_before, incoming)
    np.testing.assert_allclose(tangent_after, outgoing)
    middle, tang = _smooth_geometry(model, corner, windows)
    assert np.isfinite(middle).all()
    assert np.linalg.norm(tang) == pytest.approx(1)
    assert np.linalg.norm(middle - _point_on_model(model, corner)[0]) > 0.001


def test_rod_model_is_extended_below_40mm_stop_for_collision_screen():
    model = model_for_puzzle(Puzzle.A1)
    transform = plate_to_robot(PLATE, HEIGHT)
    segments = _rod_segments(model, ClearanceSettings(), transform)
    np.testing.assert_allclose(segments[-1, 1], [PLATE.x, PLATE.y, HEIGHT])
    assert np.isfinite(segments).all()


def test_point_segment_distance_is_exact_and_rejects_degenerate_segment():
    segments = np.array([[[0., 0., 0.], [1., 0., 0.]]])
    samples = np.array([[0.3, 2., 0.], [2., 0., 0.]])
    np.testing.assert_allclose(
        _distance_to_segments(samples, segments), [2., 1.]
    )
    with pytest.raises(ValueError, match="zero-length"):
        _distance_to_segments(samples, np.zeros((1, 2, 3)))


def test_homogeneous_poses_include_correct_ring_plane_clearance():
    model = model_for_puzzle(Puzzle.A1)
    settings = ClearanceSettings()
    rod = _rod_segments(model, settings, plate_to_robot(PLATE, HEIGHT))
    pose = np.eye(4)
    pose[:3, 3] = [PLATE.x, PLATE.y, HEIGHT]
    assert _pose_clearance(pose, rod, HEIGHT, settings) < 0


def test_so3_midpoint_and_geodesic_rotation():
    first = np.eye(4)
    last = np.eye(4)
    last[:3, :3] = np.diag([-1, -1, 1])
    last[:3, 3] = [2, 4, 6]
    halfway = _interpolate_pose(first, last)
    np.testing.assert_allclose(halfway[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(
        halfway[:3, :3].T @ halfway[:3, :3], np.eye(3), atol=1e-12
    )
    assert _rotation_angle(
        first[:3, :3], halfway[:3, :3]
    ) == pytest.approx(math.pi / 2)
    assert _rotation_angle(
        halfway[:3, :3], last[:3, :3]
    ) == pytest.approx(math.pi / 2)


def test_so3_midpoint_for_small_rotation():
    first = np.eye(4)
    last = np.eye(4)
    angle = math.radians(20)
    last[:3, :3] = [[math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    middle = _interpolate_pose(first, last)
    assert _rotation_angle(
        first[:3, :3], middle[:3, :3]
    ) == pytest.approx(angle / 2)


def test_frame_for_tangent_is_proper_pose():
    pose = _frame_for_tangent(
        np.array([0., 0., -1.]), np.array([.4, .1, .2]),
        np.array([0., 0.]), None,
    )
    np.testing.assert_allclose(pose[:3, 2], [0, 0, 1])
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def test_certification_checks_between_samples_not_just_endpoints():
    """Two individually clear poses can cross a rod in the middle."""
    rod = np.array([[[.0325, 0, .075], [.0325, 0, .125]]])
    settings = ClearanceSettings(max_subdivision_depth=6)
    start = np.eye(4)
    end = np.eye(4)
    start[:3, 3] = [0, 0, 0]
    end[:3, 3] = [0, 0, .2]
    a = _pose_clearance(start, rod, -1, settings)
    b = _pose_clearance(end, rod, -1, settings)
    assert a > settings.target_clearance_m
    assert b > settings.target_clearance_m
    assert not _interval_approved(start, end, a, b, rod, -1, settings, 0)


def test_certification_accepts_unchanged_pose_with_clearance():
    rod = np.array([[[2., 0, 0], [2., 0, 1]]])
    settings = ClearanceSettings()
    pose = np.eye(4)
    pose[:3, 3] = [0, 0, 1]
    clearance = _pose_clearance(pose, rod, -1, settings)
    assert _interval_approved(
        pose, pose, clearance, clearance, rod, -1, settings, 0
    )


def test_zero_subdivision_depth_conservatively_refuses_long_move():
    rod = np.array([[[2., 0, 0], [2., 0, 1]]])
    settings = ClearanceSettings(max_subdivision_depth=0)
    start = np.eye(4)
    end = np.eye(4)
    end[:3, 3] = [0, 0, 3]
    a = _pose_clearance(start, rod, -1, settings)
    b = _pose_clearance(end, rod, -1, settings)
    assert not _interval_approved(start, end, a, b, rod, -1, settings, 0)
