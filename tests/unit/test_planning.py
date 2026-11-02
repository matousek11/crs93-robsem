"""Tests for offline desired hoop/flange poses and conservative diagnostics."""

import dataclasses
import math

import numpy as np
import pytest

from hoop_guide.geometry import AdapterMount
from hoop_guide.geometry import plate_to_robot
from hoop_guide.planning import _holder_y_axis
from hoop_guide.planning import hoop_pose_path
from hoop_guide.plate import PlatePose
from hoop_guide.rods import Puzzle
from hoop_guide.rods import rod_centerline
from hoop_guide.tool_setup import initial_offline_tool_estimate
from hoop_guide.tool_setup import ToolCalibration
from hoop_guide.types import Pose


@pytest.mark.parametrize("puzzle", list(Puzzle))
def test_every_puzzle_produces_rigid_hoop_frames(puzzle):
    plate = PlatePose(0.45, 0.10, math.pi / 6)
    path = hoop_pose_path(puzzle, plate, 0.08, 0.01)
    points, tangents = rod_centerline(puzzle, plate, 0.08, 0.01)

    assert path.poses_robot.shape == (len(points), 4, 4)
    np.testing.assert_allclose(path.points_robot, points)
    np.testing.assert_allclose(path.tangents_robot, tangents)
    np.testing.assert_allclose(path.poses_robot[:, :3, 3], points)
    np.testing.assert_allclose(path.poses_robot[:, :3, 2], -tangents)
    np.testing.assert_allclose(
        np.transpose(path.poses_robot[:, :3, :3], (0, 2, 1))
        @ path.poses_robot[:, :3, :3],
        np.broadcast_to(np.eye(3), (len(points), 3, 3)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.det(path.poses_robot[:, :3, :3]), 1, atol=1e-12
    )
    np.testing.assert_allclose(
        path.poses_robot[:, 3],
        np.tile([0, 0, 0, 1], (len(path.poses_robot), 1)),
    )


def test_a1_holder_points_toward_robot_base_and_remains_level():
    path = hoop_pose_path(Puzzle.A1, PlatePose(0.45, 0.10, 0), 0.08, 0.02)
    expected = np.array([-0.45, -0.10, 0.0])
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        path.poses_robot[:, :3, 1],
        np.tile(expected, (len(path.poses_robot), 1)),
    )
    np.testing.assert_allclose(
        path.poses_robot[:, :3, 2],
        np.tile([0, 0, 1], (len(path.poses_robot), 1)),
    )
    assert path.sharp_corner_indices == ()
    assert path.fallback_heading_indices == ()


def test_a1_and_a2_have_identical_paths_for_same_plate_pose():
    pose = PlatePose(0.4, 0.1, 0.2)
    path_a1 = hoop_pose_path(Puzzle.A1, pose, 0.08, 0.015)
    path_a2 = hoop_pose_path(Puzzle.A2, pose, 0.08, 0.015)
    np.testing.assert_allclose(path_a1.poses_robot, path_a2.poses_robot)


def test_junction_diagnostics_detect_actual_corners_not_smooth_arc():
    pose = PlatePose(0.45, 0.1, 0.0)
    results = {
        puzzle: hoop_pose_path(puzzle, pose, 0.08, 0.01)
        for puzzle in Puzzle
    }
    assert results[Puzzle.A1].sharp_corner_indices == ()
    assert results[Puzzle.A2].sharp_corner_indices == ()
    assert len(results[Puzzle.B].sharp_corner_indices) == 2
    assert len(results[Puzzle.C].sharp_corner_indices) == 3
    # D.step: tangency of line/arc/vertical is continuous, lower kink isn't.
    assert len(results[Puzzle.D].sharp_corner_indices) == 1
    for puzzle in (Puzzle.B, Puzzle.C, Puzzle.D):
        for corner in results[puzzle].sharp_corner_indices:
            assert 0 < corner < len(results[puzzle].poses_robot) - 1


def test_fallback_direction_is_flagged_for_rod_over_robot_origin():
    # At the origin, pointing horizontally toward the base is undefined.
    path = hoop_pose_path(Puzzle.A1, PlatePose(0, 0, 0), 0, 0.02)
    assert path.fallback_heading_indices == tuple(range(len(path.poses_robot)))
    for pose in path.poses_robot:
        np.testing.assert_allclose(pose[:3, 1], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(pose[:3, 2], [0, 0, 1], atol=1e-12)


def test_fallback_reuses_previous_heading_when_projection_disappears():
    # Puzzle D starts with -X tangent, preferred base direction -X and
    # projection is undefined; later points should still have valid frames.
    path = hoop_pose_path(Puzzle.D, PlatePose(0, 0, 0), 0, 0.02)
    assert path.fallback_heading_indices
    assert 0 in path.fallback_heading_indices
    np.testing.assert_allclose(np.linalg.det(path.poses_robot[:, :3, :3]), 1)


def test_custom_robot_base_heading_changes_y_axis():
    path = hoop_pose_path(
        Puzzle.A1,
        PlatePose(0, 0, 0),
        0,
        0.02,
        robot_base_xy=(0, 1),
    )
    np.testing.assert_allclose(
        path.poses_robot[:, :3, 1],
        np.tile([0, 1, 0], (len(path.poses_robot), 1)),
    )
    assert path.fallback_heading_indices == ()


@pytest.mark.parametrize(
    ("step", "base"),
    [
        (0.0, (0.0, 0.0)),
        (-1.0, (0.0, 0.0)),
        (float("nan"), (0.0, 0.0)),
        (0.01, (float("inf"), 0.0)),
        (0.01, (1.0,)),
    ],
)
def test_invalid_path_parameters_raise(step, base):
    with pytest.raises(ValueError):
        hoop_pose_path(
            Puzzle.A1, PlatePose(0, 0, 0), 0, step, robot_base_xy=base
        )


def test_offline_mount_is_explicitly_unverified_and_uses_cad_center():
    estimate = initial_offline_tool_estimate()
    assert not estimate.is_verified
    assert "UNVERIFIED" in estimate.source
    np.testing.assert_allclose(
        estimate.mount.flange_to_hoop_center()[:3, 3],
        [0.0, -0.135, -0.0033],
    )


def test_unverified_mount_cannot_produce_nonpreview_flange_path():
    path = hoop_pose_path(Puzzle.A1, PlatePose(0.45, 0.10, 0), 0.08, 0.02)
    with pytest.raises(ValueError, match="UNVERIFIED"):
        path.flange_poses(initial_offline_tool_estimate())


def test_offline_flange_preview_round_trip_and_adapter_offset():
    path = hoop_pose_path(Puzzle.B, PlatePose(0.45, 0.1, 0), 0.08, 0.02)
    estimate = initial_offline_tool_estimate()

    flange_poses = path.flange_poses(estimate, preview_only=True)
    assert flange_poses.shape == path.poses_robot.shape
    for flange, hoop in zip(flange_poses, path.poses_robot):
        np.testing.assert_allclose(
            flange @ estimate.mount.flange_to_hoop_center(),
            hoop,
            atol=1e-12,
        )


def test_verified_mount_still_refuses_sharp_corner_path():
    estimated = initial_offline_tool_estimate()
    verified_for_test = ToolCalibration(
        mount=estimated.mount,
        is_verified=True,
        source="SYNTHETIC UNIT TEST - not a real hardware calibration",
    )
    path = hoop_pose_path(Puzzle.B, PlatePose(0.45, 0.1, 0), 0.08, 0.02)
    with pytest.raises(ValueError, match="SHARP CORNERS"):
        path.flange_poses(verified_for_test)


def test_verified_mount_still_refuses_ambiguous_holder_direction():
    estimated = initial_offline_tool_estimate()
    verified_for_test = ToolCalibration(
        mount=estimated.mount,
        is_verified=True,
        source="SYNTHETIC UNIT TEST - not a real hardware calibration",
    )
    path = hoop_pose_path(Puzzle.A1, PlatePose(0, 0, 0), 0, 0.02)
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        path.flange_poses(verified_for_test)


def test_verified_synthetic_mount_can_transform_smooth_unambiguous_path():
    estimate = initial_offline_tool_estimate()
    # Synthetic test only. This is not verification of the physical adapter.
    mount = ToolCalibration(
        mount=estimate.mount, is_verified=True, source="SYNTHETIC UNIT TEST"
    )
    path = hoop_pose_path(Puzzle.A1, PlatePose(0.45, 0.1, 0), 0.08, 0.02)
    flanges = path.flange_poses(mount)
    for flange, hoop in zip(flanges, path.poses_robot):
        np.testing.assert_allclose(
            flange @ mount.mount.flange_to_hoop_center(), hoop, atol=1e-12
        )


def test_tool_calibration_needs_nonempty_provenance():
    pose = Pose(np.eye(4))
    with pytest.raises(ValueError, match="source"):
        ToolCalibration(AdapterMount(pose, pose), False, "  ")


def test_plate_rotation_changes_tool_pose_in_robot_frame():
    plate = PlatePose(0.4, -0.2, math.pi / 2)
    path = hoop_pose_path(Puzzle.A1, plate, 0.08, 0.02)
    np.testing.assert_allclose(path.poses_robot[0, :3, 3], [0.4, -0.2, 0.28])
    np.testing.assert_allclose(
        plate_to_robot(plate, 0.08)[:3, 3], [0.4, -0.2, 0.08]
    )


def test_fallback_with_previous_axis_parallel_to_normal_uses_canonical():
    holder, fallback = _holder_y_axis(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    assert fallback
    np.testing.assert_allclose(holder, [0.0, 1.0, 0.0])


def test_large_rotation_step_detects_coarsely_sampled_bend():
    # Quarter-circle R50 sampled with two intervals has ~45-degree jumps.
    path = hoop_pose_path(Puzzle.D, PlatePose(0.45, 0.1, 0), 0.08, 0.05)
    assert path.large_rotation_step_indices
    estimate = initial_offline_tool_estimate()
    simulated = ToolCalibration(
        estimate.mount,
        is_verified=True,
        source="SYNTHETIC UNIT TEST - not physical calibration",
    )
    # Isolate large-step guard from the independent sharp-corner guard.
    no_corners = dataclasses.replace(path, sharp_corner_indices=())
    with pytest.raises(ValueError, match="ORIENTATION STEPS"):
        no_corners.flange_poses(simulated)


def test_a1_has_no_large_orientation_changes():
    path = hoop_pose_path(Puzzle.A1, PlatePose(0.45, 0.1, 0), 0.08, 0.02)
    assert path.large_rotation_step_indices == ()
