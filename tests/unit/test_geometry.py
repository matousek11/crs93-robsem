import math

import numpy as np
import pytest

from hoop_guide.geometry import AdapterMount
from hoop_guide.geometry import flange_pose_for_hoop
from hoop_guide.geometry import HOOP_CAD_CENTER_M
from hoop_guide.geometry import hoop_cad_to_center
from hoop_guide.geometry import inverse_pose
from hoop_guide.geometry import plate_to_robot
from hoop_guide.geometry import transform_directions
from hoop_guide.geometry import transform_points
from hoop_guide.plate import PlatePose
from hoop_guide.types import Directions3D
from hoop_guide.types import Points3D
from hoop_guide.types import Pose


def test_plate_frame_origin_moves_to_plate_center_and_height():
    transform = plate_to_robot(PlatePose(0.45, 0.10, 0), 0.08)

    actual = transform_points(Points3D(np.array([[0.0, 0.0, 0.0]])), transform)

    np.testing.assert_allclose(actual, [[0.45, 0.10, 0.08]])


def test_plate_frame_ninety_degree_rotation_preserves_z():
    transform = plate_to_robot(PlatePose(0.0, 0.0, math.pi / 2), 0.05)

    actual = transform_points(Points3D(np.array([[0.1, 0.0, 0.2]])), transform)

    np.testing.assert_allclose(actual, [[0.0, 0.1, 0.25]], atol=1e-12)


def test_transform_directions_ignores_translation_and_preserves_norm():
    transform = plate_to_robot(PlatePose(5.0, 6.0, math.pi / 2), 7.0)
    directions = Directions3D(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))

    actual = transform_directions(directions, transform)

    expected = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_inverse_pose_undoes_rotation_and_translation():
    transform = plate_to_robot(PlatePose(0.3, -0.2, -math.pi / 3), 0.4)

    result = inverse_pose(transform)

    np.testing.assert_allclose(result @ transform, np.eye(4), atol=1e-12)


def test_hoop_to_flange_conversion_accounts_for_adapter_rotation():
    # Flange-to-hoop includes a 90-degree mounting rotation and 15 cm offset.
    hoop_in_flange = Pose(np.eye(4))
    hoop_in_flange[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    hoop_in_flange[:3, 3] = [0.15, 0.0, 0.0]
    desired_hoop = plate_to_robot(PlatePose(0.45, 0.10, math.pi / 2), 0.30)

    flange = flange_pose_for_hoop(desired_hoop, hoop_in_flange)

    np.testing.assert_allclose(
        flange @ hoop_in_flange, desired_hoop, atol=1e-12
    )


def test_empty_points_and_directions_are_valid():
    pose = plate_to_robot(PlatePose(0.0, 0.0, 0.0), 0.0)

    assert transform_points(Points3D(np.empty((0, 3))), pose).shape == (0, 3)
    directions = Directions3D(np.empty((0, 3)))
    assert transform_directions(directions, pose).shape == (0, 3)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_plate_transform_rejects_nonfinite_pose_and_height(value):
    with pytest.raises(ValueError, match="finite"):
        plate_to_robot(PlatePose(value, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="finite"):
        plate_to_robot(PlatePose(0.0, 0.0, 0.0), value)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((3, 3)),
        np.full((4, 4), np.nan),
        np.diag([1.0, 1.0, 1.0, 2.0]),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    ],
    ids=["shape", "nonfinite", "last-row", "nonorthogonal", "reflection"],
)
def test_invalid_rigid_transform_is_rejected(invalid):
    with pytest.raises(ValueError, match="Pose"):
        inverse_pose(Pose(invalid))


@pytest.mark.parametrize(
    "invalid", [np.zeros(3), np.full((1, 3), np.nan), np.zeros((1, 2))]
)
def test_transform_points_rejects_invalid_shape_or_values(invalid):
    pose = plate_to_robot(PlatePose(0.0, 0.0, 0.0), 0.0)

    with pytest.raises(ValueError, match="Points"):
        transform_points(Points3D(invalid), pose)


@pytest.mark.parametrize(
    "invalid",
    [np.zeros(3), np.full((1, 3), np.inf), np.zeros((1, 2)), np.zeros((1, 3))],
)
def test_transform_directions_requires_valid_unit_vectors(invalid):
    pose = plate_to_robot(PlatePose(0.0, 0.0, 0.0), 0.0)

    with pytest.raises(ValueError, match="Directions"):
        transform_directions(Directions3D(invalid), pose)


def test_hoop_cad_center_matches_step_ring_geometry():
    """Hoop.step ring opening center: (0, -135, -3.3) mm."""
    assert HOOP_CAD_CENTER_M == (0.0, -0.135, -0.0033)
    pose = hoop_cad_to_center()
    np.testing.assert_allclose(pose[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose[:3, 3], HOOP_CAD_CENTER_M)


def test_adapter_mount_compounds_all_three_known_transform_steps():
    # A fabricated mounting pose, not the dimensions of a real adapter.
    flange_adapter = Pose(np.eye(4))
    flange_adapter[:3, 3] = [0.01, 0.02, 0.03]
    adapter_hoop_cad = Pose(np.eye(4))
    adapter_hoop_cad[:3, :3] = [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]
    adapter_hoop_cad[:3, 3] = [0.0, 0.04, 0.02]
    mount = AdapterMount(flange_adapter, adapter_hoop_cad)

    expected = flange_adapter @ adapter_hoop_cad @ hoop_cad_to_center()
    np.testing.assert_allclose(mount.flange_to_hoop_center(), expected)

    desired_hoop = plate_to_robot(PlatePose(0.4, 0.1, 0.3), 0.25)
    desired_flange = mount.flange_for_hoop(desired_hoop)
    np.testing.assert_allclose(
        desired_flange @ mount.flange_to_hoop_center(),
        desired_hoop,
        atol=1e-12,
    )


def test_adapter_mount_copies_supplied_matrices():
    flange_adapter = Pose(np.eye(4))
    adapter_hoop_cad = Pose(np.eye(4))
    mount = AdapterMount(flange_adapter, adapter_hoop_cad)
    flange_adapter[0, 3] = 100.0
    adapter_hoop_cad[1, 3] = 100.0
    np.testing.assert_allclose(mount.flange_to_adapter, np.eye(4))
    np.testing.assert_allclose(mount.adapter_to_hoop_cad, np.eye(4))


@pytest.mark.parametrize("index", [0, 1])
def test_adapter_mount_rejects_invalid_calibration(index):
    poses = [Pose(np.eye(4)), Pose(np.eye(4))]
    poses[index] = Pose(np.diag([2.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="Pose"):
        AdapterMount(*poses)
