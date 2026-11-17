import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hoop_guide.models.hoop_mount import HoopMount

HOOP_DISTANCE = 0.2  # along the flange z axis [m]


def make_pose(x, y, z, angle_about_y=0.0, angle_about_z=0.0):
    cos_y, sin_y = math.cos(angle_about_y), math.sin(angle_about_y)
    cos_z, sin_z = math.cos(angle_about_z), math.sin(angle_about_z)
    rotation_about_y = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]]
    )
    rotation_about_z = np.array(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]]
    )
    pose = np.eye(4)
    pose[:3, :3] = rotation_about_z @ rotation_about_y
    pose[:3, 3] = [x, y, z]
    return pose


@pytest.fixture
def hoop_mount():
    return HoopMount(make_pose(0.0, 0.0, HOOP_DISTANCE))


def test_hoop_of_unrotated_flange_is_shifted_along_robot_z(hoop_mount):
    flange_pose = make_pose(0.4, 0.1, 0.3)

    hoop_pose = hoop_mount.to_hoop_pose(flange_pose)

    np.testing.assert_allclose(
        hoop_pose, make_pose(0.4, 0.1, 0.3 + HOOP_DISTANCE), atol=1e-12
    )


def test_hoop_offset_turns_with_the_flange(hoop_mount):
    # Flange z axis points along robot x after a quarter turn about y.
    flange_pose = make_pose(0.4, 0.1, 0.3, angle_about_y=math.pi / 2)

    hoop_pose = hoop_mount.to_hoop_pose(flange_pose)

    np.testing.assert_allclose(
        hoop_pose[:3, 3], [0.4 + HOOP_DISTANCE, 0.1, 0.3], atol=1e-12
    )


def test_flange_of_unrotated_hoop_is_shifted_back_along_robot_z(hoop_mount):
    hoop_pose = make_pose(0.4, 0.1, 0.3)

    flange_pose = hoop_mount.to_flange_pose(hoop_pose)

    np.testing.assert_allclose(
        flange_pose, make_pose(0.4, 0.1, 0.3 - HOOP_DISTANCE), atol=1e-12
    )


coordinates = st.floats(-1.0, 1.0)
angles = st.floats(-math.pi, math.pi)
poses = st.builds(
    make_pose, coordinates, coordinates, coordinates, angles, angles
)


@given(flange_pose=poses, flange_to_hoop=poses)
def test_flange_pose_of_hoop_pose_is_the_original_flange_pose(
    flange_pose, flange_to_hoop
):
    hoop_mount = HoopMount(flange_to_hoop)

    result = hoop_mount.to_flange_pose(hoop_mount.to_hoop_pose(flange_pose))

    np.testing.assert_allclose(result, flange_pose, atol=1e-9)


@pytest.mark.parametrize(
    ("flange_to_hoop", "message"),
    [
        (np.eye(3), "shape"),
        (np.full((4, 4), np.nan), "finite"),
        (np.zeros((4, 4)), "invertible"),
    ],
    ids=["wrong_shape", "not_finite", "singular"],
)
def test_invalid_flange_to_hoop_raises(flange_to_hoop, message):
    with pytest.raises(ValueError, match=message):
        HoopMount(flange_to_hoop)
