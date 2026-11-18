import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hoop_guide.geometry import angles_from_rotation
from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.geometry import rotation_about_axis
from hoop_guide.geometry import rotation_from_angles
from hoop_guide.types import RotationAngles
from hoop_guide.types import RotationMatrix

X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)
Z_AXIS = (0.0, 0.0, 1.0)
QUARTER_TURN = math.pi / 2

angles = st.floats(-6, 6, allow_nan=False, allow_infinity=False)
# Angles about y where the rotated x axis is vertical.
quarter_turns_about_y = st.sampled_from([-QUARTER_TURN, QUARTER_TURN])


def to_rotation(angle_x, angle_y, angle_z):
    return rotation_from_angles(
        RotationAngles(np.array([angle_x, angle_y, angle_z]))
    )


def rotate(x, y, angle):
    cosine, sine = get_cosine_and_sine(angle)
    return (cosine * x - sine * y, sine * x + cosine * y)


def test_zero_angle_leaves_a_point_in_place():
    assert rotate(0.3, -0.4, 0.0) == pytest.approx((0.3, -0.4))


def test_quarter_turn_maps_x_axis_onto_y_axis():
    assert rotate(1.0, 0.0, math.pi / 2) == pytest.approx((0.0, 1.0))


@given(
    x=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
    y=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
    angle=st.floats(-6, 6, allow_nan=False, allow_infinity=False),
)
def test_opposite_rotation_returns_the_point(x, y, angle):
    rotated_x, rotated_y = rotate(x, y, angle)

    assert rotate(rotated_x, rotated_y, -angle) == pytest.approx((x, y))


# rotation_about_axis


@pytest.mark.parametrize(
    ("axis_index", "axis", "expected_axis"),
    [
        (0, Y_AXIS, Z_AXIS),
        (1, Z_AXIS, X_AXIS),
        (2, X_AXIS, Y_AXIS),
    ],
    ids=["about_x", "about_y", "about_z"],
)
def test_quarter_turn_about_an_axis_maps_the_next_axis_onto_the_one_after(
    axis_index, axis, expected_axis
):
    rotation = rotation_about_axis(axis_index, QUARTER_TURN)

    np.testing.assert_allclose(rotation @ axis, expected_axis, atol=1e-12)


@pytest.mark.parametrize("axis_index", [0, 1, 2], ids=["x", "y", "z"])
def test_rotation_about_an_axis_leaves_that_axis_in_place(axis_index):
    axis = np.eye(3)[axis_index]

    rotation = rotation_about_axis(axis_index, 0.7)

    np.testing.assert_allclose(rotation @ axis, axis, atol=1e-12)


@pytest.mark.parametrize("axis_index", [-1, 3], ids=["negative", "too_large"])
def test_rotation_about_an_unknown_axis_is_refused(axis_index):
    with pytest.raises(ValueError, match="Axis index"):
        rotation_about_axis(axis_index, 0.1)


# rotation_from_angles


@pytest.mark.parametrize(
    ("angle_y", "expected_z_axis"),
    [
        (0.0, (0.0, 0.0, 1.0)),
        (QUARTER_TURN, (1.0, 0.0, 0.0)),
        (math.pi, (0.0, 0.0, -1.0)),
        (-QUARTER_TURN, (-1.0, 0.0, 0.0)),
    ],
    ids=["up", "forward", "down", "backward"],
)
def test_angle_about_y_tilts_the_z_axis_in_the_xz_plane(
    angle_y, expected_z_axis
):
    rotation = to_rotation(0.0, angle_y, 0.0)

    np.testing.assert_allclose(rotation @ Z_AXIS, expected_z_axis, atol=1e-12)


def test_rotation_about_x_is_applied_before_the_one_about_z():
    rotation = to_rotation(QUARTER_TURN, 0.0, QUARTER_TURN)

    # About x: y goes to z. About z: z stays.
    np.testing.assert_allclose(rotation @ Y_AXIS, Z_AXIS, atol=1e-12)


def test_rotation_from_angles_of_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        rotation_from_angles(RotationAngles(np.zeros(2)))


@given(angle_x=angles, angle_y=angles, angle_z=angles)
def test_rotation_from_angles_is_orthonormal_and_keeps_handedness(
    angle_x, angle_y, angle_z
):
    rotation = to_rotation(angle_x, angle_y, angle_z)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


# angles_from_rotation


@pytest.mark.parametrize(
    "rotation_angles",
    [
        (0.0, 0.0, 0.0),
        (0.2, -0.5, 1.1),
        (-3.0, 1.2, 2.5),
        (0.0, QUARTER_TURN, 0.0),
        (0.0, -QUARTER_TURN, 0.8),
    ],
    ids=["identity", "general", "large", "home", "quarter_turn_down"],
)
def test_angles_from_rotation_returns_the_angles_of_the_rotation(
    rotation_angles,
):
    rotation = to_rotation(*rotation_angles)

    np.testing.assert_allclose(
        angles_from_rotation(rotation), rotation_angles, atol=1e-12
    )


@given(angle_x=angles, angle_y=angles, angle_z=angles)
def test_angles_from_rotation_give_the_same_rotation(angle_x, angle_y, angle_z):
    rotation = to_rotation(angle_x, angle_y, angle_z)

    restored_rotation = rotation_from_angles(angles_from_rotation(rotation))

    np.testing.assert_allclose(restored_rotation, rotation, atol=1e-6)


@given(angle_x=angles, angle_y=quarter_turns_about_y, angle_z=angles)
def test_angles_from_rotation_in_gimbal_lock_give_the_same_rotation(
    angle_x, angle_y, angle_z
):
    rotation = to_rotation(angle_x, angle_y, angle_z)

    rotation_angles = angles_from_rotation(rotation)

    assert rotation_angles[0] == pytest.approx(0.0)
    np.testing.assert_allclose(
        rotation_from_angles(rotation_angles), rotation, atol=1e-6
    )


def test_angles_from_rotation_of_a_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        angles_from_rotation(RotationMatrix(np.eye(4)))
