import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.geometry import rotation_about_axis

X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)
Z_AXIS = (0.0, 0.0, 1.0)
QUARTER_TURN = math.pi / 2

angles = st.floats(-6, 6, allow_nan=False, allow_infinity=False)


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


@given(axis_index=st.sampled_from([0, 1, 2]), angle=angles)
def test_rotation_about_an_axis_is_orthonormal_and_keeps_handedness(
    axis_index, angle
):
    rotation = rotation_about_axis(axis_index, angle)

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@given(axis_index=st.sampled_from([0, 1, 2]), angle=angles)
def test_opposite_rotation_about_an_axis_returns_the_orientation(
    axis_index, angle
):
    rotation = rotation_about_axis(axis_index, angle)
    opposite_rotation = rotation_about_axis(axis_index, -angle)

    np.testing.assert_allclose(
        opposite_rotation @ rotation, np.eye(3), atol=1e-12
    )
