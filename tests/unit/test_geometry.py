import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hoop_guide.geometry import get_cosine_and_sine


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
