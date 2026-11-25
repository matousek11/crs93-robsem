import math

import numpy as np
import pytest

from hoop_guide.plate import PlatePose
from hoop_guide.run_check import get_hoop_offset
from hoop_guide.run_check import get_plate_error
from hoop_guide.types import Pose

ROD_AT = (0.5, 0.0, 0.2)  # [m]
ROD_MOVES_TO_Y = (0.0, 0.02, 0.0)  # [m/s]


def pose_at(position):
    """Pose with the z axis down, along a vertical rod."""
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = position
    return Pose(pose)


@pytest.mark.parametrize(
    ("hoop_at", "expected_offset", "expected_ahead"),
    [
        ((0.5, 0.004, 0.2), 0.004, 0.004),
        ((0.5, -0.003, 0.2), 0.003, -0.003),
        ((0.503, 0.0, 0.2), 0.003, 0.0),
        ((0.5, 0.004, 0.25), 0.004, 0.004),
    ],
    ids=["ahead", "behind", "beside", "higher_on_the_rod"],
)
def test_hoop_offset_is_across_the_rod_and_signed_by_its_motion(
    hoop_at, expected_offset, expected_ahead
):
    offset, ahead = get_hoop_offset(
        pose_at(hoop_at), pose_at(ROD_AT), ROD_MOVES_TO_Y
    )

    assert offset == pytest.approx(expected_offset)
    assert ahead == pytest.approx(expected_ahead)


def test_hoop_at_a_standing_rod_is_not_ahead():
    _, ahead = get_hoop_offset(
        pose_at((0.5, 0.004, 0.2)), pose_at(ROD_AT), (0.0, 0.0, 0.0)
    )

    assert ahead == pytest.approx(0.0)


def test_plate_error_is_the_distance_and_the_heading_difference():
    distance, heading = get_plate_error(
        PlatePose(0.503, 0.004, 0.1), PlatePose(0.5, 0.0, 0.0)
    )

    assert distance == pytest.approx(0.005)
    assert heading == pytest.approx(0.1)


def test_plate_heading_error_is_the_short_way_round():
    _, heading = get_plate_error(
        PlatePose(0.5, 0.0, math.pi - 0.1), PlatePose(0.5, 0.0, -math.pi + 0.1)
    )

    assert heading == pytest.approx(-0.2)
