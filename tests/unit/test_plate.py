import math

import numpy as np
import pytest

from hoop_guide.plate import MARKER_CENTERS
from hoop_guide.plate import estimate_plate_pose
from hoop_guide.types import Homography
from hoop_guide.types import PixelPoints
from hoop_guide.vision import MarkerDetection

IDENTITY = Homography(np.eye(3))
PLATE = (1.0, 2.0)


def marker(marker_id, corners):
    return MarkerDetection(marker_id, PixelPoints(np.array(corners, float)))


def test_one_aligned_marker_gives_plate_center_and_zero_heading():
    # Marker 0 sits up-left of the plate center, so the plate center is
    # down-right of the marker center (0.9625, 2.0375).
    detected = marker(
        0,
        [
            (0.9425, 2.0575),
            (0.9825, 2.0575),
            (0.9825, 2.0175),
            (0.9425, 2.0175),
        ],
    )

    pose = estimate_plate_pose([detected], IDENTITY)

    assert pose is not None
    assert pose.x == pytest.approx(PLATE[0])
    assert pose.y == pytest.approx(PLATE[1])
    assert pose.heading == pytest.approx(0.0)


def test_marker_on_the_other_corner_gives_the_same_plate_pose():
    detected = marker(
        1,
        [
            (1.0175, 1.9825),
            (1.0575, 1.9825),
            (1.0575, 1.9425),
            (1.0175, 1.9425),
        ],
    )

    pose = estimate_plate_pose([detected], IDENTITY)

    assert pose is not None
    assert (pose.x, pose.y, pose.heading) == pytest.approx((*PLATE, 0.0))


def test_quarter_turn_maps_plate_x_axis_onto_robot_y_axis():
    # Marker center (0.9625, 1.9625), square rotated by a quarter turn.
    detected = marker(
        0,
        [
            (0.9425, 1.9425),
            (0.9425, 1.9825),
            (0.9825, 1.9825),
            (0.9825, 1.9425),
        ],
    )

    pose = estimate_plate_pose([detected], IDENTITY)

    assert pose is not None
    assert (pose.x, pose.y) == pytest.approx(PLATE)
    assert pose.heading == pytest.approx(math.pi / 2)


def test_heading_uses_both_marker_edges():
    corners = np.array(
        [
            [0.0, 0.02],
            [0.02, 0.03],
            [0.02, -0.02],
            [0.0, -0.02],
        ]
    )
    top_left, top_right, bottom_right, bottom_left = corners
    x_axis = (top_right - top_left) + (bottom_right - bottom_left)
    heading = math.atan2(x_axis[1], x_axis[0])
    center = corners.mean(axis=0)
    offset_x, offset_y = MARKER_CENTERS[0]
    cos, sin = math.cos(heading), math.sin(heading)

    pose = estimate_plate_pose([marker(0, corners)], IDENTITY)

    assert pose is not None
    assert pose.heading == pytest.approx(heading)
    assert pose.x == pytest.approx(
        center[0] - (cos * offset_x - sin * offset_y)
    )
    assert pose.y == pytest.approx(
        center[1] - (sin * offset_x + cos * offset_y)
    )


def test_homography_scales_pixel_corners_into_meters():
    detected = marker(
        0,
        [
            (942.5, 2057.5),
            (982.5, 2057.5),
            (982.5, 2017.5),
            (942.5, 2017.5),
        ],
    )
    millimeters = Homography(np.diag([0.001, 0.001, 1.0]))

    pose = estimate_plate_pose([detected], millimeters)

    assert pose is not None
    assert (pose.x, pose.y, pose.heading) == pytest.approx((*PLATE, 0.0))


def test_two_markers_agree_on_the_plate_pose():
    markers = [
        marker(
            0,
            [
                (0.9425, 2.0575),
                (0.9825, 2.0575),
                (0.9825, 2.0175),
                (0.9425, 2.0175),
            ],
        ),
        marker(
            1,
            [
                (1.0175, 1.9825),
                (1.0575, 1.9825),
                (1.0575, 1.9425),
                (1.0175, 1.9425),
            ],
        ),
    ]

    pose = estimate_plate_pose(markers, IDENTITY)

    assert pose is not None
    assert (pose.x, pose.y, pose.heading) == pytest.approx((*PLATE, 0.0))


def test_unknown_marker_is_ignored():
    known = marker(
        0,
        [
            (0.9425, 2.0575),
            (0.9825, 2.0575),
            (0.9825, 2.0175),
            (0.9425, 2.0175),
        ],
    )
    unknown = marker(7, [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)])

    pose = estimate_plate_pose([unknown, known], IDENTITY)

    assert pose is not None
    assert (pose.x, pose.y, pose.heading) == pytest.approx((*PLATE, 0.0))


@pytest.mark.parametrize(
    "markers",
    [[], [marker(7, [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)])]],
    ids=["no-markers", "unknown-only"],
)
def test_no_known_marker_returns_none(markers):
    assert estimate_plate_pose(markers, IDENTITY) is None
