import math

import numpy as np
import pytest

from hoop_guide.plate import PlatePose
from hoop_guide.types import Homography
from hoop_guide.types import PixelPoints
from hoop_guide.viewer.overlay import AXIS_X_COLOR
from hoop_guide.viewer.overlay import AXIS_Y_COLOR
from hoop_guide.viewer.overlay import MARKER_COLOR
from hoop_guide.viewer.overlay import draw_markers
from hoop_guide.viewer.overlay import draw_plate_pose
from hoop_guide.vision import MarkerDetection

# One pixel is one millimeter on the table plane.
MM_PER_PIXEL = Homography(np.diag([0.001, 0.001, 1.0]))


@pytest.fixture
def image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def marker(marker_id, left, top, size):
    corners = [
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    ]
    return MarkerDetection(marker_id, PixelPoints(np.array(corners, float)))


def color_at(image, x, y):
    return tuple(int(c) for c in image[y, x])


def test_draw_markers_outlines_marker(image):
    result = draw_markers(image, [marker(7, 50, 50, 80)])

    assert color_at(result, 90, 50) == MARKER_COLOR
    assert color_at(result, 90, 130) == MARKER_COLOR
    assert color_at(result, 90, 90) == (0, 0, 0)


def test_draw_markers_without_markers_returns_same_content(image):
    result = draw_markers(image, [])

    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize(
    ("theta", "x_axis_pixel", "y_axis_pixel"),
    [(0.0, (125, 100), (100, 125)), (math.pi / 2, (100, 125), (75, 100))],
    ids=["aligned", "quarter-turn"],
)
def test_draw_plate_pose_draws_axes_from_plate_center(
    image, theta, x_axis_pixel, y_axis_pixel
):
    pose = PlatePose(x=0.1, y=0.1, theta=theta)

    result = draw_plate_pose(image, pose, MM_PER_PIXEL)

    assert color_at(result, *x_axis_pixel) == AXIS_X_COLOR
    assert color_at(result, *y_axis_pixel) == AXIS_Y_COLOR


@pytest.mark.parametrize(
    "draw",
    [
        lambda image: draw_markers(image, [marker(1, 10, 10, 50)]),
        lambda image: draw_plate_pose(
            image, PlatePose(0.1, 0.1, 0.0), MM_PER_PIXEL
        ),
    ],
    ids=["markers", "plate-pose"],
)
def test_drawing_leaves_input_image_unchanged(image, draw):
    original = image.copy()

    draw(image)

    np.testing.assert_array_equal(image, original)
