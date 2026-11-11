import math

import numpy as np
import pytest

from hoop_guide.estimation import TableMotion
from hoop_guide.plate import PlatePose
from hoop_guide.types import Homography
from hoop_guide.types import PixelPoints
from hoop_guide.viewer.overlay import AXIS_X_COLOR
from hoop_guide.viewer.overlay import AXIS_Y_COLOR
from hoop_guide.viewer.overlay import MARKER_COLOR
from hoop_guide.viewer.overlay import TEXT_ORIGIN
from hoop_guide.viewer.overlay import draw_markers
from hoop_guide.viewer.overlay import draw_plate_pose
from hoop_guide.viewer.overlay import render
from hoop_guide.viewer.overlay import status_line
from hoop_guide.vision import Frame
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
    ("heading", "x_axis_pixel", "y_axis_pixel"),
    [(0.0, (125, 100), (100, 125)), (math.pi / 2, (100, 125), (75, 100))],
    ids=["aligned", "quarter-turn"],
)
def test_draw_plate_pose_draws_axes_from_plate_center(
    image, heading, x_axis_pixel, y_axis_pixel
):
    pose = PlatePose(x=0.1, y=0.1, heading=heading)

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


@pytest.mark.parametrize(
    ("marker_ids", "fps", "is_rotating_leftwise", "expected"),
    [
        ([1, 0], 9.96, None, "10.0 fps | markers: 0, 1 | rotation: --"),
        ([], 10.0, True, "10.0 fps | markers: none | rotation: Left"),
        ([3], None, False, "-- fps | markers: 3 | rotation: Right"),
    ],
    ids=["sorted-ids-no-rotation", "no-markers-left", "unknown-fps-right"],
)
def test_status_line_shows_fps_and_marker_ids(
    image, marker_ids, fps, is_rotating_leftwise, expected
):
    frame = Frame(image, 0.0, tuple(marker(i, 0, 0, 10) for i in marker_ids))

    assert status_line(frame, fps, is_rotating_leftwise) == expected


def test_render_draws_markers_and_status_line(image):
    frame = Frame(image, 0.0, (marker(7, 50, 100, 80),))

    result = render(frame, 10.0)

    assert color_at(result, 90, 180) == MARKER_COLOR
    assert result[: TEXT_ORIGIN[1], : TEXT_ORIGIN[1] * 3].any()


def test_render_draws_plate_pose_when_provided(image):
    frame = Frame(image, 0.0, ())
    pose = PlatePose(x=0.1, y=0.1, heading=0.0)

    result = render(frame, 10.0, plate_pose=pose, homography=MM_PER_PIXEL)

    assert color_at(result, 125, 100) == AXIS_X_COLOR
    assert color_at(result, 100, 125) == AXIS_Y_COLOR


def test_render_includes_rotation_from_table_estimation(image):
    frame = Frame(image, 0.0, ())
    motion = TableMotion(0.0, 0.0, 0.5, 0.0, PlatePose(0.0, 0.0, 0.0))

    result = render(frame, 10.0, table_estimation=motion)

    assert result[: TEXT_ORIGIN[1], : TEXT_ORIGIN[1] * 3].any()
