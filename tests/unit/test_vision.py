import math

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.vision import ARUCO_DICTIONARY
from hoop_guide.vision import detect_hoop_center
from hoop_guide.vision import detect_markers
from hoop_guide.vision import image_to_plane
from hoop_guide.vision import process_image

CANVAS_SIZE = 400
MARKER_SIZE = 100
# Detected corners are compared with the generated ones.
CORNER_TOLERANCE = 1.0  # [px]


def blank_image():
    return np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)


def add_marker(image, marker_id, left, top):
    """Pastes a marker and returns its outer corners in ArUco order."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    marker = dictionary.generateImageMarker(marker_id, MARKER_SIZE)
    image[top : top + MARKER_SIZE, left : left + MARKER_SIZE] = marker[
        ..., np.newaxis
    ]
    # Corners lie on pixel edges, half a pixel from the pixel centers.
    x0, y0 = left - 0.5, top - 0.5
    x1, y1 = x0 + MARKER_SIZE, y0 + MARKER_SIZE
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def rotate(image, corners, angle):
    """Rotates the image and corners around the image center [deg]."""
    center = (CANVAS_SIZE / 2 - 0.5, CANVAS_SIZE / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (CANVAS_SIZE, CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    rotated_corners = corners @ matrix[:, :2].T + matrix[:, 2]
    return rotated, rotated_corners


def test_marker_is_found_with_id_and_corners():
    image = blank_image()
    corners = add_marker(image, 7, left=150, top=120)

    markers = detect_markers(image)

    assert [m.marker_id for m in markers] == [7]
    np.testing.assert_allclose(
        markers[0].corners, corners, atol=CORNER_TOLERANCE
    )


def test_two_markers_are_found():
    image = blank_image()
    add_marker(image, 0, left=30, top=30)
    add_marker(image, 1, left=250, top=250)

    markers = detect_markers(image)

    assert sorted(m.marker_id for m in markers) == [0, 1]


def test_image_without_markers_gives_no_detections():
    assert detect_markers(blank_image()) == ()


@pytest.mark.parametrize("angle", [30.0, 90.0, 180.0], ids=["30", "90", "180"])
def test_corners_rotate_with_marker(angle):
    image = blank_image()
    corners = add_marker(image, 3, left=150, top=150)
    image, corners = rotate(image, corners, angle)

    markers = detect_markers(image)

    np.testing.assert_allclose(
        markers[0].corners, corners, atol=CORNER_TOLERANCE
    )


@settings(max_examples=25, deadline=None)
@given(
    marker_id=st.integers(min_value=0, max_value=49),
    angle=st.floats(min_value=-180.0, max_value=180.0),
)
def test_detected_corners_match_generated_marker(marker_id, angle):
    image = blank_image()
    corners = add_marker(image, marker_id, left=150, top=150)
    image, corners = rotate(image, corners, angle)

    markers = detect_markers(image)

    assert [m.marker_id for m in markers] == [marker_id]
    np.testing.assert_allclose(
        markers[0].corners, corners, atol=CORNER_TOLERANCE
    )


def test_image_to_plane_applies_the_homography_in_order():
    points = PixelPoints(np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 500.0]]))
    homography = Homography(np.diag([0.001, 0.002, 1.0]))

    plane = image_to_plane(points, homography)

    np.testing.assert_allclose(plane, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def test_image_to_plane_of_no_points_is_empty():
    plane = image_to_plane(PixelPoints(np.zeros((0, 2))), Homography(np.eye(3)))

    assert plane.shape == (0, 2)


def test_image_to_plane_rejects_a_bad_point_shape():
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        image_to_plane(PixelPoints(np.zeros(4)), homography)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        image_to_plane(PixelPoints(np.zeros((2, 3))), homography)


def test_image_to_plane_rejects_a_bad_homography_shape():
    points = PixelPoints(np.zeros((1, 2)))

    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        image_to_plane(points, Homography(np.eye(2)))


def test_image_to_plane_rejects_non_finite_values():
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match="finite"):
        image_to_plane(PixelPoints(np.array([[math.nan, 0.0]])), homography)

    broken = np.eye(3)
    broken[0, 0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        image_to_plane(PixelPoints(np.zeros((1, 2))), Homography(broken))


def test_process_image_keeps_image_and_timestamp_and_adds_markers():
    image = blank_image()
    add_marker(image, 5, left=150, top=150)

    frame = process_image(image, timestamp=12.5)

    assert frame.image is image
    assert frame.timestamp == pytest.approx(12.5)
    assert [m.marker_id for m in frame.markers] == [5]


def test_process_image_adds_the_detected_hoop_center():
    image = blank_image()
    cv2.circle(image, (180, 220), 40, (0, 0, 0), thickness=10)

    frame = process_image(image, timestamp=0.0)

    assert frame.hoop_center == pytest.approx((180.0, 220.0), abs=1.0)


def test_process_image_without_a_hoop_has_no_hoop_center():
    frame = process_image(blank_image(), timestamp=0.0)

    assert frame.hoop_center is None


def test_detect_hoop_center_finds_center_accurately():
    image = Image(blank_image())
    cv2.circle(image, (180, 220), 40, (0, 0, 0), thickness=10)

    center = detect_hoop_center(image)

    assert center is not None
    assert center[0] == pytest.approx(180.0, abs=1.0)
    assert center[1] == pytest.approx(220.0, abs=1.0)


def test_detect_hoop_center_ignores_small_and_angular_shapes():
    image = Image(blank_image())
    cv2.circle(image, (50, 50), 5, (0, 0, 0), thickness=-1)
    cv2.rectangle(image, (250, 60), (270, 80), (0, 0, 0), thickness=-1)
    cv2.circle(image, (160, 200), 45, (0, 0, 0), thickness=10)

    center = detect_hoop_center(image)

    assert center == pytest.approx((160.0, 200.0), abs=1.0)


def test_detect_hoop_center_prefers_the_hoop_to_a_larger_dark_gripper():
    # Reproduces the largest dark contour winning over the hoop.
    image = Image(blank_image())
    cv2.rectangle(image, (120, 10), (330, 150), (30, 30, 30), thickness=-1)
    cv2.circle(image, (200, 280), 50, (0, 0, 0), thickness=3)

    center = detect_hoop_center(image)

    assert center == pytest.approx((200.0, 280.0), abs=1.0)


def test_detect_hoop_center_finds_a_thin_light_hoop_under_uneven_light():
    # Brightness falls from 250 to 110 across the image and the hoop line is
    # only 40 gray levels darker than the table around it, so no single
    # threshold separates it.
    brightness = np.linspace(250.0, 110.0, CANVAS_SIZE)
    table = np.tile(brightness, (CANVAS_SIZE, 1))
    hoop_line = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    cv2.circle(hoop_line, (220, 190), 60, 255, thickness=3)
    gray = np.where(hoop_line > 0, table - 40.0, table).astype(np.uint8)
    image = Image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.uint8))

    center = detect_hoop_center(image)

    assert center == pytest.approx((220.0, 190.0), abs=1.0)


def test_detect_hoop_center_is_not_shifted_by_the_handle_joint():
    # A bump on one side moves the smallest enclosing circle by 4 px.
    image = Image(blank_image())
    cv2.circle(image, (200, 220), 60, (0, 0, 0), thickness=3)
    cv2.circle(image, (200, 156), 6, (0, 0, 0), thickness=-1)

    center = detect_hoop_center(image)

    assert center == pytest.approx((200.0, 220.0), abs=1.0)


def test_detect_hoop_center_returns_none_for_an_open_arc():
    image = Image(blank_image())
    cv2.ellipse(image, (200, 200), (80, 80), 0, 0, 120, (0, 0, 0), thickness=3)

    assert detect_hoop_center(image) is None


def test_detect_hoop_center_returns_none_for_a_marker_only():
    image = blank_image()
    add_marker(image, 5, left=150, top=150)

    assert detect_hoop_center(Image(image)) is None


def test_detect_hoop_center_returns_none_on_blank_image():
    image = Image(blank_image())

    assert detect_hoop_center(image) is None


def test_detect_hoop_center_returns_none_on_uniform_black_image():
    black_image = Image(np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8))

    assert detect_hoop_center(black_image) is None


def test_detect_hoop_center_returns_none_on_small_noise_only():
    image = Image(blank_image())
    cv2.circle(image, (100, 100), 3, (0, 0, 0), thickness=-1)

    assert detect_hoop_center(image) is None


def test_detect_hoop_center_rejects_invalid_image_shapes():
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        detect_hoop_center(Image(np.zeros((100, 100), dtype=np.uint8)))

    with pytest.raises(ValueError, match=r"non-zero"):
        detect_hoop_center(Image(np.zeros((0, 100, 3), dtype=np.uint8)))
