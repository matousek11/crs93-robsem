import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from hoop_guide.vision import ARUCO_DICTIONARY
from hoop_guide.vision import detect_markers
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


def test_process_image_keeps_image_and_timestamp_and_adds_markers():
    image = blank_image()
    add_marker(image, 5, left=150, top=150)

    frame = process_image(image, timestamp=12.5)

    assert frame.image is image
    assert frame.timestamp == pytest.approx(12.5)
    assert [m.marker_id for m in frame.markers] == [5]
