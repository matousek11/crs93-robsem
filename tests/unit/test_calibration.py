import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from hoop_guide.calibration import compute_reprojection_error
from hoop_guide.calibration import estimate_homography
from hoop_guide.calibration import load_homography
from hoop_guide.calibration import save_homography
from hoop_guide.types import Homography
from hoop_guide.types import PixelPoints
from hoop_guide.types import PlanePoints


def _sample_points() -> tuple[PixelPoints, PlanePoints, Homography]:
    """Generates synthetic correspondences from a known affine transform."""
    image_points = PixelPoints(
        np.array(
            [
                [100.0, 100.0],
                [500.0, 100.0],
                [500.0, 400.0],
                [100.0, 400.0],
                [300.0, 250.0],
            ],
            dtype=np.float64,
        )
    )
    # Scale px -> m and shift.
    true_matrix = np.array(
        [[0.001, 0.0, 0.2], [0.0, 0.001, -0.1], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), true_matrix
    ).reshape(-1, 2)
    plane_points = PlanePoints(np.asarray(transformed, dtype=np.float64))
    return image_points, plane_points, Homography(true_matrix)


def test_estimate_homography_reconstructs_known_transform():
    image_points, plane_points, true_homography = _sample_points()

    estimated = estimate_homography(image_points, plane_points)

    np.testing.assert_allclose(estimated, true_homography, atol=1e-9)


def test_estimate_homography_normalizes_bottom_right_entry():
    image_points, plane_points, _ = _sample_points()

    estimated = estimate_homography(image_points, plane_points)

    assert estimated[2, 2] == pytest.approx(1.0)


def test_estimate_homography_rejects_fewer_than_four_points():
    image_points = PixelPoints(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    plane_points = PlanePoints(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))

    with pytest.raises(ValueError, match="At least 4"):
        estimate_homography(image_points, plane_points)


def test_estimate_homography_rejects_mismatched_lengths():
    image_points = PixelPoints(np.zeros((5, 2)))
    plane_points = PlanePoints(np.zeros((4, 2)))

    with pytest.raises(ValueError, match="same length"):
        estimate_homography(image_points, plane_points)


def test_estimate_homography_rejects_bad_dimensions():
    bad_image = PixelPoints(np.zeros((4, 3)))
    valid_image = PixelPoints(np.zeros((4, 2)))
    bad_plane = PlanePoints(np.zeros((4, 3)))
    valid_plane = PlanePoints(np.zeros((4, 2)))

    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        estimate_homography(bad_image, valid_plane)

    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        estimate_homography(valid_image, bad_plane)


def test_estimate_homography_rejects_non_finite_values():
    image_points, plane_points, _ = _sample_points()
    broken_image = np.array(image_points)
    broken_image[0, 0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        estimate_homography(PixelPoints(broken_image), plane_points)

    broken_plane = np.array(plane_points)
    broken_plane[0, 0] = math.inf

    with pytest.raises(ValueError, match="finite"):
        estimate_homography(image_points, PlanePoints(broken_plane))


def test_estimate_homography_rejects_collinear_points():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    image_points = PixelPoints(collinear)
    plane_points = PlanePoints(collinear * 0.1)

    with pytest.raises(ValueError, match="failed"):
        estimate_homography(image_points, plane_points)


@given(
    scale=st.floats(min_value=0.0001, max_value=0.01),
    offset_x=st.floats(min_value=-1.0, max_value=1.0),
    offset_y=st.floats(min_value=-1.0, max_value=1.0),
)
@settings(max_examples=25)
def test_estimate_homography_hypothesis(
    scale: float, offset_x: float, offset_y: float
):
    image_points = PixelPoints(
        np.array(
            [
                [50.0, 50.0],
                [800.0, 50.0],
                [800.0, 600.0],
                [50.0, 600.0],
            ],
            dtype=np.float64,
        )
    )
    true_matrix = np.array(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), true_matrix
    ).reshape(-1, 2)
    plane_points = PlanePoints(np.asarray(transformed, dtype=np.float64))

    estimated = estimate_homography(image_points, plane_points)

    np.testing.assert_allclose(estimated, true_matrix, atol=1e-7)


def test_compute_reprojection_error_exact_mapping_is_zero():
    image_points, plane_points, homography = _sample_points()

    error = compute_reprojection_error(image_points, plane_points, homography)

    assert error == pytest.approx(0.0, abs=1e-9)


def test_compute_reprojection_error_known_shift():
    image_points, plane_points, homography = _sample_points()
    shift_meters = 0.005
    shifted_plane = PlanePoints(np.array(plane_points) + [shift_meters, 0.0])

    error = compute_reprojection_error(image_points, shifted_plane, homography)

    assert error == pytest.approx(shift_meters, rel=1e-4)


def test_compute_reprojection_error_rejects_empty_collection():
    empty_image = PixelPoints(np.zeros((0, 2)))
    empty_plane = PlanePoints(np.zeros((0, 2)))
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match="at least one point"):
        compute_reprojection_error(empty_image, empty_plane, homography)


def test_compute_reprojection_error_rejects_mismatched_lengths():
    image_points = PixelPoints(np.zeros((4, 2)))
    plane_points = PlanePoints(np.zeros((3, 2)))
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match="same length"):
        compute_reprojection_error(image_points, plane_points, homography)


def test_compute_reprojection_error_rejects_bad_plane_shape():
    image_points = PixelPoints(np.zeros((4, 2)))
    plane_points = PlanePoints(np.zeros((4, 3)))
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        compute_reprojection_error(image_points, plane_points, homography)


def test_compute_reprojection_error_rejects_non_finite_plane_points():
    image_points = PixelPoints(np.zeros((4, 2)))
    broken_plane = np.zeros((4, 2))
    broken_plane[0, 0] = math.nan
    homography = Homography(np.eye(3))

    with pytest.raises(ValueError, match="finite"):
        compute_reprojection_error(
            image_points, PlanePoints(broken_plane), homography
        )


def test_save_and_load_homography_roundtrip(tmp_path: Path):
    file_path = tmp_path / "subdir" / "homography.json"
    homography = Homography(
        np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]])
    )

    save_homography(file_path, homography)
    loaded = load_homography(file_path)

    np.testing.assert_allclose(loaded, homography)


def test_save_homography_rejects_bad_shape():
    broken = Homography(np.eye(2))

    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        save_homography("dummy.json", broken)


def test_save_homography_rejects_non_finite_values():
    broken = np.eye(3)
    broken[0, 0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        save_homography("dummy.json", Homography(broken))


def test_load_homography_file_not_found(tmp_path: Path):
    non_existent = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_homography(non_existent)


def test_load_homography_invalid_json(tmp_path: Path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("not json content", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_homography(file_path)


def test_load_homography_missing_key(tmp_path: Path):
    file_path = tmp_path / "no_key.json"
    file_path.write_text(json.dumps({"other": [1, 2, 3]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing 'homography' key"):
        load_homography(file_path)


def test_load_homography_bad_matrix_shape(tmp_path: Path):
    file_path = tmp_path / "bad_shape.json"
    file_path.write_text(
        json.dumps({"homography": [[1.0, 0.0], [0.0, 1.0]]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Expected homography shape"):
        load_homography(file_path)


def test_load_homography_non_finite_values(tmp_path: Path):
    file_path = tmp_path / "non_finite.json"
    file_path.write_text(
        json.dumps(
            {"homography": [[None, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        load_homography(file_path)
