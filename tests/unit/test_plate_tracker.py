import math

import numpy as np
import pytest

from hoop_guide.plate import MARKER_CENTERS
from hoop_guide.plate import MARKER_SIZE
from hoop_guide.plate import PlatePose
from hoop_guide.plate_tracker import PLATE_SAMPLE_COUNT
from hoop_guide.plate_tracker import PlateTracker
from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.vision import Frame
from hoop_guide.vision import MarkerDetection

# One pixel is one meter, so marker corners are given in the robot plane.
IDENTITY = Homography(np.eye(3))
IMAGE = Image(np.zeros((2, 2, 3), dtype=np.uint8))
PLATE_MARKER_ID = 1
FRAME_PERIOD = 0.1  # [s]
ROTATION_CENTER = (0.5, 0.1)
ROTATION_RADIUS = 0.08  # [m]


def frame_with_plate(timestamp, plate_pose):
    """Frame in which one plate marker is seen for the plate pose."""
    cosine, sine = math.cos(plate_pose.heading), math.sin(plate_pose.heading)
    center_x, center_y = MARKER_CENTERS[PLATE_MARKER_ID]
    half = MARKER_SIZE / 2
    # Top left, top right, bottom right, bottom left in the plate frame.
    corners_on_plate = [
        (center_x - half, center_y + half),
        (center_x + half, center_y + half),
        (center_x + half, center_y - half),
        (center_x - half, center_y - half),
    ]
    corners = [
        (
            plate_pose.x + cosine * x - sine * y,
            plate_pose.y + sine * x + cosine * y,
        )
        for x, y in corners_on_plate
    ]
    marker = MarkerDetection(PLATE_MARKER_ID, PixelPoints(np.array(corners)))
    return Frame(IMAGE, timestamp, (marker,))


def frame_without_plate(timestamp):
    return Frame(IMAGE, timestamp, ())


def rotating_plate_pose(exposure_time, angular_velocity):
    """Pose of a plate turning about ROTATION_CENTER, facing outwards."""
    angle = angular_velocity * exposure_time
    return PlatePose(
        x=ROTATION_CENTER[0] + ROTATION_RADIUS * math.cos(angle),
        y=ROTATION_CENTER[1] + ROTATION_RADIUS * math.sin(angle),
        heading=angle,
    )


def track(
    tracker, frame_count, angular_velocity, start=0.0, camera_latency=0.0
):
    """Feeds frames of a rotating plate, returns the last exposure time.

    Each frame is stamped camera_latency after the exposure of its image.
    """
    exposure_time = start
    for index in range(frame_count):
        exposure_time = start + index * FRAME_PERIOD
        tracker.update(
            frame_with_plate(
                exposure_time + camera_latency,
                rotating_plate_pose(exposure_time, angular_velocity),
            )
        )
    return exposure_time


def test_new_tracker_knows_nothing():
    tracker = PlateTracker(IDENTITY)

    assert tracker.plate_pose is None
    assert tracker.motion is None


def test_plate_pose_comes_from_the_newest_frame():
    tracker = PlateTracker(IDENTITY)
    tracker.update(frame_with_plate(0.0, PlatePose(0.4, 0.1, 0.0)))

    tracker.update(frame_with_plate(0.1, PlatePose(0.5, -0.2, 0.3)))

    plate_pose = tracker.plate_pose
    assert plate_pose is not None
    assert (plate_pose.x, plate_pose.y, plate_pose.heading) == pytest.approx(
        (0.5, -0.2, 0.3)
    )


def test_plate_pose_is_none_when_the_plate_is_hidden():
    tracker = PlateTracker(IDENTITY)
    tracker.update(frame_with_plate(0.0, PlatePose(0.4, 0.1, 0.0)))

    tracker.update(frame_without_plate(0.1))

    assert tracker.plate_pose is None


def test_motion_is_none_until_all_plate_poses_are_collected():
    tracker = PlateTracker(IDENTITY)

    track(tracker, PLATE_SAMPLE_COUNT - 1, angular_velocity=0.3)

    assert tracker.motion is None


def test_frames_without_the_plate_do_not_count():
    tracker = PlateTracker(IDENTITY)
    track(tracker, PLATE_SAMPLE_COUNT - 1, angular_velocity=0.3)

    tracker.update(frame_without_plate(100.0))

    assert tracker.motion is None


@pytest.mark.parametrize(
    "angular_velocity", [0.3, -0.3], ids=["leftwise", "rightwise"]
)
def test_motion_of_a_rotating_plate_has_its_center_and_velocity(
    angular_velocity,
):
    tracker = PlateTracker(IDENTITY)

    track(tracker, PLATE_SAMPLE_COUNT, angular_velocity)

    motion = tracker.motion
    assert motion is not None
    assert motion.angular_velocity == pytest.approx(angular_velocity)
    assert (motion.center_x, motion.center_y) == pytest.approx(ROTATION_CENTER)
    assert motion.is_rotating_leftwise is (angular_velocity > 0)


def test_motion_of_a_still_plate_has_zero_velocity():
    tracker = PlateTracker(IDENTITY)

    for index in range(PLATE_SAMPLE_COUNT):
        tracker.update(
            frame_with_plate(index * FRAME_PERIOD, PlatePose(0.4, 0.1, 0.2))
        )

    motion = tracker.motion
    assert motion is not None
    assert motion.angular_velocity == pytest.approx(0.0, abs=1e-9)
    assert (motion.pose_ref.x, motion.pose_ref.y) == pytest.approx((0.4, 0.1))


def test_motion_does_not_change_after_it_is_fitted():
    tracker = PlateTracker(IDENTITY)
    last_exposure_time = track(tracker, PLATE_SAMPLE_COUNT, 0.3)
    fitted_motion = tracker.motion

    track(
        tracker,
        PLATE_SAMPLE_COUNT,
        angular_velocity=-0.3,
        start=last_exposure_time + FRAME_PERIOD,
    )

    assert tracker.motion is fitted_motion


def test_plate_pose_keeps_updating_after_the_motion_is_fitted():
    tracker = PlateTracker(IDENTITY)
    last_exposure_time = track(tracker, PLATE_SAMPLE_COUNT, 0.3)

    tracker.update(
        frame_with_plate(
            last_exposure_time + FRAME_PERIOD, PlatePose(0.5, -0.2, 0.3)
        )
    )

    plate_pose = tracker.plate_pose
    assert plate_pose is not None
    assert (plate_pose.x, plate_pose.y) == pytest.approx((0.5, -0.2))


def test_motion_predicts_the_plate_without_new_frames():
    tracker = PlateTracker(IDENTITY)
    last_exposure_time = track(tracker, PLATE_SAMPLE_COUNT, 0.3)
    motion = tracker.motion
    assert motion is not None

    predicted = motion.predict(last_exposure_time + 20.0)

    expected = rotating_plate_pose(last_exposure_time + 20.0, 0.3)
    assert (predicted.x, predicted.y) == pytest.approx(
        (expected.x, expected.y), abs=1e-6
    )


def test_camera_latency_is_removed_from_the_frame_timestamps():
    camera_latency = 0.25
    tracker = PlateTracker(IDENTITY, camera_latency=camera_latency)
    last_exposure_time = track(
        tracker, PLATE_SAMPLE_COUNT, 0.3, camera_latency=camera_latency
    )
    motion = tracker.motion
    assert motion is not None

    predicted = motion.predict(last_exposure_time + 20.0)

    expected = rotating_plate_pose(last_exposure_time + 20.0, 0.3)
    assert (predicted.x, predicted.y) == pytest.approx(
        (expected.x, expected.y), abs=1e-6
    )


def test_frame_older_than_the_last_plate_pose_raises():
    tracker = PlateTracker(IDENTITY)
    tracker.update(frame_with_plate(1.0, PlatePose(0.4, 0.1, 0.0)))

    with pytest.raises(ValueError, match="order of their timestamps"):
        tracker.update(frame_with_plate(0.5, PlatePose(0.4, 0.1, 0.0)))
