import threading
import time

import numpy as np

from hoop_guide.camera_stream import CameraStream
from hoop_guide.vision import Frame

WAIT_TIMEOUT = 2.0


class FakeCamera:
    """Returns the given images, then repeats the last one or fails."""

    def __init__(self, images, should_repeat_last=False):
        self._images = list(images)
        self._should_repeat_last = should_repeat_last
        self.grab_count = 0

    def grab_image(self):
        self.grab_count += 1
        if self.grab_count <= len(self._images):
            return self._images[self.grab_count - 1]
        if self._should_repeat_last:
            return self._images[-1]
        raise RuntimeError("camera disconnected")


def make_image(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


def make_frame(image, timestamp):
    return Frame(image, timestamp, ())


def ignore(frame):
    pass


def wait_until(condition):
    deadline = time.monotonic() + WAIT_TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.001)


def run_until_stopped(camera, process=make_frame, on_frame=ignore):
    with CameraStream(camera, process, on_frame) as stream:
        wait_until(lambda: not stream.is_running)
    return stream


def test_frames_are_passed_to_callback_in_order():
    images = [make_image(1), make_image(2), make_image(3)]
    received: list[Frame] = []

    run_until_stopped(FakeCamera(images), on_frame=received.append)

    assert [int(frame.image[0, 0, 0]) for frame in received] == [1, 2, 3]


def test_frames_have_increasing_timestamps():
    images = [make_image(1), make_image(2), make_image(3)]
    received: list[Frame] = []

    run_until_stopped(FakeCamera(images), on_frame=received.append)

    timestamps = [frame.timestamp for frame in received]
    assert timestamps == sorted(timestamps)


def test_frames_come_from_process():
    processed = Frame(make_image(9), 0.0, ())
    received: list[Frame] = []

    run_until_stopped(
        FakeCamera([make_image(1)]),
        process=lambda image, timestamp: processed,
        on_frame=received.append,
    )

    assert received == [processed]


def test_latest_is_last_frame():
    images = [make_image(1), make_image(2)]

    stream = run_until_stopped(FakeCamera(images))

    assert stream.latest is not None
    np.testing.assert_array_equal(stream.latest.image, make_image(2))


def test_latest_is_none_before_first_frame():
    stream = CameraStream(FakeCamera([make_image(1)]), make_frame, ignore)

    assert stream.latest is None


def test_camera_failure_stops_stream_and_is_logged(caplog):
    run_until_stopped(FakeCamera([]))

    assert "Camera stream failed" in caplog.text


def test_callback_failure_stops_stream_and_is_logged(caplog):
    def failing_callback(frame):
        raise ValueError("viewer broken")

    camera = FakeCamera([make_image(1)], should_repeat_last=True)
    run_until_stopped(camera, on_frame=failing_callback)

    assert camera.grab_count == 1
    assert "Camera stream failed" in caplog.text


def test_exit_stops_grabbing():
    is_grabbing = threading.Event()
    camera = FakeCamera([make_image(1)], should_repeat_last=True)

    with CameraStream(
        camera, make_frame, lambda frame: is_grabbing.set()
    ) as stream:
        assert is_grabbing.wait(WAIT_TIMEOUT)

    assert not stream.is_running
