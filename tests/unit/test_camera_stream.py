import threading
import time

import numpy as np

from hoop_guide.camera_stream import CameraStream

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


def wait_until(condition):
    deadline = time.monotonic() + WAIT_TIMEOUT
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.001)


def test_images_are_passed_to_callback_in_order():
    images = [make_image(1), make_image(2), make_image(3)]
    received: list[np.ndarray] = []

    with CameraStream(FakeCamera(images), received.append) as stream:
        wait_until(lambda: not stream.is_running)

    assert [int(image[0, 0, 0]) for image in received] == [1, 2, 3]


def test_latest_is_last_grabbed_image():
    images = [make_image(1), make_image(2)]

    with CameraStream(FakeCamera(images), lambda image: None) as stream:
        wait_until(lambda: not stream.is_running)

    assert stream.latest is not None
    np.testing.assert_array_equal(stream.latest, make_image(2))


def test_latest_is_none_before_first_image():
    stream = CameraStream(FakeCamera([make_image(1)]), lambda image: None)

    assert stream.latest is None


def test_camera_failure_stops_stream_and_is_logged(caplog):
    with CameraStream(FakeCamera([]), lambda image: None) as stream:
        wait_until(lambda: not stream.is_running)

    assert "Camera stream failed" in caplog.text


def test_callback_failure_stops_stream_and_is_logged(caplog):
    def failing_callback(image):
        raise ValueError("viewer broken")

    camera = FakeCamera([make_image(1)], should_repeat_last=True)
    with CameraStream(camera, failing_callback) as stream:
        wait_until(lambda: not stream.is_running)

    assert camera.grab_count == 1
    assert "Camera stream failed" in caplog.text


def test_exit_stops_grabbing():
    is_grabbing = threading.Event()
    camera = FakeCamera([make_image(1)], should_repeat_last=True)

    with CameraStream(camera, lambda image: is_grabbing.set()) as stream:
        assert is_grabbing.wait(WAIT_TIMEOUT)

    assert not stream.is_running
