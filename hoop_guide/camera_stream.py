"""Background grabbing and processing of camera images."""

import logging
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Self

from hoop_guide.hardware.interfaces import Camera
from hoop_guide.types import Image
from hoop_guide.vision import Frame

logger = logging.getLogger(__name__)

JOIN_TIMEOUT = 1.0  # [s]


class CameraStream:
    """Grabs and processes camera images in a background thread.

    Runs while used as a context. Every image is timestamped right after
    the grab and turned into a Frame by process. The newest Frame is kept
    as the latest one and passed to on_frame. When the camera, process or
    on_frame fails, the error is logged and grabbing stops. The camera must
    not be used by anything else while the stream runs.

    Args:
        camera: Source of images.
        process: Turns an image and its timestamp [s] into a Frame.
        on_frame: Called with every new Frame, from the grabbing thread.
    """

    def __init__(
        self,
        camera: Camera,
        process: Callable[[Image, float], Frame],
        on_frame: Callable[[Frame], None],
    ) -> None:
        self._camera = camera
        self._process = process
        self._on_frame = on_frame
        self._latest: Frame | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="camera", daemon=True
        )

    @property
    def latest(self) -> Frame | None:
        """The newest Frame, None before the first one arrives."""
        return self._latest

    @property
    def is_running(self) -> bool:
        """True while the background thread grabs images."""
        return self._thread.is_alive()

    def __enter__(self) -> Self:
        """Starts grabbing."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stops grabbing."""
        self._stop_event.set()
        # A grab in progress can take up to the camera timeout, the daemon
        # thread does not keep the application alive.
        self._thread.join(JOIN_TIMEOUT)

    def _run(self) -> None:
        logger.info("Camera stream started.")
        while not self._stop_event.is_set():
            try:
                image = self._camera.grab_image()
                frame = self._process(image, time.monotonic())
                self._latest = frame
                self._on_frame(frame)
            except Exception:
                logger.exception("Camera stream failed, grabbing stopped.")
                return
        logger.info("Camera stream stopped.")
