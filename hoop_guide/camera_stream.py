"""Background grabbing of camera images."""

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Self

from hoop_guide.hardware.interfaces import Camera
from hoop_guide.types import Image

logger = logging.getLogger(__name__)

JOIN_TIMEOUT = 1.0  # [s]


class CameraStream:
    """Grabs camera images in a background thread while used as a context.

    Every new image is kept as the latest one and passed to on_image. When
    the camera or on_image fails, the error is logged and grabbing stops.
    The camera must not be used by anything else while the stream runs.

    Args:
        camera: Source of images.
        on_image: Called with every new image, from the grabbing thread.
    """

    def __init__(
        self, camera: Camera, on_image: Callable[[Image], None]
    ) -> None:
        self._camera = camera
        self._on_image = on_image
        self._latest: Image | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="camera", daemon=True
        )

    @property
    def latest(self) -> Image | None:
        """The newest image, None before the first one arrives."""
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
                self._latest = image
                self._on_image(image)
            except Exception:
                logger.exception("Camera stream failed, grabbing stopped.")
                return
        logger.info("Camera stream stopped.")
