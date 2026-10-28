"""Camera window running in a separate process.

The OpenCV window needs the main thread of its process, which in the
application belongs to the menu and the Ctrl+C handling.
"""

import contextlib
import multiprocessing
import queue
import signal
from multiprocessing.synchronize import Event
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from hoop_guide.types import Image

WINDOW_NAME = "hoop_guide camera"
MAX_DISPLAY_WIDTH = 1280  # [px]
POLL_PERIOD = 0.05  # [s]
KEY_WAIT = 1  # [ms]
CLOSE_TIMEOUT = 2.0  # [s]


class Viewer:
    """Window showing the latest image passed to show, while used as a context.

    Only the newest image waits for the window, older ones are dropped, so
    show never blocks. The window is display only: keys in it are ignored
    and closing it does not affect the application.
    """

    def __init__(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._images: multiprocessing.Queue[Image] = context.Queue(maxsize=1)
        # Images left in the queue on exit can be thrown away
        self._images.cancel_join_thread()
        # Connects stop events between processes
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_window_loop,
            args=(self._images, self._stop_event),
            name="viewer",
            # The window process is killed when the main process ends.
            daemon=True,
        )

    def __enter__(self) -> Self:
        """Starts the window process."""
        self._process.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stops the window process."""
        self._stop_event.set()
        self._process.join(CLOSE_TIMEOUT)
        # Kill process if it is still running after CLOSE_TIMEOUT
        if self._process.is_alive():
            self._process.terminate()

    def show(self, image: Image) -> None:
        """Sends the image to the window, replacing one not shown yet."""
        height, width = image.shape[:2]
        if width > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / width
            image = Image(
                cv2.resize(
                    image, (MAX_DISPLAY_WIDTH, round(height * scale))
                ).astype(np.uint8, copy=False)
            )
        # remove old picture from queue
        with contextlib.suppress(queue.Empty):
            self._images.get_nowait()
        # put new picture into queue
        with contextlib.suppress(queue.Full):
            self._images.put_nowait(image)


def _window_loop(images: "multiprocessing.Queue[Image]", stop: Event) -> None:
    # Ignore Ctrl+C, the main process closes this one after the robot
    # returns home.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    has_image = False
    try:
        while not stop.is_set():
            with contextlib.suppress(queue.Empty):
                cv2.imshow(WINDOW_NAME, images.get(timeout=POLL_PERIOD))
                has_image = True
            cv2.waitKey(KEY_WAIT)
            # Some backends report an empty window as not visible.
            is_closed = (
                cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            )
            if has_image and is_closed:
                return
    finally:
        cv2.destroyAllWindows()
