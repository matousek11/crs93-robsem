"""Frame rate of the camera stream shown in the window."""

import collections

# Number of frames the rate is averaged over, so the number does not jump.
FPS_WINDOW = 10


class FpsMeter:
    """Measures the frame rate from frame timestamps."""

    def __init__(self) -> None:
        self._timestamps: collections.deque[float] = collections.deque(
            maxlen=FPS_WINDOW
        )

    def tick(self, timestamp: float) -> float | None:
        """Adds a frame and returns the frame rate.

        Args:
            timestamp: Time of the frame [s].

        Returns:
            Frames per second over the last FPS_WINDOW frames, None until
            there are two frames with different timestamps.
        """
        self._timestamps.append(timestamp)
        duration = self._timestamps[-1] - self._timestamps[0]
        if duration <= 0:
            return None
        return (len(self._timestamps) - 1) / duration
