import pytest

from hoop_guide.viewer.fps import FPS_WINDOW
from hoop_guide.viewer.fps import FpsMeter


def tick_all(meter, timestamps):
    return [meter.tick(t) for t in timestamps][-1]


def test_first_frame_has_unknown_rate():
    assert FpsMeter().tick(1.0) is None


def test_equal_timestamps_have_unknown_rate():
    assert tick_all(FpsMeter(), [1.0, 1.0]) is None


def test_regular_frames_give_their_rate():
    timestamps = [i * 0.1 for i in range(5)]

    assert tick_all(FpsMeter(), timestamps) == pytest.approx(10.0)


def test_rate_uses_only_last_frames():
    slow = [i * 1.0 for i in range(FPS_WINDOW)]
    fast = [slow[-1] + (i + 1) * 0.1 for i in range(FPS_WINDOW)]

    assert tick_all(FpsMeter(), slow + fast) == pytest.approx(10.0)
