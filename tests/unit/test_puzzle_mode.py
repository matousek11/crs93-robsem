import math

import numpy as np
import pytest

from hoop_guide.keyboard import Key
from hoop_guide.planning import SYNC_DISTANCE
from hoop_guide.planning import NoJointConfigError
from hoop_guide.plate import MARKER_CENTERS
from hoop_guide.plate import MARKER_SIZE
from hoop_guide.plate import PlatePose
from hoop_guide.puzzle_mode import BOTTOM_WAIT
from hoop_guide.puzzle_mode import MEASUREMENT_TIMEOUT
from hoop_guide.puzzle_mode import MOTION_POLL_PERIOD
from hoop_guide.puzzle_mode import MOVE_OVERHEAD
from hoop_guide.puzzle_mode import PuzzleMode
from hoop_guide.puzzle_mode import SystemClock
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import PixelPoints
from hoop_guide.vision import Frame
from hoop_guide.vision import MarkerDetection

# One pixel is one meter, so marker corners are given in the robot plane.
IDENTITY = Homography(np.eye(3))
IMAGE = Image(np.zeros((2, 2, 3), dtype=np.uint8))
PLATE_MARKER_ID = 1
PLATE_HEIGHT = 0.08  # [m]
FRAME_PERIOD = 0.1  # [s]
ROTATION_CENTER = (0.40, 0.03)  # [m]
ROTATION_RADIUS = 0.06  # [m]
ANGULAR_VELOCITY = 0.2  # [rad/s]
MOVING_CHECKS = 10  # each step takes 10 * MOTION_POLL_PERIOD
SYNC_HEIGHT = PLATE_HEIGHT + 0.2 + SYNC_DISTANCE
BOTTOM_HEIGHT = PLATE_HEIGHT + STOP_HEIGHT
FAKE_MOVE_TIME = 0.25  # what the fake robot reports for every move [s]
ENTER_TIME = 8.0  # after the measurement of about 5 s [s]
STOP_ON_ROD_TIME = 8.5  # a few steps down the rod [s]


def plate_pose_at(time):
    """Pose of a plate turning about ROTATION_CENTER, facing outwards."""
    angle = ANGULAR_VELOCITY * time
    return PlatePose(
        x=ROTATION_CENTER[0] + ROTATION_RADIUS * math.cos(angle),
        y=ROTATION_CENTER[1] + ROTATION_RADIUS * math.sin(angle),
        heading=angle,
    )


def plate_marker(plate_pose):
    cosine, sine = math.cos(plate_pose.heading), math.sin(plate_pose.heading)
    center_x, center_y = MARKER_CENTERS[PLATE_MARKER_ID]
    half = MARKER_SIZE / 2
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
    return MarkerDetection(PLATE_MARKER_ID, PixelPoints(np.array(corners)))


class FakeClock:
    def __init__(self):
        self.time = 0.0

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class FakeFrames:
    """Camera with a new frame every FRAME_PERIOD of the fake clock."""

    def __init__(self, clock, is_plate_visible=True):
        self._clock = clock
        self._is_plate_visible = is_plate_visible

    @property
    def latest(self):
        timestamp = (
            math.floor(self._clock.time / FRAME_PERIOD + 1e-9) * FRAME_PERIOD
        )
        markers = (
            (plate_marker(plate_pose_at(timestamp)),)
            if self._is_plate_visible
            else ()
        )
        return Frame(IMAGE, timestamp, markers)


class FakeMotion:
    """Robot whose every move takes MOVING_CHECKS questions.

    Args:
        clock: Fake clock, its time is recorded with every move.
        is_reachable: Rule on the hoop pose and the time of the move.
    """

    def __init__(self, clock, is_reachable=lambda pose, time: True):
        self._clock = clock
        self._is_reachable = is_reachable
        self._remaining_checks = 0
        self.moves = []

    def get_move_time(self, hoop_pose):
        if not self._is_reachable(hoop_pose, self._clock.time):
            raise NoJointConfigError("No joint configuration.")
        return FAKE_MOVE_TIME

    def move_to_ik(self, hoop_pose):
        if not self._is_reachable(hoop_pose, self._clock.time):
            raise NoJointConfigError("No joint configuration.")
        self.moves.append((self._clock.time, hoop_pose.copy()))
        self._remaining_checks = MOVING_CHECKS

    def is_in_motion(self):
        if self._remaining_checks == 0:
            return False
        self._remaining_checks -= 1
        return True

    def heights(self):
        return [pose[2, 3] for _, pose in self.moves]


class FakeRobot:
    def __init__(self):
        self.home_count = 0

    def soft_home(self):
        self.home_count += 1


class FakeKeyboard:
    """Keys pressed at given times of the fake clock."""

    def __init__(self, clock, timed_keys=()):
        self._clock = clock
        self._timed_keys = sorted(
            timed_keys, key=lambda timed_key: timed_key[0]
        )
        self.events = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("exit")

    def poll_key(self):
        if self._timed_keys and self._timed_keys[0][0] <= self._clock.time:
            return self._timed_keys.pop(0)[1]
        return None

    def discard_pending(self):
        self._timed_keys = [
            timed_key
            for timed_key in self._timed_keys
            if timed_key[0] > self._clock.time
        ]


class FakeReporter:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


class Run:
    """Everything a puzzle run used, for the asserts."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        timed_keys=(),
        motion_class=None,
        is_reachable=lambda pose, time: True,
        has_camera=True,
        is_plate_visible=True,
        homography=IDENTITY,
        plate_height=PLATE_HEIGHT,
    ):
        self.clock = FakeClock()
        self.motion = (motion_class or FakeMotion)(self.clock, is_reachable)
        self.robot = FakeRobot()
        self.keyboard = FakeKeyboard(self.clock, timed_keys)
        self.reporter = FakeReporter()
        frames = (
            FakeFrames(self.clock, is_plate_visible) if has_camera else None
        )
        self.mode = PuzzleMode(
            self.motion,
            self.robot,
            frames,
            self.keyboard,
            self.reporter,
            homography=homography,
            plate_height=plate_height,
            clock=self.clock,
        )

    def run(self, puzzle=Puzzle.A1):
        self.mode.run(puzzle)
        return self


def run_puzzle(**options):
    return Run(**options).run()


def complete_run():
    return run_puzzle(timed_keys=[(ENTER_TIME, Key.ENTER)])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"has_camera": False}, "camera"),
        ({"homography": None}, "homography"),
        ({"plate_height": None}, "PLATE_HEIGHT"),
    ],
    ids=["no_camera", "no_homography", "no_plate_height"],
)
def test_missing_requirement_is_reported_before_anything_moves(
    options, message
):
    run = run_puzzle(**options)

    assert message in run.reporter.warnings[0]
    assert run.motion.moves == []
    assert run.keyboard.events == []


def test_complete_run_goes_down_to_the_stop_height_and_back_home():
    run = complete_run()

    heights = run.motion.heights()
    assert heights[0] == pytest.approx(SYNC_HEIGHT)
    assert min(heights) == pytest.approx(BOTTOM_HEIGHT)
    assert heights[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1
    assert run.reporter.warnings == []
    assert run.keyboard.events == ["enter", "exit"]


def test_hoop_follows_the_sync_point_until_enter():
    run = complete_run()

    before_enter = [
        pose[2, 3] for time, pose in run.motion.moves if time < ENTER_TIME
    ]
    assert len(before_enter) > 1
    assert before_enter == pytest.approx([SYNC_HEIGHT] * len(before_enter))


def test_moves_aim_at_the_plate_predicted_for_the_arrival():
    run = complete_run()

    sync_moves = [
        (time, pose)
        for time, pose in run.motion.moves
        if pose[2, 3] == pytest.approx(SYNC_HEIGHT)
    ]
    for time, pose in sync_moves:
        plate = plate_pose_at(time + FAKE_MOVE_TIME + MOVE_OVERHEAD)
        np.testing.assert_allclose(pose[:2, 3], (plate.x, plate.y), atol=1e-6)


def test_hoop_goes_down_then_follows_the_bottom_then_goes_back_up():
    run = complete_run()

    after_enter = [
        pose[2, 3] for time, pose in run.motion.moves if time >= ENTER_TIME
    ]
    lowest = after_enter.index(min(after_enter))
    assert np.all(np.diff(after_enter[: lowest + 1]) < 0)
    at_bottom = [
        time
        for time, pose in run.motion.moves
        if pose[2, 3] == pytest.approx(BOTTOM_HEIGHT)
    ]
    step_time = MOVING_CHECKS * MOTION_POLL_PERIOD
    assert at_bottom[-1] - at_bottom[0] >= BOTTOM_WAIT - step_time
    rest = after_enter[lowest + len(at_bottom) :]
    assert np.all(np.diff(rest) > 0)


def test_stop_during_the_measurement_ends_without_moving():
    run = run_puzzle(timed_keys=[(1.0, Key.STOP)])

    assert run.motion.moves == []
    assert "Stopped." in run.reporter.infos
    assert run.robot.home_count == 0


def test_plate_not_seen_ends_after_the_timeout_without_moving():
    run = run_puzzle(is_plate_visible=False)

    assert "not seen" in run.reporter.warnings[0]
    assert run.clock.time >= MEASUREMENT_TIMEOUT
    assert run.motion.moves == []


def test_stop_at_the_sync_point_leaves_the_hoop_there():
    run = run_puzzle(timed_keys=[(ENTER_TIME, Key.STOP)])

    assert min(run.motion.heights()) == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 0


def test_stop_on_the_rod_goes_back_out_and_home():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (STOP_ON_ROD_TIME, Key.STOP)]
    )

    heights = run.motion.heights()
    assert min(heights) > BOTTOM_HEIGHT + 0.05
    assert heights[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_stop_while_following_the_bottom_goes_back_out_and_home():
    # Ten steps of 0.2 s down the rod A1 end at about 10 s.
    stop_at_bottom_time = ENTER_TIME + 2.5
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (stop_at_bottom_time, Key.STOP)]
    )

    assert "Stopped, going back out." in run.reporter.infos
    assert min(run.motion.heights()) == pytest.approx(BOTTOM_HEIGHT)
    assert run.motion.heights()[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_going_back_out_starts_one_point_above_the_reached_one():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (STOP_ON_ROD_TIME, Key.STOP)]
    )

    heights = run.motion.heights()
    lowest = heights.index(min(heights))
    assert heights[lowest + 1] == pytest.approx(heights[lowest - 1])
    assert heights[lowest + 1] > heights[lowest]


def test_unreachable_first_point_on_the_rod_goes_home_from_the_sync_point():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        is_reachable=lambda pose, time: pose[2, 3] > SYNC_HEIGHT - 0.01,
    )

    assert "out of reach" in run.reporter.warnings[0]
    assert run.motion.heights() == pytest.approx(
        [SYNC_HEIGHT] * len(run.motion.moves)
    )
    assert run.robot.home_count == 1


def test_stop_while_going_back_out_is_ignored():
    run = run_puzzle(
        timed_keys=[
            (ENTER_TIME, Key.ENTER),
            (STOP_ON_ROD_TIME, Key.STOP),
            (STOP_ON_ROD_TIME + 0.3, Key.STOP),
        ]
    )

    assert run.motion.heights()[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_unreachable_point_on_the_rod_goes_back_out_and_home():
    lowest_reachable = PLATE_HEIGHT + 0.1
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        is_reachable=lambda pose, time: pose[2, 3] >= lowest_reachable,
    )

    assert "out of reach" in run.reporter.warnings[0]
    heights = run.motion.heights()
    assert min(heights) >= lowest_reachable
    assert heights[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_unreachable_sync_point_ends_without_going_down():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        is_reachable=lambda pose, time: False,
    )

    assert "Synchronization point is out of reach" in run.reporter.warnings[0]
    assert run.motion.moves == []
    assert run.robot.home_count == 0


def test_failed_way_back_out_warns_and_does_not_go_home():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (STOP_ON_ROD_TIME, Key.STOP)],
        is_reachable=lambda pose, time: time < STOP_ON_ROD_TIME,
    )

    assert "Going back out failed" in run.reporter.warnings[0]
    assert run.robot.home_count == 0
    assert run.keyboard.events == ["enter", "exit"]


def test_keys_pressed_before_the_run_are_dropped():
    run = run_puzzle(timed_keys=[(0.0, Key.STOP), (ENTER_TIME, Key.ENTER)])

    assert run.robot.home_count == 1


def test_keyboard_is_closed_when_the_robot_fails():
    class FailingMotion(FakeMotion):
        def is_in_motion(self):
            raise RuntimeError("Check ready: error.")

    run = Run(timed_keys=[(ENTER_TIME, Key.ENTER)], motion_class=FailingMotion)

    with pytest.raises(RuntimeError, match="Check ready"):
        run.run()

    assert run.keyboard.events == ["enter", "exit"]


def test_system_clock_sleeps_for_the_time():
    clock = SystemClock()
    start = clock.now()

    clock.sleep(0.01)

    assert clock.now() - start >= 0.01
