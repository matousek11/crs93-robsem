import math
import re

import numpy as np
import pytest

from hoop_guide.control import MOVE_OVERHEAD
from hoop_guide.estimation import TableMotion
from hoop_guide.keyboard import Key
from hoop_guide.planning import AIM_AHEAD
from hoop_guide.planning import SYNC_DISTANCE
from hoop_guide.planning import NoJointConfigError
from hoop_guide.planning import RodPlanner
from hoop_guide.plate import MARKER_CENTERS
from hoop_guide.plate import MARKER_SIZE
from hoop_guide.plate import PlatePose
from hoop_guide.puzzle_mode import INITIAL_SEARCH_TIME
from hoop_guide.puzzle_mode import MEASUREMENT_TIMEOUT
from hoop_guide.puzzle_mode import MOTION_POLL_PERIOD
from hoop_guide.puzzle_mode import SEARCH_TIME_MARGIN
from hoop_guide.puzzle_mode import PuzzleMode
from hoop_guide.puzzle_mode import SystemClock
from hoop_guide.rods import STOP_HEIGHT
from hoop_guide.rods import Puzzle
from hoop_guide.run_schedule import BOTTOM_WAIT
from hoop_guide.types import Homography
from hoop_guide.types import Image
from hoop_guide.types import JointConfig
from hoop_guide.types import JointSpeeds
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
# Each step takes 15 * MOTION_POLL_PERIOD = 0.3 s, about as long as a step of
# the start search with the fast fake joints: MOVE_OVERHEAD plus the ready
# flag delay.
MOVING_CHECKS = 15
SYNC_HEIGHT = PLATE_HEIGHT + 0.2 + SYNC_DISTANCE
BOTTOM_HEIGHT = PLATE_HEIGHT + STOP_HEIGHT
# One turn of the table after 8 s, so the plate is where the tests expect
# it and the measurement of PLATE_SAMPLE_COUNT frames is over [s].
ENTER_TIME = 8.0 + 2 * math.pi / ANGULAR_VELOCITY
# The first search ends at once on the fake clock, but the earliest start is
# where the expected search time puts it.
FIRST_START_DELAY = SEARCH_TIME_MARGIN * INITIAL_SEARCH_TIME  # [s]
# A few steps down the rod after the start [s].
STOP_ON_ROD_TIME = ENTER_TIME + FIRST_START_DELAY + 1.5
LOW_HEIGHT = PLATE_HEIGHT + 0.1  # rules make points below this unreachable


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
    """Camera with a new frame every FRAME_PERIOD of the fake clock.

    Args:
        clock: Fake clock.
        is_plate_visible: False for frames without the plate.
        plate_hidden_after: From this time the plate is not in the frames
            [s], None for never.
        frames_stop_after: From this time there is no frame [s], None for
            never.
    """

    def __init__(
        self,
        clock,
        is_plate_visible=True,
        plate_hidden_after=None,
        frames_stop_after=None,
    ):
        self._clock = clock
        self._is_plate_visible = is_plate_visible
        self._plate_hidden_after = plate_hidden_after
        self._frames_stop_after = frames_stop_after

    @property
    def latest(self):
        now = self._clock.time
        if (
            self._frames_stop_after is not None
            and now > self._frames_stop_after
        ):
            return None
        is_hidden = (
            self._plate_hidden_after is not None
            and now > self._plate_hidden_after
        )
        timestamp = math.floor(now / FRAME_PERIOD + 1e-9) * FRAME_PERIOD
        markers = (
            (plate_marker(plate_pose_at(timestamp)),)
            if self._is_plate_visible and not is_hidden
            else ()
        )
        return Frame(IMAGE, timestamp, markers)


class FakeMotion:
    """Robot whose every move takes MOVING_CHECKS questions.

    Its joint configuration is the hoop position, so the joints are fast.

    Args:
        clock: Fake clock, its time is recorded with every move.
        is_reachable: Rule on the hoop pose and the time of the move.
        is_reachable_in_search: Rule the start search sees, is_reachable
            when None.
        flip_after: From this time points below LOW_HEIGHT need the fourth
            joint turned by half a turn [s], None for never.
        start_joint_angles: Joint angles before the first move [rad].
        choice_time: Time every choice of a joint configuration takes, so
            a search takes time on the fake clock [s].
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        clock,
        is_reachable=lambda pose, time: True,
        is_reachable_in_search=None,
        flip_after=None,
        start_joint_angles=None,
        choice_time=0.0,
    ):
        self._clock = clock
        self._choice_time = choice_time
        self._is_reachable = is_reachable
        self._is_reachable_in_search = is_reachable_in_search or is_reachable
        self._flip_after = flip_after
        self._remaining_checks = 0
        self._joint_angles = JointConfig(
            np.zeros(6) if start_joint_angles is None else start_joint_angles
        )
        self.moves = []

    @property
    def joint_speeds(self):
        return JointSpeeds(np.full(6, 1000.0))

    @property
    def joint_limits(self):
        return JointConfig(np.full(6, -10.0)), JointConfig(np.full(6, 10.0))

    def get_joint_angles(self):
        return self._joint_angles

    def get_hoop_pose(self):
        return self.moves[-1][1].copy()

    def choose_joint_config(self, hoop_pose, joint_angles):
        self._clock.time += self._choice_time
        if not self._is_reachable_in_search(hoop_pose, self._clock.time):
            raise NoJointConfigError("No joint configuration.")
        is_flipped = (
            self._flip_after is not None
            and self._clock.time > self._flip_after
            and hoop_pose[2, 3] < LOW_HEIGHT
        )
        fourth_joint = math.pi if is_flipped else 0.0
        # The fifth joint is bent, zero would be the wrist singularity.
        return JointConfig(np.r_[hoop_pose[:3, 3], fourth_joint, 1.0, 0.0])

    def move_to_ik(self, hoop_pose):
        if not self._is_reachable(hoop_pose, self._clock.time):
            raise NoJointConfigError("No joint configuration.")
        self.moves.append((self._clock.time, hoop_pose.copy()))
        self._joint_angles = JointConfig(np.r_[hoop_pose[:3, 3], 0.0, 1.0, 0.0])
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
        is_reachable_in_search=None,
        flip_after=None,
        start_joint_angles=None,
        choice_time=0.0,
        has_camera=True,
        is_plate_visible=True,
        plate_hidden_after=None,
        frames_stop_after=None,
        homography=IDENTITY,
        plate_height=PLATE_HEIGHT,
    ):
        self.clock = FakeClock()
        self.motion = (motion_class or FakeMotion)(
            self.clock,
            is_reachable,
            is_reachable_in_search,
            flip_after,
            start_joint_angles,
            choice_time,
        )
        self.robot = FakeRobot()
        self.keyboard = FakeKeyboard(self.clock, timed_keys)
        self.reporter = FakeReporter()
        frames = (
            FakeFrames(
                self.clock,
                is_plate_visible,
                plate_hidden_after,
                frames_stop_after,
            )
            if has_camera
            else None
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


def test_moves_aim_at_the_plate_predicted_a_bit_after_the_arrival():
    run = complete_run()

    sync_moves = [
        (time, pose)
        for time, pose in run.motion.moves
        if pose[2, 3] == pytest.approx(SYNC_HEIGHT)
    ]
    for time, pose in sync_moves:
        # The fake joints are fast, the move itself takes almost no time.
        plate = plate_pose_at(time + MOVE_OVERHEAD + AIM_AHEAD)
        np.testing.assert_allclose(pose[:2, 3], (plate.x, plate.y), atol=1e-4)


def test_hoop_goes_down_then_follows_the_bottom_then_goes_back_up():
    run = complete_run()

    heights = run.motion.heights()
    # From the last move at the sync point on.
    first_down = next(
        index
        for index, height in enumerate(heights)
        if height < SYNC_HEIGHT - 1e-9
    )
    after_enter = heights[first_down - 1 :]
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


def test_operator_messages_are_also_logged(caplog):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        run = complete_run()

    assert set(run.reporter.infos) <= {r.getMessage() for r in caplog.records}


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
    # Enter is noticed within one step after ENTER_TIME, one more step follows
    # the sync point, then every point down the rod takes one step. Stop
    # early in BOTTOM_WAIT.
    still_table = TableMotion(0.5, 0.0, 0.0, 0.0, PlatePose(0.5, 0.0, 0.0))
    path = RodPlanner(Puzzle.A1, still_table, PLATE_HEIGHT).update(0.0, 0)
    step_time = MOVING_CHECKS * MOTION_POLL_PERIOD
    latest_bottom_time = (
        ENTER_TIME + FIRST_START_DELAY + (len(path) + 1) * step_time
    )
    stop_at_bottom_time = latest_bottom_time + 0.2
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
        is_reachable_in_search=lambda pose, time: True,
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


def test_point_out_of_reach_only_in_reality_goes_back_out_and_home():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        is_reachable=lambda pose, time: pose[2, 3] >= LOW_HEIGHT,
        is_reachable_in_search=lambda pose, time: True,
    )

    assert "out of reach" in run.reporter.warnings[0]
    heights = run.motion.heights()
    assert min(heights) >= LOW_HEIGHT
    assert heights[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_joint_flip_on_the_rod_is_not_sent_and_the_hoop_goes_back_out():
    # The start search does not see the flip, it appears after the start.
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        flip_after=ENTER_TIME + FIRST_START_DELAY + 0.5,
    )

    assert "Joint 4 would turn by 180 deg" in run.reporter.warnings[0]
    assert min(run.motion.heights()) >= LOW_HEIGHT
    assert run.motion.heights()[-1] == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 1


def test_move_from_home_to_the_sync_point_may_turn_a_joint_far():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)],
        start_joint_angles=np.full(6, 3.0),
    )

    assert run.reporter.warnings == []
    assert run.robot.home_count == 1


def test_joint_angles_of_every_step_are_logged(caplog):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        complete_run()

    step_messages = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Step to path point")
    ]
    assert step_messages
    assert all("joints [" in message for message in step_messages)


def test_simulated_run_is_logged_step_by_step(caplog):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        complete_run()

    messages = [r.getMessage() for r in caplog.records]
    simulated = [m for m in messages if m.startswith("Simulated step")]
    assert simulated
    assert any(m.startswith("Simulated run:") for m in messages)


def test_every_step_on_the_rod_is_checked_in_the_log(caplog):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        complete_run()

    messages = [r.getMessage() for r in caplog.records]
    checks = [m for m in messages if m.startswith("Check ")]
    steps_on_rod = [
        m for m in messages if re.match(r"Step to path point [1-9]", m)
    ]
    assert len(checks) >= len(steps_on_rod) > 0
    assert all("mm off the rod" in m for m in checks)
    assert any("against the simulation" in m for m in checks)
    assert any("off the prediction" in m for m in checks)


@pytest.mark.parametrize(
    ("camera_option", "expected"),
    [
        ("plate_hidden_after", "plate not seen"),
        ("frames_stop_after", "no camera frame"),
    ],
    ids=["plate_hidden", "no_frames"],
)
def test_check_says_when_the_camera_cannot_see_the_plate(
    caplog, camera_option, expected
):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        run_puzzle(
            timed_keys=[(ENTER_TIME, Key.ENTER)],
            **{camera_option: ENTER_TIME},
        )

    checks = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Check ")
    ]
    assert checks
    assert all(expected in check for check in checks)


def test_tilted_hoop_is_logged_with_its_tilt(caplog):
    # Only a hoop axis tilted away from the vertical rod is reachable.
    def is_tilted(pose, time):
        return pose[2, 2] > -math.cos(math.radians(5.0))

    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        run = run_puzzle(
            timed_keys=[(ENTER_TIME, Key.ENTER)],
            is_reachable_in_search=is_tilted,
        )

    assert run.robot.home_count == 1
    assert any(
        "tilt 10 deg toward" in record.getMessage() for record in caplog.records
    )


def test_run_goes_down_exactly_at_the_simulated_start(caplog):
    with caplog.at_level("INFO", logger="hoop_guide.puzzle_mode"):
        complete_run()

    first_check = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Check down at path point 1")
    )
    match = re.search(r"sent ([+-][\d.]+) s against", first_check)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(0.0, abs=0.01)


def test_start_passed_during_a_long_search_does_not_go_down():
    # Every choice of joints takes 0.1 s, the search outlasts its estimate.
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (ENTER_TIME + 1000.0, Key.STOP)],
        choice_time=0.1,
    )

    assert "the start has passed" in run.reporter.warnings[0]
    assert min(run.motion.heights()) == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 0


def test_without_a_safe_start_the_hoop_stays_at_the_sync_point():
    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (ENTER_TIME + 3.0, Key.STOP)],
        is_reachable=lambda pose, time: pose[2, 3] >= LOW_HEIGHT,
    )

    assert "No run down the rod goes through" in run.reporter.warnings[0]
    assert min(run.motion.heights()) == pytest.approx(SYNC_HEIGHT)
    assert run.robot.home_count == 0


def test_enter_searches_again_after_no_safe_start():
    def is_reachable(pose, time):
        return pose[2, 3] >= LOW_HEIGHT or time > ENTER_TIME + 2.0

    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER), (ENTER_TIME + 3.0, Key.ENTER)],
        is_reachable=is_reachable,
    )

    assert len(run.reporter.warnings) == 1
    assert min(run.motion.heights()) == pytest.approx(BOTTOM_HEIGHT)
    assert run.robot.home_count == 1


def test_hoop_waits_at_the_sync_point_until_a_safe_start():
    # Low points are out of reach while the rod is nearer to the robot than
    # the table axis. At ENTER_TIME the plate is there and turns away.
    def is_reachable(pose, time):
        return pose[2, 3] >= LOW_HEIGHT or pose[0, 3] >= ROTATION_CENTER[0]

    run = run_puzzle(
        timed_keys=[(ENTER_TIME, Key.ENTER)], is_reachable=is_reachable
    )

    low_moves = [
        (time, pose)
        for time, pose in run.motion.moves
        if pose[2, 3] < LOW_HEIGHT
    ]
    assert low_moves
    assert low_moves[0][0] > ENTER_TIME + 5.0
    assert all(pose[0, 3] >= ROTATION_CENTER[0] for _, pose in low_moves)
    assert any("Going down the rod in" in info for info in run.reporter.infos)
    assert run.reporter.warnings == []
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
