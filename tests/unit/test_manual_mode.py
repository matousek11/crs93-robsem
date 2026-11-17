import cv2
import numpy as np
import pytest

from hoop_guide.calibration import CALIBRATION_POINT_COUNT
from hoop_guide.calibration import load_homography
from hoop_guide.keyboard import Key
from hoop_guide.manual_mode import MANUAL_STEP
from hoop_guide.manual_mode import ManualCommand
from hoop_guide.manual_mode import ManualMode
from hoop_guide.manual_mode import ManualRow
from hoop_guide.manual_mode import ManualState
from hoop_guide.planning import NoJointConfigError
from hoop_guide.types import PixelPoints
from hoop_guide.types import Point3D
from hoop_guide.vision import Frame
from hoop_guide.vision import image_to_plane

START_POSITION = (0.4, 0.0, 0.05)
PIXELS_PER_METER = 4000.0
START_PIXEL = 200.0
IMAGE_SIZE = 800
HOOP_RADIUS = 30
GRID_STEP = 0.05  # distance of neighbouring calibration points [m]
STEPS_PER_GRID_STEP = round(GRID_STEP / MANUAL_STEP)
EXIT_KEYS = [Key.DOWN, Key.DOWN, Key.DOWN, Key.ENTER]


def make_state(row=ManualRow.X):
    return ManualState(row, Point3D(np.array(START_POSITION)))


def make_hoop_pose(position=START_POSITION):
    # Hoop turned by a quarter turn about z, to see that it is kept.
    hoop_pose = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    hoop_pose[:3, 3] = position
    return hoop_pose


def to_pixel(position):
    """Where the fake camera sees a hoop at the position."""
    return (
        (position[0] - START_POSITION[0]) * PIXELS_PER_METER + START_PIXEL,
        (position[1] - START_POSITION[1]) * PIXELS_PER_METER + START_PIXEL,
    )


def make_image(hoop_pixel=None):
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    if hoop_pixel is not None:
        center = (round(hoop_pixel[0]), round(hoop_pixel[1]))
        cv2.circle(image, center, HOOP_RADIUS, (0, 0, 0), thickness=6)
    return image


class FakeMotion:
    def __init__(self, moving_checks=0, is_out_of_reach=False):
        self.hoop_pose = make_hoop_pose()
        self.moves = []
        self._moving_checks = moving_checks
        self._remaining_moving_checks = 0
        self._is_out_of_reach = is_out_of_reach

    def move_to_ik(self, hoop_pose):
        if self._is_out_of_reach:
            raise NoJointConfigError("No joint configuration.")
        self.moves.append(hoop_pose.copy())
        self.hoop_pose = hoop_pose.copy()
        self._remaining_moving_checks = self._moving_checks

    def get_hoop_pose(self):
        return self.hoop_pose.copy()

    def is_in_motion(self):
        if self._remaining_moving_checks == 0:
            return False
        self._remaining_moving_checks -= 1
        return True


class FakeFrames:
    """Camera that sees the hoop where the fake motion has put it.

    Args:
        motion: Fake motion the hoop position is taken from.
        has_frame: False for a camera that has not delivered a frame yet.
        is_hoop_visible: False for images without the hoop.
        shifted_frame_index: Frame in which the hoop is drawn in a wrong
            place, counted from 0. None for no such frame.
    """

    def __init__(
        self,
        motion,
        has_frame=True,
        is_hoop_visible=True,
        shifted_frame_index=None,
    ):
        self._motion = motion
        self._has_frame = has_frame
        self._is_hoop_visible = is_hoop_visible
        self._shifted_frame_index = shifted_frame_index
        self._frame_index = 0

    @property
    def latest(self):
        if not self._has_frame:
            return None
        hoop_pixel = None
        if self._is_hoop_visible:
            hoop_pixel = to_pixel(self._motion.hoop_pose[:3, 3])
            if self._frame_index == self._shifted_frame_index:
                hoop_pixel = (hoop_pixel[0] + 300.0, hoop_pixel[1])
        self._frame_index += 1
        return Frame(make_image(hoop_pixel), timestamp=0.0, markers=())


class FakeKeyboard:
    def __init__(self, keys):
        self._keys = list(keys)
        self.events = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("exit")

    def read_key(self):
        return self._keys.pop(0)

    def discard_pending(self):
        self.events.append("discard_pending")


class FakeDisplay:
    def __init__(self):
        self.views = []
        self.events = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("exit")

    def show(self, view):
        self.views.append(view)


def run_mode(tmp_path, keys, motion=None, **frames_options):
    motion = motion or FakeMotion()
    keyboard = FakeKeyboard([*keys, *EXIT_KEYS])
    display = FakeDisplay()
    homography_path = tmp_path / "homography.json"
    mode = ManualMode(
        motion,
        FakeFrames(motion, **frames_options),
        keyboard,
        display,
        homography_path,
    )
    mode.run()
    return motion, keyboard, display, homography_path


def move_keys(x_grid_steps, y_grid_steps):
    """Keys that move the hoop by whole grid steps, from and to the x row."""

    def arrows(grid_steps):
        key = Key.RIGHT if grid_steps > 0 else Key.LEFT
        return [key] * (abs(grid_steps) * STEPS_PER_GRID_STEP)

    return [
        *arrows(x_grid_steps),
        Key.DOWN,
        *arrows(y_grid_steps),
        Key.UP,
        Key.ENTER,
    ]


def grid_calibration_keys():
    """Keys that record a 3 x 3 grid of points, starting where the hoop is."""
    grid_moves = [
        (0, 0),
        (1, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (-1, 0),
        (0, 1),
        (1, 0),
        (1, 0),
    ]
    keys = []
    for x_grid_steps, y_grid_steps in grid_moves:
        keys += [*move_keys(x_grid_steps, y_grid_steps), Key.SPACE]
    return keys


# ManualState.handle_key


@pytest.mark.parametrize(
    ("row", "key", "expected_row"),
    [
        (ManualRow.X, Key.DOWN, ManualRow.Y),
        (ManualRow.Z, Key.DOWN, ManualRow.EXIT),
        (ManualRow.EXIT, Key.UP, ManualRow.Z),
        (ManualRow.X, Key.UP, ManualRow.X),
        (ManualRow.EXIT, Key.DOWN, ManualRow.EXIT),
    ],
    ids=["down", "down_to_exit", "up", "up_at_first", "down_at_last"],
)
def test_up_and_down_select_the_neighbouring_row(row, key, expected_row):
    state, command = make_state(row).handle_key(key)

    assert state.selected_row is expected_row
    assert command is None
    np.testing.assert_allclose(state.target_position, START_POSITION)


@pytest.mark.parametrize(
    ("row", "key", "expected_change"),
    [
        (ManualRow.X, Key.RIGHT, (MANUAL_STEP, 0.0, 0.0)),
        (ManualRow.X, Key.LEFT, (-MANUAL_STEP, 0.0, 0.0)),
        (ManualRow.Y, Key.RIGHT, (0.0, MANUAL_STEP, 0.0)),
        (ManualRow.Z, Key.LEFT, (0.0, 0.0, -MANUAL_STEP)),
        (ManualRow.EXIT, Key.RIGHT, (0.0, 0.0, 0.0)),
    ],
    ids=["x_right", "x_left", "y_right", "z_left", "exit_row"],
)
def test_left_and_right_change_the_selected_axis(row, key, expected_change):
    state, command = make_state(row).handle_key(key)

    np.testing.assert_allclose(
        state.target_position, np.add(START_POSITION, expected_change)
    )
    assert state.selected_row is row
    assert command is None


def test_changing_the_target_keeps_the_previous_state():
    previous_state = make_state()

    previous_state.handle_key(Key.RIGHT)

    np.testing.assert_allclose(previous_state.target_position, START_POSITION)


@pytest.mark.parametrize(
    ("row", "key", "expected_command"),
    [
        (ManualRow.X, Key.ENTER, ManualCommand.MOVE),
        (ManualRow.Z, Key.ENTER, ManualCommand.MOVE),
        (ManualRow.EXIT, Key.ENTER, ManualCommand.EXIT),
        (ManualRow.Y, Key.SPACE, ManualCommand.RECORD),
        (ManualRow.EXIT, Key.SPACE, ManualCommand.RECORD),
    ],
    ids=[
        "enter_on_x",
        "enter_on_z",
        "enter_on_exit",
        "space_on_axis",
        "space_on_exit",
    ],
)
def test_enter_and_space_give_a_command(row, key, expected_command):
    previous_state = make_state(row)

    state, command = previous_state.handle_key(key)

    assert command is expected_command
    assert state is previous_state


# ManualMode.run


def test_first_view_shows_the_measured_position_as_target_and_current(
    tmp_path,
):
    _, _, display, _ = run_mode(tmp_path, [])

    view = display.views[0]
    assert view.state.selected_row is ManualRow.X
    np.testing.assert_allclose(view.state.target_position, START_POSITION)
    np.testing.assert_allclose(view.current_position, START_POSITION)
    assert view.recorded_points == 0
    assert not view.message


def test_exit_leaves_without_moving_the_robot(tmp_path):
    motion, _, _, _ = run_mode(tmp_path, [Key.RIGHT])

    assert motion.moves == []


def test_arrows_change_the_target_and_keep_the_current_position(tmp_path):
    _, _, display, _ = run_mode(tmp_path, [Key.RIGHT, Key.RIGHT])

    view = display.views[2]
    np.testing.assert_allclose(
        view.state.target_position,
        np.add(START_POSITION, (2 * MANUAL_STEP, 0.0, 0.0)),
    )
    np.testing.assert_allclose(view.current_position, START_POSITION)


def test_enter_moves_the_hoop_to_the_whole_target_and_keeps_orientation(
    tmp_path,
):
    keys = [Key.RIGHT, Key.DOWN, Key.LEFT, Key.ENTER]

    motion, _, display, _ = run_mode(tmp_path, keys)

    expected_position = np.add(START_POSITION, (MANUAL_STEP, -MANUAL_STEP, 0.0))
    assert len(motion.moves) == 1
    np.testing.assert_allclose(motion.moves[0][:3, 3], expected_position)
    np.testing.assert_allclose(
        motion.moves[0][:3, :3], make_hoop_pose()[:3, :3]
    )
    np.testing.assert_allclose(
        display.views[-1].current_position, expected_position
    )


def test_second_move_starts_from_the_target_of_the_first_one(tmp_path):
    keys = [Key.RIGHT, Key.ENTER, Key.RIGHT, Key.ENTER]

    motion, _, _, _ = run_mode(tmp_path, keys)

    np.testing.assert_allclose(
        motion.moves[1][:3, 3],
        np.add(START_POSITION, (2 * MANUAL_STEP, 0.0, 0.0)),
    )


def test_move_waits_for_the_robot_and_then_discards_pressed_keys(tmp_path):
    motion = FakeMotion(moving_checks=1)

    _, keyboard, display, _ = run_mode(tmp_path, [Key.RIGHT, Key.ENTER], motion)

    assert not motion.is_in_motion()
    assert keyboard.events == ["enter", "discard_pending", "exit"]
    messages = [view.message for view in display.views]
    assert messages.index("Moving...") < messages.index("Move done.")


def test_target_out_of_reach_is_reported_and_the_mode_goes_on(tmp_path):
    motion = FakeMotion(is_out_of_reach=True)

    _, _, display, _ = run_mode(tmp_path, [Key.RIGHT, Key.ENTER], motion)

    view = display.views[-1]
    assert "out of reach" in view.message
    np.testing.assert_allclose(view.current_position, START_POSITION)


def test_space_records_a_point(tmp_path):
    _, _, display, _ = run_mode(tmp_path, [Key.SPACE])

    view = display.views[1]
    assert view.recorded_points == 1
    assert view.message == f"Point 1/{CALIBRATION_POINT_COUNT} recorded."


@pytest.mark.parametrize(
    ("frames_options", "message"),
    [
        ({"has_frame": False}, "No camera image"),
        ({"is_hoop_visible": False}, "Hoop not found"),
    ],
    ids=["no_frame", "no_hoop"],
)
def test_point_without_a_detected_hoop_is_not_recorded(
    tmp_path, frames_options, message
):
    _, _, display, _ = run_mode(tmp_path, [Key.SPACE], **frames_options)

    view = display.views[1]
    assert view.recorded_points == 0
    assert message in view.message


def test_point_at_another_height_is_not_recorded(tmp_path):
    raise_hoop = [Key.DOWN, Key.DOWN, *[Key.RIGHT] * 5, Key.ENTER]

    _, _, display, _ = run_mode(tmp_path, [Key.SPACE, *raise_hoop, Key.SPACE])

    view = display.views[-1]
    assert view.recorded_points == 1
    assert "differs from the first point" in view.message


def test_last_point_saves_a_homography_that_maps_the_image_to_the_robot(
    tmp_path,
):
    _, _, display, homography_path = run_mode(tmp_path, grid_calibration_keys())

    homography = load_homography(homography_path)
    far_position = np.add(START_POSITION, (2 * GRID_STEP, 2 * GRID_STEP, 0.0))
    image_points = PixelPoints(
        np.array([to_pixel(START_POSITION), to_pixel(far_position)])
    )
    np.testing.assert_allclose(
        np.asarray(image_to_plane(image_points, homography)),
        np.array([START_POSITION[:2], far_position[:2]]),
        atol=5e-4,
    )
    view = display.views[-1]
    assert "Homography saved" in view.message
    assert "Restart the application" in view.message
    assert view.recorded_points == 0


def test_homography_with_a_large_error_is_not_saved(tmp_path):
    _, _, display, homography_path = run_mode(
        tmp_path, grid_calibration_keys(), shifted_frame_index=4
    )

    view = display.views[-1]
    assert not homography_path.exists()
    assert "homography not saved" in view.message
    assert view.recorded_points == 0


def test_points_recorded_at_one_place_do_not_save_a_homography(tmp_path):
    keys = [Key.SPACE] * CALIBRATION_POINT_COUNT

    _, _, display, homography_path = run_mode(tmp_path, keys)

    view = display.views[-1]
    assert not homography_path.exists()
    assert "Homography not saved" in view.message
    assert view.recorded_points == 0


def test_next_run_starts_without_the_points_of_the_previous_one(tmp_path):
    motion = FakeMotion()
    display = FakeDisplay()
    keyboard = FakeKeyboard([Key.SPACE, *EXIT_KEYS, *EXIT_KEYS])
    mode = ManualMode(
        motion, FakeFrames(motion), keyboard, display, tmp_path / "h.json"
    )
    mode.run()
    views_of_first_run = len(display.views)

    mode.run()

    assert display.views[views_of_first_run - 1].recorded_points == 1
    assert display.views[views_of_first_run].recorded_points == 0


def test_keyboard_and_display_are_closed_when_the_robot_fails(tmp_path):
    class FailingMotion(FakeMotion):
        def is_in_motion(self):
            raise RuntimeError("Check ready: error.")

    motion = FailingMotion()
    keyboard = FakeKeyboard([Key.ENTER])
    display = FakeDisplay()
    mode = ManualMode(
        motion, FakeFrames(motion), keyboard, display, tmp_path / "h.json"
    )

    with pytest.raises(RuntimeError, match="Check ready"):
        mode.run()

    assert keyboard.events == ["enter", "exit"]
    assert display.events == ["enter", "exit"]
