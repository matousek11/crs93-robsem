import contextlib
import signal

import pytest

from hoop_guide.app import App
from hoop_guide.cli import Config
from hoop_guide.rods import Puzzle

MODE_KEYS = {"1": Puzzle.A1, "2": Puzzle.A2, "3": Puzzle.B, "4": Puzzle.C, "5": Puzzle.D}
HARDWARE_ACTIONS = {"6": "soft_home", "7": "release", "8": "reset_motors"}


class FakeRobot:
    def __init__(self, log, is_real=True, failures=None):
        self.name = "CRS93"
        self.is_real = is_real
        self._log = log
        self._failures = failures or {}
        self.ctrl_c_handler_during_shutdown = None

    def _call(self, name):
        self._log.append(("robot", name))
        if name in self._failures:
            raise self._failures[name]

    def initialize(self, should_home):
        self._call(f"initialize(should_home={should_home})")

    def soft_home(self):
        self._call("soft_home")

    def release(self):
        self._call("release")

    def reset_motors(self):
        self._call("reset_motors")

    def shutdown(self):
        self.ctrl_c_handler_during_shutdown = signal.getsignal(signal.SIGINT)
        self._call("shutdown")


class FakeUI:
    def __init__(self, log, answers=()):
        self._log = log
        self._answers = list(answers)

    def header(self, status):
        self._log.append(("header", status.state))

    def info(self, message):
        self._log.append(("info", message))

    def warning(self, message):
        self._log.append(("warning", message))

    def error(self, message):
        self._log.append(("error", message))

    def confirm(self, prompt):
        self._log.append(("confirm", prompt))

    def menu(self, options):
        answer = self._answers.pop(0) if self._answers else EOFError()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @contextlib.contextmanager
    def busy(self, message):
        self._log.append(("busy", message))
        yield


class FakePuzzleMode:
    def __init__(self, log, failure=None):
        self._log = log
        self._failure = failure

    def run(self, puzzle):
        self._log.append(("puzzle", puzzle))
        if self._failure is not None:
            raise self._failure


def run_app(  # noqa: PLR0913, PLR0917
    tmp_path,
    answers=(),
    is_real=True,
    should_home=True,
    failures=None,
    extra_actions=(),
    puzzle_failure=None,
):
    log: list[tuple[str, object]] = []
    robot = FakeRobot(log, is_real=is_real, failures=failures)
    ui = FakeUI(log, answers)
    config = Config(
        is_real=is_real,
        robot="crs93",
        should_home=should_home,
        should_show_view=False,
    )
    puzzle_mode = FakePuzzleMode(log, puzzle_failure)
    App(
        config,
        robot,
        ui,
        tmp_path,
        puzzle_mode=puzzle_mode,
        extra_actions=extra_actions,
    ).run()
    return log, robot


def robot_calls(log):
    return [name for source, name in log if source == "robot"]


def messages(log, kind):
    return [text for source, text in log if source == kind]


def test_virtual_robot_skips_initialization_and_shutdown(tmp_path):
    log, _ = run_app(tmp_path, is_real=False)

    assert robot_calls(log) == []
    assert "Virtual robot, initialization skipped." in messages(log, "info")


def test_real_robot_warns_and_waits_for_enter_before_initialization(tmp_path):
    log, _ = run_app(tmp_path)

    warning = log.index(("warning", messages(log, "warning")[0]))
    confirm = next(i for i, entry in enumerate(log) if entry[0] == "confirm")
    initialize = log.index(("robot", "initialize(should_home=True)"))
    assert "REAL robot" in log[warning][1]
    assert warning < confirm < initialize


@pytest.mark.parametrize("should_home", [True, False], ids=["home", "no-home"])
def test_initialization_respects_home_option(tmp_path, should_home):
    log, _ = run_app(tmp_path, should_home=should_home)

    assert robot_calls(log)[0] == f"initialize(should_home={should_home})"


def test_quit_shuts_robot_down(tmp_path):
    log, _ = run_app(tmp_path, answers=["0"])

    assert robot_calls(log) == ["initialize(should_home=True)", "shutdown"]


def test_end_of_input_quits(tmp_path):
    log, _ = run_app(tmp_path, answers=[EOFError()])

    assert robot_calls(log)[-1] == "shutdown"


def test_ctrl_c_in_menu_shuts_robot_down(tmp_path):
    log, _ = run_app(tmp_path, answers=[KeyboardInterrupt()])

    assert robot_calls(log)[-1] == "shutdown"
    assert "Interrupted by Ctrl+C." in messages(log, "info")


def test_ctrl_c_during_initialization_shuts_robot_down(tmp_path):
    failures = {"initialize(should_home=True)": KeyboardInterrupt()}

    log, _ = run_app(tmp_path, failures=failures)

    assert robot_calls(log) == ["initialize(should_home=True)", "shutdown"]


def test_ctrl_c_is_ignored_during_shutdown_and_restored_after(tmp_path):
    before = signal.getsignal(signal.SIGINT)

    _, robot = run_app(tmp_path, answers=["0"])

    assert robot.ctrl_c_handler_during_shutdown == signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == before


def test_failed_initialization_skips_menu_and_shuts_down(tmp_path):
    failures = {"initialize(should_home=True)": RuntimeError("no controller")}

    log, _ = run_app(tmp_path, answers=["6"], failures=failures)

    assert robot_calls(log) == ["initialize(should_home=True)", "shutdown"]
    assert messages(log, "error")[0].startswith("Initialization failed.")


def test_failed_shutdown_shows_error(tmp_path):
    failures = {"shutdown": RuntimeError("serial closed")}

    log, _ = run_app(tmp_path, answers=["0"], failures=failures)

    assert messages(log, "error")[0].startswith("Shutdown failed")


def test_invalid_option_shows_warning_and_returns_to_menu(tmp_path):
    log, _ = run_app(tmp_path, answers=["42", "6", "0"])

    assert "Invalid option '42'." in messages(log, "warning")
    assert "soft_home" in robot_calls(log)


@pytest.mark.parametrize(
    ("key", "puzzle"),
    MODE_KEYS.items(),
    ids=[p.value for p in MODE_KEYS.values()],
)
def test_mode_runs_the_puzzle_of_its_key(tmp_path, key, puzzle):
    log, _ = run_app(tmp_path, answers=[key, "0"])

    assert messages(log, "puzzle") == [puzzle]


def test_mode_not_implemented_yet_is_reported_and_the_menu_goes_on(tmp_path):
    log, _ = run_app(
        tmp_path, answers=["1", "1", "0"], puzzle_failure=NotImplementedError()
    )

    assert (
        messages(log, "warning")[-2:] == ["Mode A1 is not implemented yet."] * 2
    )


def test_failed_mode_is_reported_and_the_menu_goes_on(tmp_path):
    log, _ = run_app(
        tmp_path, answers=["3", "3", "0"], puzzle_failure=RuntimeError("stop")
    )

    assert len(messages(log, "error")) == 2
    assert messages(log, "error")[0].startswith("Mode B failed.")


@pytest.mark.parametrize(("key", "call"), HARDWARE_ACTIONS.items())
def test_hardware_action_calls_robot(tmp_path, key, call):
    log, _ = run_app(tmp_path, answers=[key, "0"])

    assert robot_calls(log) == [
        "initialize(should_home=True)",
        call,
        "shutdown",
    ]


@pytest.mark.parametrize("key", HARDWARE_ACTIONS)
def test_hardware_action_is_blocked_on_virtual_robot(tmp_path, key):
    log, _ = run_app(tmp_path, answers=[key, "0"], is_real=False)

    assert robot_calls(log) == []
    assert messages(log, "warning")[-1].endswith(
        "is not available with the virtual robot."
    )


def test_release_asks_for_confirmation_first(tmp_path):
    log, _ = run_app(tmp_path, answers=["7", "0"])

    confirms = [i for i, entry in enumerate(log) if entry[0] == "confirm"]
    assert confirms[-1] < log.index(("robot", "release"))


def test_failed_hardware_action_shows_error_and_returns_to_menu(tmp_path):
    failures = {"soft_home": RuntimeError("motor error")}

    log, _ = run_app(tmp_path, answers=["6", "8", "0"], failures=failures)

    assert messages(log, "error")[0].startswith("Soft home failed.")
    assert "reset_motors" in robot_calls(log)


EXTRA_ACTION_KEY = "9"


def test_extra_action_is_offered_after_the_built_in_ones(tmp_path):
    calls = []
    extra_actions = [("Manual move", lambda: calls.append("manual move"))]

    run_app(tmp_path, answers=[EXTRA_ACTION_KEY], extra_actions=extra_actions)

    assert calls == ["manual move"]


def test_without_extra_actions_their_key_is_invalid(tmp_path):
    log, _ = run_app(tmp_path, answers=[EXTRA_ACTION_KEY])

    assert f"Invalid option '{EXTRA_ACTION_KEY}'." in messages(log, "warning")


def test_failed_extra_action_is_reported_and_the_menu_goes_on(tmp_path):
    calls = []

    def action():
        calls.append("manual move")
        raise RuntimeError("Check ready: error.")

    log, _ = run_app(
        tmp_path,
        answers=[EXTRA_ACTION_KEY, EXTRA_ACTION_KEY],
        extra_actions=[("Manual move", action)],
    )

    assert calls == ["manual move", "manual move"]
    assert any("Manual move failed." in text for text in messages(log, "error"))
    assert robot_calls(log)[-1] == "shutdown"
