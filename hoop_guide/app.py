"""Application flow: startup, main menu and shutdown."""

import contextlib
import functools
import logging
import signal
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

from hoop_guide.cli import Config
from hoop_guide.hardware.interfaces import RobotControl
from hoop_guide.logs import LOG_FILE_NAME
from hoop_guide.ui import MenuOption
from hoop_guide.ui import Status
from hoop_guide.ui import UserInterface

logger = logging.getLogger(__name__)

PUZZLE_MODES = ("A1", "A2", "B", "C", "D")
QUIT_KEY = "0"
NO_MODE = "None"


@contextlib.contextmanager
def _ignore_ctrl_c() -> Iterator[None]:
    # set handler in case signal comes and keep the prev handler in var
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


class App:
    """Terminal application controlling one robot.

    Args:
        config: Configuration from the command line.
        robot: Real or virtual robot.
        ui: User interface.
        run_dir: Folder with logs of this run.
    """

    def __init__(
        self,
        config: Config,
        robot: RobotControl,
        ui: UserInterface,
        run_dir: Path,
    ) -> None:
        self._config = config
        self._robot = robot
        self._ui = ui
        self._run_dir = run_dir
        self._actions = self._build_actions()

    def run(self) -> None:
        """Runs until the operator quits, then always shuts the robot down."""
        try:
            if self._start():
                self._menu_loop()
        except KeyboardInterrupt:
            self._info("Interrupted by Ctrl+C.")
        finally:
            self._shutdown()

    def _build_actions(self) -> dict[str, tuple[str, Callable[[], None]]]:
        entries: list[tuple[str, Callable[[], None]]] = [
            (f"Mode {mode}", functools.partial(self._run_mode, mode))
            for mode in PUZZLE_MODES
        ]
        entries += [
            ("Soft home", self._soft_home),
            ("Release", self._release),
            ("Reset motors", self._reset_motors),
        ]
        return {str(key): entry for key, entry in enumerate(entries, start=1)}

    def _start(self) -> bool:
        self._show("Starting")
        if not self._robot.is_real:
            self._info("Virtual robot, initialization skipped.")
            return True
        self._warning(
            f"Connected to the REAL robot {self._robot.name}. It will move,"
            " keep the Emergency Stop button within reach."
        )
        self._show("Waiting for operator")
        self._ui.confirm("Press Enter to initialize the robot...")
        self._show("Initializing")
        try:
            with self._ui.busy("Initializing robot..."):
                self._robot.initialize(should_home=self._config.should_home)
        except Exception:
            self._fail("Initialization failed.")
            return False
        self._info("Robot initialized.")
        return True

    def _menu_loop(self) -> None:
        options = [
            MenuOption(key, label) for key, (label, _) in self._actions.items()
        ]
        options.append(MenuOption(QUIT_KEY, "Quit"))
        while True:
            self._show("Idle")
            try:
                choice = self._ui.menu(options)
            except EOFError:
                choice = QUIT_KEY
            if choice == QUIT_KEY:
                logger.info("Quit selected.")
                return
            if choice not in self._actions:
                self._warning(f"Invalid option '{choice}'.")
                continue
            label, action = self._actions[choice]
            logger.info("Selected %s.", label)
            action()

    def _run_mode(self, mode: str) -> None:
        self._warning(f"Mode {mode} is not implemented yet.")

    def _soft_home(self) -> None:
        self._hardware_action("Soft home", self._robot.soft_home)

    def _release(self) -> None:
        self._hardware_action(
            "Release",
            self._robot.release,
            confirmation="The arm can slightly drop, make sure it is not above"
            " the work surface. Press Enter to release...",
        )

    def _reset_motors(self) -> None:
        self._hardware_action("Reset motors", self._robot.reset_motors)

    def _hardware_action(
        self,
        label: str,
        operation: Callable[[], None],
        confirmation: str | None = None,
    ) -> None:
        if not self._robot.is_real:
            self._warning(f"{label} is not available with the virtual robot.")
            return
        if confirmation is not None:
            self._ui.confirm(confirmation)
        self._show(label)
        try:
            with self._ui.busy(f"{label}..."):
                operation()
        except Exception:
            self._fail(f"{label} failed.")
            return
        self._info(f"{label} done.")

    def _shutdown(self) -> None:
        with _ignore_ctrl_c():
            if self._robot.is_real:
                self._show("Shutting down")
                try:
                    with self._ui.busy(
                        "Returning robot home and closing the connection..."
                    ):
                        self._robot.shutdown()
                except Exception:
                    self._fail("Shutdown failed, check the robot.")
                else:
                    self._info("Robot is home, connection closed.")
            self._info("Bye.")

    def _show(self, state: str) -> None:
        logger.info("State: %s.", state)
        self._ui.header(
            Status(
                robot_name=self._robot.name,
                is_real=self._robot.is_real,
                mode=NO_MODE,
                state=state,
                run_dir=self._run_dir,
            )
        )

    def _info(self, message: str) -> None:
        logger.info(message)
        self._ui.info(message)

    def _warning(self, message: str) -> None:
        logger.warning(message)
        self._ui.warning(message)

    def _fail(self, message: str) -> None:
        logger.exception(message)
        self._ui.error(f"{message} Details in {self._run_dir / LOG_FILE_NAME}.")
