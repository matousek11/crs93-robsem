"""Terminal user interface."""

import contextlib
import dataclasses
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Protocol
from typing import Self

from rich.console import Console
from rich.console import Group
from rich.console import RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hoop_guide.calibration import CALIBRATION_POINT_COUNT
from hoop_guide.manual_mode import ManualRow
from hoop_guide.manual_mode import ManualView

MANUAL_MODE_KEYS = (
    "Up/Down row   Left/Right value   Enter move or exit   Space record point"
)
_AXIS_ROWS = (ManualRow.X, ManualRow.Y, ManualRow.Z)


@dataclasses.dataclass(frozen=True)
class Status:
    """What the header shows to the operator.

    Attributes:
        robot_name: Display name of the robot.
        is_real: True for the real robot.
        mode: Running mode, e.g. "A1".
        state: What the application is doing, e.g. "Initializing".
        run_dir: Folder with logs of this run.
    """

    robot_name: str
    is_real: bool
    mode: str
    state: str
    run_dir: Path


@dataclasses.dataclass(frozen=True)
class MenuOption:
    """One item of the main menu.

    Attributes:
        key: What the operator types to select it.
        label: Description shown in the menu.
    """

    key: str
    label: str


class UserInterface(Protocol):
    """Everything the application shows to or asks from the operator."""

    def header(self, status: Status) -> None:
        """Shows the current status."""

    def info(self, message: str) -> None:
        """Shows an information message."""

    def warning(self, message: str) -> None:
        """Shows a warning."""

    def error(self, message: str) -> None:
        """Shows an error."""

    def confirm(self, prompt: str) -> None:
        """Waits until the operator presses Enter."""

    def menu(self, options: Sequence[MenuOption]) -> str:
        """Shows the menu and returns what the operator typed."""

    def busy(self, message: str) -> AbstractContextManager[None]:
        """Shows the message while the block runs."""


class RichUI:
    """User interface in the terminal built on rich."""

    def __init__(self) -> None:
        self._console = Console()

    def header(self, status: Status) -> None:
        """Shows the current status."""
        connection = (
            "[bold red]REAL[/]" if status.is_real else "[bold green]VIRTUAL[/]"
        )
        grid = Table.grid(padding=(0, 2))
        grid.add_row("Robot", f"{status.robot_name} {connection}")
        grid.add_row("Mode", status.mode)
        grid.add_row("State", status.state)
        grid.add_row("Logs", str(status.run_dir))
        self._console.print(Panel(grid, title="hoop_guide", expand=False))

    def info(self, message: str) -> None:
        """Shows an information message."""
        self._console.print(message, markup=False)

    def warning(self, message: str) -> None:
        """Shows a warning."""
        self._console.print(message, style="bold yellow", markup=False)

    def error(self, message: str) -> None:
        """Shows an error."""
        self._console.print(message, style="bold red", markup=False)

    def confirm(self, prompt: str) -> None:
        """Waits until the operator presses Enter."""
        self._console.input(prompt)

    def menu(self, options: Sequence[MenuOption]) -> str:
        """Shows the menu and returns what the operator typed."""
        for option in options:
            self._console.print(f"  {option.key}  {option.label}", markup=False)
        return self._console.input("Select option and press Enter: ").strip()

    @contextlib.contextmanager
    def busy(self, message: str) -> Iterator[None]:
        """Shows the message with a spinner while the block runs."""
        with self._console.status(message):
            yield


class RichManualDisplay:
    """Manual move table in the terminal, redrawn in place.

    The table is on the screen while the display is used as a context.
    """

    def __init__(self) -> None:
        self._live = Live(console=Console(), auto_refresh=False)

    def __enter__(self) -> Self:
        """Starts drawing the table."""
        self._live.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stops drawing the table and leaves its last state on the screen."""
        self._live.stop()

    def show(self, view: ManualView) -> None:
        """Redraws the table."""
        self._live.update(_render_manual_view(view), refresh=True)


def _render_manual_view(view: ManualView) -> RenderableType:
    table = Table.grid(padding=(0, 3))
    table.add_row("", "", "target [m]", "current [m]", style="dim")
    for row in _AXIS_ROWS:
        table.add_row(
            _selection_mark(view, row),
            row.name.lower(),
            f"{view.state.target_position[row.value]:.3f}",
            f"{view.current_position[row.value]:.3f}",
            style=_row_style(view, row),
        )
    table.add_row(
        _selection_mark(view, ManualRow.EXIT),
        "Exit",
        style=_row_style(view, ManualRow.EXIT),
    )
    points = f"points {view.recorded_points}/{CALIBRATION_POINT_COUNT}"
    content = Group(
        table,
        Text(""),
        Text(view.message),
        Text(MANUAL_MODE_KEYS, style="dim"),
    )
    return Panel(content, title=f"Manual move   {points}", expand=False)


def _selection_mark(view: ManualView, row: ManualRow) -> str:
    return ">" if view.state.selected_row is row else " "


def _row_style(view: ManualView, row: ManualRow) -> str:
    return "bold" if view.state.selected_row is row else ""
