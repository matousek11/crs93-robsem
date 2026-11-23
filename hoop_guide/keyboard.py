"""Single key presses read from the terminal."""

import enum
import os
import select
import sys
import termios
import tty
from types import TracebackType
from typing import Any
from typing import Self

ESCAPE = "\x1b"
ESCAPE_SEQUENCE_LENGTH = 3
ESCAPE_SEQUENCE_TIMEOUT = 0.05
"""How long the rest of an arrow key sequence may take to arrive [s]."""


class Key(enum.Enum):
    """Keys the application reacts to."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    ENTER = "enter"
    SPACE = "space"
    STOP = "stop"


# Arrows start with ESC [ in the normal and ESC O in the application mode
# of the terminal.
_KEY_BY_CHARACTERS = {
    "\x1b[A": Key.UP,
    "\x1b[B": Key.DOWN,
    "\x1b[C": Key.RIGHT,
    "\x1b[D": Key.LEFT,
    "\x1bOA": Key.UP,
    "\x1bOB": Key.DOWN,
    "\x1bOC": Key.RIGHT,
    "\x1bOD": Key.LEFT,
    "\r": Key.ENTER,
    "\n": Key.ENTER,
    " ": Key.SPACE,
    "c": Key.STOP,
    "C": Key.STOP,
}


def parse_key(characters: str) -> Key | None:
    """Maps what the terminal sent for one key press to a Key.

    Args:
        characters: One character, or the whole escape sequence of an arrow.

    Returns:
        The key, None for a key the application does not use.
    """
    return _KEY_BY_CHARACTERS.get(characters)


class TerminalKeyReader:
    """Reads key presses one by one, without waiting for Enter.

    While used as a context, the terminal sends every key press at once and
    does not print it. The previous terminal settings come back on exit.

    Args:
        file_descriptor: Terminal to read from, the standard input when None.
    """

    def __init__(self, file_descriptor: int | None = None) -> None:
        self._file_descriptor = (
            sys.stdin.fileno() if file_descriptor is None else file_descriptor
        )
        self._previous_settings: list[Any] | None = None

    def __enter__(self) -> Self:
        """Switches the terminal to single key presses."""
        self._previous_settings = termios.tcgetattr(self._file_descriptor)
        tty.setcbreak(self._file_descriptor)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restores the terminal settings."""
        if self._previous_settings is not None:
            termios.tcsetattr(
                self._file_descriptor,
                termios.TCSADRAIN,
                self._previous_settings,
            )
            self._previous_settings = None

    def read_key(self) -> Key:
        """Waits for the next key press, keys without a meaning are skipped.

        Raises:
            EOFError: The terminal was closed.
        """
        while True:
            key = parse_key(self._read_key_characters())
            if key is not None:
                return key

    def poll_key(self) -> Key | None:
        """Returns a waiting key press at once, keys without a meaning skipped.

        Returns:
            The key, None when no key with a meaning is waiting.

        Raises:
            EOFError: The terminal was closed.
        """
        while self._has_input(timeout=0.0):
            key = parse_key(self._read_key_characters())
            if key is not None:
                return key
        return None

    def discard_pending(self) -> None:
        """Drops key presses that arrived while nobody was reading."""
        termios.tcflush(self._file_descriptor, termios.TCIFLUSH)

    def _read_key_characters(self) -> str:
        characters = self._read_character()
        if characters != ESCAPE:
            return characters
        while len(characters) < ESCAPE_SEQUENCE_LENGTH and self._has_input(
            timeout=ESCAPE_SEQUENCE_TIMEOUT
        ):
            characters += self._read_character()
        return characters

    def _read_character(self) -> str:
        data = os.read(self._file_descriptor, 1)
        if not data:
            raise EOFError("Terminal was closed.")
        return data.decode("latin-1")

    def _has_input(self, timeout: float) -> bool:
        readable, _, _ = select.select([self._file_descriptor], [], [], timeout)
        return bool(readable)
