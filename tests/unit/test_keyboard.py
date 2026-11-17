import os
import pty
import termios

import pytest

from hoop_guide.keyboard import Key
from hoop_guide.keyboard import TerminalKeyReader
from hoop_guide.keyboard import parse_key


@pytest.mark.parametrize(
    ("characters", "key"),
    [
        ("\x1b[A", Key.UP),
        ("\x1b[B", Key.DOWN),
        ("\x1b[C", Key.RIGHT),
        ("\x1b[D", Key.LEFT),
        ("\x1bOA", Key.UP),
        ("\r", Key.ENTER),
        ("\n", Key.ENTER),
        (" ", Key.SPACE),
    ],
    ids=[
        "up",
        "down",
        "right",
        "left",
        "up_application_mode",
        "carriage_return",
        "line_feed",
        "space",
    ],
)
def test_parse_key_of_a_used_key(characters, key):
    assert parse_key(characters) is key


@pytest.mark.parametrize(
    "characters",
    ["a", "\x1b", "\x1b[Z", ""],
    ids=["letter", "escape", "other_sequence", "nothing"],
)
def test_parse_key_of_an_unused_key_is_none(characters):
    assert parse_key(characters) is None


@pytest.fixture
def terminal():
    """File descriptors of a pseudo terminal: what is typed, what reads it."""
    typing_end, reading_end = pty.openpty()
    yield typing_end, reading_end
    os.close(typing_end)
    os.close(reading_end)


def line_and_echo_flags(file_descriptor):
    """Whether the terminal waits for Enter and prints typed keys."""
    local_flags = termios.tcgetattr(file_descriptor)[3]
    return local_flags & (termios.ICANON | termios.ECHO)


def test_reader_returns_keys_in_the_order_they_were_pressed(terminal):
    typing_end, reading_end = terminal

    with TerminalKeyReader(reading_end) as reader:
        os.write(typing_end, b"\x1b[A \r")
        keys = [reader.read_key() for _ in range(3)]

    assert keys == [Key.UP, Key.SPACE, Key.ENTER]


def test_reader_skips_keys_without_a_meaning(terminal):
    typing_end, reading_end = terminal

    with TerminalKeyReader(reading_end) as reader:
        os.write(typing_end, b"ab\x1b[D")
        key = reader.read_key()

    assert key is Key.LEFT


def test_reader_discards_keys_pressed_before_the_call(terminal):
    typing_end, reading_end = terminal

    with TerminalKeyReader(reading_end) as reader:
        os.write(typing_end, b"\r\r")
        reader.discard_pending()
        os.write(typing_end, b" ")
        key = reader.read_key()

    assert key is Key.SPACE


def test_reader_restores_the_terminal_settings_on_exit(terminal):
    _, reading_end = terminal
    flags_before = line_and_echo_flags(reading_end)

    with TerminalKeyReader(reading_end):
        flags_inside = line_and_echo_flags(reading_end)

    assert flags_before == termios.ICANON | termios.ECHO
    assert flags_inside == 0
    assert line_and_echo_flags(reading_end) == flags_before


def test_reader_on_a_closed_terminal_raises_end_of_file():
    reading_end, writing_end = os.pipe()
    os.close(writing_end)
    reader = TerminalKeyReader(reading_end)

    with pytest.raises(EOFError):
        reader.read_key()

    os.close(reading_end)
