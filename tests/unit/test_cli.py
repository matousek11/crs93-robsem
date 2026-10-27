import pytest

from hoop_guide.cli import Config
from hoop_guide.cli import parse_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], Config(is_real=False, robot="crs93", should_home=True)),
        (["--real"], Config(is_real=True, robot="crs93", should_home=True)),
        (
            ["--real", "--robot", "crs97"],
            Config(is_real=True, robot="crs97", should_home=True),
        ),
        (
            ["--real", "--no-home"],
            Config(is_real=True, robot="crs93", should_home=False),
        ),
    ],
    ids=["default-virtual", "real", "real-crs97", "real-no-home"],
)
def test_parse_args_returns_config(argv, expected):
    assert parse_args(argv) == expected


def test_parse_args_rejects_unknown_robot():
    with pytest.raises(SystemExit):
        parse_args(["--robot", "crs42"])
