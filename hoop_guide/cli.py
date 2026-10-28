"""Command line arguments, see MANUAL.md."""

import argparse
import dataclasses
from collections.abc import Sequence

from hoop_guide.hardware.robot import ROBOT_CLASSES

DEFAULT_ROBOT = "crs93"


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration from the command line.

    Attributes:
        is_real: Connect to the real robot instead of the virtual one.
        robot: Robot name, crs93 or crs97.
        should_home: If True, initialization performs hard home (finding
            the joint limits) followed by soft home (moving to the home
            position). If False, both are skipped, which is only safe when
            the control unit stayed powered on since the last homing.
        should_show_view: Show the camera window. Only the real robot has
            a camera.
    """

    is_real: bool
    robot: str
    should_home: bool
    should_show_view: bool


def parse_args(argv: Sequence[str] | None = None) -> Config:
    """Parses command line arguments.

    Args:
        argv: Arguments without the program name, sys.argv if None.

    Returns:
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description="Guides the CRS robot hoop along a rotating rod.",
        epilog="See MANUAL.md for details.",
    )
    parser.add_argument(
        "--real",
        dest="is_real",
        action="store_true",
        help="connect to the real robot (default: virtual robot)",
    )
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_CLASSES),
        default=DEFAULT_ROBOT,
        help=f"robot station (default: {DEFAULT_ROBOT})",
    )
    parser.add_argument(
        "--no-home",
        dest="should_home",
        action="store_false",
        help="skip hard and soft home during initialization",
    )
    parser.add_argument(
        "--no-view",
        dest="should_show_view",
        action="store_false",
        help="do not show the camera window",
    )
    args = parser.parse_args(argv)
    return Config(
        is_real=args.is_real,
        robot=args.robot,
        should_home=args.should_home,
        should_show_view=args.should_show_view,
    )
