"""Entry point of the application, see MANUAL.md."""

import contextlib
import logging
import sys

from hoop_guide.app import App
from hoop_guide.camera_stream import CameraStream
from hoop_guide.cli import parse_args
from hoop_guide.hardware.robot import Robot
from hoop_guide.logs import LOG_FILE_NAME
from hoop_guide.logs import setup_logging
from hoop_guide.ui import RichUI
from hoop_guide.viewer.window import Viewer

logger = logging.getLogger(__name__)


def main() -> int:
    """Wires the application together and runs it.

    Returns:
        Exit code.
    """
    config = parse_args()
    run_dir = setup_logging()
    logger.info("Started with %s.", config)
    ui = RichUI()
    try:
        robot = Robot(config.robot, config.is_real)
    except Exception:
        logger.exception("Connection to the robot failed.")
        ui.error(
            "Connection to the robot failed."
            f" Details in {run_dir / LOG_FILE_NAME}."
        )
        return 1
    with contextlib.ExitStack() as stack:
        if config.should_show_view and robot.is_real:
            viewer = stack.enter_context(Viewer())
            stack.enter_context(CameraStream(robot, viewer.show))
        App(config, robot, ui, run_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
