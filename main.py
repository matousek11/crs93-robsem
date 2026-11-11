"""Entry point of the application, see MANUAL.md."""

import contextlib
import logging
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path

from hoop_guide.app import App
from hoop_guide.calibration import load_homography
from hoop_guide.camera_stream import CameraStream
from hoop_guide.cli import parse_args
from hoop_guide.estimation import fit_table_motion
from hoop_guide.hardware.robot import Robot
from hoop_guide.logs import LOG_FILE_NAME
from hoop_guide.logs import setup_logging
from hoop_guide.plate import PlatePose
from hoop_guide.plate import estimate_plate_pose
from hoop_guide.types import Homography
from hoop_guide.ui import RichUI
from hoop_guide.viewer.fps import FpsMeter
from hoop_guide.viewer.overlay import render
from hoop_guide.viewer.window import Viewer
from hoop_guide.vision import Frame
from hoop_guide.vision import process_image

logger = logging.getLogger(__name__)

DEFAULT_HOMOGRAPHY_FILE = Path("config/homography.json")


def _try_load_homography() -> Homography | None:
    if not DEFAULT_HOMOGRAPHY_FILE.is_file():
        return None
    try:
        homography = load_homography(DEFAULT_HOMOGRAPHY_FILE)
        logger.info("Loaded homography from %s.", DEFAULT_HOMOGRAPHY_FILE)
        return homography
    except Exception:
        logger.exception(
            "Failed to load homography from %s.", DEFAULT_HOMOGRAPHY_FILE
        )
        return None


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
        if robot.is_real:
            on_frame = _create_frame_handler(stack, config.should_show_view)
            stack.enter_context(CameraStream(robot, process_image, on_frame))
        App(config, robot, ui, run_dir).run()
    return 0


def _create_frame_handler(
    stack: contextlib.ExitStack[bool | None], should_show_view: bool
) -> Callable[[Frame], None]:
    """Returns what to do with every Frame, opens the window if shown.

    The window is added to the stack before the camera stream, so the stream
    stops first and never sends images to a closed window.
    """
    if not should_show_view:
        return lambda frame: None
    viewer = stack.enter_context(Viewer())
    fps_meter = FpsMeter()
    homography = _try_load_homography()
    samples: deque[tuple[float, PlatePose]] = deque(maxlen=10)

    def show(frame: Frame) -> None:
        plate_pose = (
            estimate_plate_pose(frame.markers, homography)
            if homography is not None
            else None
        )
        if plate_pose is not None:
            samples.append((frame.timestamp, plate_pose))
        table_estimation = fit_table_motion(samples)
        viewer.show(
            render(
                frame,
                fps_meter.tick(frame.timestamp),
                plate_pose=plate_pose,
                homography=homography,
                table_estimation=table_estimation,
            )
        )

    return show


if __name__ == "__main__":
    sys.exit(main())
