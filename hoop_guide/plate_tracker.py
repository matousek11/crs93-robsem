"""Table motion measured once from the first camera frames with the plate."""

from hoop_guide.estimation import TableMotion
from hoop_guide.estimation import fit_table_motion
from hoop_guide.plate import PlatePose
from hoop_guide.plate import estimate_plate_pose
from hoop_guide.types import Homography
from hoop_guide.vision import Frame

# The camera gives about 4.3 frames per second, so 130 frames take about
# 30 s. In the lab on 2026-11-25, 50 frames (11.5 s) gave the angular
# velocity 0.5 to 3 % off, the plate was up to 14 deg off its prediction
# 40 s later, when the run ends. The error of the fitted slope falls with
# the measured time to the power of 1.5, so 30 s give it about 4 times
# smaller.
PLATE_SAMPLE_COUNT = 130
"""Number of plate poses the table motion is fitted to."""

# TODO: measure in the lab.
CAMERA_LATENCY = 0.0
"""Time from the exposure of an image to the timestamp of its frame [s]."""


class PlateTracker:
    """Measures the table motion once and keeps the newest plate pose.

    The plate poses of the first PLATE_SAMPLE_COUNT frames that show the
    plate are collected, then the motion is fitted to them and does not
    change any more. Later frames only update plate_pose. The motion
    predicts the plate pose from the time alone, so it is used while the
    arm hides the markers.

    update is called from the camera thread, the properties can be read
    from another one. They hold immutable values replaced as a whole, so a
    reader gets either the previous or the new one.

    Args:
        homography: Calibrated image to table plane homography.
        camera_latency: Subtracted from frame timestamps, so the motion is
            fitted and predicts in time.monotonic of the exposure [s].
    """

    def __init__(
        self, homography: Homography, camera_latency: float = CAMERA_LATENCY
    ) -> None:
        self._homography = homography
        self._camera_latency = camera_latency
        self._samples: list[tuple[float, PlatePose]] = []
        self._plate_pose: PlatePose | None = None
        self._motion: TableMotion | None = None

    @property
    def plate_pose(self) -> PlatePose | None:
        """Plate pose in the newest frame, None when no plate marker is seen."""
        return self._plate_pose

    @property
    def motion(self) -> TableMotion | None:
        """Motion fitted to the first PLATE_SAMPLE_COUNT plate poses.

        None until they are collected.
        """
        return self._motion

    def update(self, frame: Frame) -> None:
        """Estimates the plate pose and collects it until the motion is fitted.

        Args:
            frame: Newest camera frame.

        Raises:
            ValueError: While the motion is not fitted yet, the frame is not
                newer than the last one in which the plate was seen.
        """
        plate_pose = estimate_plate_pose(frame.markers, self._homography)
        if plate_pose is not None and self._motion is None:
            self._collect(frame.timestamp - self._camera_latency, plate_pose)
        self._plate_pose = plate_pose

    def _collect(self, exposure_time: float, plate_pose: PlatePose) -> None:
        if self._samples and exposure_time <= self._samples[-1][0]:
            raise ValueError(
                "Frames must come in the order of their timestamps."
            )
        self._samples.append((exposure_time, plate_pose))
        if len(self._samples) >= PLATE_SAMPLE_COUNT:
            self._motion = fit_table_motion(self._samples)
