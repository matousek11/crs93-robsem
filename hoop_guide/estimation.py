"""Table motion: center of rotation, angular velocity and pose prediction.

The table turns at a constant angular velocity about an unknown vertical
axis. The axis need not pass through the plate. Samples are plate poses
in the robot frame, already estimated from markers.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.plate import PlatePose

MIN_SAMPLES = 3
# Unwrapped turn from the first sample to the last. Below this the center
# of rotation is not observable [rad].
MIN_ROTATION_FOR_CENTER = 0.1


@dataclasses.dataclass(frozen=True)
class TableMotion:
    """Constant rotation of the table about a fixed vertical axis.

    The center may lie anywhere in the robot plane, including outside the
    plate. When the samples turn by less than MIN_ROTATION_FOR_CENTER, the
    center is not observable and is stored as the mean plate position.
    predict then keeps that position and only advances the heading.

    Attributes:
        center_x: Center of rotation along the robot x axis [m].
        center_y: Center of rotation along the robot y axis [m].
        angular_velocity: Rate of change of plate heading about z, positive
            counterclockwise [rad/s].
        t_ref: time.monotonic time at which pose_ref holds [s].
        pose_ref: Plate pose fitted at t_ref, not a raw sample.
    """

    center_x: float
    center_y: float
    angular_velocity: float
    t_ref: float
    pose_ref: PlatePose

    def predict(self, t: float) -> PlatePose:
        """Returns the plate pose at time t.

        The heading is not wrapped into (-pi, pi]. After a full turn it
        differs from pose_ref.heading by 2 pi, while the position matches.

        Args:
            t: time.monotonic time [s].

        Returns:
            Predicted plate pose. At angular_velocity 0 the position stays
            at pose_ref and the stored center has no effect.
        """
        turned_angle = self.angular_velocity * (t - self.t_ref)
        cosine, sine = get_cosine_and_sine(turned_angle)
        offset_x = self.pose_ref.x - self.center_x
        offset_y = self.pose_ref.y - self.center_y
        return PlatePose(
            x=self.center_x + cosine * offset_x - sine * offset_y,
            y=self.center_y + sine * offset_x + cosine * offset_y,
            heading=self.pose_ref.heading + turned_angle,
        )

    @property
    def is_rotating_leftwise(self) -> bool:
        """True when the table turns counterclockwise (leftwise)."""
        return self.angular_velocity > 0


def fit_table_motion(
    samples: Sequence[tuple[float, PlatePose]],
) -> TableMotion | None:
    """Fits a constant angular velocity about a fixed vertical axis.

    Args:
        samples: Timed plate poses, oldest first. The time is
            time.monotonic [s], the same clock as Frame.timestamp.

    Returns:
        The fitted motion, or None when there are fewer than MIN_SAMPLES
        samples.

    Raises:
        ValueError: A value is not finite, or the times are not strictly
            increasing. Checked even when there are fewer than MIN_SAMPLES
            samples.
    """
    plate_pose_times, plate_center_x, plate_center_y, headings = _columns(
        samples
    )
    _validate(plate_pose_times, plate_center_x, plate_center_y, headings)
    if len(plate_pose_times) < MIN_SAMPLES:
        return None

    # Shortest step. A jump of pi or more between samples is not detectable.
    unwrapped = np.unwrap(headings)
    t_ref = float(plate_pose_times[-1])
    heading_ref, angular_velocity = _fit_heading(
        plate_pose_times, unwrapped, t_ref
    )
    if abs(float(unwrapped[-1] - unwrapped[0])) < MIN_ROTATION_FOR_CENTER:
        position_x = float(plate_center_x.mean())
        position_y = float(plate_center_y.mean())
        center_x, center_y = position_x, position_y
    else:
        center_x, center_y, position_x, position_y = _fit_center(
            plate_pose_times,
            plate_center_x,
            plate_center_y,
            angular_velocity,
            t_ref,
        )
    return TableMotion(
        center_x=center_x,
        center_y=center_y,
        angular_velocity=angular_velocity,
        t_ref=t_ref,
        pose_ref=PlatePose(position_x, position_y, heading_ref),
    )


def _columns(
    samples: Sequence[tuple[float, PlatePose]],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    plate_pose_times = np.array([time for time, _ in samples], dtype=np.float64)
    plate_center_x = np.array([pose.x for _, pose in samples], dtype=np.float64)
    plate_center_y = np.array([pose.y for _, pose in samples], dtype=np.float64)
    headings = np.array([pose.heading for _, pose in samples], dtype=np.float64)
    return plate_pose_times, plate_center_x, plate_center_y, headings


def _validate(
    plate_pose_times: NDArray[np.float64],
    plate_center_x: NDArray[np.float64],
    plate_center_y: NDArray[np.float64],
    headings: NDArray[np.float64],
) -> None:
    """Validate whether all poses are allowed for usage."""
    finite = (
        np.isfinite(plate_pose_times).all()
        and np.isfinite(plate_center_x).all()
        and np.isfinite(plate_center_y).all()
        and np.isfinite(headings).all()
    )
    if not finite:
        raise ValueError("Sample times and plate poses must be finite.")
    if np.any(np.diff(plate_pose_times) <= 0):
        raise ValueError("Sample times must be strictly increasing.")


def _fit_heading(
    plate_pose_times: NDArray[np.float64],
    unwrapped: NDArray[np.float64],
    t_ref: float,
) -> tuple[float, float]:
    """Fits the measured plate headings with a straight line in time.

    Returns the heading of that line at t_ref and the angular velocity
    [rad/s]. The velocity is the slope through every unwrapped heading.
    """
    equation_coefficients = np.column_stack(
        (np.ones(len(plate_pose_times)), plate_pose_times - t_ref)
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        equation_coefficients, unwrapped, rcond=None
    )
    return float(coefficients[0]), float(coefficients[1])


def _fit_center(
    plate_pose_times: NDArray[np.float64],
    plate_center_x: NDArray[np.float64],
    plate_center_y: NDArray[np.float64],
    angular_velocity: float,
    ref_time: float,
) -> tuple[float, float, float, float]:
    """Fits the center and the plate position at ref_time.

    Each sample says the measured plate center is the reference position
    rotated about the center by angular_velocity * (t - ref_time).
    """
    num_of_measurements = len(plate_pose_times)
    equation_coefficients = np.empty(
        (num_of_measurements * 2, 4), dtype=np.float64
    )
    measured_data = np.empty(num_of_measurements * 2, dtype=np.float64)
    samples = zip(plate_pose_times, plate_center_x, plate_center_y, strict=True)
    for index, (pose_time, measured_x, measured_y) in enumerate(samples):
        delta_time = float(pose_time) - ref_time
        turned_angle = angular_velocity * delta_time
        cosine, sine = get_cosine_and_sine(turned_angle)
        equation_coefficients[2 * index] = (
            cosine,
            -sine,
            1.0 - cosine,
            sine,
        )
        equation_coefficients[2 * index + 1] = (
            sine,
            cosine,
            -sine,
            1.0 - cosine,
        )
        measured_data[2 * index] = measured_x
        measured_data[2 * index + 1] = measured_y
    solution, _, _, _ = np.linalg.lstsq(
        equation_coefficients, measured_data, rcond=None
    )
    position_x, position_y, center_x, center_y = (float(v) for v in solution)
    return center_x, center_y, position_x, position_y
