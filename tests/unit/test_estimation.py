import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hoop_guide.estimation import MIN_SAMPLES
from hoop_guide.estimation import TableMotion
from hoop_guide.estimation import fit_table_motion
from hoop_guide.geometry import get_cosine_and_sine
from hoop_guide.plate import PlatePose

CENTER_TOLERANCE = 1e-6


def pose_on_circle(center, offset, heading, turned):
    cosine, sine = get_cosine_and_sine(turned)
    return PlatePose(
        center[0] + cosine * offset[0] - sine * offset[1],
        center[1] + sine * offset[0] + cosine * offset[1],
        heading,
    )


def test_fit_finds_a_center_outside_the_plate():
    samples = [
        (10.0, PlatePose(0.4, 0.0, 0.0)),
        (11.0, PlatePose(0.0, 0.4, math.pi / 2)),
        (12.0, PlatePose(-0.4, 0.0, math.pi)),
    ]

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.center_x == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert motion.center_y == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert motion.angular_velocity == pytest.approx(math.pi / 2)
    assert motion.t_ref == pytest.approx(12.0)
    assert motion.pose_ref.x == pytest.approx(-0.4)
    assert motion.pose_ref.y == pytest.approx(0.0)
    assert motion.pose_ref.heading == pytest.approx(math.pi)
    earlier = motion.predict(10.0)
    assert earlier.x == pytest.approx(0.4)
    assert earlier.y == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert earlier.heading == pytest.approx(0.0)


def test_fit_of_negative_velocity_turns_the_other_way():
    samples = [
        (10.0, PlatePose(0.4, 0.0, 0.0)),
        (11.0, PlatePose(0.0, -0.4, -math.pi / 2)),
        (12.0, PlatePose(-0.4, 0.0, -math.pi)),
    ]

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.center_x == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert motion.center_y == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert motion.angular_velocity == pytest.approx(-math.pi / 2)


def test_fit_when_plate_center_is_the_rotation_center():
    samples = [
        (0.0, PlatePose(0.2, 0.3, 0.0)),
        (1.0, PlatePose(0.2, 0.3, 0.4)),
        (2.0, PlatePose(0.2, 0.3, 0.8)),
    ]

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.center_x == pytest.approx(0.2)
    assert motion.center_y == pytest.approx(0.3)
    assert motion.angular_velocity == pytest.approx(0.4)
    assert motion.pose_ref.heading == pytest.approx(0.8)


def test_small_turn_stores_the_mean_position_as_the_center():
    samples = [
        (0.0, PlatePose(0.30, 0.40, 0.00)),
        (1.0, PlatePose(0.31, 0.39, 0.02)),
        (2.0, PlatePose(0.29, 0.41, 0.04)),
    ]

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.angular_velocity == pytest.approx(0.02)
    assert motion.center_x == pytest.approx(0.30)
    assert motion.center_y == pytest.approx(0.40)
    assert motion.pose_ref.x == pytest.approx(0.30)
    assert motion.pose_ref.y == pytest.approx(0.40)
    assert motion.pose_ref.heading == pytest.approx(0.04)


def test_heading_is_unwrapped_across_the_branch_cut():
    def wrap(heading):
        return math.atan2(math.sin(heading), math.cos(heading))

    start = 3.10
    samples = [
        (0.0, PlatePose(0.2, 0.3, wrap(start))),
        (1.0, PlatePose(0.2, 0.3, wrap(start + 0.04))),
        (2.0, PlatePose(0.2, 0.3, wrap(start + 0.08))),
    ]

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.angular_velocity == pytest.approx(0.04)
    assert motion.pose_ref.heading == pytest.approx(start + 0.08)


def test_predict_half_turn_mirrors_position_about_the_center():
    motion = TableMotion(0.0, 0.0, math.pi / 2, 10.0, PlatePose(0.4, 0.0, 0.2))

    pose = motion.predict(12.0)

    assert pose.x == pytest.approx(-0.4)
    assert pose.y == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert pose.heading == pytest.approx(0.2 + math.pi)


def test_predict_negative_quarter_turn_goes_below_the_center():
    motion = TableMotion(0.0, 0.0, -math.pi / 2, 10.0, PlatePose(0.4, 0.0, 0.2))

    pose = motion.predict(11.0)

    assert pose.x == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert pose.y == pytest.approx(-0.4)
    assert pose.heading == pytest.approx(0.2 - math.pi / 2)


def test_predict_full_turn_keeps_position_and_adds_two_pi():
    motion = TableMotion(0.0, 0.0, math.pi / 2, 10.0, PlatePose(0.4, 0.0, 0.2))

    pose = motion.predict(14.0)

    assert pose.x == pytest.approx(0.4)
    assert pose.y == pytest.approx(0.0, abs=CENTER_TOLERANCE)
    assert pose.heading == pytest.approx(0.2 + 2 * math.pi)


def test_predict_with_zero_velocity_keeps_the_reference_pose():
    motion = TableMotion(4.0, -2.0, 0.0, 3.0, PlatePose(0.25, 0.5, 1.0))

    pose = motion.predict(100.0)

    assert pose.x == pytest.approx(0.25)
    assert pose.y == pytest.approx(0.5)
    assert pose.heading == pytest.approx(1.0)


@pytest.mark.parametrize("count", [0, 1, MIN_SAMPLES - 1])
def test_fewer_than_three_samples_returns_none(count):
    samples = [(float(i), PlatePose(0.1, 0.2, 0.3)) for i in range(count)]

    assert fit_table_motion(samples) is None


def test_non_finite_sample_raises_even_when_too_short():
    samples = [(0.0, PlatePose(0.0, 0.0, math.nan))]

    with pytest.raises(ValueError, match="finite"):
        fit_table_motion(samples)


@pytest.mark.parametrize(
    "times",
    [(0.0, 1.0, 1.0), (0.0, 2.0, 1.0)],
    ids=["duplicate", "decreasing"],
)
def test_times_that_are_not_strictly_increasing_raise(times):
    samples = [(time, PlatePose(0.1, 0.2, 0.3)) for time in times]

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_table_motion(samples)


coordinate = st.floats(-1, 1, allow_nan=False, allow_infinity=False)
plate_offset = st.floats(-0.4, 0.4, allow_nan=False, allow_infinity=False)


@given(
    center=st.tuples(coordinate, coordinate),
    offset=st.tuples(plate_offset, plate_offset),
    heading0=st.floats(
        -math.pi, math.pi, allow_nan=False, allow_infinity=False
    ),
    omega=st.floats(-2, 2, allow_nan=False, allow_infinity=False).filter(
        lambda speed: abs(speed) >= 0.3
    ),
)
def test_fit_recovers_motion_and_full_turn_matches_position(
    center, offset, heading0, omega
):
    times = (0.0, 0.2, 0.4, 0.6, 0.8)
    center_x, center_y = center
    samples = []
    for time in times:
        turned = omega * time
        heading = math.atan2(
            math.sin(heading0 + turned), math.cos(heading0 + turned)
        )
        samples.append((time, pose_on_circle(center, offset, heading, turned)))

    motion = fit_table_motion(samples)

    assert motion is not None
    assert motion.center_x == pytest.approx(center_x, abs=CENTER_TOLERANCE)
    assert motion.center_y == pytest.approx(center_y, abs=CENTER_TOLERANCE)
    assert motion.angular_velocity == pytest.approx(omega, abs=CENTER_TOLERANCE)
    turned_end = omega * times[-1]
    end = pose_on_circle(center, offset, heading0 + turned_end, turned_end)
    assert motion.pose_ref.x == pytest.approx(end.x, abs=CENTER_TOLERANCE)
    assert motion.pose_ref.y == pytest.approx(end.y, abs=CENTER_TOLERANCE)
    assert motion.pose_ref.heading == pytest.approx(
        end.heading, abs=CENTER_TOLERANCE
    )
    future = motion.predict(motion.t_ref + 2 * math.pi / omega)
    assert future.x == pytest.approx(motion.pose_ref.x, abs=CENTER_TOLERANCE)
    assert future.y == pytest.approx(motion.pose_ref.y, abs=CENTER_TOLERANCE)
    assert future.heading == pytest.approx(
        motion.pose_ref.heading + 2 * math.pi, abs=CENTER_TOLERANCE
    )
