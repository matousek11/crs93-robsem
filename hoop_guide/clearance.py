"""Offline-only geometry screening of a hoop travelling along bent rods.

This is a *conservative surrogate* based on the supplied STEP/drawings, not
exact CAD solid collision checking or a robot motion planner. It covers the
nominal rod (including square-profile corners), the Hoop.step ring AND its
integral handle, and the infinite horizontal plane of the plate top.

The result MUST NOT be used to command real hardware: there is no check of
CRS adapter, robot links, joint limits, assembly measurements or table motion.
Unresolved intervals remain explicitly flagged; never assume they are safe.
"""

import dataclasses
import math

import numpy as np
from numpy.typing import NDArray

from hoop_guide.geometry import plate_to_robot
from hoop_guide.plate import PlatePose
from hoop_guide.planning import _holder_y_axis
from hoop_guide.rods import model_for_puzzle
from hoop_guide.rods import Puzzle
from hoop_guide.rods import RodModel
from hoop_guide.tool_setup import ToolCalibration
from hoop_guide.types import Pose

# Geometry from the attached Hoop.step and the supplied technical drawing.
# A ring of 60 mm inner diameter and ~69.6 mm outer diameter has a maximum
# half-thickness 3.3 mm. This enclosing toroidal tube includes its 3D volume.
_RING_MEAN_RADIUS_M = 0.0325
_RING_TUBE_BOUND_M = math.hypot(0.035 - _RING_MEAN_RADIUS_M, 0.0033)
# Conservative piecewise capsules of the Hoop.step handle in the hoop frame:
# +Y runs from center of opening toward mount; checked against STEP mesh
# vertices with 0.15 mm tessellation. NOT a formal full-solid CAD proof.
_HANDLE_PARTS = (
    (0.035, 0.045, 0.022),
    (0.045, 0.105, 0.013),
    (0.105, 0.155, 0.010),
)
_HANDLE_MAX_REACH_M = 0.177
_MIN_RING_POINTS = 16
_MIN_HEADING_SAMPLES = 8


@dataclasses.dataclass(frozen=True)
class ClearanceSettings:
    """Offline-only conservative nominal envelope / sampling settings.

    The rod bounding radius 8 mm is larger than sqrt(2)*5 mm: the actual
    STEP mesh reaches about 7.37 mm at square-profile mitered bends.
    The values do NOT include mounting, manufacturing and motion errors.
    The 2 mm clearance target is a BEST-GUESS for OFFLINE EXPLORATION,
    NOT a validated safety margin. A real margin needs measured errors.
    """

    path_step_m: float = 0.005
    blend_half_length_m: float = 0.018
    rod_sample_step_m: float = 0.002
    rod_radius_bound_m: float = 0.008
    target_clearance_m: float = 0.002
    ring_points: int = 96
    max_subdivision_depth: int = 7
    heading_samples: int = 24
    # The numerical check may terminate with unresolved intervals; never
    # silently treat them as collision-free.

    def __post_init__(self) -> None:
        values = (
            self.path_step_m,
            self.blend_half_length_m,
            self.rod_sample_step_m,
            self.rod_radius_bound_m,
            self.target_clearance_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(
                "All clearance lengths must be finite and positive."
            )
        if (
            self.ring_points < _MIN_RING_POINTS
            or self.max_subdivision_depth < 0
            or self.heading_samples < _MIN_HEADING_SAMPLES
        ):
            raise ValueError("Ring resolution or subdivision depth is invalid.")


@dataclasses.dataclass(frozen=True)
class ClearanceResult:
    """An offline nominal geometric path, with explicit verification limits.

    Every pose is in CRS93 frame R. `geometric_screen_passed` says ONLY that
    this conservatively approximated hoop/handle-to-rod/plate check passed,
    including a bounded interpolation check between stored poses. It never
    certifies the assembled tool or full robot motion.
    """

    poses_robot: NDArray[np.float64]
    progress_m: NDArray[np.float64]
    clearances_m: NDArray[np.float64]
    unresolved_intervals: tuple[int, ...]
    blended_corner_indices: tuple[int, ...]
    restricted_corner_indices: tuple[int, ...]
    heading_offsets_rad: NDArray[np.float64]
    settings: ClearanceSettings

    @property
    def geometric_screen_passed(self) -> bool:
        """Whether every interval passed the nominal conservative screen."""
        return not self.unresolved_intervals and bool(np.all(
            self.clearances_m >= self.settings.target_clearance_m
        ))

    def flange_preview(
        self, calibration: ToolCalibration
    ) -> NDArray[np.float64]:
        """OFFLINE ONLY: calculate flange poses, even if screen is unresolved.

        Unverified mounting is deliberately allowed for diagnostics. This
        method provides NO motion permission and NEVER sends commands.
        """
        return np.stack([
            calibration.mount.flange_for_hoop(Pose(pose))
            for pose in self.poses_robot
        ])


def _point_on_model(
    model: RodModel, progress: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Arc-length lookup with a deterministic outgoing tangent at joints."""
    remaining = progress
    for segment in model.segments[:-1]:
        if remaining < segment.length - 1e-12:
            return segment.point_at(remaining), segment.tangent_at(remaining)
        remaining -= segment.length
    segment = model.segments[-1]
    remaining = float(np.clip(remaining, 0.0, segment.length))
    return segment.point_at(remaining), segment.tangent_at(remaining)


def _bend_windows(
    model: RodModel, requested_half_length: float
) -> list[tuple[float, float, NDArray[np.float64], NDArray[np.float64]]]:
    """Nonoverlapping neighborhoods around true tangent discontinuities."""
    windows = []
    distance = 0.0
    for first, second in zip(model.segments, model.segments[1:]):
        distance += first.length
        u = first.tangent_at(first.length)
        v = second.tangent_at(0)
        if float(np.dot(u, v)) > math.cos(math.radians(2)):
            continue
        # Preserve at least 60% of short adjacent segments. The final
        # corner of D is close to the stop point and is therefore restricted.
        width = min(
            requested_half_length, 0.4 * first.length, 0.4 * second.length
        )
        windows.append((distance, width, u, v))
    return windows


def _smooth_geometry(
    model: RodModel,
    progress: float,
    windows: list[
        tuple[float, float, NDArray[np.float64], NDArray[np.float64]]
    ],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Blend each sharp centerline corner with a C1 cubic Hermite curve.

    The center may move slightly OFF the nominal rod centerline around bends.
    That is intentional; collision geometry must decide whether it is valid.
    """
    for corner, half_width, incoming, outgoing in windows:
        if corner - half_width < progress < corner + half_width:
            start, _ = _point_on_model(model, corner - half_width)
            end, _ = _point_on_model(model, corner + half_width)
            u = (progress - (corner - half_width)) / (2 * half_width)
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            velocity_start = 2 * half_width * incoming
            velocity_end = 2 * half_width * outgoing
            point = (h00 * start + h10 * velocity_start
                     + h01 * end + h11 * velocity_end)
            du = ((6 * u**2 - 6 * u) * start
                  + (3 * u**2 - 4 * u + 1) * velocity_start
                  + (-6 * u**2 + 6 * u) * end
                  + (3 * u**2 - 2 * u) * velocity_end)
            norm = float(np.linalg.norm(du))
            if norm < 1e-12:
                raise ValueError("Corner blend has a degenerate tangent.")
            return point, du / norm
    return _point_on_model(model, progress)


def _frame_for_tangent(
    tangent_robot: NDArray[np.float64],
    point_robot: NDArray[np.float64],
    base_xy: NDArray[np.float64],
    previous_y: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    normal = -tangent_robot
    holder_y, _ = _holder_y_axis(point_robot, normal, base_xy, previous_y)
    axis_x = np.cross(holder_y, normal)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    rotation = np.column_stack((axis_x, axis_y, normal))
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = point_robot
    return result


def _rod_segments(
    model: RodModel,
    settings: ClearanceSettings,
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dense rod polyline; extend last tangent to plate as a conservative GUESS.

    Below the 40 mm stop the supplied RodModel does not contain the rod.
    Extending its last tangent to z=0 is an OFFLINE ESTIMATE that MUST be
    validated against the original solid, especially on puzzle D.
    """
    vertices: list[NDArray[np.float64]] = []
    for segment in model.segments:
        count = math.ceil(segment.length / settings.rod_sample_step_m)
        positions = np.linspace(0.0, segment.length, count + 1)
        vertices.extend(segment.point_at(float(s)) for s in positions[1:])
        if len(vertices) == count:  # first segment: add its start at head
            vertices.insert(0, segment.point_at(0.0))
    last = vertices[-1]
    last_tangent = model.segments[-1].tangent_at(model.segments[-1].length)
    if last_tangent[2] < -1e-10 and last[2] > 0:
        extension = last + last_tangent * (-last[2] / last_tangent[2])
        vertices.append(extension)
    points = np.stack(vertices)
    return np.stack((
        points[:-1] @ transform[:3, :3].T + transform[:3, 3],
        points[1:] @ transform[:3, :3].T + transform[:3, 3],
    ), axis=1)



def _distance_to_segments(
    samples: NDArray[np.float64], segments: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Minimum exact point-to-segment distance for each sample."""
    start = segments[:, 0]
    delta = segments[:, 1] - start
    denom = np.sum(delta * delta, axis=1)
    if np.any(denom <= 0):
        raise ValueError("Rod polyline contains zero-length segments.")
    relative = samples[:, None, :] - start[None, :, :]
    t = np.clip(np.sum(relative * delta[None, :, :], axis=2) / denom, 0, 1)
    closest = start[None, :, :] + t[:, :, None] * delta[None, :, :]
    return np.min(np.linalg.norm(samples[:, None, :] - closest, axis=2), axis=1)


def _pose_clearance(
    pose: NDArray[np.float64],
    rod_segments: NDArray[np.float64],
    plate_height: float,
    settings: ClearanceSettings,
) -> float:
    """Conservative lower bound of nominal hoop/handle-to-rod/plate distance.

    Ring points approximate its centerline; the half-step chord bound is
    subtracted to cover unsampled portions of the whole ring. The rod STEP
    miter radius is included in `rod_radius_bound_m` and the interpolated arc
    sagitta is bounded by the rod sampling step (R50 is the tightest arc).
    """
    angle = (2 * np.pi / settings.ring_points) * np.arange(settings.ring_points)
    circle_local = np.column_stack((
        _RING_MEAN_RADIUS_M * np.cos(angle),
        _RING_MEAN_RADIUS_M * np.sin(angle),
        np.zeros_like(angle),
    ))
    origin = pose[:3, 3]
    rot = pose[:3, :3]
    circle = circle_local @ rot.T + origin
    angular_gap = (
        2 * _RING_MEAN_RADIUS_M
        * math.sin(math.pi / (2 * settings.ring_points))
    )
    # Rod STEP D includes R50, sagitta <= step^2/(8*R) for curvature R.
    rod_sagitta = settings.rod_sample_step_m**2 / (8 * 0.05)
    ring_rod = float(np.min(_distance_to_segments(circle, rod_segments))) - (
        _RING_TUBE_BOUND_M + settings.rod_radius_bound_m
        + angular_gap + rod_sagitta
    )
    ring_plate = float(np.min(circle[:, 2])) - (
        plate_height + _RING_TUBE_BOUND_M + angular_gap
    )
    minimum = min(ring_rod, ring_plate)
    for start, end, tube_radius in _HANDLE_PARTS:
        # Sampling also has a conservative half-step error bound for the
        # portions of the capsule axis between successive samples.
        n = max(2, math.ceil((end-start) / 0.003) + 1)
        ys = np.linspace(start, end, n)
        local = np.column_stack((np.zeros(n), ys, np.zeros(n)))
        handle = local @ rot.T + origin
        sample_gap = (end - start) / (2 * (n - 1))
        handle_rod = float(
            np.min(_distance_to_segments(handle, rod_segments))
        ) - (
            tube_radius + settings.rod_radius_bound_m + rod_sagitta + sample_gap
        )
        handle_plate = float(np.min(handle[:, 2])) - (
            plate_height + tube_radius + sample_gap
        )
        minimum = min(minimum, handle_rod, handle_plate)
    return minimum


def _rotation_angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    relative = a.T @ b
    return math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))


def _interpolate_pose(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """SE(3) halfway point: linear translation, SO(3) geodesic midpoint."""
    relative = a[:3, :3].T @ b[:3, :3]
    # For rotations not near pi, R + I has an orthogonal polar factor equal
    # to the geodesic square root. If near pi, use Rodrigues with robust axis.
    if np.trace(relative) > -0.999999:
        u, _, vt = np.linalg.svd(np.eye(3) + relative)
        midway = u @ vt
        if np.linalg.det(midway) < 0:
            u[:, -1] *= -1
            midway = u @ vt
    else:
        # Deterministic pi-rotation axis, with component signs resolved by
        # off-diagonal entries of the rotation matrix.
        axis = np.sqrt(np.maximum((np.diag(relative) + 1) / 2, 0))
        idx = int(np.argmax(axis))
        if axis[idx] < 1e-9:
            raise ValueError("Degenerate rotation interpolation.")
        for j in range(3):
            if j != idx:
                axis[j] = (
                    (relative[idx, j] + relative[j, idx]) / (4 * axis[idx])
                )
        axis /= np.linalg.norm(axis)
        # rotate 90 degrees about axis
        skew = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        midway = np.outer(axis, axis) + skew
    result = np.eye(4)
    result[:3, :3] = a[:3, :3] @ midway
    result[:3, 3] = (a[:3, 3] + b[:3, 3]) / 2
    return result


def _interval_approved(  # noqa: PLR0913
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    clear_a: float,
    clear_b: float,
    rod: NDArray[np.float64],
    height: float,
    settings: ClearanceSettings,
    depth: int,
) -> bool:
    """Prove lower distance bound or recursively evaluate subintervals.

    Any rigid point on the modeled hoop/handle moves at most
    ||translation|| + max_reach * theta between either endpoint and an
    intermediate interpolated pose. Since clearance is 1-Lipschitz under
    this displacement, lower bounds can certify continuous interpolation.
    """
    displacement = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = _rotation_angle(a[:3, :3], b[:3, :3])
    bound = displacement + _HANDLE_MAX_REACH_M * rotation
    if min(clear_a, clear_b) - bound >= settings.target_clearance_m:
        return True
    if depth >= settings.max_subdivision_depth:
        return False
    middle = _interpolate_pose(a, b)
    mid_clear = _pose_clearance(middle, rod, height, settings)
    if mid_clear < settings.target_clearance_m:
        return False
    return (
        _interval_approved(
            a, middle, clear_a, mid_clear, rod, height, settings, depth + 1
        )
        and _interval_approved(
            middle, b, mid_clear, clear_b, rod, height, settings, depth + 1
        )
    )



def _choose_holder_headings(  # noqa: PLR0914
    poses: NDArray[np.float64],
    rod: NDArray[np.float64],
    height: float,
    settings: ClearanceSettings,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find a continuous collision-aware holder orientation around each normal.

    Dynamic programming selects among discrete rotations about local +Z_H.
    Staying close to the preferred base-facing direction is secondary to
    nominal clearance and continuity. NO adapter/robot collisions are checked.
    """
    angles = (
        2 * math.pi * np.arange(settings.heading_samples)
        / settings.heading_samples
    )
    cosine = np.cos(angles)
    sine = np.sin(angles)
    angle_matrices = np.stack([
        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        for c, s in zip(cosine, sine)
    ])
    n = len(poses)
    m = settings.heading_samples
    scores = np.full((n, m), np.inf)
    clearances = np.empty((n, m))
    parents = np.zeros((n, m), dtype=np.intp)
    for index, pose in enumerate(poses):
        for j, rotate in enumerate(angle_matrices):
            candidate = pose.copy()
            candidate[:3, :3] = pose[:3, :3] @ rotate
            clearances[index, j] = _pose_clearance(
                candidate, rod, height, settings
            )
        # A failed collision screen is extremely costly. If none of the
        # candidate headings passes, still return the least bad diagnostic.
        violation = np.maximum(
            0.0, settings.target_clearance_m - clearances[index]
        )
        node_costs = (1e6 * violation**2 + 0.005 * (1.0 - cosine)
                      - 0.020 * np.clip(
                          (clearances[index] - settings.target_clearance_m)
                          / 0.004, 0, 1
                      ))
        if index == 0:
            scores[0] = node_costs
            continue
        # Angular difference of absolute holder headings accounts for changes
        # of the preferred heading frame between adjacent candidate poses.
        # Compute distances on SO(3), rather than differences of local angles.
        prev_rotations = poses[index-1, :3, :3] @ angle_matrices
        current_rotations = pose[:3, :3] @ angle_matrices
        relative_trace = np.einsum(
            "aij,bij->ab", prev_rotations, current_rotations
        )
        rotation_cost = np.arccos(np.clip((relative_trace-1)/2, -1, 1))**2
        all_costs = scores[index-1, :, None] + 0.13 * rotation_cost
        parents[index] = np.argmin(all_costs, axis=0)
        scores[index] = node_costs + np.min(all_costs, axis=0)
    selected = np.empty(n, dtype=np.intp)
    selected[-1] = int(np.argmin(scores[-1]))
    for i in range(n-1, 0, -1):
        selected[i-1] = parents[i, selected[i]]
    result = poses.copy()
    for i, j in enumerate(selected):
        result[i, :3, :3] = poses[i, :3, :3] @ angle_matrices[j]
    wrapped = (angles[selected] + math.pi) % (2 * math.pi) - math.pi
    return result, wrapped


def _failed_edges(
    transforms: NDArray[np.float64],
    clears: NDArray[np.float64],
    rod: NDArray[np.float64],
    height: float,
    settings: ClearanceSettings,
) -> list[int]:
    failures = []
    for i in range(len(transforms) - 1):
        if (min(clears[i], clears[i + 1]) < settings.target_clearance_m
                or not _interval_approved(
                    transforms[i], transforms[i+1],
                    float(clears[i]), float(clears[i+1]),
                    rod, height, settings, 0,
                )):
            failures.append(i)
    return failures


def _global_holder_search(  # noqa: PLR0914
    poses: NDArray[np.float64],
    headings: NDArray[np.float64],
    rod: NDArray[np.float64],
    height: float,
    settings: ClearanceSettings,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None
):
    """Global discrete heading search, lazily rejecting unsafe swept edges.

    A low-cost orientation chain is selected from collision-free nodes. Every
    edge of the selected chain is then checked continuously. Any bad edge is
    removed and the graph is solved again. This is a bounded offline search;
    returning None means UNRESOLVED, not safe.
    """
    angles = np.linspace(
        -math.pi, math.pi, settings.heading_samples, endpoint=False
    )
    m = len(angles)
    n = len(poses)
    candidate_poses = np.empty((n, m, 4, 4), dtype=np.float64)
    candidate_clearance = np.empty((n, m), dtype=np.float64)
    for i, pose in enumerate(poses):
        for j, angle in enumerate(angles):
            relative = float(angle - headings[i])
            c, sn = math.cos(relative), math.sin(relative)
            rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
            candidate = pose.copy()
            candidate[:3, :3] = pose[:3, :3] @ rotation
            candidate_poses[i, j] = candidate
            candidate_clearance[i, j] = _pose_clearance(
                candidate, rod, height, settings
            )
    active = candidate_clearance >= settings.target_clearance_m
    forbidden: set[tuple[int, int, int]] = set()
    # Bounded search, never infinite loop. A failure stays explicitly reported.
    for _ in range(80):
        scores = np.full((n, m), np.inf)
        parents = np.full((n, m), -1, dtype=np.intp)
        scores[0, active[0]] = (0.005 * (1-np.cos(angles[active[0]]))
             - 0.02 * np.clip((candidate_clearance[0, active[0]]
                             - settings.target_clearance_m) / 0.004, 0, 1))
        for i in range(1, n):
            for j in np.flatnonzero(active[i]):
                current = candidate_poses[i, j]
                for k in np.flatnonzero(np.isfinite(scores[i-1])):
                    if (i-1, int(k), int(j)) in forbidden:
                        continue
                    previous = candidate_poses[i-1, k]
                    spin = _rotation_angle(previous[:3, :3], current[:3, :3])
                    if spin > math.radians(35):
                        continue
                    score = (float(scores[i-1, k]) + 0.05 * spin**2
                             + 0.002 * (1-math.cos(float(angles[j])))
                             - 0.02 * float(np.clip(
                                 (candidate_clearance[i, j]
                                  - settings.target_clearance_m)
                             / 0.004, 0, 1)))
                    if score < scores[i, j]:
                        scores[i, j] = score
                        parents[i, j] = k
            if not np.isfinite(scores[i]).any():
                return None
        ids = np.empty(n, dtype=np.intp)
        ids[-1] = int(np.argmin(scores[-1]))
        for i in range(n-1, 0, -1):
            ids[i-1] = parents[i, ids[i]]
        path = np.stack([candidate_poses[i, ids[i]] for i in range(n)])
        clear = np.array([candidate_clearance[i, ids[i]] for i in range(n)])
        failures = _failed_edges(path, clear, rod, height, settings)
        if not failures:
            return path, angles[ids], clear
        for index in failures:
            forbidden.add((index, int(ids[index]), int(ids[index+1])))
    return None


def screen_hoop_path(  # noqa: PLR0914
    puzzle: Puzzle,
    plate: PlatePose,
    plate_height: float,
    *,
    settings: ClearanceSettings | None = None,
    robot_base_xy: tuple[float, float] = (0.0, 0.0),
) -> ClearanceResult:
    """Build a corner-blended candidate and screen its nominal swept volume.

    BEST GUESS / OFFLINE ONLY: no fitted adapter, real robot, or clearance
    calibration. A failed or unresolved interval must NOT be executed.
    Return `geometric_screen_passed` only for the surrogate geometry.
    """
    conf = settings if settings is not None else ClearanceSettings()
    if len(robot_base_xy) != 2 or not np.isfinite(robot_base_xy).all():
        raise ValueError("robot_base_xy must be two finite coordinates.")
    model = model_for_puzzle(puzzle)
    windows = _bend_windows(model, conf.blend_half_length_m)
    length = sum(segment.length for segment in model.segments)
    count = math.ceil(length / conf.path_step_m)
    samples = list(np.linspace(0, length, count + 1))
    for corner, half_width, _, _ in windows:
        samples.extend((corner-half_width, corner, corner+half_width))
    progress = np.unique(np.round(samples, 12))
    frame = np.asarray(plate_to_robot(plate, plate_height))
    base = np.asarray(robot_base_xy, dtype=np.float64)
    poses = []
    prev_y = None
    for s in progress:
        local_p, local_t = _smooth_geometry(model, float(s), windows)
        global_p = frame[:3, :3] @ local_p + frame[:3, 3]
        global_t = frame[:3, :3] @ local_t
        pose = _frame_for_tangent(global_t, global_p, base, prev_y)
        poses.append(pose)
        prev_y = pose[:3, 1]
    transforms = np.stack(poses)
    rod = _rod_segments(model, conf, frame)
    transforms, heading_offsets = _choose_holder_headings(
        transforms, rod, plate_height, conf
    )
    clears = np.asarray([
        _pose_clearance(pose, rod, plate_height, conf) for pose in transforms
    ])
    bad = _failed_edges(transforms, clears, rod, plate_height, conf)
    if bad:
        global_solution = _global_holder_search(
            transforms, heading_offsets, rod, plate_height, conf
        )
        if global_solution is not None:
            transforms, heading_offsets, clears = global_solution
            bad = _failed_edges(transforms, clears, rod, plate_height, conf)
    corners_idx = tuple(
        int(np.argmin(np.abs(progress - corner)))
        for corner, _, _, _ in windows
    )
    restricted = tuple(int(np.argmin(np.abs(progress-corner)))
                       for corner, width, _, _ in windows
                       if width < conf.blend_half_length_m * 0.5)
    return ClearanceResult(
        poses_robot=transforms,
        progress_m=progress,
        clearances_m=clears,
        unresolved_intervals=tuple(bad),
        blended_corner_indices=corners_idx,
        restricted_corner_indices=restricted,
        heading_offsets_rad=heading_offsets,
        settings=conf,
    )
