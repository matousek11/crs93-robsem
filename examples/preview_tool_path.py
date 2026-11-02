"""Print an OFFLINE-only nominal hoop/flange pose preview for one puzzle.

Run from repository root:
    uv run python -m examples.preview_tool_path A1

This never opens a controller connection or commands a robot.
"""

import argparse

import numpy as np

from hoop_guide.planning import hoop_pose_path
from hoop_guide.plate import PlatePose
from hoop_guide.rods import Puzzle
from hoop_guide.tool_setup import initial_offline_tool_estimate


def main() -> None:
    """Print preview poses and problems requiring verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", choices=[item.value for item in Puzzle])
    parser.add_argument("--plate-x", type=float, default=0.45)
    parser.add_argument("--plate-y", type=float, default=0.10)
    parser.add_argument("--plate-theta", type=float, default=0.0)
    parser.add_argument("--plate-height", type=float, default=0.08)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    # BEST GUESS / OFFLINE ONLY: (0.45, 0.10, 0.08) m are EXAMPLE plate
    # coordinates for demonstration, NOT measured CRS93 installation values.
    # Replace with measured calibration before interpreting geometrical results.
    pose = PlatePose(args.plate_x, args.plate_y, args.plate_theta)
    path = hoop_pose_path(
        Puzzle(args.puzzle), pose, args.plate_height, args.step
    )
    estimate = initial_offline_tool_estimate()
    flanges = path.flange_poses(estimate, preview_only=True)
    np.set_printoptions(precision=4, suppress=True)
    print("=== OFFLINE GEOMETRY PREVIEW - NOT SAFE FOR ROBOT MOTION ===")
    print(f"Puzzle: {args.puzzle}; sampled poses: {len(path.poses_robot)}")
    print(f"Sharp corners (indices): {path.sharp_corner_indices}")
    print(
        f"Ambiguous holder heading (indices): {path.fallback_heading_indices}"
    )
    print("Large orientation steps:", path.large_rotation_step_indices)
    print("Tool mounting VERIFIED:", estimate.is_verified)
    print("NOTE:", estimate.source)
    print("First desired hoop pose R_T_H:\n", path.poses_robot[0])
    print("Last desired hoop pose R_T_H:\n", path.poses_robot[-1])
    print("First nominal flange pose R_T_F:\n", flanges[0])
    print("Last nominal flange pose R_T_F:\n", flanges[-1])
    print("No IK, collision checking, time parametrization or commands.")


if __name__ == "__main__":
    main()
