"""Offline conservative hoop/handle-to-rod/plate clearance preview.

Example from the repository root:
    uv run python -m examples.preview_clearance_path C

No robot connections, inverse kinematics, trajectory timing, or commands.
The example plate coordinates are NOT measured on a CRS93 workstation.
"""

import argparse

import numpy as np

from hoop_guide.clearance import ClearanceSettings
from hoop_guide.clearance import screen_hoop_path
from hoop_guide.plate import PlatePose
from hoop_guide.rods import Puzzle
from hoop_guide.tool_setup import initial_offline_tool_estimate


def main() -> None:
    """Print clearance diagnostics and preview poses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle", choices=[p.value for p in Puzzle])
    parser.add_argument("--plate-x", type=float, default=0.45)
    parser.add_argument("--plate-y", type=float, default=0.10)
    parser.add_argument("--plate-theta", type=float, default=0.0)
    parser.add_argument("--plate-height", type=float, default=0.08)
    parser.add_argument("--step", type=float, default=0.005)
    args = parser.parse_args()

    # BEST GUESS / OFFLINE DEMO ONLY: values (0.45, 0.10, 0.08) are NOT
    # calibrated measurements of the physical CRS93 table installation.
    result = screen_hoop_path(
        Puzzle(args.puzzle),
        PlatePose(args.plate_x, args.plate_y, args.plate_theta),
        args.plate_height,
        settings=ClearanceSettings(path_step_m=args.step),
    )
    tool = initial_offline_tool_estimate()
    flange = result.flange_preview(tool)
    np.set_printoptions(precision=4, suppress=True)
    print("=== OFFLINE APPROXIMATE CLEARANCE SCREEN — NO ROBOT COMMANDS ===")
    print("Puzzle:", args.puzzle)
    print("Number of hoop poses:", len(result.poses_robot))
    print("Corners blended (indices):", result.blended_corner_indices)
    print(
        "Restricted corner blends (indices):",
        result.restricted_corner_indices,
    )
    print("Unresolved swept intervals:", result.unresolved_intervals)
    print(
        "Minimum nominal modeled clearance [mm]:",
        round(1000 * float(result.clearances_m.min()), 2),
    )
    print("Nominal geometric screen passed:", result.geometric_screen_passed)
    print("Actual flange/adapter mounting verified:", tool.is_verified)
    print("First hoop pose R_T_H:\n", result.poses_robot[0])
    print("Last flange PREVIEW pose R_T_F:\n", flange[-1])
    print(
        "WARNING: no actual CAD-solid contact guarantee, robot collision, "
        "IK, real motion or table tracking."
    )


if __name__ == "__main__":
    main()
