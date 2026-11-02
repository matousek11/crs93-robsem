"""Tool geometry estimates and verification state for offline planning.

The physical CRS93 flange, adapter and Hoop.step frames must be aligned using
measurements / assembly CAD before ANY robot motion. These functions do not
connect to or control a robot.
"""

import dataclasses

import numpy as np

from hoop_guide.geometry import AdapterMount
from hoop_guide.types import Pose


@dataclasses.dataclass(frozen=True)
class ToolCalibration:
    """Flange-to-hoop mounting data with an explicit verification status.

    Args:
        mount: Fixed transformations for the assembled adapter and hoop.
        verified: Whether a human has independently verified the transforms
            against the real assembly and the FK/IK output frame. This flag is
            NOT a collision or motion-safety certification.
        source: Description of the measurements or the offline estimate.
    """

    mount: AdapterMount
    is_verified: bool
    source: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Calibration source must be documented.")


def initial_offline_tool_estimate() -> ToolCalibration:
    """Returns explicitly UNVERIFIED placeholder frames for offline previews.

    BEST GUESS / OFFLINE ONLY - NOT FINAL CALIBRATION:
      1) F -> A = identity: choose the adapter reference frame A to coincide
         with the CRS93 FK/IK flange frame F. This is a coordinate convention,
         NOT a measurement of the physical adapter.
      2) A -> C = identity: assume the Hoop.step CAD axes align with those of
         the adapter. This is an ARBITRARY INITIAL APPROXIMATION, not a CAD-
         verified assembled mounting. It MUST be replaced with a transform
         measured from the actual adapter/hoop assembly.
      3) C -> H is applied by AdapterMount from Hoop.step: translation
         (0, -0.135, -0.0033) m, with the CAD axes unchanged. This is the
         CAD-derived ring-center geometry, not a guessed mounting offset.

    Also verify the meaning of the FK/IK tool frame returned by ctu_crs,
    the plate height, attachment angle, and mounted puzzle geometry.
    NEVER use these defaults to send motion commands to real hardware.
    """
    return ToolCalibration(
        mount=AdapterMount(
            flange_to_adapter=Pose(np.eye(4, dtype=np.float64)),
            adapter_to_hoop_cad=Pose(np.eye(4, dtype=np.float64)),
        ),
        is_verified=False,
        source=(
            "UNVERIFIED OFFLINE GUESS: F->A and A->C identities are frame "
            "placeholders; C->H uses Hoop.step dimensions. Verify real "
            "mounting and ctu_crs FK/IK flange frame."
        ),
    )
