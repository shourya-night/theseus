"""
State vector: position + velocity anchored to a reference frame and epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from theseus.coordinates.frames import ReferenceFrame


@dataclass
class StateVector:
    """
    Cartesian state vector (position + velocity) in a specific frame.

    Parameters
    ----------
    position : np.ndarray
        [x, y, z] position (m).
    velocity : np.ndarray
        [vx, vy, vz] velocity (m/s).
    frame : ReferenceFrame
        Reference frame of the vectors.
    epoch_jd : float | None
        Julian Date at which this state is valid (optional).
    """
    position: np.ndarray
    velocity: np.ndarray
    frame: ReferenceFrame = ReferenceFrame.ICRF
    epoch_jd: float | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")
        if self.velocity.shape != (3,):
            raise ValueError(f"Velocity must be shape (3,), got {self.velocity.shape}")
        if not np.all(np.isfinite(self.position)):
            raise ValueError(f"Position contains non-finite values: {self.position}")
        if not np.all(np.isfinite(self.velocity)):
            raise ValueError(f"Velocity contains non-finite values: {self.velocity}")

    @property
    def r(self) -> float:
        """Position magnitude (m)."""
        return float(np.linalg.norm(self.position))

    @property
    def v(self) -> float:
        """Velocity magnitude (m/s)."""
        return float(np.linalg.norm(self.velocity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": self.position.tolist(),
            "velocity_m_s": self.velocity.tolist(),
            "frame": self.frame.value,
            "epoch_jd": self.epoch_jd,
        }
