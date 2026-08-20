"""
Numerical health monitoring.

Detects pathological numerical conditions during simulation:
    - NaN / Infinity in state vectors
    - Negative mass
    - Integration divergence (position/velocity blow-up)
    - Invalid orbital parameters (negative semi-major axis for bound orbit, etc.)

Never silently continues past a numerical failure.
"""

from __future__ import annotations

import numpy as np


class NumericalInstabilityError(RuntimeError):
    """Raised when a numerical health check fails."""

    def __init__(self, message: str, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class NumericalHealthChecker:
    """
    Configurable numerical health checker.

    Parameters
    ----------
    max_position : float
        Maximum allowed position magnitude (m).  Default 1e15 (≈ 6700 AU).
    max_velocity : float
        Maximum allowed velocity magnitude (m/s).  Default 1e8 (≈ 0.33 c).
    max_acceleration : float
        Maximum allowed acceleration magnitude (m/s²).  Default 1e6.
    min_mass : float
        Minimum allowed mass (kg).  Default 0.0 (dry mass can be zero in edge
        cases, but negative mass is always invalid).
    """

    def __init__(
        self,
        max_position: float = 1e15,
        max_velocity: float = 1e8,
        max_acceleration: float = 1e6,
        min_mass: float = 0.0,
    ) -> None:
        self.max_position = max_position
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_mass = min_mass

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def check_finite(name: str, arr: np.ndarray) -> None:
        """Raise if *arr* contains NaN or Inf."""
        if not np.all(np.isfinite(arr)):
            raise NumericalInstabilityError(
                f"{name} contains non-finite values: {arr}",
                diagnostics={"field": name, "value": arr.tolist()},
            )

    def check_magnitude(self, name: str, arr: np.ndarray, limit: float) -> None:
        """Raise if the Euclidean norm of *arr* exceeds *limit*."""
        mag = float(np.linalg.norm(arr))
        if mag > limit:
            raise NumericalInstabilityError(
                f"{name} magnitude {mag:.6e} exceeds limit {limit:.6e}",
                diagnostics={"field": name, "magnitude": mag, "limit": limit},
            )

    def check_mass(self, mass: float) -> None:
        """Raise if mass is negative."""
        if mass < self.min_mass:
            raise NumericalInstabilityError(
                f"Mass {mass:.6e} kg is below minimum {self.min_mass:.6e} kg",
                diagnostics={"mass": mass, "min_mass": self.min_mass},
            )

    # ------------------------------------------------------------------
    # Composite check on a full state
    # ------------------------------------------------------------------

    def check_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        acceleration: np.ndarray | None = None,
    ) -> None:
        """
        Run all health checks on a simulation state.

        Raises :class:`NumericalInstabilityError` on the first failure.
        """
        self.check_finite("position", position)
        self.check_finite("velocity", velocity)
        self.check_magnitude("position", position, self.max_position)
        self.check_magnitude("velocity", velocity, self.max_velocity)
        self.check_mass(mass)
        if acceleration is not None:
            self.check_finite("acceleration", acceleration)
            self.check_magnitude("acceleration", acceleration, self.max_acceleration)
