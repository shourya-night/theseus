"""
Conservation-law diagnostics.

Computes and tracks conserved quantities (energy, angular momentum)
across a simulation to detect numerical integration drift.

Usage
-----
>>> diag = ConservationDiagnostics(mu=3.986004418e14)
>>> diag.record(t, r, v)       # at each step
>>> diag.energy_drift()        # relative drift from initial value
>>> diag.angular_momentum_drift()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ConservationRecord:
    """A single conservation-law sample."""
    time: float
    specific_energy: float          # J/kg = m²/s²
    angular_momentum: np.ndarray    # m²/s  (3-vector)
    angular_momentum_mag: float     # m²/s  (scalar)


class ConservationDiagnostics:
    """
    Track energy and angular-momentum conservation during propagation.

    Parameters
    ----------
    mu : float
        Gravitational parameter of the central body (m³/s²).
    """

    def __init__(self, mu: float) -> None:
        self.mu = mu
        self._records: list[ConservationRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, time: float, position: np.ndarray, velocity: np.ndarray) -> ConservationRecord:
        """
        Compute and store conserved quantities at a single instant.

        Parameters
        ----------
        time : float
            Simulation time (s).
        position : np.ndarray
            Position vector (m).
        velocity : np.ndarray
            Velocity vector (m/s).

        Returns
        -------
        ConservationRecord
        """
        r = float(np.linalg.norm(position))
        v = float(np.linalg.norm(velocity))

        # Specific orbital energy: ε = v²/2 − μ/r
        energy = 0.5 * v * v - self.mu / r

        # Specific angular momentum: h = r × v
        h_vec = np.cross(position, velocity)
        h_mag = float(np.linalg.norm(h_vec))

        rec = ConservationRecord(
            time=time,
            specific_energy=energy,
            angular_momentum=h_vec,
            angular_momentum_mag=h_mag,
        )
        self._records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[ConservationRecord]:
        return list(self._records)

    @property
    def initial(self) -> Optional[ConservationRecord]:
        """Return the first recorded sample, or None."""
        return self._records[0] if self._records else None

    def energy_drift(self) -> np.ndarray:
        """
        Relative energy drift from the initial value.

        Returns an array of (ε(t) − ε₀) / |ε₀| for each recorded sample.
        """
        if len(self._records) < 2:
            return np.array([])
        e0 = self._records[0].specific_energy
        if abs(e0) < 1e-30:
            # Parabolic orbit — use absolute drift
            return np.array([r.specific_energy - e0 for r in self._records])
        return np.array([(r.specific_energy - e0) / abs(e0) for r in self._records])

    def angular_momentum_drift(self) -> np.ndarray:
        """
        Relative angular-momentum magnitude drift from the initial value.

        Returns an array of (|h(t)| − |h₀|) / |h₀| for each sample.
        """
        if len(self._records) < 2:
            return np.array([])
        h0 = self._records[0].angular_momentum_mag
        if h0 < 1e-30:
            return np.array([r.angular_momentum_mag for r in self._records])
        return np.array(
            [(r.angular_momentum_mag - h0) / h0 for r in self._records]
        )

    def max_energy_drift(self) -> float:
        """Maximum absolute relative energy drift."""
        drift = self.energy_drift()
        return float(np.max(np.abs(drift))) if len(drift) > 0 else 0.0

    def max_angular_momentum_drift(self) -> float:
        """Maximum absolute relative angular-momentum drift."""
        drift = self.angular_momentum_drift()
        return float(np.max(np.abs(drift))) if len(drift) > 0 else 0.0

    def summary(self) -> dict:
        """Return a serialisable diagnostics summary."""
        return {
            "num_samples": len(self._records),
            "max_energy_drift_relative": self.max_energy_drift(),
            "max_angular_momentum_drift_relative": self.max_angular_momentum_drift(),
            "initial_energy_J_per_kg": self._records[0].specific_energy if self._records else None,
            "initial_h_m2_per_s": self._records[0].angular_momentum_mag if self._records else None,
        }
