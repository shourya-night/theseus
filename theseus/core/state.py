"""
Simulation state and state-history containers.

SimulationState holds all physical quantities at a single instant.
StateHistory is an ordered, indexable collection of states that the
future 3-D renderer consumes directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationState:
    """
    Complete simulation state at one instant.

    All quantities are in SI units (m, m/s, m/s², kg, s).

    Attributes
    ----------
    time : float
        Elapsed simulation time from epoch (s).
    position : np.ndarray
        Position vector [x, y, z] (m).
    velocity : np.ndarray
        Velocity vector [vx, vy, vz] (m/s).
    acceleration : np.ndarray
        Total acceleration vector (m/s²).
    mass : float
        Total spacecraft mass (kg).
    fuel_mass : float
        Remaining propellant mass (kg).
    metadata : dict
        Extensible key-value store for auxiliary quantities
        (altitude, dynamic pressure, heating, …).
    """
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 0.0
    fuel_mass: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- derived convenience properties ------------------------------------

    @property
    def speed(self) -> float:
        """Scalar speed |v| (m/s)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def altitude(self) -> float | None:
        """Altitude above a reference body, if stored in metadata (m)."""
        return self.metadata.get("altitude")

    @property
    def r_mag(self) -> float:
        """Distance from coordinate origin (m)."""
        return float(np.linalg.norm(self.position))

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "time": self.time,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "mass": self.mass,
            "fuel_mass": self.fuel_mass,
            "speed": self.speed,
            "r_mag": self.r_mag,
            "metadata": self.metadata,
        }


class StateHistory:
    """
    Ordered collection of :class:`SimulationState` objects.

    Designed for efficient append-only recording during propagation
    and random-access playback by the future renderer.
    """

    def __init__(self) -> None:
        self._states: list[SimulationState] = []

    def append(self, state: SimulationState) -> None:
        """Append a state to the history."""
        self._states.append(state)

    @property
    def states(self) -> list[SimulationState]:
        return self._states

    @property
    def times(self) -> np.ndarray:
        """Array of all recorded times (s)."""
        return np.array([s.time for s in self._states])

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) array of position vectors (m)."""
        return np.array([s.position for s in self._states])

    @property
    def velocities(self) -> np.ndarray:
        """(N, 3) array of velocity vectors (m/s)."""
        return np.array([s.velocity for s in self._states])

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> SimulationState:
        return self._states[idx]

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise the entire history."""
        return [s.to_dict() for s in self._states]
