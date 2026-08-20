"""
Simple circular-orbit ephemeris provider.

Computes positions of celestial bodies using circular-orbit
approximations around their parent body.  Explicitly marked as
**approximate** — suitable for testing and order-of-magnitude checks.

Precision
---------
~10–30 % position error for realistic scenarios.
NOT suitable for precision trajectory design.

Reason for existence
--------------------
Works without any external data files or network access, making it
ideal for unit tests and quick prototyping.
"""

from __future__ import annotations

import math

import numpy as np

from theseus.ephemeris.provider import EphemerisProvider
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)
from theseus.time.epochs import JD_J2000

# Mean orbital radii (m) and periods (s) — approximate circular orbits
# Source: JPL Planetary Fact Sheets (mean values)
_ORBIT_DATA: dict[str, dict[str, float]] = {
    "Sun":     {"radius": 0.0, "period": 0.0, "parent": ""},
    "Mercury": {"radius": 5.791e10,  "period": 7.600e6,  "parent": "Sun"},
    "Venus":   {"radius": 1.082e11,  "period": 1.941e7,  "parent": "Sun"},
    "Earth":   {"radius": 1.496e11,  "period": 3.156e7,  "parent": "Sun"},
    "Moon":    {"radius": 3.844e8,   "period": 2.361e6,  "parent": "Earth"},
    "Mars":    {"radius": 2.279e11,  "period": 5.936e7,  "parent": "Sun"},
    "Jupiter": {"radius": 7.785e11,  "period": 3.743e8,  "parent": "Sun"},
    "Saturn":  {"radius": 1.427e12,  "period": 9.295e8,  "parent": "Sun"},
    "Uranus":  {"radius": 2.871e12,  "period": 2.651e9,  "parent": "Sun"},
    "Neptune": {"radius": 4.498e12,  "period": 5.200e9,  "parent": "Sun"},
}

_FIDELITY = ModelFidelity(
    model_name="SimpleCircularEphemeris",
    level=FidelityLevel.SIMPLIFIED,
    assumptions=[
        Assumption("circular_orbits", "All orbits are circular and coplanar"),
        Assumption("fixed_epoch", "Orbital phases referenced to J2000.0"),
        Assumption("no_inclination", "All orbits in ecliptic/equatorial plane (i=0)"),
    ],
    valid_domain="Solar system, order-of-magnitude only",
    source="JPL Planetary Fact Sheet (mean orbital radii)",
    limitations="~10-30% position error; no inclination, eccentricity, or perturbations",
)


class SimpleEphemerisProvider(EphemerisProvider):
    """
    Circular-orbit approximation ephemeris.

    Bodies orbit their parent in circles in the xy-plane.
    """

    def __init__(self) -> None:
        FidelityRegistry.get().register(_FIDELITY)

    @property
    def name(self) -> str:
        return "SimpleCircularEphemeris"

    @property
    def source(self) -> str:
        return "Circular-orbit approximation (JPL mean radii)"

    @property
    def precision_description(self) -> str:
        return "~10-30% position error. Order-of-magnitude only."

    def _body_position_velocity(
        self, body_name: str, epoch_jd: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity in parent-centred frame."""
        key = body_name.strip().capitalize()
        if key not in _ORBIT_DATA:
            # Try case-insensitive
            for k in _ORBIT_DATA:
                if k.lower() == body_name.strip().lower():
                    key = k
                    break
            else:
                raise KeyError(f"Unknown body: {body_name!r}")

        data = _ORBIT_DATA[key]
        r = data["radius"]
        T = data["period"]

        if r == 0.0 or T == 0.0:
            return np.zeros(3), np.zeros(3)

        dt = (epoch_jd - JD_J2000) * 86400.0  # seconds since J2000
        omega = 2.0 * math.pi / T  # rad/s
        theta = omega * dt

        pos = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
        vel = np.array([-r * omega * math.sin(theta), r * omega * math.cos(theta), 0.0])
        return pos, vel

    def get_position(self, body_name: str, epoch_jd: float) -> np.ndarray:
        """Position in ICRF (m), geocentric for Moon, heliocentric for planets."""
        pos, _ = self._body_position_velocity(body_name, epoch_jd)
        return pos

    def get_state(self, body_name: str, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
        return self._body_position_velocity(body_name, epoch_jd)
