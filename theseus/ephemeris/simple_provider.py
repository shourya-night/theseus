"""
Keplerian elliptical-orbit ephemeris provider.

Computes positions of celestial bodies using Keplerian orbital elements
matching the frontend celestial catalog for 100% mathematical consistency.
"""

from __future__ import annotations

import math
import numpy as np

from theseus.ephemeris.provider import EphemerisProvider
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)
from theseus.time.epochs import JD_J2000
from theseus.coordinates.transformations import perifocal_to_eci_matrix

AU_METERS = 149597870700.0

# Keplerian Elements, matching the frontend celestial catalog.  `varpi_deg`
# is longitude of perihelion, therefore ω = ϖ − Ω before the standard
# Rz(Ω)·Rx(i)·Rz(ω) perifocal-to-inertial transformation.
PLANETARY_ORBITAL_ELEMENTS: dict[str, dict[str, float]] = {
    "Sun":     {"a_m": 0.0, "e": 0.0, "inc_deg": 0.0, "raan_deg": 0.0, "varpi_deg": 0.0, "period_days": 0.0, "m0_deg": 0.0},
    "Mercury": {"a_m": 0.38709893 * AU_METERS, "e": 0.20563069, "inc_deg": 7.00487, "raan_deg": 48.33167, "varpi_deg": 77.45645, "period_days": 87.9691, "m0_deg": 174.7947},
    "Venus":   {"a_m": 0.72333199 * AU_METERS, "e": 0.00677323, "inc_deg": 3.39471, "raan_deg": 76.68069, "varpi_deg": 131.53298, "period_days": 224.701, "m0_deg": 50.115},
    "Earth":   {"a_m": 1.00000011 * AU_METERS, "e": 0.01671022, "inc_deg": 0.00005, "raan_deg": -11.26064, "varpi_deg": 102.94719, "period_days": 365.25636, "m0_deg": 358.617},
    # The simple provider has no lunar node/inclination source. Preserve its
    # historical planar Moon approximation rather than inventing elements.
    "Moon":    {"a_m": 384400000.0, "e": 0.0, "inc_deg": 0.0, "raan_deg": 0.0, "varpi_deg": 83.353, "period_days": 27.32166, "m0_deg": 135.27},
    "Mars":    {"a_m": 1.52366231 * AU_METERS, "e": 0.09341233, "inc_deg": 1.85061, "raan_deg": 49.57854, "varpi_deg": 336.04084, "period_days": 686.971, "m0_deg": 19.373},
    "Jupiter": {"a_m": 5.20336301 * AU_METERS, "e": 0.04839266, "inc_deg": 1.3053, "raan_deg": 100.55615, "varpi_deg": 14.75385, "period_days": 4332.59, "m0_deg": 20.020},
    "Saturn":  {"a_m": 9.53707032 * AU_METERS, "e": 0.05415060, "inc_deg": 2.48446, "raan_deg": 113.71504, "varpi_deg": 92.43194, "period_days": 10759.22, "m0_deg": 317.020},
    "Uranus":  {"a_m": 19.19126393 * AU_METERS, "e": 0.04716771, "inc_deg": 0.76986, "raan_deg": 74.22988, "varpi_deg": 170.96424, "period_days": 30685.4, "m0_deg": 142.2386},
    "Neptune": {"a_m": 30.06896348 * AU_METERS, "e": 0.00858587, "inc_deg": 1.76917, "raan_deg": 131.72169, "varpi_deg": 44.97135, "period_days": 60189.0, "m0_deg": 256.228},
}

_FIDELITY = ModelFidelity(
    model_name="KeplerianEphemeris",
    level=FidelityLevel.SIMPLIFIED,
    assumptions=[
        Assumption("keplerian_orbits", "Planets move on Keplerian ellipses"),
        Assumption("fixed_epoch", "Orbital phases referenced to J2000.0"),
    ],
    valid_domain="Solar system heliocentric motion",
    source="JPL Orbital Elements",
    limitations="Two-body Keplerian elements with fixed J2000 orientations",
)

def _solve_kepler(M: float, e: float) -> float:
    """Solve M = E - e*sin(E) using Newton-Raphson method."""
    M = M % (2.0 * math.pi)
    if M < 0:
        M += 2.0 * math.pi
    E = M + e * math.sin(M)
    for _ in range(15):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-11:
            break
    return E


class SimpleEphemerisProvider(EphemerisProvider):
    """Keplerian-orbit ephemeris provider."""

    def __init__(self) -> None:
        FidelityRegistry.get().register(_FIDELITY)

    @property
    def name(self) -> str:
        return "KeplerianEphemeris"

    @property
    def source(self) -> str:
        return "Keplerian-orbit analytical provider (JPL orbital elements)"

    @property
    def precision_description(self) -> str:
        return "< 0.1% position agreement with Keplerian analytical catalog."

    def _body_position_velocity(
        self, body_name: str, epoch_jd: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity in parent-centred frame."""
        key = body_name.strip().capitalize()
        if key not in PLANETARY_ORBITAL_ELEMENTS:
            for k in PLANETARY_ORBITAL_ELEMENTS:
                if k.lower() == body_name.strip().lower():
                    key = k
                    break
            else:
                raise KeyError(f"Unknown body: {body_name!r}")

        data = PLANETARY_ORBITAL_ELEMENTS[key]
        a = data["a_m"]
        e = data["e"]
        if a == 0.0 or data["period_days"] == 0.0:
            return np.zeros(3), np.zeros(3)

        dt = (epoch_jd - JD_J2000) * 86400.0  # seconds since J2000
        T_sec = data["period_days"] * 86400.0
        n = (2.0 * math.pi) / T_sec
        M = math.radians(data["m0_deg"]) + n * dt
        E = _solve_kepler(M, e)

        nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0), math.sqrt(1.0 - e) * math.cos(E / 2.0))
        r = a * (1.0 - e * math.cos(E))
        raan = math.radians(data["raan_deg"])
        inc = math.radians(data["inc_deg"])
        argp = math.radians(data["varpi_deg"] - data["raan_deg"])
        rotation = perifocal_to_eci_matrix(raan, inc, argp)

        # Position and velocity are both rotated from the same perifocal
        # frame. This preserves a self-consistent 3D state for Lambert input.
        pos_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
        pos = rotation @ pos_pqw

        mu_sun = 1.32712440018e20
        h = math.sqrt(mu_sun * a * (1.0 - e**2)) if a > 0 else 1.0
        vr = (mu_sun / h) * e * math.sin(nu)
        vtheta = (mu_sun / h) * (1.0 + e * math.cos(nu))
        vel_pqw = np.array([
            vr * math.cos(nu) - vtheta * math.sin(nu),
            vr * math.sin(nu) + vtheta * math.cos(nu),
            0.0,
        ])
        vel = rotation @ vel_pqw

        return pos, vel

    def get_position(self, body_name: str, epoch_jd: float) -> np.ndarray:
        pos, _ = self._body_position_velocity(body_name, epoch_jd)
        return pos

    def get_state(self, body_name: str, epoch_jd: float) -> tuple[np.ndarray, np.ndarray]:
        return self._body_position_velocity(body_name, epoch_jd)
