"""
Solar radiation pressure (SRP) force model.

P_srp = L_sun / (4π c r_sun²)
a_SRP = P_srp × Cr × A / m × r̂_sun

where:
    L_sun   = solar luminosity (W)
    c       = speed of light (m/s)
    r_sun   = distance from Sun to spacecraft (m)
    Cr      = reflectivity coefficient (1.0 absorbing, 2.0 fully specular)
    A       = cross-section area (m²)
    m       = spacecraft mass (kg)
    r̂_sun  = unit vector from Sun to spacecraft

Shadow model: cylindrical shadow — SRP is zero when the spacecraft
is in Earth's shadow cylinder.

Reference
---------
Montenbruck & Gill, "Satellite Orbits", §3.4.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from theseus.dynamics.force_model import ForceModel
from theseus.ephemeris.provider import EphemerisProvider
from theseus.constants.physical import L_SUN_VAL, C_VAL
from theseus.core.fidelity import (
    ModelFidelity, FidelityLevel, Assumption, FidelityRegistry,
)
from theseus.time.epochs import JD_J2000
from theseus.bodies.catalog import EARTH


class SolarRadiationPressure(ForceModel):
    """
    Solar radiation pressure acceleration.

    Parameters
    ----------
    ephemeris : EphemerisProvider
        Provides the Sun's position (geocentric ICRF).
    cr : float
        Reflectivity coefficient (1.0–2.0).
    area : float
        Cross-section area (m²).
    epoch_jd_t0 : float
        Julian Date corresponding to simulation t=0.
    shadow_body_radius : float or None
        If given, cylindrical shadow model using this radius (m).
    """

    def __init__(
        self,
        ephemeris: EphemerisProvider,
        cr: float = 1.5,
        area: float = 10.0,
        epoch_jd_t0: float = JD_J2000,
        shadow_body_radius: Optional[float] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="SolarRadiationPressure", enabled=enabled)
        self.ephemeris = ephemeris
        self.cr = cr
        self.area = area
        self.epoch_jd_t0 = epoch_jd_t0
        self.shadow_radius = shadow_body_radius if shadow_body_radius is not None else EARTH.radius
        self._fidelity = ModelFidelity(
            model_name="SolarRadiationPressure",
            level=FidelityLevel.MODERATE,
            assumptions=[
                Assumption("flat_plate", "Spacecraft modeled as flat plate perpendicular to Sun"),
                Assumption("cylindrical_shadow", "Eclipse modeled as cylindrical shadow"),
                Assumption("constant_solar_flux", "Solar luminosity constant"),
            ],
            source="Montenbruck & Gill §3.4",
        )
        FidelityRegistry.get().register(self._fidelity)

    def _get_geocentric_sun_position(self, jd: float) -> np.ndarray:
        """Get geocentric Sun position from ephemeris provider."""
        sun_pos = self.ephemeris.get_position("Sun", jd)
        sun_norm = float(np.linalg.norm(sun_pos))
        if sun_norm < 1.0:
            # Ephemeris provider is heliocentric with Sun at [0,0,0]
            # In geocentric frame, Sun is at -r_earth(t)
            try:
                earth_pos = self.ephemeris.get_position("Earth", jd)
                if np.linalg.norm(earth_pos) > 1e6:
                    return -earth_pos
            except Exception:
                pass
        return sun_pos

    def _in_shadow(self, position: np.ndarray, sun_pos: np.ndarray) -> bool:
        """
        Cylindrical shadow model.

        Returns True if spacecraft is in Earth's shadow.
        """
        sun_norm = float(np.linalg.norm(sun_pos))
        if sun_norm < 1.0 or self.shadow_radius <= 0.0:
            return False

        # Unit vector from Earth (origin) to Sun
        sun_dir = sun_pos / sun_norm

        # Project spacecraft position onto Sun direction
        proj = float(np.dot(position, sun_dir))
        if proj > 0:
            # Spacecraft is on the Sun-side of Earth
            return False

        # Distance from the Earth-Sun line
        perp = position - proj * sun_dir
        perp_dist = float(np.linalg.norm(perp))
        return perp_dist < self.shadow_radius

    def compute_acceleration(
        self, t: float, position: np.ndarray, velocity: np.ndarray, mass: float,
    ) -> np.ndarray:
        jd = self.epoch_jd_t0 + t / 86400.0
        sun_pos = self._get_geocentric_sun_position(jd)  # geocentric Sun position

        sun_norm = float(np.linalg.norm(sun_pos))
        if sun_norm < 1.0 or mass <= 0.0:
            return np.zeros(3)

        # Check shadow
        if self._in_shadow(position, sun_pos):
            return np.zeros(3)

        # Vector from Sun to spacecraft (in geocentric frame)
        # Sun is at sun_pos relative to Earth. Spacecraft is at `position` relative to Earth.
        # Vector from Sun to spacecraft = position - sun_pos
        r_sun_to_sc = position - sun_pos
        r_sun = float(np.linalg.norm(r_sun_to_sc))
        if r_sun < 1.0:
            return np.zeros(3)

        # SRP at this distance
        # P = L_sun / (4π c r²)
        P_srp = L_SUN_VAL / (4.0 * np.pi * C_VAL * r_sun ** 2)

        # Acceleration: away from Sun
        r_hat = r_sun_to_sc / r_sun
        a_srp = P_srp * self.cr * self.area / mass * r_hat
        return a_srp
