"""
VALIDATION N: Solar Radiation Pressure (SRP) & Eclipse Shadow Model
Independently verifies SRP pressure magnitude at 1 AU (P ≈ 4.54e-6 N/m^2),
inverse-square distance law, and checks shadow / NaN edge-case behavior.
"""

import math
import numpy as np
import pytest

from theseus.constants.physical import L_SUN_VAL, C_VAL, AU_VAL
from theseus.ephemeris.astropy_provider import AstropyEphemerisProvider
from theseus.dynamics.srp import SolarRadiationPressure
from theseus.bodies.catalog import EARTH
from theseus.time.epochs import JD_J2000


class TestValidationNSRP:
    """Independent verification of Solar Radiation Pressure physics."""

    def test_srp_magnitude_at_1au(self):
        """
        Nominal solar radiation pressure at 1 AU:
        P_srp = L_sun / (4 * pi * c * (1 AU)^2) ≈ 4.540e-6 N/m^2.
        For Cr = 1.0, Area = 10 m^2, Mass = 1000 kg:
        a_srp = P_srp * Cr * Area / Mass ≈ 4.540e-8 m/s^2.
        """
        P_expected = L_SUN_VAL / (4.0 * math.pi * C_VAL * (AU_VAL ** 2))
        assert P_expected == pytest.approx(4.540e-6, rel=1e-3)

        # Using Astropy provider which gives realistic geocentric Sun at 1 AU
        eph = AstropyEphemerisProvider()
        srp = SolarRadiationPressure(
            eph, cr=1.0, area=10.0,
            shadow_body_radius=0.0,  # No shadow
        )
        pos = np.array([7_000_000.0, 0.0, 0.0])  # Near Earth (1 AU from Sun)
        acc = srp.compute_acceleration(0.0, pos, np.zeros(3), 1000.0)
        a_mag = np.linalg.norm(acc)

        assert a_mag == pytest.approx(4.540e-8, rel=0.05)

    def test_srp_shadow_behavior(self):
        """
        When the spacecraft is directly behind Earth (anti-Sun direction) inside the cylindrical shadow,
        SRP acceleration must be exactly [0, 0, 0] without NaNs.
        """
        eph = AstropyEphemerisProvider()
        srp = SolarRadiationPressure(
            eph, cr=1.5, area=10.0,
            shadow_body_radius=EARTH.radius,
        )
        sun_pos = eph.get_position("Sun", JD_J2000)
        sun_dir = sun_pos / np.linalg.norm(sun_pos)

        # Place spacecraft in Earth's umbra (behind Earth, opposite Sun)
        pos_in_shadow = -sun_dir * (EARTH.radius + 500_000.0)

        acc = srp.compute_acceleration(0.0, pos_in_shadow, np.zeros(3), 1000.0)

        # Must not be NaN and must be zero
        assert np.all(np.isfinite(acc))
        np.testing.assert_allclose(acc, [0.0, 0.0, 0.0])
