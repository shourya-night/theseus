"""
VALIDATION A: Physical & Astronomical Constants
Independently verifies fundamental physical constants, units, and consistency relations.
"""

import math
import pytest

from theseus.constants.physical import (
    G, SPEED_OF_LIGHT, STEFAN_BOLTZMANN, BOLTZMANN, PLANCK,
    STANDARD_GRAVITY, STANDARD_ATMOSPHERE, SOLAR_LUMINOSITY,
    SOLAR_IRRADIANCE_1AU, ASTRONOMICAL_UNIT,
    G_VAL, C_VAL, G0_VAL, AU_VAL, L_SUN_VAL, S0_VAL,
)
from theseus.bodies.catalog import ALL_BODIES, EARTH, SUN, MOON, MARS, JUPITER


class TestValidationAConstants:
    """Independent verification of physical constants."""

    def test_speed_of_light_exact(self):
        # 2019 SI definition: exact 299792458 m/s
        assert SPEED_OF_LIGHT.value == 299_792_458.0
        assert C_VAL == 299_792_458.0

    def test_astronomical_unit_exact(self):
        # IAU 2012 Resolution B2: exact 149597870700 m
        assert ASTRONOMICAL_UNIT.value == 149_597_870_700.0
        assert AU_VAL == 149_597_870_700.0

    def test_standard_gravity_exact(self):
        # ISO 80000-3: exact 9.80665 m/s^2
        assert STANDARD_GRAVITY.value == 9.80665
        assert G0_VAL == 9.80665

    def test_solar_irradiance_consistency(self):
        """
        Solar irradiance at 1 AU: S0 = L_sun / (4 * pi * AU^2).
        L_sun = 3.828e26 W (IAU 2015 nominal)
        AU = 149597870700 m
        Expected S0 = 3.828e26 / (4 * pi * 149597870700^2) ≈ 1361.16 W/m^2.
        Compare with published value S0 = 1361.0 W/m^2.
        """
        calculated_s0 = L_SUN_VAL / (4.0 * math.pi * AU_VAL ** 2)
        assert calculated_s0 == pytest.approx(S0_VAL, rel=1e-3)
        assert S0_VAL == pytest.approx(1361.0, abs=1.0)

    def test_gravitational_constant_codata2018(self):
        # CODATA 2018: G = 6.67430(15)e-11 m^3 kg^-1 s^-2
        assert G.value == pytest.approx(6.67430e-11, rel=1e-5)
        assert G_VAL == pytest.approx(6.67430e-11, rel=1e-5)

    def test_planetary_gm_values_reference(self):
        """
        Verify authoritative planetary GM values against DE430/DE440.
        Earth: 3.986004418e14 m^3/s^2 (EGM2008 / GRS80)
        Sun: 1.32712440018e20 m^3/s^2
        Moon: 4.9028695e12 m^3/s^2
        Mars: 4.282837e13 m^3/s^2
        Jupiter: 1.26686534e17 m^3/s^2
        """
        assert EARTH.mu == pytest.approx(3.986004418e14, rel=1e-9)
        assert SUN.mu == pytest.approx(1.32712440018e20, rel=1e-9)
        assert MOON.mu == pytest.approx(4.9028695e12, rel=1e-6)
        assert MARS.mu == pytest.approx(4.282837e13, rel=1e-6)
        assert JUPITER.mu == pytest.approx(1.26686534e17, rel=1e-6)
