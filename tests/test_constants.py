"""Tests for physical constants and unit conversions."""

import math
import pytest

from theseus.constants.physical import (
    G, SPEED_OF_LIGHT, STEFAN_BOLTZMANN, BOLTZMANN, PLANCK,
    STANDARD_GRAVITY, STANDARD_ATMOSPHERE, SOLAR_LUMINOSITY,
    ASTRONOMICAL_UNIT, UNIVERSAL_GAS_CONSTANT,
    G_VAL, C_VAL, G0_VAL, AU_VAL,
)
from theseus.constants.units import (
    km_to_m, m_to_km, deg_to_rad, rad_to_deg,
    hours_to_seconds, seconds_to_hours,
    days_to_seconds, seconds_to_days,
    au_to_m, m_to_au,
    M_PER_KM, RAD_PER_DEG,
)


# ---------------------------------------------------------------------------
# Physical constants — verify against CODATA / IAU reference values
# ---------------------------------------------------------------------------

class TestPhysicalConstants:

    def test_gravitational_constant(self):
        """G = 6.67430e-11 m³ kg⁻¹ s⁻² (CODATA 2018)."""
        assert G.value == pytest.approx(6.67430e-11, rel=1e-4)
        assert G.unit == "m^3 kg^-1 s^-2"
        assert "CODATA" in G.source

    def test_speed_of_light(self):
        """c = 299 792 458 m/s (exact)."""
        assert SPEED_OF_LIGHT.value == 299_792_458.0
        assert SPEED_OF_LIGHT.uncertainty == 0.0

    def test_boltzmann(self):
        """k_B = 1.380649e-23 J/K (exact by SI definition)."""
        assert BOLTZMANN.value == 1.380649e-23

    def test_stefan_boltzmann(self):
        """σ = 5.670374419e-8 W m⁻² K⁻⁴."""
        assert STEFAN_BOLTZMANN.value == pytest.approx(5.670374419e-8, rel=1e-8)

    def test_standard_gravity(self):
        """g₀ = 9.80665 m/s² (exact)."""
        assert STANDARD_GRAVITY.value == 9.80665

    def test_standard_atmosphere(self):
        """1 atm = 101325 Pa (exact)."""
        assert STANDARD_ATMOSPHERE.value == 101_325.0

    def test_astronomical_unit(self):
        """1 AU = 149 597 870 700 m (IAU 2012, exact)."""
        assert ASTRONOMICAL_UNIT.value == 149_597_870_700.0

    def test_solar_luminosity(self):
        """L☉ = 3.828e26 W (IAU 2015 nominal)."""
        assert SOLAR_LUMINOSITY.value == pytest.approx(3.828e26, rel=1e-3)

    def test_convenience_values_match(self):
        """Bare-value aliases must equal the PhysicalConstant.value."""
        assert G_VAL == G.value
        assert C_VAL == SPEED_OF_LIGHT.value
        assert G0_VAL == STANDARD_GRAVITY.value
        assert AU_VAL == ASTRONOMICAL_UNIT.value


# ---------------------------------------------------------------------------
# Unit conversions — round-trip fidelity
# ---------------------------------------------------------------------------

class TestUnitConversions:

    def test_km_m_roundtrip(self):
        assert m_to_km(km_to_m(42.0)) == pytest.approx(42.0)

    def test_km_to_m_value(self):
        assert km_to_m(1.0) == 1000.0
        assert km_to_m(384.4) == pytest.approx(384_400.0)

    def test_deg_rad_roundtrip(self):
        assert rad_to_deg(deg_to_rad(180.0)) == pytest.approx(180.0)
        assert rad_to_deg(deg_to_rad(360.0)) == pytest.approx(360.0)

    def test_deg_to_rad_known(self):
        assert deg_to_rad(180.0) == pytest.approx(math.pi)
        assert deg_to_rad(90.0) == pytest.approx(math.pi / 2)

    def test_hours_seconds_roundtrip(self):
        assert seconds_to_hours(hours_to_seconds(2.5)) == pytest.approx(2.5)

    def test_hours_to_seconds(self):
        assert hours_to_seconds(1.0) == 3600.0

    def test_days_seconds_roundtrip(self):
        assert seconds_to_days(days_to_seconds(1.0)) == pytest.approx(1.0)

    def test_days_to_seconds(self):
        assert days_to_seconds(1.0) == 86400.0

    def test_au_m_roundtrip(self):
        assert m_to_au(au_to_m(1.0)) == pytest.approx(1.0)

    def test_au_to_m(self):
        assert au_to_m(1.0) == 149_597_870_700.0
