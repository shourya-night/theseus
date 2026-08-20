"""
VALIDATION L: Atmospheric Models (Exponential & US Standard 1976)
Independently verifies sea level density, temperature lapse rates, ideal gas law consistency,
and monotonic density decrease up to 86 km.
"""

import math
import numpy as np
import pytest

from theseus.atmosphere.models import ExponentialAtmosphere, US1976StandardAtmosphere
from theseus.constants.physical import R_GAS_VAL, M_AIR_VAL


class TestValidationLAtmosphere:
    """Independent verification of atmospheric density and pressure profiles."""

    def test_sea_level_properties(self):
        """Sea level: rho = 1.2250 kg/m^3, P = 101325 Pa, T = 288.15 K."""
        us76 = US1976StandardAtmosphere()
        props = us76.get_properties(0.0)

        assert props.temperature == pytest.approx(288.15, abs=1e-2)
        assert props.pressure == pytest.approx(101325.0, abs=1.0)
        assert props.density == pytest.approx(1.2250, abs=0.01)

    def test_ideal_gas_law_consistency(self):
        """P = rho * R_gas * T / M_air across various altitudes in US76."""
        us76 = US1976StandardAtmosphere()
        for alt in [0.0, 5000.0, 11000.0, 25000.0, 50000.0, 80000.0]:
            props = us76.get_properties(alt)
            calc_P = props.density * R_GAS_VAL * props.temperature / M_AIR_VAL
            assert calc_P == pytest.approx(props.pressure, rel=1e-5)

    def test_temperature_lapse_rates(self):
        """
        US76 Layer 1 (Troposphere, 0-11 km): dT/dh = -0.0065 K/m.
        At h = 11 km: T = 288.15 - 0.0065 * 11000 = 216.65 K.
        US76 Layer 2 (Tropopause, 11-20 km): Isothermal T = 216.65 K.
        """
        us76 = US1976StandardAtmosphere()
        assert us76.get_properties(11000.0).temperature == pytest.approx(216.65, abs=1e-2)
        assert us76.get_properties(15000.0).temperature == pytest.approx(216.65, abs=1e-2)
        assert us76.get_properties(20000.0).temperature == pytest.approx(216.65, abs=1e-2)

    def test_monotonic_density_decrease(self):
        """Density must strictly decrease with increasing altitude."""
        us76 = US1976StandardAtmosphere()
        altitudes = np.linspace(0, 86000, 100)
        densities = [us76.density(h) for h in altitudes]

        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1]
