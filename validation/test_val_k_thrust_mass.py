"""
VALIDATION K: Thrust, Propellant Mass Flow, and Rocket Equation
Independently verifies Tsiolkovsky rocket equation, thrust acceleration scaling,
mass flow rate m_dot = F / (Isp * g0), and fuel depletion bounds.
"""

import math
import numpy as np
import pytest

from theseus.constants.physical import G0_VAL
from theseus.spacecraft.vehicle import Spacecraft
from theseus.dynamics.thrust import ThrustModel, ThrustDirection
from theseus.maneuvers.burns import impulsive_burn, finite_burn_duration, fuel_for_delta_v, delta_v_from_fuel


class TestValidationKThrustMass:
    """Independent verification of spacecraft propulsion and mass accounting."""

    def test_tsiolkovsky_rocket_equation_exact(self):
        """
        Spacecraft: m_dry = 500 kg, m_fuel = 500 kg (m0 = 1000 kg), Isp = 300 s.
        ve = 300 * 9.80665 = 2941.995 m/s.
        Delta-v = ve * ln(1000 / 500) = 2941.995 * ln(2) = 2039.239 m/s.
        """
        m_dry = 500.0
        m_fuel = 500.0
        isp = 300.0

        sc = Spacecraft(dry_mass=m_dry, fuel_mass=m_fuel, specific_impulse=isp)
        ve_expected = isp * G0_VAL
        dv_expected = ve_expected * math.log((m_dry + m_fuel) / m_dry)

        assert sc.exhaust_velocity == pytest.approx(ve_expected, rel=1e-12)
        assert sc.delta_v_available() == pytest.approx(dv_expected, rel=1e-12)

    def test_mass_flow_rate(self):
        """
        m_dot = F / (Isp * g0).
        F = 1000 N, Isp = 250 s -> m_dot = 1000 / (250 * 9.80665) = 0.407886 kg/s.
        """
        sc = Spacecraft(thrust=1000.0, specific_impulse=250.0, num_engines=1)
        expected_mdot = 1000.0 / (250.0 * G0_VAL)
        assert sc.mass_flow_rate == pytest.approx(expected_mdot, rel=1e-12)

    def test_finite_burn_duration_exact(self):
        """
        t_burn = (m0 * ve / F) * (1 - exp(-Delta_v / ve))
        m0 = 1000 kg, F = 500 N, Isp = 300 s, Delta_v = 500 m/s.
        """
        m0 = 1000.0
        F = 500.0
        isp = 300.0
        dv = 500.0

        ve = isp * G0_VAL
        expected_duration = (m0 * ve / F) * (1.0 - math.exp(-dv / ve))

        duration = finite_burn_duration(dv, F, m0, isp)
        assert duration == pytest.approx(expected_duration, rel=1e-12)

    def test_fuel_consumption_bounds(self):
        """Fuel consumed cannot exceed available fuel mass."""
        sc = Spacecraft(dry_mass=500.0, fuel_mass=50.0)
        consumed = sc.consume_fuel(100.0)
        assert consumed == 50.0
        assert sc.fuel_mass == 0.0
        assert sc.total_mass == 500.0  # Dry mass preserved
