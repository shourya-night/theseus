"""
VALIDATION B: Orbital Mechanics Fundamentals
Independently calculates circular, elliptic, and hyperbolic orbital properties
and compares with THESEUS OrbitalElements outputs.
"""

import math
import pytest

from theseus.bodies.catalog import EARTH
from theseus.orbital.elements import OrbitalElements


class TestValidationBOrbitalMechanics:
    """Independent verification of two-body orbital mechanics formulas."""

    def test_circular_orbit_exact_values(self):
        """
        r = 6778.137 km (400 km altitude LEO)
        Earth mu = 3.986004418e14 m^3/s^2
        Independently calculated reference values:
        - v_c = sqrt(mu / r) = 7668.55836 m/s
        - T = 2 * pi * sqrt(r^3 / mu) = 5553.64205 s (92.5607 min)
        - epsilon = -mu / (2 * r) = -29403392.24 J/kg
        - h = r * v_c = 51978537905.38 m^2/s
        """
        r = 6_778_137.0
        mu = EARTH.mu

        v_circ_ref = math.sqrt(mu / r)
        period_ref = 2.0 * math.pi * math.sqrt(r**3 / mu)
        energy_ref = -mu / (2.0 * r)
        h_ref = r * v_circ_ref

        oe = OrbitalElements(a=r, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=mu)

        assert oe.period == pytest.approx(period_ref, rel=1e-12)
        assert oe.specific_energy == pytest.approx(energy_ref, rel=1e-12)
        assert oe.specific_angular_momentum == pytest.approx(h_ref, rel=1e-12)
        assert oe.periapsis_radius == pytest.approx(r, rel=1e-12)
        assert oe.apoapsis_radius == pytest.approx(r, rel=1e-12)

    def test_elliptical_orbit_vis_viva_and_conservation(self):
        """
        GTO orbit: a = 24396 km, e = 0.7306
        Independently verify vis-viva equation:
        v(r) = sqrt(mu * (2/r - 1/a))
        h = sqrt(mu * a * (1 - e^2))
        """
        a = 24_396_000.0
        e = 0.7306
        mu = EARTH.mu

        r_p_ref = a * (1.0 - e)
        r_a_ref = a * (1.0 + e)
        p_ref = a * (1.0 - e**2)
        h_ref = math.sqrt(mu * p_ref)
        energy_ref = -mu / (2.0 * a)

        oe = OrbitalElements(a=a, e=e, i=math.radians(28.5), raan=1.0, argp=0.5, nu=0.0, mu=mu)

        assert oe.periapsis_radius == pytest.approx(r_p_ref, rel=1e-12)
        assert oe.apoapsis_radius == pytest.approx(r_a_ref, rel=1e-12)
        assert oe.semi_latus_rectum == pytest.approx(p_ref, rel=1e-12)
        assert oe.specific_angular_momentum == pytest.approx(h_ref, rel=1e-12)
        assert oe.specific_energy == pytest.approx(energy_ref, rel=1e-12)

        # Check radius at true anomaly nu = 90 deg: r = p / (1 + e*cos(nu)) = p
        oe_90 = OrbitalElements(a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=math.pi / 2, mu=mu)
        assert oe_90.radius == pytest.approx(p_ref, rel=1e-12)

    def test_hyperbolic_trajectory(self):
        """
        Hyperbolic trajectory: a = -10000 km, e = 1.5
        Energy must be strictly positive: epsilon = -mu / (2a) > 0
        v_inf = sqrt(2 * epsilon) = sqrt(-mu / a)
        Period and apoapsis must be None.
        """
        a = -10_000_000.0
        e = 1.5
        mu = EARTH.mu

        oe = OrbitalElements(a=a, e=e, i=0.2, raan=0.5, argp=0.1, nu=0.3, mu=mu)

        assert oe.specific_energy > 0
        assert oe.specific_energy == pytest.approx(-mu / (2.0 * a), rel=1e-12)
        assert oe.period is None
        assert oe.apoapsis_radius is None
        assert oe.periapsis_radius == pytest.approx(a * (1.0 - e), rel=1e-12)
