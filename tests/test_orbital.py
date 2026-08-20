"""Tests for orbital mechanics: elements, conversions, Kepler solver."""

import math

import numpy as np
import pytest

from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import state_to_elements, elements_to_state
from theseus.orbital.kepler import solve_kepler, eccentric_to_true


# Earth GM
MU_EARTH = 3.986004418e14  # m³/s²


class TestOrbitalElements:
    """Test derived properties of OrbitalElements."""

    def test_circular_leo(self):
        """Circular LEO: a = 6778 km, e = 0."""
        a = 6_778_000.0
        oe = OrbitalElements(a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
        assert oe.semi_latus_rectum == pytest.approx(a)
        assert oe.periapsis_radius == pytest.approx(a)
        assert oe.apoapsis_radius == pytest.approx(a)
        # Period ≈ 92.4 min for 400 km altitude
        assert oe.period == pytest.approx(5553.6, rel=0.01)  # ~92.6 min

    def test_elliptic_orbit(self):
        """GTO-like orbit: a = 24396 km, e = 0.7306."""
        a = 24_396_000.0
        e = 0.7306
        oe = OrbitalElements(a=a, e=e, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
        # Periapsis ≈ 6578 km, apoapsis ≈ 42214 km
        assert oe.periapsis_radius == pytest.approx(6_578_000.0, rel=0.01)
        assert oe.apoapsis_radius == pytest.approx(42_214_000.0, rel=0.01)

    def test_period_circular(self):
        """Circular orbit at 6371+400 = 6771 km: T ≈ 5549 s ≈ 92.5 min."""
        r = 6_771_000.0
        oe = OrbitalElements(a=r, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
        T = oe.period
        expected = 2 * math.pi * math.sqrt(r**3 / MU_EARTH)
        assert T == pytest.approx(expected, rel=1e-10)

    def test_specific_energy_circular(self):
        """ε = −μ/(2a)."""
        a = 7_000_000.0
        oe = OrbitalElements(a=a, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=0.0, mu=MU_EARTH)
        assert oe.specific_energy == pytest.approx(-MU_EARTH / (2 * a))

    def test_hyperbolic_no_period(self):
        """Hyperbolic orbit has no period."""
        oe = OrbitalElements(a=-10_000_000.0, e=1.5, i=0.0, raan=0.0, argp=0.0, nu=0.5, mu=MU_EARTH)
        assert oe.period is None
        assert oe.apoapsis_radius is None


class TestStateElementsConversion:
    """Round-trip state → elements → state."""

    def test_circular_equatorial(self):
        """Circular equatorial orbit: r = 7000 km, v = circular speed."""
        r_mag = 7_000_000.0
        v_circ = math.sqrt(MU_EARTH / r_mag)
        r = np.array([r_mag, 0.0, 0.0])
        v = np.array([0.0, v_circ, 0.0])

        oe = state_to_elements(r, v, MU_EARTH)

        assert oe.a == pytest.approx(r_mag, rel=1e-8)
        assert oe.e == pytest.approx(0.0, abs=1e-8)
        assert oe.i == pytest.approx(0.0, abs=1e-8)

    def test_roundtrip_inclined_elliptic(self):
        """Inclined elliptic orbit: roundtrip r,v → elements → r,v."""
        r = np.array([6_524_834.0, 6_862_875.0, 6_448_296.0])
        v = np.array([4_901.327, 5_533.756, -1_976.341])

        oe = state_to_elements(r, v, MU_EARTH)
        r2, v2 = elements_to_state(oe)

        np.testing.assert_allclose(r2, r, rtol=1e-8)
        np.testing.assert_allclose(v2, v, rtol=1e-8)

    def test_roundtrip_circular_inclined(self):
        """Circular inclined orbit: roundtrip fidelity."""
        r_mag = 7_200_000.0
        v_circ = math.sqrt(MU_EARTH / r_mag)

        # 45° inclination, spacecraft at ascending node
        i = math.radians(45)
        r = np.array([r_mag, 0.0, 0.0])
        v = np.array([0.0, v_circ * math.cos(i), v_circ * math.sin(i)])

        oe = state_to_elements(r, v, MU_EARTH)
        r2, v2 = elements_to_state(oe)

        np.testing.assert_allclose(r2, r, rtol=1e-7)
        np.testing.assert_allclose(v2, v, rtol=1e-7)

    def test_energy_conservation_roundtrip(self):
        """Energy is identical before and after roundtrip."""
        r = np.array([8_000_000.0, 3_000_000.0, 1_000_000.0])
        v = np.array([-1_000.0, 6_000.0, 3_000.0])

        e1 = 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)
        oe = state_to_elements(r, v, MU_EARTH)
        r2, v2 = elements_to_state(oe)
        e2 = 0.5 * np.dot(v2, v2) - MU_EARTH / np.linalg.norm(r2)

        assert e1 == pytest.approx(e2, rel=1e-10)


class TestKeplerSolver:
    """Test Kepler equation solver."""

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.999])
    def test_roundtrip_elliptic(self, e: float):
        """M → E → verify M = E − e sin E."""
        M = 1.5  # rad
        sol = solve_kepler(M, e)
        assert sol.converged
        E = sol.eccentric_anomaly
        M_check = E - e * math.sin(E)
        # Wrap M_check to [0, 2π)
        M_check = M_check % (2 * math.pi)
        M_wrapped = M % (2 * math.pi)
        assert M_check == pytest.approx(M_wrapped, abs=1e-10)

    def test_circular_orbit(self):
        """e = 0 → E = M."""
        M = 2.3
        sol = solve_kepler(M, 0.0)
        assert sol.converged
        assert sol.eccentric_anomaly == pytest.approx(M % (2 * math.pi), abs=1e-12)

    def test_hyperbolic(self):
        """Hyperbolic Kepler equation: M = e sinh H − H."""
        e = 2.0
        M = 5.0
        sol = solve_kepler(M, e)
        assert sol.converged
        H = sol.eccentric_anomaly
        M_check = e * math.sinh(H) - H
        assert M_check == pytest.approx(M, abs=1e-10)

    def test_convergence_diagnostics(self):
        """Solver returns iteration count and residual."""
        sol = solve_kepler(1.0, 0.5)
        assert sol.converged
        assert sol.iterations > 0
        assert sol.residual < 1e-12

    def test_eccentric_to_true_circular(self):
        """For e = 0, ν = E."""
        E = 1.7
        nu = eccentric_to_true(E, 0.0)
        assert nu == pytest.approx(E, abs=1e-12)

    def test_eccentric_to_true_roundtrip(self):
        """ν → E → ν roundtrip via OrbitalElements."""
        e = 0.3
        nu = 1.2
        oe = OrbitalElements(a=7e6, e=e, i=0, raan=0, argp=0, nu=nu, mu=MU_EARTH)
        E = oe.eccentric_anomaly
        nu_back = eccentric_to_true(E, e)
        # Wrap both
        assert (nu_back % (2 * math.pi)) == pytest.approx(nu % (2 * math.pi), abs=1e-10)
