"""Tests for numerical integrators and propagation."""

import math

import numpy as np
import pytest

from theseus.propagation.integrators import RK4Integrator, RKF45Integrator
from theseus.propagation.analytical import propagate_twobody
from theseus.propagation.numerical import NumericalPropagator
from theseus.core.diagnostics import ConservationDiagnostics


MU_EARTH = 3.986004418e14  # m³/s²


# ===================================================================
# RK4
# ===================================================================

class TestRK4:

    def test_exponential_decay(self):
        """dy/dt = -y  →  y(t) = exp(-t).  Verify at t=1."""
        def f(t, y):
            return -y
        rk4 = RK4Integrator(dt=0.01)
        result = rk4.integrate(f, np.array([1.0]), (0.0, 1.0))
        y_final = result.states[-1][0]
        assert y_final == pytest.approx(math.exp(-1.0), rel=1e-8)

    def test_simple_harmonic_oscillator(self):
        """x'' + x = 0 → x(t)=cos(t), v(t)=-sin(t).  Check at t=2π."""
        def f(t, y):
            return np.array([y[1], -y[0]])
        rk4 = RK4Integrator(dt=0.01)
        result = rk4.integrate(f, np.array([1.0, 0.0]), (0.0, 2 * math.pi))
        x_final = result.states[-1][0]
        v_final = result.states[-1][1]
        assert x_final == pytest.approx(1.0, abs=1e-6)
        assert v_final == pytest.approx(0.0, abs=1e-5)

    def test_circular_orbit_energy_conservation(self):
        """Circular orbit: energy conserved within tolerance over 10 orbits."""
        r0 = 7_000_000.0  # m
        v0 = math.sqrt(MU_EARTH / r0)
        T = 2 * math.pi * math.sqrt(r0**3 / MU_EARTH)

        def two_body(t, y):
            r = y[:3]
            v = y[3:6]
            r_mag = np.linalg.norm(r)
            a = -MU_EARTH / r_mag**3 * r
            return np.concatenate([v, a])

        y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
        rk4 = RK4Integrator(dt=10.0)
        result = rk4.integrate(two_body, y0, (0.0, 10 * T))

        # Check energy conservation
        e0 = 0.5 * v0**2 - MU_EARTH / r0
        y_final = result.states[-1]
        r_f = np.linalg.norm(y_final[:3])
        v_f = np.linalg.norm(y_final[3:6])
        e_f = 0.5 * v_f**2 - MU_EARTH / r_f
        assert abs((e_f - e0) / abs(e0)) < 1e-8


# ===================================================================
# RKF45
# ===================================================================

class TestRKF45:

    def test_exponential_decay(self):
        """dy/dt = -y, adaptive.  Should match exp(-1)."""
        def f(t, y):
            return -y
        rkf = RKF45Integrator(atol=1e-12, rtol=1e-12, dt_initial=0.1)
        result = rkf.integrate(f, np.array([1.0]), (0.0, 1.0))
        y_final = result.states[-1][0]
        assert y_final == pytest.approx(math.exp(-1.0), rel=1e-10)

    def test_circular_orbit_energy(self):
        """Adaptive integrator conserves energy over 5 orbits."""
        r0 = 7_000_000.0
        v0 = math.sqrt(MU_EARTH / r0)
        T = 2 * math.pi * math.sqrt(r0**3 / MU_EARTH)

        def two_body(t, y):
            r = y[:3]; v = y[3:6]
            r_mag = np.linalg.norm(r)
            a = -MU_EARTH / r_mag**3 * r
            return np.concatenate([v, a])

        y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
        rkf = RKF45Integrator(atol=1e-12, rtol=1e-12, dt_initial=60.0)
        result = rkf.integrate(two_body, y0, (0.0, 5 * T))

        e0 = 0.5 * v0**2 - MU_EARTH / r0
        y_final = result.states[-1]
        r_f = np.linalg.norm(y_final[:3])
        v_f = np.linalg.norm(y_final[3:6])
        e_f = 0.5 * v_f**2 - MU_EARTH / r_f
        assert abs((e_f - e0) / abs(e0)) < 1e-10


# ===================================================================
# Analytical propagation
# ===================================================================

class TestAnalyticalPropagation:

    def test_circular_orbit_returns_to_start(self):
        """After one period, position should return to initial."""
        r0_mag = 7_000_000.0
        v_circ = math.sqrt(MU_EARTH / r0_mag)
        T = 2 * math.pi * math.sqrt(r0_mag**3 / MU_EARTH)

        r0 = np.array([r0_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_circ, 0.0])
        times = [0.0, T / 4, T / 2, 3 * T / 4, T]

        history = propagate_twobody(r0, v0, MU_EARTH, times)
        assert len(history) == 5

        # After one period, back to start
        np.testing.assert_allclose(history[-1].position, r0, atol=1.0)  # < 1 m
        np.testing.assert_allclose(history[-1].velocity, v0, atol=1e-4)

    def test_energy_conservation(self):
        """Energy is conserved exactly (analytical propagation)."""
        r0 = np.array([8_000_000.0, 0.0, 0.0])
        v0 = np.array([0.0, 6_500.0, 2_000.0])
        e0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)

        times = np.linspace(0, 20000, 50)
        history = propagate_twobody(r0, v0, MU_EARTH, times)

        for state in history.states:
            e = 0.5 * np.dot(state.velocity, state.velocity) - MU_EARTH / np.linalg.norm(state.position)
            assert e == pytest.approx(e0, rel=1e-8)


# ===================================================================
# Numerical propagator (two-body)
# ===================================================================

class TestNumericalPropagator:

    def test_twobody_conservation(self):
        """Numerical propagator conserves energy and angular momentum."""
        r0_mag = 7_000_000.0
        v_circ = math.sqrt(MU_EARTH / r0_mag)
        T = 2 * math.pi * math.sqrt(r0_mag**3 / MU_EARTH)

        def accel(t, pos, vel, mass):
            r_mag = np.linalg.norm(pos)
            return -MU_EARTH / r_mag**3 * pos

        prop = NumericalPropagator(
            acceleration_fn=accel,
            integrator="rkf45",
            dt=60.0,
            atol=1e-12,
            rtol=1e-12,
            mu=MU_EARTH,
        )
        r0 = np.array([r0_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_circ, 0.0])

        history, events, diag = prop.propagate(r0, v0, (0.0, 2 * T))

        assert len(history) > 10
        assert diag is not None
        assert diag.max_energy_drift() < 1e-9
        assert diag.max_angular_momentum_drift() < 1e-9
