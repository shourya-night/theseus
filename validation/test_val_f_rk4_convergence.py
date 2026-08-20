"""
VALIDATION F: Classical RK4 Convergence Rate
Tests fixed-step RK4 integrator against known analytical ODE dy/dt = -y, y(0)=1
and verifies fourth-order asymptotic error scaling (Error ~ O(h^4)).
"""

import math
import numpy as np
import pytest

from theseus.propagation.integrators import RK4Integrator


class TestValidationFRK4Convergence:
    """Independent verification of RK4 mathematical order of convergence."""

    def test_rk4_fourth_order_convergence_rate(self):
        """
        ODE: dy/dt = -y, y(0) = 1.
        Analytical solution at t = 1: y(1) = 1/e = 0.36787944117144233.
        Compute global error for dt in [0.2, 0.1, 0.05, 0.025].
        For a 4th-order method, halving the timestep should reduce the error by ~ 16x (ratio = 2^4 = 16).
        """
        def f(t, y):
            return -y

        y_exact = math.exp(-1.0)
        timesteps = [0.2, 0.1, 0.05, 0.025]
        errors = []

        for dt in timesteps:
            integrator = RK4Integrator(dt=dt)
            res = integrator.integrate(f, np.array([1.0]), (0.0, 1.0))
            y_num = res.states[-1][0]
            err = abs(y_num - y_exact)
            errors.append(err)

        # Check error reduction ratios
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            # Ratio should be approximately 2^4 = 16 (allowing 14.5 to 17.5 due to higher order terms)
            assert 14.0 < ratio < 18.0, f"Step halving ratio {ratio:.2f} is not close to theoretical 16.0 (4th order)"

    def test_rk4_second_order_oscillator(self):
        """
        Harmonic oscillator: x'' + omega^2 x = 0 with omega = 2 rad/s.
        y = [x, v]. dy/dt = [v, -omega^2 x].
        Initial: x(0)=1, v(0)=0. Exact: x(t)=cos(omega*t), v(t)=-omega*sin(omega*t).
        Verify at t = pi (one half-period, omega*t = 2*pi).
        """
        omega = 2.0
        def f(t, y):
            return np.array([y[1], -omega**2 * y[0]])

        dt = 0.005
        integrator = RK4Integrator(dt=dt)
        res = integrator.integrate(f, np.array([1.0, 0.0]), (0.0, math.pi))

        x_final = res.states[-1][0]
        v_final = res.states[-1][1]

        # At t=pi, x = cos(2*pi) = 1.0, v = -2*sin(2*pi) = 0.0
        assert x_final == pytest.approx(1.0, abs=1e-5)
        assert v_final == pytest.approx(0.0, abs=1e-5)
