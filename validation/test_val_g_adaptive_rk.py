"""
VALIDATION G: Adaptive Step-Size Integrator (RKF45)
Tests RKF45 error estimation, adaptive step adjustments, tolerance scaling (rtol=1e-5, 1e-8, 1e-11),
and minimum/maximum timestep handling.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.propagation.integrators import RKF45Integrator


MU = EARTH.mu


class TestValidationGAdaptiveRK:
    """Independent verification of RKF45 adaptive step size behavior."""

    def test_tolerance_scaling_improves_accuracy(self):
        """
        Integrate a two-body orbit for 1 period at tolerances rtol in [1e-4, 1e-7, 1e-10].
        Verify that tighter tolerance results in lower position error and smaller energy drift.
        """
        r0_mag = 7_000_000.0
        v_circ = math.sqrt(MU / r0_mag)
        T = 2.0 * math.pi * math.sqrt(r0_mag**3 / MU)

        def deriv(t, y):
            r = y[:3]
            v = y[3:6]
            r_mag = np.linalg.norm(r)
            a = -MU / r_mag**3 * r
            return np.concatenate([v, a])

        y0 = np.array([r0_mag, 0.0, 0.0, 0.0, v_circ, 0.0])
        e0 = 0.5 * v_circ**2 - MU / r0_mag

        tolerances = [1e-4, 1e-7, 1e-10]
        energy_drifts = []
        pos_errors = []
        step_counts = []

        for tol in tolerances:
            rkf = RKF45Integrator(atol=tol, rtol=tol, dt_initial=60.0)
            res = rkf.integrate(deriv, y0, (0.0, T))
            step_counts.append(res.steps_taken)

            y_final = res.states[-1]
            pos_final = y_final[:3]
            vel_final = y_final[3:6]

            e_final = 0.5 * np.dot(vel_final, vel_final) - MU / np.linalg.norm(pos_final)
            energy_drifts.append(abs((e_final - e0) / abs(e0)))
            pos_errors.append(np.linalg.norm(pos_final - y0[:3]))

        # Step count should increase as tolerance tightens
        assert step_counts[0] < step_counts[1] < step_counts[2]
        # Energy drift should strictly decrease
        assert energy_drifts[0] > energy_drifts[1] > energy_drifts[2]
        # Position error should strictly decrease
        assert pos_errors[0] > pos_errors[1] > pos_errors[2]

    def test_dt_limits_enforced(self):
        """Verify dt_min and dt_max limits are respected during integration."""
        def f(t, y):
            return -y

        rkf = RKF45Integrator(dt_min=0.01, dt_max=0.5, dt_initial=0.1)
        res = rkf.integrate(f, np.array([1.0]), (0.0, 5.0))
        dts = np.diff(res.times)

        assert np.all(dts >= 0.01 - 1e-12)
        assert np.all(dts <= 0.5 + 1e-12)
