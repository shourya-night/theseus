"""
VALIDATION O: Lambert Solver & Independent Trajectory Propagation Verification
Tests Lambert two-point boundary value problem and INDEPENDENTLY PROPAGATES the solved velocity
vector to verify if the spacecraft actually reaches the target position r2 at the requested time of flight.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.orbital.lambert import solve_lambert
from theseus.propagation.integrators import RKF45Integrator


MU = EARTH.mu


def _propagate_and_check_arrival(r1: np.ndarray, v1: np.ndarray, tof: float, mu: float) -> np.ndarray:
    """
    Independently integrate two-body equations of motion with high precision from r1 with velocity v1
    for duration tof, and return final position r2_actual.
    """
    def deriv(t, y):
        r = y[:3]
        v = y[3:6]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag ** 3) * r
        return np.concatenate([v, a])

    y0 = np.concatenate([r1, v1])
    rkf = RKF45Integrator(atol=1e-12, rtol=1e-12, dt_initial=min(10.0, tof / 20.0))
    res = rkf.integrate(deriv, y0, (0.0, tof))
    return res.states[-1][:3], res.states[-1][3:6]


class TestValidationOLambert:
    """Independent verification of Lambert solver and physical trajectory arrival."""

    def test_lambert_quarter_orbit_transfer_and_propagation(self):
        """
        90 degree transfer on circular orbit: r1 = [7000 km, 0, 0], r2 = [0, 7000 km, 0].
        TOF = quarter of circular orbit period.
        Solve Lambert -> propagate trajectory -> verify it reaches r2 exactly!
        """
        r_mag = 7_000_000.0
        T = 2.0 * math.pi * math.sqrt(r_mag**3 / MU)
        tof = T / 4.0

        r1 = np.array([r_mag, 0.0, 0.0])
        r2 = np.array([0.0, r_mag, 0.0])

        sol = solve_lambert(r1, r2, tof, MU, prograde=True)
        assert sol.converged, f"Lambert did not converge: residual={sol.residual}"

        # INDEPENDENT VERIFICATION: Propagate from r1 with sol.v1 for tof seconds
        r2_reached, v2_reached = _propagate_and_check_arrival(r1, sol.v1, tof, MU)

        pos_miss = np.linalg.norm(r2_reached - r2)
        vel_miss = np.linalg.norm(v2_reached - sol.v2)

        assert pos_miss < 10.0, f"Spacecraft missed target position r2 by {pos_miss:.3f} metres!"
        assert vel_miss < 0.01, f"Arrival velocity mismatch {vel_miss:.4f} m/s"

    def test_lambert_elliptical_transfer_and_propagation(self):
        """
        Non-circular transfer: r1 = [6800 km, 1000 km, 500 km], r2 = [-2000 km, 15000 km, 3000 km], TOF = 3000 s.
        """
        r1 = np.array([6_800_000.0, 1_000_000.0, 500_000.0])
        r2 = np.array([-2_000_000.0, 15_000_000.0, 3_000_000.0])
        tof = 3000.0

        sol = solve_lambert(r1, r2, tof, MU, prograde=True)
        assert sol.converged

        # Independent propagation
        r2_reached, v2_reached = _propagate_and_check_arrival(r1, sol.v1, tof, MU)
        pos_miss = np.linalg.norm(r2_reached - r2)

        assert pos_miss < 50.0, f"Spacecraft missed target by {pos_miss:.3f} m"

    def test_lambert_180_degree_transfer_behavior(self):
        """
        Hohmann-type 180 deg transfer: r1 = [7000 km, 0, 0], r2 = [-42164 km, 0, 0].
        This test determines whether the Lambert solver handles 180 deg or crashes due to sin(dtheta)=0.
        """
        r1 = np.array([7_000_000.0, 0.0, 0.0])
        r2 = np.array([-42_164_000.0, 0.0, 0.0])
        a_t = (7_000_000.0 + 42_164_000.0) / 2.0
        tof = math.pi * math.sqrt(a_t**3 / MU)

        try:
            sol = solve_lambert(r1, r2, tof, MU, prograde=True)
            assert sol.converged
        except ValueError as e:
            pytest.fail(f"Lambert solver failed on 180 degree Hohmann transfer: {e}")
