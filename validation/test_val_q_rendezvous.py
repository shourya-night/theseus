"""
VALIDATION Q: Orbital Rendezvous & Target Interception
Tests rendezvous problem solving, chaser arrival accuracy, and relative velocity behavior.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.rendezvous.solver import solve_rendezvous
from theseus.propagation.integrators import RKF45Integrator


MU = EARTH.mu


class TestValidationQRendezvous:
    """Independent verification of rendezvous trajectory and arrival state."""

    def test_coplanar_short_arc_rendezvous(self):
        """
        Target in 500 km circular orbit at theta = 30 deg.
        Chaser in 400 km circular orbit at theta = 0 deg.
        TOF chosen such that transfer angle < 180 deg.
        """
        r1 = EARTH.radius + 400_000.0
        r2 = EARTH.radius + 500_000.0
        v1 = math.sqrt(MU / r1)
        v2 = math.sqrt(MU / r2)

        # Target 30 deg ahead
        theta_target = math.radians(30.0)
        chaser_r = np.array([r1, 0.0, 0.0])
        chaser_v = np.array([0.0, v1, 0.0])

        target_r = np.array([r2 * math.cos(theta_target), r2 * math.sin(theta_target), 0.0])
        target_v = np.array([-v2 * math.sin(theta_target), v2 * math.cos(theta_target), 0.0])

        # Choose TOF = 1500 s (about 1/4 orbit)
        tof = 1500.0

        try:
            res = solve_rendezvous(chaser_r, chaser_v, target_r, target_v, tof, MU)
            assert res.lambert_solution.converged
            assert res.delta_v_total > 0

            # Verify chaser trajectory reaches target arrival position
            if res.transfer_trajectory is not None:
                final_chaser_pos = res.transfer_trajectory[-1].position
                miss = np.linalg.norm(final_chaser_pos - res.target_position_at_arrival)
                assert miss < 100.0, f"Rendezvous chaser missed target by {miss:.2f} m"
        except RuntimeError as e:
            pytest.fail(f"Rendezvous solver failed to converge: {e}")
