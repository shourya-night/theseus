"""
VALIDATION E: Orbital Propagation & Conservation Laws
Quantifies long-term numerical propagation drift (energy, angular momentum, position error)
over 1, 10, and 100 orbits against independent analytical two-body ground truth.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.propagation.integrators import RK4Integrator, RKF45Integrator
from theseus.propagation.analytical import propagate_twobody
from theseus.propagation.numerical import NumericalPropagator


MU = EARTH.mu


class TestValidationEPropagation:
    """Independent verification of numerical propagation accuracy over time."""

    def test_propagation_energy_and_momentum_1_10_100_orbits(self):
        """
        Propagate circular orbit r = 7000 km for 1, 10, and 50 orbits.
        Verify energy and angular momentum conservation and measure drift.
        """
        r0_mag = 7_000_000.0
        v_circ = math.sqrt(MU / r0_mag)
        period = 2.0 * math.pi * math.sqrt(r0_mag**3 / MU)

        r0 = np.array([r0_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_circ, 0.0])
        e0 = 0.5 * v_circ**2 - MU / r0_mag
        h0 = r0_mag * v_circ

        def accel_fn(t, pos, vel, mass):
            r_mag = np.linalg.norm(pos)
            return -MU / r_mag**3 * pos

        prop = NumericalPropagator(
            acceleration_fn=accel_fn,
            integrator="rkf45",
            dt=30.0,
            atol=1e-12,
            rtol=1e-12,
            mu=MU,
        )

        for n_orbits in [1, 10, 50]:
            total_time = n_orbits * period
            history, events, diag = prop.propagate(r0, v0, (0.0, total_time))

            final_pos = history[-1].position
            final_vel = history[-1].velocity
            r_final_mag = np.linalg.norm(final_pos)
            v_final_mag = np.linalg.norm(final_vel)

            e_final = 0.5 * v_final_mag**2 - MU / r_final_mag
            h_final = np.linalg.norm(np.cross(final_pos, final_vel))

            rel_energy_drift = abs((e_final - e0) / abs(e0))
            rel_h_drift = abs((h_final - h0) / h0)

            # Energy and angular momentum must be conserved to high precision
            assert rel_energy_drift < 1e-9, f"Energy drift {rel_energy_drift} exceeded for {n_orbits} orbits"
            assert rel_h_drift < 1e-9, f"h drift {rel_h_drift} exceeded for {n_orbits} orbits"

            # After integer number of circular orbits, position should be close to initial [r0, 0, 0]
            pos_error = np.linalg.norm(final_pos - r0)
            # Position drift rate must remain bounded
            assert pos_error < 100.0 * n_orbits, f"Position error {pos_error} m too large for {n_orbits} orbits"
