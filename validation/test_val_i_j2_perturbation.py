"""
VALIDATION I: J2 Perturbation & Nodal Precession
Independently verifies J2 oblateness acceleration equations and compares
secular RAAN nodal precession rate against first-order analytical perturbation theory.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.dynamics.gravity import J2Perturbation, PointMassGravity
from theseus.dynamics.force_model import CompositeForceModel
from theseus.propagation.numerical import NumericalPropagator
from theseus.orbital.conversions import state_to_elements


MU = EARTH.mu


class TestValidationIJ2Perturbation:
    """Independent verification of J2 perturbation dynamics."""

    def test_j2_acceleration_analytic_components(self):
        """
        Vallado Eq. 8-30:
        a_J2 = 3/2 * mu * J2 * Re^2 / r^5 * [ x*(5z^2/r^2 - 1), y*(5z^2/r^2 - 1), z*(5z^2/r^2 - 3) ]
        Verify for a point in space [x, y, z] = [4000 km, 3000 km, 5000 km].
        """
        j2_model = J2Perturbation(EARTH)
        pos = np.array([4_000_000.0, 3_000_000.0, 5_000_000.0])
        r = float(np.linalg.norm(pos))
        x, y, z = pos[0], pos[1], pos[2]

        factor = 1.5 * EARTH.mu * EARTH.J2 * (EARTH.radius ** 2) / (r ** 5)
        ax_ref = factor * x * (5.0 * (z / r)**2 - 1.0)
        ay_ref = factor * y * (5.0 * (z / r)**2 - 1.0)
        az_ref = factor * z * (5.0 * (z / r)**2 - 3.0)
        a_ref = np.array([ax_ref, ay_ref, az_ref])

        a_actual = j2_model.compute_acceleration(0.0, pos, np.zeros(3), 100.0)
        np.testing.assert_allclose(a_actual, a_ref, rtol=1e-12)

    def test_secular_raan_precession_rate(self):
        """
        First-order secular RAAN precession:
        dOmega/dt = -3/2 * J2 * (Re/p)^2 * n * cos(i)
        Orbit: a = 7000 km, e = 0.01, i = 45 deg.
        """
        a = 7_000_000.0
        e = 0.01
        inc = math.radians(45.0)
        p = a * (1.0 - e**2)
        n = math.sqrt(MU / a**3)

        domega_dt_analytical = -1.5 * EARTH.J2 * (EARTH.radius / p)**2 * n * math.cos(inc)  # rad/s

        # Propagate for 10 orbits (~16 hours) with point mass + J2
        composite = CompositeForceModel([
            PointMassGravity(EARTH),
            J2Perturbation(EARTH),
        ])
        prop = NumericalPropagator(
            acceleration_fn=composite.compute_acceleration,
            integrator="rkf45",
            dt=30.0,
            atol=1e-11,
            rtol=1e-11,
            mu=MU,
        )

        # Initial state at ascending node
        r0 = np.array([p / (1.0 + e), 0.0, 0.0])
        v_mag0 = math.sqrt(MU * (2.0 / r0[0] - 1.0 / a))
        v0 = np.array([0.0, v_mag0 * math.cos(inc), v_mag0 * math.sin(inc)])

        T = 2.0 * math.pi / n
        duration = 10 * T
        history, events, _ = prop.propagate(r0, v0, (0.0, duration))

        # Check RAAN rate from numerical propagation
        # Extract initial and final elements
        # Note: nodal drift is secular, periodic oscillations are small
        oe_initial = state_to_elements(history[0].position, history[0].velocity, MU)
        oe_final = state_to_elements(history[-1].position, history[-1].velocity, MU)

        domega_numerical = (oe_final.raan - oe_initial.raan)
        # Wrap angle
        if domega_numerical > math.pi:
            domega_numerical -= 2.0 * math.pi
        elif domega_numerical < -math.pi:
            domega_numerical += 2.0 * math.pi

        domega_dt_numerical = domega_numerical / duration

        # Analytical rate is first-order secular approximation (~5-10% agreement with numerical osculating)
        assert domega_dt_numerical == pytest.approx(domega_dt_analytical, rel=0.15)
        # Sign must be negative (retrograde precession for prograde orbit i < 90)
        assert domega_dt_numerical < 0
