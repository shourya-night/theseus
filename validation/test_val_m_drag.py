"""
VALIDATION M: Atmospheric Drag & Aerodynamic Scaling
Independently verifies drag acceleration formula a_D = -0.5 * rho * v_rel * Cd * A / m * v_rel_hat,
scaling laws (v^2, area, density), and Earth atmospheric co-rotation.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.atmosphere.models import ExponentialAtmosphere
from theseus.dynamics.drag import DragModel


class TestValidationMDrag:
    """Independent verification of atmospheric drag acceleration."""

    def test_drag_formula_exact_value(self):
        """
        Spacecraft at polar position (no atmospheric co-rotation effect):
        r = [0, 0, Re + 200 km], v = [7500 m/s, 0, 0].
        Cd = 2.2, Area = 10 m^2, Mass = 1000 kg.
        """
        alt = 200_000.0
        atm = ExponentialAtmosphere(rho0=1.225, scale_height=8500.0)
        drag = DragModel(atm, cd=2.2, area=10.0, body_radius=EARTH.radius, body_rotation_rate=0.0)

        pos = np.array([0.0, 0.0, EARTH.radius + alt])
        vel = np.array([7500.0, 0.0, 0.0])
        mass = 1000.0

        rho = 1.225 * math.exp(-alt / 8500.0)
        v_mag = 7500.0
        a_expected_mag = 0.5 * rho * (v_mag ** 2) * 2.2 * 10.0 / mass
        a_expected = -a_expected_mag * np.array([1.0, 0.0, 0.0])

        a_actual = drag.compute_acceleration(0.0, pos, vel, mass)
        np.testing.assert_allclose(a_actual, a_expected, rtol=1e-10)

    def test_drag_scaling_laws(self):
        """Verify drag scaling: 2x area -> 2x drag; 2x density -> 2x drag."""
        atm1 = ExponentialAtmosphere(rho0=1.0, scale_height=8500.0)
        atm2 = ExponentialAtmosphere(rho0=2.0, scale_height=8500.0)

        drag_base = DragModel(atm1, cd=2.0, area=10.0, body_rotation_rate=0.0)
        drag_double_area = DragModel(atm1, cd=2.0, area=20.0, body_rotation_rate=0.0)
        drag_double_rho = DragModel(atm2, cd=2.0, area=10.0, body_rotation_rate=0.0)

        pos = np.array([0.0, 0.0, EARTH.radius + 300_000.0])
        vel = np.array([7000.0, 0.0, 0.0])
        mass = 1000.0

        a_base = np.linalg.norm(drag_base.compute_acceleration(0, pos, vel, mass))
        a_area = np.linalg.norm(drag_double_area.compute_acceleration(0, pos, vel, mass))
        a_rho = np.linalg.norm(drag_double_rho.compute_acceleration(0, pos, vel, mass))

        assert a_area / a_base == pytest.approx(2.0, rel=1e-10)
        assert a_rho / a_base == pytest.approx(2.0, rel=1e-10)

    def test_corotating_atmosphere_effect(self):
        """
        In an equatorial prograde orbit, atmosphere moves eastward at v_atm = omega * r.
        v_rel = v - v_atm < v -> drag should be slightly LOWER than for a retrograde orbit.
        """
        atm = ExponentialAtmosphere()
        drag = DragModel(atm, cd=2.2, area=10.0, body_rotation_rate=7.2921159e-5)

        r = EARTH.radius + 300_000.0
        pos = np.array([r, 0.0, 0.0])  # on x axis
        v_prograde = np.array([0.0, 7700.0, 0.0])    # moving in +y
        v_retrograde = np.array([0.0, -7700.0, 0.0])  # moving in -y

        a_pro = np.linalg.norm(drag.compute_acceleration(0, pos, v_prograde, 1000))
        a_retro = np.linalg.norm(drag.compute_acceleration(0, pos, v_retrograde, 1000))

        # Retrograde should experience higher drag due to headwind from co-rotating atmosphere
        assert a_retro > a_pro
