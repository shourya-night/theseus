"""
VALIDATION H: Gravity Acceleration & Inverse-Square Law
Independently calculates Newtonian point-mass gravity at Earth surface,
400 km, 1000 km, and 10,000 km altitudes, verifying inverse-square law behavior.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.dynamics.gravity import PointMassGravity


class TestValidationHGravity:
    """Independent verification of point-mass gravitational acceleration."""

    def test_earth_gravity_at_known_altitudes(self):
        """
        Earth mu = 3.986004418e14 m^3/s^2, Re = 6378137.0 m
        Theoretical g(r) = mu / r^2:
        - Surface (h=0 km, r=6378137 m): g = 9.798285 m/s^2
        - 400 km (r=6778137 m): g = 8.677519 m/s^2
        - 1000 km (r=7378137 m): g = 7.322079 m/s^2
        - 10000 km (r=16378137 m): g = 1.485989 m/s^2
        """
        grav = PointMassGravity(EARTH)
        altitudes_km = [0.0, 400.0, 1000.0, 10000.0]

        for alt_km in altitudes_km:
            r = EARTH.radius + alt_km * 1000.0
            pos = np.array([r, 0.0, 0.0])
            acc = grav.compute_acceleration(0.0, pos, np.zeros(3), 100.0)
            g_actual = np.linalg.norm(acc)

            g_expected = EARTH.mu / (r ** 2)
            assert g_actual == pytest.approx(g_expected, rel=1e-10)

    def test_inverse_square_scaling(self):
        """Doubling distance must reduce acceleration by exactly 4x."""
        grav = PointMassGravity(EARTH)
        r1 = 7_000_000.0
        r2 = 14_000_000.0
        r3 = 28_000_000.0

        a1 = np.linalg.norm(grav.compute_acceleration(0, np.array([r1, 0, 0]), np.zeros(3), 1))
        a2 = np.linalg.norm(grav.compute_acceleration(0, np.array([r2, 0, 0]), np.zeros(3), 1))
        a3 = np.linalg.norm(grav.compute_acceleration(0, np.array([r3, 0, 0]), np.zeros(3), 1))

        assert a1 / a2 == pytest.approx(4.0, rel=1e-12)
        assert a1 / a3 == pytest.approx(16.0, rel=1e-12)
