"""
VALIDATION C: Orbital Element & State Vector Conversions
Tests state -> orbital elements -> state round-trip across all classical edge cases.
"""

import math
import numpy as np
import pytest

from theseus.bodies.catalog import EARTH
from theseus.orbital.elements import OrbitalElements
from theseus.orbital.conversions import state_to_elements, elements_to_state


MU = EARTH.mu


class TestValidationCElementConversion:
    """Independent verification of Cartesian <-> Keplerian element conversions."""

    def test_roundtrip_circular_equatorial(self):
        """Circular equatorial: i=0, e=0."""
        r_in = np.array([7_000_000.0, 0.0, 0.0])
        v_circ = math.sqrt(MU / 7_000_000.0)
        v_in = np.array([0.0, v_circ, 0.0])

        oe = state_to_elements(r_in, v_in, MU)
        r_out, v_out = elements_to_state(oe)

        np.testing.assert_allclose(r_out, r_in, atol=1e-3)
        np.testing.assert_allclose(v_out, v_in, atol=1e-6)

    def test_roundtrip_elliptic_equatorial(self):
        """Elliptic equatorial: i=0, e=0.3."""
        r_in = np.array([7_000_000.0, 0.0, 0.0])
        v_in = np.array([0.0, 8_500.0, 0.0])

        oe = state_to_elements(r_in, v_in, MU)
        r_out, v_out = elements_to_state(oe)

        np.testing.assert_allclose(r_out, r_in, atol=1e-3)
        np.testing.assert_allclose(v_out, v_in, atol=1e-6)

    def test_roundtrip_circular_inclined_45deg(self):
        """Circular inclined: i=45 deg, e=0."""
        r_mag = 7_200_000.0
        v_circ = math.sqrt(MU / r_mag)
        i = math.radians(45.0)

        r_in = np.array([r_mag, 0.0, 0.0])
        v_in = np.array([0.0, v_circ * math.cos(i), v_circ * math.sin(i)])

        oe = state_to_elements(r_in, v_in, MU)
        r_out, v_out = elements_to_state(oe)

        # NOTE: This will expose whether the rotation matrix between PQW and ECI is correct!
        np.testing.assert_allclose(r_out, r_in, atol=1e-2)
        np.testing.assert_allclose(v_out, v_in, atol=1e-4)

    def test_roundtrip_inclined_elliptic(self):
        """General inclined elliptic orbit."""
        r_in = np.array([6_524_834.0, 6_862_875.0, 6_448_296.0])
        v_in = np.array([4_901.327, 5_533.756, -1_976.341])

        oe = state_to_elements(r_in, v_in, MU)
        r_out, v_out = elements_to_state(oe)

        np.testing.assert_allclose(r_out, r_in, atol=1e-2)
        np.testing.assert_allclose(v_out, v_in, atol=1e-4)

    def test_roundtrip_polar_orbit(self):
        """Polar orbit: i = 90 deg."""
        r_mag = 7_000_000.0
        v_circ = math.sqrt(MU / r_mag)

        r_in = np.array([r_mag, 0.0, 0.0])
        v_in = np.array([0.0, 0.0, v_circ])

        oe = state_to_elements(r_in, v_in, MU)
        assert oe.i == pytest.approx(math.pi / 2.0, abs=1e-6)

        r_out, v_out = elements_to_state(oe)
        np.testing.assert_allclose(r_out, r_in, atol=1e-2)
        np.testing.assert_allclose(v_out, v_in, atol=1e-4)

    def test_roundtrip_retrograde_orbit(self):
        """Retrograde orbit: i = 135 deg."""
        r_mag = 7_000_000.0
        v_circ = math.sqrt(MU / r_mag)
        i = math.radians(135.0)

        r_in = np.array([r_mag, 0.0, 0.0])
        v_in = np.array([0.0, v_circ * math.cos(i), v_circ * math.sin(i)])

        oe = state_to_elements(r_in, v_in, MU)
        assert oe.i == pytest.approx(i, abs=1e-6)

        r_out, v_out = elements_to_state(oe)
        np.testing.assert_allclose(r_out, r_in, atol=1e-2)
        np.testing.assert_allclose(v_out, v_in, atol=1e-4)
