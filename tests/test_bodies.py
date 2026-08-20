"""Tests for celestial body catalog."""

import math
import pytest

from theseus.bodies import (
    CelestialBody, SUN, MERCURY, VENUS, EARTH, MOON, MARS,
    JUPITER, SATURN, URANUS, NEPTUNE, get_body, ALL_BODIES,
)
from theseus.constants.physical import G_VAL


class TestCelestialBodies:

    def test_catalog_completeness(self):
        """All 10 required bodies are present."""
        expected = {"Sun", "Mercury", "Venus", "Earth", "Moon",
                    "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}
        assert set(ALL_BODIES.keys()) == expected

    def test_get_body_case_insensitive(self):
        assert get_body("earth") is EARTH
        assert get_body("MARS") is MARS
        assert get_body("Moon") is MOON

    def test_unknown_body_raises(self):
        with pytest.raises(KeyError):
            get_body("Pluto")

    # -- Earth -------------------------------------------------------

    def test_earth_mu(self):
        """Earth GM = 3.986004418e14 m³/s² (GRS80)."""
        assert EARTH.mu == pytest.approx(3.986004418e14, rel=1e-8)

    def test_earth_radius(self):
        """Earth equatorial radius = 6 378 137 m (WGS84)."""
        assert EARTH.radius == 6_378_137.0

    def test_earth_j2(self):
        """Earth J2 ≈ 1.08263e-3."""
        assert EARTH.J2 == pytest.approx(1.08263e-3, rel=1e-3)

    def test_earth_surface_gravity(self):
        """g ≈ μ/R² ≈ 9.798 m/s² (equatorial, geometric only)."""
        g = EARTH.surface_gravity
        assert g == pytest.approx(9.798, abs=0.02)

    def test_earth_escape_velocity(self):
        """v_esc ≈ √(2μ/R) ≈ 11 180 m/s."""
        assert EARTH.escape_velocity == pytest.approx(11_180, rel=0.005)

    def test_earth_rotation_rate(self):
        """ω ≈ 7.2921e-5 rad/s."""
        assert EARTH.rotation_rate == pytest.approx(7.2921e-5, rel=1e-3)

    def test_earth_has_atmosphere(self):
        assert EARTH.atmosphere.has_atmosphere is True
        assert EARTH.atmosphere.surface_pressure == 101_325.0
        assert EARTH.atmosphere.surface_density == pytest.approx(1.225, abs=0.01)

    # -- Moon --------------------------------------------------------

    def test_moon_mu(self):
        assert MOON.mu == pytest.approx(4.9028695e12, rel=1e-5)

    def test_moon_parent(self):
        assert MOON.parent_name == "Earth"

    # -- Sun ---------------------------------------------------------

    def test_sun_mu(self):
        assert SUN.mu == pytest.approx(1.32712440018e20, rel=1e-9)

    # -- General: GM > 0 and radius > 0 for every body ---------------

    @pytest.mark.parametrize("body", ALL_BODIES.values(), ids=lambda b: b.name)
    def test_positive_mu(self, body: CelestialBody):
        assert body.mu > 0

    @pytest.mark.parametrize("body", ALL_BODIES.values(), ids=lambda b: b.name)
    def test_positive_radius(self, body: CelestialBody):
        assert body.radius > 0

    # -- Independent Planetary Physical Property Validation -----------

    @pytest.mark.parametrize(
        "body_name, expected_mass, expected_mu",
        [
            ("Earth", 5.9722e24, 3.986004418e14),
            ("Sun", 1.9885e30, 1.32712440018e20),
            ("Moon", 7.342e22, 4.9048695e12),
            ("Mars", 6.4171e23, 4.282837e13),
            ("Jupiter", 1.8982e27, 1.26686534e17),
        ],
    )
    def test_independent_mass_and_gm(self, body_name: str, expected_mass: float, expected_mu: float):
        """Validate celestial body mass and GM against independent IAU/NASA published values."""
        body = get_body(body_name)
        assert body.mu == pytest.approx(expected_mu, rel=1e-3)
        if body.mass is not None:
            assert body.mass == pytest.approx(expected_mass, rel=1e-3)
