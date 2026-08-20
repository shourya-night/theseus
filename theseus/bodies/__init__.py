"""Celestial body definitions and catalog."""

from theseus.bodies.body import CelestialBody  # noqa: F401
from theseus.bodies.catalog import (  # noqa: F401
    SUN, MERCURY, VENUS, EARTH, MOON, MARS,
    JUPITER, SATURN, URANUS, NEPTUNE,
    get_body, ALL_BODIES,
)
