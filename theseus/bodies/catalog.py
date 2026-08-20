"""
Catalog of celestial bodies with authoritative numerical values.

Sources
-------
GM (gravitational parameter):
    DE430/DE440 — JPL Planetary and Lunar Ephemerides
    Folkner et al., "The Planetary and Lunar Ephemerides DE430 and DE431",
    IPN Progress Report 42-196, 2014.

Radii:
    IAU Working Group on Cartographic Coordinates and Rotational Elements,
    "Report of the IAU Working Group", Celestial Mechanics and Dynamical
    Astronomy, 2018.

J2:
    GRS80 / EGM2008 for Earth; Jacobson et al. for other planets.

Rotation periods, axial tilts:
    JPL Solar System Dynamics fact sheets.
    https://ssd.jpl.nasa.gov/planets/phys_par.html

Atmospheres:
    NASA Planetary Fact Sheets.

All values are in SI: m, kg, s, rad, Pa, kg/m³.
"""

from __future__ import annotations

import math

from theseus.bodies.body import AtmosphereParams, CelestialBody
from theseus.constants.physical import G_VAL

# ===================================================================
# Sun
# ===================================================================
SUN = CelestialBody(
    name="Sun",
    mu=1.32712440018e20,          # m³/s²  (DE430)
    radius=6.9634e8,              # m      (nominal)
    mass=1.32712440018e20 / G_VAL,
    rotation_period=2.1642e6,     # s  ≈ 25.05 days (equatorial, sidereal)
    axial_tilt=math.radians(7.25),
    parent_name=None,
    data_source="DE430, IAU 2015",
)

# ===================================================================
# Mercury
# ===================================================================
MERCURY = CelestialBody(
    name="Mercury",
    mu=2.2032e13,
    radius=2_439_700.0,
    mass=2.2032e13 / G_VAL,
    rotation_period=5.0674e6,       # s ≈ 58.646 days
    axial_tilt=math.radians(0.034),
    J2=5.03e-5,
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Anderson et al. 1987 J2",
)

# ===================================================================
# Venus
# ===================================================================
VENUS = CelestialBody(
    name="Venus",
    mu=3.24859e14,
    radius=6_051_800.0,
    mass=3.24859e14 / G_VAL,
    rotation_period=-2.0997e7,      # s ≈ −243.025 days (retrograde)
    axial_tilt=math.radians(177.36),
    J2=4.458e-6,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=9.2e6,     # Pa (92 atm)
        surface_density=65.0,       # kg/m³
        scale_height=15_900.0,      # m
        composition={"CO2": 0.965, "N2": 0.035},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Konopliv et al. 1999 J2",
)

# ===================================================================
# Earth
# ===================================================================
EARTH = CelestialBody(
    name="Earth",
    mu=3.986004418e14,              # m³/s²  (GRS80 / EGM2008)
    radius=6_378_137.0,             # m      (WGS84 equatorial)
    radius_polar=6_356_752.3142,    # m      (WGS84 polar)
    mass=3.986004418e14 / G_VAL,
    rotation_period=86_164.0905,    # s      (sidereal day, IERS)
    axial_tilt=math.radians(23.4393),
    J2=1.08263e-3,                  # GRS80 / EGM96
    J3=-2.5327e-6,                  # EGM96
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=101_325.0,
        surface_density=1.225,
        scale_height=8_500.0,
        composition={"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093},
    ),
    parent_name="Sun",
    data_source="GRS80, WGS84, EGM2008, IERS Conventions 2010",
)

# ===================================================================
# Moon
# ===================================================================
MOON = CelestialBody(
    name="Moon",
    mu=4.9028695e12,
    radius=1_737_400.0,
    mass=4.9028695e12 / G_VAL,
    rotation_period=2.3606e6,       # s ≈ 27.322 days (synchronous)
    axial_tilt=math.radians(6.687),
    J2=2.033e-4,
    parent_name="Earth",
    data_source="DE430, IAU 2018 radii, Konopliv et al. 2001 J2",
)

# ===================================================================
# Mars
# ===================================================================
MARS = CelestialBody(
    name="Mars",
    mu=4.282837e13,
    radius=3_396_200.0,
    radius_polar=3_376_200.0,
    mass=4.282837e13 / G_VAL,
    rotation_period=88_642.663,     # s ≈ 24h 37m (sidereal)
    axial_tilt=math.radians(25.19),
    J2=1.9555e-3,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=636.0,       # Pa (mean)
        surface_density=0.020,        # kg/m³
        scale_height=11_100.0,
        composition={"CO2": 0.9532, "N2": 0.027, "Ar": 0.016},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Konopliv et al. 2016 J2",
)

# ===================================================================
# Jupiter
# ===================================================================
JUPITER = CelestialBody(
    name="Jupiter",
    mu=1.26686534e17,
    radius=71_492_000.0,
    radius_polar=66_854_000.0,
    mass=1.26686534e17 / G_VAL,
    rotation_period=35_730.0,       # s ≈ 9.925 h (System III)
    axial_tilt=math.radians(3.13),
    J2=1.4736e-2,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=None,          # No solid surface
        composition={"H2": 0.898, "He": 0.102},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Jacobson 2003 J2",
)

# ===================================================================
# Saturn
# ===================================================================
SATURN = CelestialBody(
    name="Saturn",
    mu=3.7931187e16,
    radius=60_268_000.0,
    radius_polar=54_364_000.0,
    mass=3.7931187e16 / G_VAL,
    rotation_period=38_361.6,       # s ≈ 10.656 h
    axial_tilt=math.radians(26.73),
    J2=1.6298e-2,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=None,
        composition={"H2": 0.963, "He": 0.0325},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Jacobson 2006 J2",
)

# ===================================================================
# Uranus
# ===================================================================
URANUS = CelestialBody(
    name="Uranus",
    mu=5.793939e15,
    radius=25_559_000.0,
    radius_polar=24_973_000.0,
    mass=5.793939e15 / G_VAL,
    rotation_period=-62_063.7,      # s ≈ −17.24 h (retrograde)
    axial_tilt=math.radians(97.77),
    J2=3.343e-3,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=None,
        composition={"H2": 0.825, "He": 0.152, "CH4": 0.023},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Jacobson 2014 J2",
)

# ===================================================================
# Neptune
# ===================================================================
NEPTUNE = CelestialBody(
    name="Neptune",
    mu=6.836529e15,
    radius=24_764_000.0,
    radius_polar=24_341_000.0,
    mass=6.836529e15 / G_VAL,
    rotation_period=57_996.0,       # s ≈ 16.11 h
    axial_tilt=math.radians(28.32),
    J2=3.411e-3,
    atmosphere=AtmosphereParams(
        has_atmosphere=True,
        surface_pressure=None,
        composition={"H2": 0.80, "He": 0.19, "CH4": 0.015},
    ),
    parent_name="Sun",
    data_source="DE430, IAU 2018 radii, Jacobson 2009 J2",
)

# ===================================================================
# Catalog look-up
# ===================================================================

ALL_BODIES: dict[str, CelestialBody] = {
    b.name: b for b in [
        SUN, MERCURY, VENUS, EARTH, MOON, MARS,
        JUPITER, SATURN, URANUS, NEPTUNE,
    ]
}


def get_body(name: str) -> CelestialBody:
    """
    Look up a celestial body by name (case-insensitive).

    Raises
    ------
    KeyError
        If the body is not in the catalog.
    """
    key = name.strip().capitalize()
    # Handle multi-word edge cases
    for k, v in ALL_BODIES.items():
        if k.lower() == name.strip().lower():
            return v
    raise KeyError(f"Unknown celestial body: {name!r}")
