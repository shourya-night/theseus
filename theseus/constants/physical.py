"""
Physical constants used throughout THESEUS.

All values in SI units (m, kg, s, N, rad, W, K, J, Pa).

Sources:
    CODATA 2018: NIST SP 961 (May 2019)
    IAU 2015:    IAU 2015 Resolution B3 — Nominal Solar and Planetary Values
    IERS:        IERS Conventions (2010), IERS Technical Note No. 36
    WGS84:       NIMA TR8350.2, Third Edition, 4 July 1997
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PhysicalConstant:
    """A physical constant with metadata for traceability."""
    symbol: str
    value: float
    unit: str
    source: str
    uncertainty: Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# ---------------------------------------------------------------------------

G = PhysicalConstant(
    symbol="G",
    value=6.67430e-11,
    unit="m^3 kg^-1 s^-2",
    source="CODATA 2018",
    uncertainty=1.5e-15,
    description="Newtonian constant of gravitation",
)

SPEED_OF_LIGHT = PhysicalConstant(
    symbol="c",
    value=299_792_458.0,
    unit="m s^-1",
    source="SI definition (exact)",
    uncertainty=0.0,
    description="Speed of light in vacuum",
)

STEFAN_BOLTZMANN = PhysicalConstant(
    symbol="sigma",
    value=5.670374419e-8,
    unit="W m^-2 K^-4",
    source="CODATA 2018 (derived, exact in 2019 SI)",
    uncertainty=0.0,
    description="Stefan-Boltzmann constant",
)

BOLTZMANN = PhysicalConstant(
    symbol="k_B",
    value=1.380649e-23,
    unit="J K^-1",
    source="SI definition (exact)",
    uncertainty=0.0,
    description="Boltzmann constant",
)

PLANCK = PhysicalConstant(
    symbol="h",
    value=6.62607015e-34,
    unit="J s",
    source="SI definition (exact)",
    uncertainty=0.0,
    description="Planck constant",
)

# ---------------------------------------------------------------------------
# Standard / reference values
# ---------------------------------------------------------------------------

STANDARD_GRAVITY = PhysicalConstant(
    symbol="g_0",
    value=9.80665,
    unit="m s^-2",
    source="ISO 80000-3:2006 (exact by definition)",
    uncertainty=0.0,
    description="Standard acceleration of gravity",
)

STANDARD_ATMOSPHERE = PhysicalConstant(
    symbol="atm",
    value=101_325.0,
    unit="Pa",
    source="ISO 2533:1975 (exact by definition)",
    uncertainty=0.0,
    description="Standard atmospheric pressure at sea level",
)

# ---------------------------------------------------------------------------
# Solar constants (IAU 2015 Resolution B3)
# ---------------------------------------------------------------------------

SOLAR_LUMINOSITY = PhysicalConstant(
    symbol="L_sun",
    value=3.828e26,
    unit="W",
    source="IAU 2015 Resolution B3 (nominal)",
    description="Nominal total solar irradiance luminosity",
)

SOLAR_IRRADIANCE_1AU = PhysicalConstant(
    symbol="S_0",
    value=1361.0,
    unit="W m^-2",
    source="IAU 2015 / Kopp & Lean 2011 (Total Solar Irradiance)",
    uncertainty=0.5,
    description="Total solar irradiance at 1 AU (mean)",
)

# ---------------------------------------------------------------------------
# Astronomical constants
# ---------------------------------------------------------------------------

ASTRONOMICAL_UNIT = PhysicalConstant(
    symbol="AU",
    value=149_597_870_700.0,
    unit="m",
    source="IAU 2012 Resolution B2 (exact)",
    uncertainty=0.0,
    description="Astronomical unit",
)

# ---------------------------------------------------------------------------
# Air / atmosphere
# ---------------------------------------------------------------------------

MOLAR_MASS_DRY_AIR = PhysicalConstant(
    symbol="M_air",
    value=0.0289644,
    unit="kg mol^-1",
    source="US Standard Atmosphere 1976",
    description="Mean molar mass of dry air",
)

UNIVERSAL_GAS_CONSTANT = PhysicalConstant(
    symbol="R_star",
    value=8.314462618,
    unit="J mol^-1 K^-1",
    source="CODATA 2018 (exact in 2019 SI)",
    uncertainty=0.0,
    description="Universal (molar) gas constant",
)


# ---------------------------------------------------------------------------
# Convenience: bare numeric values for hot-path calculations
# ---------------------------------------------------------------------------

G_VAL: float = G.value
C_VAL: float = SPEED_OF_LIGHT.value
G0_VAL: float = STANDARD_GRAVITY.value
AU_VAL: float = ASTRONOMICAL_UNIT.value
SIGMA_SB_VAL: float = STEFAN_BOLTZMANN.value
L_SUN_VAL: float = SOLAR_LUMINOSITY.value
S0_VAL: float = SOLAR_IRRADIANCE_1AU.value
R_GAS_VAL: float = UNIVERSAL_GAS_CONSTANT.value
M_AIR_VAL: float = MOLAR_MASS_DRY_AIR.value
