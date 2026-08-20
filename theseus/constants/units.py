"""
Explicit unit conversion utilities.

THESEUS uses SI units internally:
    Length:   metres  (m)
    Mass:     kilograms  (kg)
    Time:     seconds  (s)
    Force:    Newtons  (N)
    Angle:    radians  (rad)

All conversions are explicit function calls — never silently mix units.
"""

import math

# ---------------------------------------------------------------------------
# Conversion factors (exact where possible)
# ---------------------------------------------------------------------------

# Length
KM_PER_M: float = 1e-3
M_PER_KM: float = 1e3

# Angle
DEG_PER_RAD: float = 180.0 / math.pi
RAD_PER_DEG: float = math.pi / 180.0

# Time
SECONDS_PER_MINUTE: float = 60.0
SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86400.0
SECONDS_PER_JULIAN_YEAR: float = 365.25 * SECONDS_PER_DAY
DAYS_PER_JULIAN_CENTURY: float = 36525.0

# Astronomical unit (m) — imported from physical for convenience
from theseus.constants.physical import AU_VAL as _AU  # noqa: E402
M_PER_AU: float = _AU
AU_PER_M: float = 1.0 / _AU


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def km_to_m(km: float) -> float:
    """Convert kilometres to metres."""
    return km * M_PER_KM


def m_to_km(m: float) -> float:
    """Convert metres to kilometres."""
    return m * KM_PER_M


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * RAD_PER_DEG


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * DEG_PER_RAD


def hours_to_seconds(hours: float) -> float:
    """Convert hours to seconds."""
    return hours * SECONDS_PER_HOUR


def seconds_to_hours(seconds: float) -> float:
    """Convert seconds to hours."""
    return seconds / SECONDS_PER_HOUR


def days_to_seconds(days: float) -> float:
    """Convert days to seconds."""
    return days * SECONDS_PER_DAY


def seconds_to_days(seconds: float) -> float:
    """Convert seconds to days."""
    return seconds / SECONDS_PER_DAY


def au_to_m(au: float) -> float:
    """Convert astronomical units to metres."""
    return au * M_PER_AU


def m_to_au(m: float) -> float:
    """Convert metres to astronomical units."""
    return m * AU_PER_M
