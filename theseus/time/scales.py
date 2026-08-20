"""
Astronomical time-scale definitions.

References
----------
IERS Conventions (2010), Chapter 10.
USNO Circular 179, Kaplan (2005).
"""

from enum import Enum


class TimeScale(Enum):
    """
    Supported astronomical time scales.

    UTC : Coordinated Universal Time.  Civil time; includes leap seconds.
    TT  : Terrestrial Time.  Uniform time for Earth-surface observations.
          TT = TAI + 32.184 s.
    TAI : International Atomic Time.  Continuous atomic time scale.
    TDB : Barycentric Dynamical Time.  The independent variable of solar-
          system ephemerides (DE430/DE440).  Quasi-uniform with TT
          (periodic terms < 1.7 ms).

    For orbital mechanics we primarily use **TDB** (ephemeris calculations)
    and **UTC** (human-facing timestamps).
    """
    UTC = "UTC"
    TT = "TT"
    TAI = "TAI"
    TDB = "TDB"
