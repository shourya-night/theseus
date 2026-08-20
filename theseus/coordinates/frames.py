"""
Reference-frame definitions.

Every vector quantity in THESEUS must be associated with a reference frame.
Mixing frames silently is forbidden.

Frames
------
ICRF
    International Celestial Reference Frame ≈ J2000 Earth-centred inertial (ECI).
    Origin: Earth centre-of-mass (or solar-system barycentre, depending on context).
    Axes: aligned with J2000 equator and equinox.
    Primary frame for orbital mechanics.

ECEF
    Earth-Centred Earth-Fixed (co-rotating with Earth).
    Origin: Earth centre-of-mass.
    Axes: x toward Greenwich meridian, z toward geographic north pole.

PERIFOCAL
    Perifocal (PQW) frame of an orbit.
    Origin: central body centre.
    P-axis: toward periapsis.  Q-axis: in orbital plane, 90° ahead.
    W-axis: along angular-momentum vector.

RTN
    Radial-Transverse-Normal (or RSW) frame.
    R: radial (outward from central body).
    T: transverse (along-track in orbital plane, ≈ velocity direction for circular orbits).
    N: normal (cross-track, completes right-hand triad = h-direction).

LVLH
    Local Vertical / Local Horizontal.
    Equivalent to RTN for the initial engine; listed separately for future
    distinction if attitude dynamics are added.
"""

from enum import Enum


class ReferenceFrame(Enum):
    """Supported reference frames."""
    ICRF = "ICRF"          # ≈ J2000 ECI
    ECEF = "ECEF"          # Earth-centred Earth-fixed
    PERIFOCAL = "PERIFOCAL"  # PQW
    RTN = "RTN"            # Radial-Transverse-Normal
    LVLH = "LVLH"          # Local Vertical / Local Horizontal
    BODY_CENTERED_INERTIAL = "BCI"  # generic body-centred inertial
