"""
Coordinate-frame transformations.

Supported conversions:
    ICRF (ECI) ↔ ECEF         via Earth rotation angle
    Cartesian ↔ Spherical
    ECI → Perifocal            via orbital elements
    ECI → RTN                  via state vector

Mathematics
-----------
All rotations use standard right-hand rotation matrices:

    Rx(θ) = [[1, 0, 0], [0, cos θ, −sin θ], [0, sin θ, cos θ]]
    Ry(θ) = [[cos θ, 0, sin θ], [0, 1, 0], [−sin θ, 0, cos θ]]
    Rz(θ) = [[cos θ, −sin θ, 0], [sin θ, cos θ, 0], [0, 0, 1]]

Reference frames are documented in ``frames.py``.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Fundamental rotation matrices
# ---------------------------------------------------------------------------

def rotation_x(angle: float) -> np.ndarray:
    """3×3 rotation matrix about the x-axis (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  -s ],
        [0.0,  s,   c ],
    ])


def rotation_y(angle: float) -> np.ndarray:
    """3×3 rotation matrix about the y-axis (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [ c,  0.0,  s ],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c ],
    ])


def rotation_z(angle: float) -> np.ndarray:
    """3×3 rotation matrix about the z-axis (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [ c,  -s, 0.0],
        [ s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# ECI ↔ ECEF
# ---------------------------------------------------------------------------

def gmst_from_jd(jd: float) -> float:
    """
    Greenwich Mean Sidereal Time angle (rad) from Julian Date (UT1).

    Uses the IAU 1982 GMST model (adequate for moderate-precision work).

    Parameters
    ----------
    jd : float
        Julian Date (UT1 scale).

    Returns
    -------
    float
        GMST in radians, wrapped to [0, 2π).

    Reference
    ---------
    Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.,
    Eq. 3-47.
    """
    T = (jd - 2_451_545.0) / 36525.0
    # GMST in seconds of time
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    # Convert seconds-of-time to radians
    gmst_rad = (gmst_sec % 86400.0) / 86400.0 * 2.0 * math.pi
    return gmst_rad % (2.0 * math.pi)


def eci_to_ecef(position_eci: np.ndarray, jd: float) -> np.ndarray:
    """
    Rotate an ECI position vector to ECEF.

    Parameters
    ----------
    position_eci : np.ndarray
        [x, y, z] in ICRF/ECI (m).
    jd : float
        Julian Date (UT1).

    Returns
    -------
    np.ndarray
        [x, y, z] in ECEF (m).
    """
    theta = gmst_from_jd(jd)
    return rotation_z(-theta) @ position_eci


def ecef_to_eci(position_ecef: np.ndarray, jd: float) -> np.ndarray:
    """
    Rotate an ECEF position vector to ECI.

    Parameters
    ----------
    position_ecef : np.ndarray
        [x, y, z] in ECEF (m).
    jd : float
        Julian Date (UT1).

    Returns
    -------
    np.ndarray
        [x, y, z] in ICRF/ECI (m).
    """
    theta = gmst_from_jd(jd)
    return rotation_z(theta) @ position_ecef


# ---------------------------------------------------------------------------
# Cartesian ↔ Spherical
# ---------------------------------------------------------------------------

def cartesian_to_spherical(xyz: np.ndarray) -> tuple[float, float, float]:
    """
    Convert Cartesian [x, y, z] to spherical (r, θ, φ).

    Returns
    -------
    r : float
        Radial distance (same unit as input).
    theta : float
        Elevation / latitude angle (rad), measured from the xy-plane
        (−π/2 to +π/2).  Positive toward +z.
    phi : float
        Azimuth / longitude angle (rad), measured from +x toward +y
        (0 to 2π).
    """
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-30:
        return 0.0, 0.0, 0.0
    theta = math.asin(z / r)
    phi = math.atan2(y, x) % (2.0 * math.pi)
    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical (r, θ, φ) to Cartesian [x, y, z].

    Parameters
    ----------
    r : float
        Radial distance.
    theta : float
        Elevation angle (rad).
    phi : float
        Azimuth angle (rad).
    """
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    return np.array([r * ct * cp, r * ct * sp, r * st])


# ---------------------------------------------------------------------------
# Perifocal (PQW) ↔ ECI
# ---------------------------------------------------------------------------

def perifocal_to_eci_matrix(raan: float, inc: float, argp: float) -> np.ndarray:
    """
    Rotation matrix from perifocal (PQW) to ECI (IJK).

    R = Rz(Ω) · Rx(i) · Rz(ω)

    Parameters
    ----------
    raan : float   Ω — Right ascension of ascending node (rad).
    inc  : float   i — Inclination (rad).
    argp : float   ω — Argument of periapsis (rad).

    Returns
    -------
    np.ndarray
        3×3 rotation matrix whose columns are P̂, Q̂, Ŵ in ECI coordinates.
    """
    return rotation_z(raan) @ rotation_x(inc) @ rotation_z(argp)


def eci_to_perifocal_matrix(raan: float, inc: float, argp: float) -> np.ndarray:
    """Rotation matrix from ECI to perifocal.  Transpose of perifocal_to_eci_matrix."""
    return perifocal_to_eci_matrix(raan, inc, argp).T


# ---------------------------------------------------------------------------
# RTN frame from state vector
# ---------------------------------------------------------------------------

def eci_to_rtn_matrix(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Rotation matrix from ECI to RTN (radial-transverse-normal).

    R̂ = r / |r|            (radial, outward)
    N̂ = (r × v) / |r × v|  (normal, orbit-normal / angular-momentum direction)
    T̂ = N̂ × R̂              (transverse, roughly along-track)

    Parameters
    ----------
    position : np.ndarray  ECI position (m).
    velocity : np.ndarray  ECI velocity (m/s).

    Returns
    -------
    np.ndarray
        3×3 matrix whose rows are R̂, T̂, N̂ in ECI components.
        Multiply ``M @ vec_eci`` to get ``vec_rtn``.
    """
    r_hat = position / np.linalg.norm(position)
    h = np.cross(position, velocity)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([r_hat, t_hat, n_hat])  # rows = RTN axes
