"""
Conversions between Cartesian state vectors and classical orbital elements.

Mathematics
-----------
State → Elements (Vallado Algorithm 9):
    h  = r × v                       angular momentum
    n  = ẑ × h                       node vector
    e_vec = ((v²−μ/r)r − (r·v)v)/μ  eccentricity vector
    e  = |e_vec|
    ε  = v²/2 − μ/r                 specific energy
    a  = −μ/(2ε)                    semi-major axis
    i  = arccos(h_z / |h|)          inclination
    Ω  = arccos(n_x / |n|)         RAAN  (adjust quadrant by n_y)
    ω  = arccos(n̂·ê)              arg periapsis (adjust by e_z)
    ν  = arccos(ê·r̂)              true anomaly (adjust by r·v)

Elements → State:
    Compute r, v in perifocal frame, then rotate to ECI.

Edge cases
----------
* Circular orbits (e ≈ 0):  ω is undefined; use ω = 0 and report
  argument of latitude u = ω + ν.
* Equatorial orbits (i ≈ 0 or π):  Ω is undefined; use Ω = 0 and
  report longitude of periapsis ω̃ = Ω + ω.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from theseus.core.trace import CalculationTrace, TraceContext
from theseus.orbital.elements import OrbitalElements

# Threshold below which an orbit is treated as circular / equatorial
_ECC_TOL = 1e-10
_INC_TOL = 1e-10  # rad


def state_to_elements(
    r: np.ndarray,
    v: np.ndarray,
    mu: float,
    *,
    trace: bool = False,
) -> OrbitalElements:
    """
    Convert Cartesian state vector to classical orbital elements.

    Parameters
    ----------
    r : np.ndarray   Position vector (m), shape (3,).
    v : np.ndarray   Velocity vector (m/s), shape (3,).
    mu : float       Gravitational parameter (m³/s²).
    trace : bool     If True, emit a CalculationTrace.

    Returns
    -------
    OrbitalElements

    Reference frame
    ---------------
    Input must be in an inertial frame (ECI / ICRF).
    The z-axis defines the reference plane (equator).
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))

    # --- angular momentum ---
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))

    # --- node vector ---
    z_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(z_hat, h_vec)
    n_mag = float(np.linalg.norm(n_vec))

    # --- eccentricity vector ---
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))

    # --- specific energy ---
    energy = 0.5 * v_mag**2 - mu / r_mag

    # --- semi-major axis ---
    if abs(1.0 - e) < 1e-12:
        # Parabolic — a is infinite; use semi-latus rectum instead
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # --- inclination ---
    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag)))

    # --- RAAN (Ω) ---
    if n_mag > _INC_TOL:
        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n_mag)))
        if n_vec[1] < 0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0  # equatorial orbit — Ω undefined, set to 0

    # --- argument of periapsis (ω) ---
    if e > _ECC_TOL and n_mag > _INC_TOL:
        n_hat = n_vec / n_mag
        e_hat = e_vec / e
        cos_argp = max(-1.0, min(1.0, float(np.dot(n_hat, e_hat))))
        argp = math.acos(cos_argp)
        if e_vec[2] < 0:
            argp = 2.0 * math.pi - argp
    elif e > _ECC_TOL:
        # Equatorial: use longitude of periapsis (account for retrograde equatorial)
        if h_vec[2] < 0:
            argp = math.atan2(-e_vec[1], e_vec[0])
        else:
            argp = math.atan2(e_vec[1], e_vec[0])
        if argp < 0:
            argp += 2.0 * math.pi
    else:
        argp = 0.0  # circular — ω undefined

    # --- true anomaly (ν) ---
    if e > _ECC_TOL:
        e_hat = e_vec / e
        r_hat = r / r_mag
        cos_nu = max(-1.0, min(1.0, float(np.dot(e_hat, r_hat))))
        nu = math.acos(cos_nu)
        if np.dot(r, v) < 0:
            nu = 2.0 * math.pi - nu
    elif n_mag > _INC_TOL:
        # Circular, inclined: use argument of latitude u = ω + ν
        n_hat = n_vec / n_mag
        r_hat = r / r_mag
        cos_u = max(-1.0, min(1.0, float(np.dot(n_hat, r_hat))))
        u = math.acos(cos_u)
        if r[2] < 0:
            u = 2.0 * math.pi - u
        nu = u - argp
        if nu < 0:
            nu += 2.0 * math.pi
    else:
        # Circular equatorial: use true longitude
        if h_vec[2] < 0:
            nu = math.atan2(-r[1], r[0])
        else:
            nu = math.atan2(r[1], r[0])
        if nu < 0:
            nu += 2.0 * math.pi

    oe = OrbitalElements(a=a, e=e, i=inc, raan=raan, argp=argp, nu=nu, mu=mu)

    if trace:
        ct = CalculationTrace(
            operation="state_to_elements",
            equation="h = r×v; e_vec = ((v²−μ/r)r − (r·v)v)/μ; a = −μ/(2ε)",
            inputs={"r_m": r.tolist(), "v_m_s": v.tolist(), "mu": mu},
            result=oe.summary(),
        )
        ct.add_step("angular_momentum", h=h_mag)
        ct.add_step("eccentricity", e=e)
        ct.add_step("energy", epsilon=energy)
        ct.add_step("semi_major_axis", a=a)
        TraceContext.emit(ct)

    return oe


def elements_to_state(
    oe: OrbitalElements,
    *,
    trace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements to Cartesian state vector.

    Returns
    -------
    (r, v) : tuple[np.ndarray, np.ndarray]
        Position (m) and velocity (m/s) in ECI (ICRF).

    Mathematics
    -----------
    1. Compute r, v in perifocal (PQW) frame:
        r_pqw = (p / (1+e cos ν)) [cos ν, sin ν, 0]
        v_pqw = √(μ/p) [−sin ν, e + cos ν, 0]
    2. Rotate PQW → ECI via  R = Rz(−Ω) Rx(−i) Rz(−ω).
    """
    p = oe.semi_latus_rectum
    mu = oe.mu
    e = oe.e
    nu = oe.nu

    # Perifocal frame
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)

    r_mag_pqw = p / (1.0 + e * cos_nu)
    r_pqw = np.array([r_mag_pqw * cos_nu, r_mag_pqw * sin_nu, 0.0])

    sqrt_mu_over_p = math.sqrt(mu / p)
    v_pqw = np.array([
        -sqrt_mu_over_p * sin_nu,
        sqrt_mu_over_p * (e + cos_nu),
        0.0,
    ])

    # Rotation PQW → ECI
    from theseus.coordinates.transformations import perifocal_to_eci_matrix
    R = perifocal_to_eci_matrix(oe.raan, oe.i, oe.argp)

    r_eci = R @ r_pqw
    v_eci = R @ v_pqw

    if trace:
        ct = CalculationTrace(
            operation="elements_to_state",
            equation="r_PQW = p/(1+e cos ν) [cos ν, sin ν, 0]; "
                     "v_PQW = √(μ/p) [−sin ν, e+cos ν, 0]",
            inputs=oe.summary(),
            result={"r_m": r_eci.tolist(), "v_m_s": v_eci.tolist()},
        )
        TraceContext.emit(ct)

    return r_eci, v_eci
